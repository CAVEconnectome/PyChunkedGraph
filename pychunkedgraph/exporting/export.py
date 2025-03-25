import numpy as np
import cloudvolume
import itertools
import dill

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend.utils import serializers, column_keys
from multiwrapper import multiprocessing_utils as mu


def get_sv_to_root_id_mapping_chunk(cg, chunk_coords, vol=None):
    """ Acquires a svid -> rootid dictionary for a chunk

    :param cg: chunkedgraph instance
    :param chunk_coords: list
    :return: dict
    """
    sv_to_root_mapping = {}

    chunk_coords = np.array(chunk_coords, dtype=int)

    if np.any((chunk_coords % cg.chunk_size) != 0):
        raise Exception("Chunk coords have to match a chunk corner exactly")

    chunk_coords = chunk_coords / cg.chunk_size
    chunk_coords = chunk_coords.astype(int)
    bb = np.array([chunk_coords, chunk_coords + 1], dtype=int)

    remapped_vol = None
    vol_shape = None

    if vol is not None:
        vol_shape = vol.shape
        vol = vol.flatten()
        remapped_vol = np.zeros_like(vol)

    atomic_rows = cg.range_read_chunk(layer=1, x=chunk_coords[0],
                                      y=chunk_coords[1], z=chunk_coords[2],
                                      columns=column_keys.Hierarchy.Parent)
    for atomic_id in atomic_rows.keys():
        # Check if already found the root for this supervoxel
        if atomic_id in sv_to_root_mapping:
            continue

        # Find root
        root_id = cg.get_root(atomic_id)
        sv_to_root_mapping[atomic_id] = root_id

        # Add atomic children of root_id
        atomic_ids = cg.get_subgraph_nodes(root_id, bounding_box=bb,
                                           bb_is_coordinate=False)
        sv_to_root_mapping.update(dict(zip(atomic_ids,
                                           [root_id] * len(atomic_ids))))

        if remapped_vol is not None:
            remapped_vol[np.in1d(vol, atomic_ids)] = root_id

    if remapped_vol is not None:
        remapped_vol = remapped_vol.reshape(vol_shape)
        return sv_to_root_mapping, remapped_vol
    else:
        return sv_to_root_mapping


def _write_flat_segmentation_thread(args):
    """ Helper of write_flat_segmentation """
    cg_info, start_block, end_block, from_url, to_url, mip = args

    assert 'segmentation' in to_url
    assert 'svenmd' in to_url

    from_cv = cloudvolume.CloudVolume(from_url, mip=mip)
    to_cv = cloudvolume.CloudVolume(to_url, mip=mip)

    cg = chunkedgraph.ChunkedGraph(table_id=cg_info["table_id"],
                                   instance_id=cg_info["instance_id"],
                                   project_id=cg_info["project_id"],
                                   credentials=cg_info["credentials"])

    for block_z in range(start_block[2], end_block[2]):
        z_start = block_z * cg.chunk_size[2]
        z_end = (block_z + 1) * cg.chunk_size[2]
        for block_y in range(start_block[1], end_block[1]):
            y_start = block_y * cg.chunk_size[1]
            y_end = (block_y + 1) * cg.chunk_size[1]
            for block_x in range(start_block[0], end_block[0]):
                x_start = block_x * cg.chunk_size[0]
                x_end = (block_x + 1) * cg.chunk_size[0]

                block = from_cv[x_start: x_end, y_start: y_end, z_start: z_end]

                _, remapped_block = get_sv_to_root_id_mapping_chunk(cg, [x_start, y_start, z_start], block)

                to_cv[x_start: x_end, y_start: y_end, z_start: z_end] = remapped_block


def write_flat_segmentation(cg, dataset_name, bounding_box=None, block_factor=2,
                            n_threads=1, mip=0):
    """ Applies the mapping in the chunkedgraph to the supervoxels to create
        a flattened segmentation

    :param cg: chunkedgraph instance
    :param dataset_name: str
    :param bounding_box: np.array
    :param block_factor: int
    :param n_threads: int
    :param mip: int
    :return: bool
    """

    if dataset_name == "pinky":
        from_url = "gs://neuroglancer/svenmd/pinky40_v11/watershed/"
        to_url = "gs://neuroglancer/svenmd/pinky40_v11/segmentation/"
    elif dataset_name == "basil":
        from_url = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed/"
        to_url = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/segmentation/"
    else:
        raise Exception("Dataset unknown")

    from_cv = cloudvolume.CloudVolume(from_url, mip=mip)

    dataset_bounding_box = np.array(from_cv.bounds.to_list())

    block_bounding_box_cg = \
        [np.floor(dataset_bounding_box[:3] / cg.chunk_size).astype(int),
         np.ceil(dataset_bounding_box[3:] / cg.chunk_size).astype(int)]

    if bounding_box is not None:
        bounding_box_cg = \
            [np.floor(bounding_box[0] / cg.chunk_size).astype(int),
             np.ceil(bounding_box[1] / cg.chunk_size).astype(int)]

        m = block_bounding_box_cg[0] < bounding_box_cg[0]
        block_bounding_box_cg[0][m] = bounding_box_cg[0][m]

        m = block_bounding_box_cg[1] > bounding_box_cg[1]
        block_bounding_box_cg[1][m] = bounding_box_cg[1][m]

    block_iter = itertools.product(np.arange(block_bounding_box_cg[0][0],
                                             block_bounding_box_cg[1][0],
                                             block_factor),
                                   np.arange(block_bounding_box_cg[0][1],
                                             block_bounding_box_cg[1][1],
                                             block_factor),
                                   np.arange(block_bounding_box_cg[0][2],
                                             block_bounding_box_cg[1][2],
                                             block_factor))
    blocks = np.array(list(block_iter))

    cg_info = cg.get_serialized_info()

    multi_args = []
    for start_block in blocks:
        end_block = start_block + block_factor
        m = end_block > block_bounding_box_cg[1]
        end_block[m] = block_bounding_box_cg[1][m]

        multi_args.append([cg_info, start_block, end_block,
                           from_url, to_url, mip])

    # Run parallelizing
    if n_threads == 1:
        mu.multiprocess_func(_write_flat_segmentation_thread, multi_args,
                             n_threads=n_threads, verbose=True,
                             debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_write_flat_segmentation_thread, multi_args,
                                n_threads=n_threads)


def export_changelog(cg, path=None):
    """ Exports all changes to binary dill file

    :param cg: ChunkedGraph instance
    :param path: str
    :return: bool
    """

    operations_d = cg.read_log_rows(np.arange(0, cg.get_max_operation_id()),
                                    dtype=np.uint64)

    if path is not None:
        with open(path, "wb") as f:
            dill.dump(operations_d, f)
    else:
        return operations_d


def load_changelog(path):
    """ Loads stored changelog

    :param path: str
    :return:
    """

    with open(path, "rb") as f:
        operations = dill.load(f)

    # Dill can marshall the `serializer` functions used for each column, but their address
    # won't match anymore, which breaks the hash lookup for `_Column`s. Hence, we simply create
    # new `_Column`s from the old `family_id` and `key`
    for operation_id, column_dict in operations.items():
        operations[operation_id] = \
            {column_keys.from_key(k.family_id, k.key): v for (k, v) in column_dict.items()}

    return operations


def get_log_diff(log_old, log_new):
    """ Computes a simple difference between two logs

    :param log_old: dict
    :param log_new: dict
    :return: dict
    """
    log = log_new.copy()

    for k in log_old:
        del log[k]

    return log
