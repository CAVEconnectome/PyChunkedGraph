import sys
import os
import numpy as np
import itertools

import cloudvolume
from igneous.tasks import MeshTask

sys.path.insert(0, os.path.join(sys.path[0], '../..'))
os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend import multiprocessing_utils as mu


def chunk_mesh_task(cg, chunk_id, ws_path, mip=3):
    """ Computes the meshes for a single chunk

    :param cg: ChunkedGraph instance
    :param chunk_id: int
    :param ws_path: str
    :param mip: int
    """
    sv_to_node_mapping = {}
    for seg_id in range(1, cg.get_max_node_id(chunk_id) + np.uint64(1)):
        node_id = cg.get_node_id(np.uint64(seg_id), chunk_id)

        atomic_ids = cg.get_subgraph(node_id, verbose=False)
        
        sv_to_node_mapping.update(dict(zip(atomic_ids,
                                           [node_id] * len(atomic_ids))))

    if len(sv_to_node_mapping):
        return

    mesh_block_shape = cg.chunk_size // np.array([2**mip, 2**mip, 1])
    chunk_offset = cg.get_chunk_coordinates(chunk_id) * mesh_block_shape

    task = MeshTask(
        mesh_block_shape,
        chunk_offset,
        ws_path,
        mip=mip,
        simplification_factor=10,
        max_simplification_error=1,
        remap_table=sv_to_node_mapping,
        generate_manifests=True,
        low_padding=1,
        high_padding=1
    )
    task.execute()


def _mesh_dataset_thread(args):
    """ Helper for mesh_dataset """
    cg_info, start_block, end_block, ws_path, mip, layer = args

    cg = chunkedgraph.ChunkedGraph(table_id=cg_info["table_id"],
                                   instance_id=cg_info["instance_id"],
                                   project_id=cg_info["project_id"])

    for block_z in range(start_block[2], end_block[2]):
        for block_y in range(start_block[1], end_block[1]):
            for block_x in range(start_block[0], end_block[0]):
                chunk_id = cg.get_chunk_id(x=block_x, y=block_y, z=block_z,
                                           layer=layer)

                chunk_mesh_task(cg=cg, chunk_id=chunk_id, ws_path=ws_path,
                                mip=mip)


def mesh_dataset(table_id, layer, mip=3, bounding_box=None, block_factor=2,
                 n_threads=1):
    """ Computes meshes for a single layer in a chunkedgraph

    :param table_id: str
    :param mip: int
        mip used for meshes
    :param bounding_box: 2 x 3 array or None
        [[x_low, y_low, z_low], [x_high, y_high, z_high]]
    :param block_factor: int
        scales workload per thread (goes with block_factor^3)
    :param n_threads: int
    """

    if "pinky" in table_id:
        ws_path = "gs://neuroglancer/svenmd/pinky40_v11/watershed/"
    elif "basil" in table_id:
        ws_path = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed/"
    else:
        raise Exception("Dataset unknown")

    cg = chunkedgraph.ChunkedGraph(table_id=table_id)

    ws_cv_mip1 = cloudvolume.CloudVolume(ws_path, mip=1)
    dataset_bounding_box = np.array(ws_cv_mip1.bounds.to_list())

    block_bounding_box_cg = \
        [np.floor(dataset_bounding_box[:3] / cg.chunk_size).astype(np.int),
         np.ceil(dataset_bounding_box[3:] / cg.chunk_size).astype(np.int)]

    if bounding_box is not None:
        bounding_box_cg = \
            [np.floor(bounding_box[0] / cg.chunk_size).astype(np.int),
             np.ceil(bounding_box[1] / cg.chunk_size).astype(np.int)]

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
    del(cg_info['credentials'])

    multi_args = []
    for start_block in blocks:
        end_block = start_block + block_factor
        m = end_block > block_bounding_box_cg[1]
        end_block[m] = block_bounding_box_cg[1][m]

        multi_args.append([cg_info, start_block, end_block, ws_path, mip,
                           layer])

    # Run multiprocessing
    if n_threads == 1:
        mu.multiprocess_func(_mesh_dataset_thread, multi_args,
                             n_threads=n_threads, verbose=True,
                             debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_mesh_dataset_thread, multi_args,
                                n_threads=n_threads)


def mesh_dataset_all_layers(table_id, excempt_layers=[1], mip=3,
                            bounding_box=None, block_factor=2, n_threads=1):
    """ Computes meshes for all layers in a chunkedgraph

    :param table_id: str
    :param excempt_layers: list of ints
        list layer ids for which meshes should not be computed
    :param mip: int
        mip used for meshes
    :param bounding_box: 2 x 3 array or None
        [[x_low, y_low, z_low], [x_high, y_high, z_high]]
    :param block_factor: int
        scales workload per thread (goes with block_factor^3)
    :param n_threads: int
    """
    cg = chunkedgraph.ChunkedGraph(table_id=table_id)

    for layer in range(1, cg.n_layers):
        if layer in excempt_layers:
            continue

        mesh_dataset(table_id, layer, mip=mip, bounding_box=bounding_box,
                     block_factor=block_factor, n_threads=n_threads)
