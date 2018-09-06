import sys
import os
import numpy as np
import json

from cloudvolume import CloudVolume, Storage
from functools import lru_cache
from igneous.tasks import MeshTask

sys.path.insert(0, os.path.join(sys.path[0], '../..'))
os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"

from pychunkedgraph.backend.chunkedgraph import ChunkedGraph  # noqa


@lru_cache(maxsize=None)
def get_segmentation_info(cg: ChunkedGraph) -> dict:
    return CloudVolume(cg.cv_path).info


def get_mesh_block_shape(cg: ChunkedGraph, graphlayer: int, source_mip: int) -> np.ndarray:
    """
    Calculate the dimensions of a segmentation block at `source_mip` that covers
    the same region as a ChunkedGraph chunk at layer `graphlayer`.
    """
    info = get_segmentation_info(cg)

    # Segmentation is not always uniformly downsampled in all directions.
    scale_0 = info['scales'][0]
    scale_mip = info['scales'][source_mip]
    distortion = np.floor_divide(scale_mip['resolution'], scale_0['resolution'])

    graphlayer_chunksize = cg.chunk_size * cg.fan_out ** np.max([0, graphlayer - 2])

    return np.floor_divide(graphlayer_chunksize, distortion)


def get_sv_to_node_mapping(cg, chunk_id):
    """ Reads sv_id -> root_id mapping for a chunk from the chunkedgraph

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :return: dict
    """

    layer = cg.get_chunk_layer(chunk_id)

    sv_to_node_mapping = {}

    if layer > 1:
        # Mapping a chunk containing agglomeration IDs - need to retrieve
        # atomic IDs within the chunk, as well as all connected or unbreakable
        # cross_chunk_edges to adjacent chunks.
        for seg_id in range(1, cg.get_max_node_id(chunk_id) + np.uint64(1)):
            node_id = cg.get_node_id(np.uint64(seg_id), chunk_id)

            atomic_ids = cg.get_subgraph(node_id, verbose=False)
            cross_chunk_edges = cg.get_atomic_cross_edge_dict(
                node_id,
                deserialize_node_ids=True)

            # Collects every 2nd element of every cross_edge list in the
            # dictionary (the atomic partner IDs in adjacent chunks)
            cross_chunk_partners = {n: node_id for
                                    arr in cross_chunk_edges.values() for
                                    n in arr[1::2]}

            sv_to_node_mapping.update(
                dict(zip(atomic_ids, [node_id] * len(atomic_ids))))
            sv_to_node_mapping.update(cross_chunk_partners)
    else:
        # Mapping a chunk containing supervoxels - need to retrieve
        # potential unbreakable counterparts for each supervoxel
        # (Look for atomic edges with infinite edge weight)
        x, y, z = cg.get_chunk_coordinates(chunk_id)
        seg_ids = cg.range_read_chunk(1, x, y, z)

        sv_to_node_mapping = {seg_id: seg_id for seg_id in
                              [np.uint64(x) for x in seg_ids.keys()]}

        for seg_id, data in seg_ids.items():
            partners = np.frombuffer(
                data.cells['0'][b'atomic_connected_partners'][0].value,
                dtype=np.uint64)
            affinities = np.frombuffer(
                data.cells['0'][b'atomic_connected_affinities'][0].value,
                dtype=np.float32)
            partners = partners[affinities == np.inf]
            sv_to_node_mapping.update(
                dict(zip(partners, [seg_id] * len(partners))))

    return sv_to_node_mapping


def chunk_mesh_task(sv_to_node_mapping, cg, chunk_id, cv_path,
                    cv_mesh_dir=None, mip=3):
    """ Computes the meshes for a single chunk

    :param get_sv_to_node_mapping: dict
    :param cg: ChunkedGraph instance
    :param chunk_id: int
    :param cv_path: str
    :param cv_mesh_dir: str or None
    :param mip: int
    """

    layer = cg.get_chunk_layer(chunk_id)

    if len(sv_to_node_mapping) == 0:
        print("Nothing to do", cg.get_chunk_coordinates(chunk_id))
        return
    # else:
        # print("Something to do", cg.get_chunk_coordinates(chunk_id))

    mesh_block_shape = get_mesh_block_shape(cg, layer, mip)

    chunk_offset = cg.get_chunk_coordinates(chunk_id) * mesh_block_shape

    task = MeshTask(
        mesh_block_shape,
        chunk_offset,
        cv_path,
        mip=mip,
        simplification_factor=999999,  # Simplify as much as possible ...
        max_simplification_error=40,    # ... staying below max error.
        remap_table=sv_to_node_mapping,
        generate_manifests=True,
        low_padding=0,                 # One voxel overlap to exactly line up
        high_padding=1,                # vertex boundaries.
        mesh_dir=cv_mesh_dir
    )
    task.execute()

    print("Layer %d -- finished:" % layer, cg.get_chunk_coordinates(chunk_id))


def mesh_single_component(node_id, cg, cv_path, cv_mesh_dir=None, mip=3):
    """ Computes the mesh for a single component

    :param node_id: uint64
    :param cg: chunkedgraph instance
    :param cv_path: str
    :param cv_mesh_dir: str
    :param mip: int
    """
    layer = cg.get_chunk_layer(node_id)

    atomic_ids = [node_id]
    if layer > 1:
        atomic_ids = cg.get_subgraph(node_id, verbose=False)

    if len(atomic_ids):
        sv_to_node_mapping = dict(zip(atomic_ids, [node_id] * len(atomic_ids)))
        chunk_mesh_task(sv_to_node_mapping, cg, cg.get_chunk_id(node_id),
                        cv_path, cv_mesh_dir=cv_mesh_dir, mip=mip)
    else:
        raise Exception("Could not find atomic nodes -- does this node id "
                        "exist?")


def _mesh_layer_thread(args):
    cg_info, start_block, end_block, cv_path, cv_mesh_dir, mip, layer = args

    cg = ChunkedGraph(table_id=cg_info["table_id"],
                      instance_id=cg_info["instance_id"],
                      project_id=cg_info["project_id"])

    for block_z in range(start_block[2], end_block[2]):
        for block_y in range(start_block[1], end_block[1]):
            for block_x in range(start_block[0], end_block[0]):
                chunk_id = cg.get_chunk_id(x=block_x, y=block_y, z=block_z,
                                           layer=layer)

                chunk_mesh_task(get_sv_to_node_mapping(cg, chunk_id),
                                cg=cg, chunk_id=chunk_id, cv_path=cv_path,
                                mip=mip, cv_mesh_dir=cv_mesh_dir)


def mesh_node_and_parents(node_id, cg, cv_path, cv_mesh_dir=None, mip=3,
                          highest_mesh_level=1, create_manifest_root=True,
                          lod=0):
    layer = cg.get_chunk_layer(node_id)

    parents = [node_id] + list(cg.get_all_parents(node_id))
    for i_layer in range(layer, highest_mesh_level + 1):
        mesh_single_component(parents[i_layer], cg=cg, cv_path=cv_path,
                              cv_mesh_dir=cv_mesh_dir, mip=mip)

    if create_manifest_root:
        with Storage(cv_path) as cv_storage:
            create_manifest_file(cg=cg, cv_storage=cv_storage,
                                 cv_mesh_dir=cv_mesh_dir, node_id=parents[-1],
                                 highest_mesh_level=highest_mesh_level,
                                 mip=mip, lod=lod)


def create_manifest_file(cg, cv_storage, cv_mesh_dir, node_id,
                         highest_mesh_level=1, mip=3, lod=0):
    """ Creates the manifest file for any node

    :param cg: ChunkedGraph instance
    :param cv_storage: Storage instance
    :param cv_mesh_dir: str
    :param node_id: uint64
    :param mip: int
    :param highest_mesh_level: int
    :param lod: int
    """
    highest_mesh_level_children = _find_highest_children_with_mesh(
        cg, cv_storage, cv_mesh_dir, [],
        cg.get_subgraph(node_id, stop_lvl=highest_mesh_level, verbose=False).tolist())

    print("%d -- number of children: %d" %
          (node_id, len(highest_mesh_level_children)))

    mesh_block_shape = get_mesh_block_shape(cg, highest_mesh_level, mip)

    frags = []
    for child_id in highest_mesh_level_children:
        lower_b = cg.get_chunk_coordinates(child_id) * mesh_block_shape
        upper_b = lower_b + mesh_block_shape

        lower_b = np.array(lower_b, dtype=np.int)
        upper_b = np.array(upper_b, dtype=np.int)

        bounds = '{}-{}_{}-{}_{}-{}'.format(lower_b[0], upper_b[0],
                                            lower_b[1], upper_b[1],
                                            lower_b[2], upper_b[2])

        frags.append('{}:{}:{}'.format(child_id, lod, bounds))

    cv_storage.put_file(file_path='{}/{}:{}'.format(cv_mesh_dir, node_id, lod),
                        content=json.dumps({"fragments": frags}),
                        content_type='application/json')


def _find_highest_children_with_mesh(cg, cv_storage, cv_mesh_dir,
                                     validated_children, test_children):
    if len(test_children) == 0:
        return validated_children

    next_child_id = test_children[0]
    del test_children[0]

    manifest_file_name = str(next_child_id) + ':0'
    test_path = "".join([cv_mesh_dir, "/", manifest_file_name])
    existence_test = cv_storage.files_exist([test_path])

    if list(existence_test.values())[0]:
        validated_children.append(next_child_id)
    else:
        if cg.get_chunk_layer(next_child_id) > 2:
            test_children.extend(cg.get_children(next_child_id))

    # print(len(test_children), len(validated_children), cv_mesh_dir, test_path,
    #       list(existence_test.values())[0])

    if len(test_children) == 0:
        return validated_children
    else:
        return _find_highest_children_with_mesh(cg, cv_storage, cv_mesh_dir,
                                                validated_children,
                                                test_children)


def _create_manifest_files_thread(args):
    cg_info, cv_path, cv_mesh_dir, root_id_start, root_id_end, \
        highest_mesh_level = args

    cg = ChunkedGraph(**cg_info)

    with Storage(cv_path) as cv_storage:
        for root_seg_id in range(root_id_start, root_id_end):

            root_id = cg.get_node_id(np.uint64(root_seg_id),
                                     cg.get_chunk_id(layer=int(cg.n_layers),
                                                     x=0, y=0, z=0))

    # cv = CloudVolume(cv_path)
    #
    # # mesh_dir = cv.info['mesh']
    #
    # paths = []
    # for seg_id in seg_ids:
    #     mesh_json_file_name = str(seg_id) + ':0'
    #     paths.append(os.path.join(mesh_dir, mesh_json_file_name))
    #
    # with Storage(cv.layer_cloudpath, progress=cv.progress) as stor:
    #     existence_test = stor.files_exist(paths)
    #
    # print("Success rate: %d / %d @" %
    #       (np.sum(list(existence_test.values())), len(seg_ids)),
    #       cg.get_chunk_coordinates(chunk_id))
