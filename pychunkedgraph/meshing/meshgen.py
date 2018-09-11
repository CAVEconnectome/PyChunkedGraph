import sys
import os
import numpy as np
import json
import re

from cloudvolume import CloudVolume, Storage, EmptyVolumeException
from cloudvolume.meshservice import decode_mesh_buffer
from functools import lru_cache
from igneous.tasks import MeshTask

sys.path.insert(0, os.path.join(sys.path[0], '../..'))
os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"

from pychunkedgraph.backend.chunkedgraph import ChunkedGraph, serialize_uint64  # noqa


def str_to_slice(slice_str: str):
    match = re.match(r"(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)", slice_str)
    return (slice(int(match.group(1)), int(match.group(2))),
            slice(int(match.group(3)), int(match.group(4))),
            slice(int(match.group(5)), int(match.group(6))))


def slice_to_str(slices) -> str:
    if isinstance(slices, slice):
        return "%d-%d" % (slices.start, slices.stop)
    else:
        return '_'.join(map(slice_to_str, slices))


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

    return np.floor_divide(graphlayer_chunksize, distortion, dtype=np.int, casting='unsafe')


def get_sv_to_node_mapping(cg, chunk_id):
    """ Reads sv_id -> root_id mapping for a chunk from the chunkedgraph

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :return: dict
    """

    layer = cg.get_chunk_layer(chunk_id)
    assert layer <= 2

    sv_to_node_mapping = {}

    if layer == 1:
        # Mapping a chunk containing supervoxels - need to retrieve
        # potential unbreakable counterparts for each supervoxel
        # (Look for atomic edges with infinite edge weight)
        # Also need to check the edges and the corner!
        x, y, z = cg.get_chunk_coordinates(chunk_id)

        center_chunk_max_node_id = cg.get_node_id(
            segment_id=cg.get_segment_id_limit(chunk_id), chunk_id=chunk_id)
        xyz_plus_chunk_min_node_id = cg.get_chunk_id(
            layer=1, x=x + 1, y=y + 1, z=z + 1)

        seg_ids_center = cg.range_read_chunk(1, x, y, z)

        seg_ids_face_neighbors = {}
        seg_ids_face_neighbors.update(
            cg.range_read_chunk(1, x + 1, y, z, row_keys=['atomic_connected_partners', 'atomic_connected_affinities']))
        seg_ids_face_neighbors.update(
            cg.range_read_chunk(1, x, y + 1, z, row_keys=['atomic_connected_partners', 'atomic_connected_affinities']))
        seg_ids_face_neighbors.update(
            cg.range_read_chunk(1, x, y, z + 1, row_keys=['atomic_connected_partners', 'atomic_connected_affinities']))

        seg_ids_edge_neighbors = {}
        seg_ids_edge_neighbors.update(
            cg.range_read_chunk(1, x + 1, y + 1, z, row_keys=['atomic_connected_partners', 'atomic_connected_affinities']))
        seg_ids_edge_neighbors.update(
            cg.range_read_chunk(1, x, y + 1, z + 1, row_keys=['atomic_connected_partners', 'atomic_connected_affinities']))
        seg_ids_edge_neighbors.update(
            cg.range_read_chunk(1, x + 1, y, z + 1, row_keys=['atomic_connected_partners', 'atomic_connected_affinities']))

        # Retrieve all face-adjacent, unbreakable supervoxel
        one_hop_neighbors = {}
        for seg_id_b, data in seg_ids_center.items():
            partners = np.frombuffer(
                data.cells['0'][b'atomic_connected_partners'][0].value,
                dtype=np.uint64)

            affinities = np.frombuffer(
                data.cells['0'][b'atomic_connected_affinities'][0].value,
                dtype=np.float32)

            # Only keep unbreakable segments, and only within the "positive" adjacent chunks
            partners = partners[(affinities == np.inf) &
                                (partners > chunk_id)]
            one_hop_neighbors.update(dict(zip(partners, [seg_id_b] * len(partners))))

        # Retrieve all edge-adjacent, unbreakable supervoxel
        two_hop_neighbors = {}
        for seg_id, base_id in one_hop_neighbors.items():
            seg_id_b = serialize_uint64(seg_id)
            partners = np.frombuffer(
                seg_ids_face_neighbors[seg_id_b].cells['0'][b'atomic_connected_partners'][0].value,
                dtype=np.uint64)
            affinities = np.frombuffer(
                seg_ids_face_neighbors[seg_id_b].cells['0'][b'atomic_connected_affinities'][0].value,
                dtype=np.float32)

            # Only keep unbreakable segments, doesn't completely exclude unnecessary chunks,
            # but is good/fast enough to cut down the number of segments
            partners = partners[(affinities == np.inf) &
                                (partners > center_chunk_max_node_id) &
                                (partners < xyz_plus_chunk_min_node_id)]
            two_hop_neighbors.update(dict(zip(partners, [base_id] * len(partners))))

        # Retrieve all corner-adjacent, unbreakable supervoxel
        three_hop_neighbors = {}
        for seg_id, base_id in two_hop_neighbors.items():
            seg_id_b = serialize_uint64(seg_id)
            if seg_id_b not in seg_ids_edge_neighbors:
                continue

            partners = np.frombuffer(
                seg_ids_edge_neighbors[seg_id_b].cells['0'][b'atomic_connected_partners'][0].value,
                dtype=np.uint64)
            affinities = np.frombuffer(
                seg_ids_edge_neighbors[seg_id_b].cells['0'][b'atomic_connected_affinities'][0].value,
                dtype=np.float32)

            # Only keep unbreakable segments - we are only interested in the single corner voxel,
            # but based on the neighboring supervoxels, there might be a few more supervoxel - doesn't matter.
            partners = partners[(affinities == np.inf) &
                                (partners > xyz_plus_chunk_min_node_id)]
            three_hop_neighbors.update(dict(zip(partners, [base_id] * len(partners))))

        sv_to_node_mapping = {seg_id: seg_id for seg_id in
                              [np.uint64(x) for x in seg_ids_center.keys()]}
        sv_to_node_mapping.update(one_hop_neighbors)
        sv_to_node_mapping.update(two_hop_neighbors)
        sv_to_node_mapping.update(three_hop_neighbors)
        sv_to_node_mapping = {np.uint64(k): np.uint64(v) for k, v in sv_to_node_mapping.items()}

    elif layer == 2:
        # Get supervoxel mapping
        x, y, z = cg.get_chunk_coordinates(chunk_id)
        sv_to_node_mapping = get_sv_to_node_mapping(cg, chunk_id=cg.get_chunk_id(layer=1, x=x, y=y, z=z))

        # Update supervoxel with their parents
        seg_ids = cg.range_read_chunk(1, x, y, z, row_keys=['parents'])

        for sv_id, base_sv_id in sv_to_node_mapping.items():
            base_sv_id = serialize_uint64(base_sv_id)
            agg_id = np.frombuffer(seg_ids[base_sv_id].cells['0'][b'parents'][0].value, dtype=np.uint64)[0]

            sv_to_node_mapping[sv_id] = agg_id

    return sv_to_node_mapping


def merge_meshes(meshes):
    num_vertices = 0
    vertices, faces = [], []

    # Dumb merge
    for mesh in meshes:
        vertices.extend(mesh['vertices'])
        faces.extend([x + num_vertices for x in mesh['faces']])
        num_vertices += len(mesh['vertices'])

    if num_vertices > 0:
        # Remove duplicate vertices
        vertex_representation = np.array(vertices)[faces]
        vertices, faces = np.unique(vertex_representation,
                                    return_inverse=True, axis=0)

    return {
        'num_vertices': len(vertices),
        'vertices': list(map(tuple, vertices)),
        'faces': list(faces)
    }


def chunk_mesh_task(cg, chunk_id, cv_path,
                    cv_mesh_dir=None, mip=3, max_err=40):
    """ Computes the meshes for a single chunk

    :param cg: ChunkedGraph instance
    :param chunk_id: int
    :param cv_path: str
    :param cv_mesh_dir: str or None
    :param mip: int
    :param max_err: float
    """

    layer = cg.get_chunk_layer(chunk_id)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)

    if layer <= 2:
        # Level 1 or 2 chunk - fetch supervoxel mapping from ChunkedGraph, and
        # generate an igneous MeshTask, which will:
        # 1) Relabel the segmentation based on the sv_to_node_mapping
        # 2) Run Marching Cubes,
        # 3) simply each mesh using error quadrics to control the edge collapse
        # 4) upload a single mesh file for each segment of this chunk
        # 5) upload a manifest file for each segment of this chunk,
        #    pointing to the mesh for the segment

        print("Retrieving remap table for chunk %s -- (%s, %s, %s, %s)" % (chunk_id, layer, cx, cy, cz))
        sv_to_node_mapping = get_sv_to_node_mapping(cg, chunk_id)
        print("Remapped %s segments to %s agglomerations. Start meshing..." % (len(sv_to_node_mapping), len(np.unique(list(sv_to_node_mapping.values())))))

        if len(sv_to_node_mapping) == 0:
            print("Nothing to do", cx, cy, cz)
            return

        mesh_block_shape = get_mesh_block_shape(cg, layer, mip)

        chunk_offset = (cx, cy, cz) * mesh_block_shape

        task = MeshTask(
            mesh_block_shape,
            chunk_offset,
            cv_path,
            mip=mip,
            simplification_factor=999999,     # Simplify as much as possible ...
            max_simplification_error=max_err,  # ... staying below max error.
            remap_table=sv_to_node_mapping,
            generate_manifests=True,
            low_padding=0,                    # One voxel overlap to exactly line up
            high_padding=1,                   # vertex boundaries.
            mesh_dir=cv_mesh_dir
        )
        task.execute()

        print("Layer %d -- finished:" % layer, cx, cy, cz)

    else:
        # Layer 3+ chunk
        # 1) Load all manifests of next lower layer for this chunk
        # 2a) Merge mesh fragments of child chunks (3 <= layer < n_layers-3), or
        # 2b) Combine the manifests without creating new meshes

        manifests_to_fetch = {np.uint64(x): [] for x in node_ids.keys()}

        mesh_dir = cv_mesh_dir or get_segmentation_info(cg)['mesh']

        chunk_block_shape = get_mesh_block_shape(cg, layer, mip)
        bbox_start = cg.get_chunk_coordinates(chunk_id) * chunk_block_shape
        bbox_end = bbox_start + chunk_block_shape
        chunk_bbox_str = slice_to_str(slice(bbox_start[i], bbox_end[i]) for i in range(3))

        for node_id_b, data in node_ids.items():
            node_id = np.uint64(node_id_b)
            children = np.frombuffer(data.cells['0'][b'children'][0].value, dtype=np.uint64)
            manifests_to_fetch[node_id].extend((f'{c}:0' for c in children))

        with Storage(os.path.join(cg.cv_path, mesh_dir)) as storage:
            print("Downloading Manifests...")
            manifest_content = storage.get_files((m for manifests in manifests_to_fetch.values() for m in manifests))
            print("Decoding Manifests...")
            manifest_content = {x['filename']: json.loads(x['content']) for x in manifest_content if x['content'] is not None and x['error'] is None}

            # Only collect fragment filenames for nodes which consist of
            # more than one fragment, skipping the ones without a manifest
            print("Collect fragments to download...")
            fragments_to_fetch = [
                fragment for manifests in manifests_to_fetch.values()
                if len(manifests) > 1
                for manifest in manifests
                if manifest in manifest_content
                for fragment in manifest_content[manifest]['fragments']]

            print("Downloading Fragments...")
            fragments_content = storage.get_files(fragments_to_fetch)

            print("Decoding Fragments...")
            fragments_content = {x['filename']: decode_mesh_buffer(x['filename'], x['content']) for x in fragments_content if x['content'] is not None and x['error'] is None}

        fragments_to_upload = []
        manifests_to_upload = []
        for node_id, manifests in manifests_to_fetch.items():
            manifest_filename = f'{node_id}:0'
            fragment_filename = f'{node_id}:0:{chunk_bbox_str}'

            fragments = [
                fragment for manifest in manifests
                if manifest in manifest_content
                for fragment in manifest_content[manifest]['fragments']]

            if len(fragments) == 0:
                # No fragments - could be caused by tiny supervoxels near the
                # chunk boundary that were lost when meshing a downsampled
                # layer 1 chunk.
                print(f"Skipped {node_id} - no mesh fragments (node probably near chunk boundary)")
                continue
            elif len(fragments) == 1:
                # Only a single fragment - new manifest will simply point to
                # existing fragment, to save storage
                manifests_to_upload.append((
                    manifest_filename,
                    '{"fragments": ["%s"]}' % fragments[0]
                ))
            else:
                # Merge fragments into one file, removing duplicate vertices
                mesh = merge_meshes(map(lambda x: fragments_content[x] if x in fragments_content else {'num_vertices': 0, 'vertices': [], 'faces': []}, fragments))

                fragments_to_upload.append((
                    fragment_filename,
                    b''.join([
                        np.uint32(mesh['num_vertices']).tobytes(),
                        np.array(mesh['vertices'], dtype=np.float32).tobytes(),
                        np.array(mesh['faces'], dtype=np.uint32).tobytes()
                    ])
                ))

                manifests_to_upload.append((
                    manifest_filename,
                    '{"fragments": ["%s"]}' % fragment_filename
                ))

        with Storage(os.path.join(cg.cv_path, mesh_dir)) as storage:
            storage.put_files(fragments_to_upload, content_type='application/octet-stream', compress=True, cache_control=False)
            storage.put_files(manifests_to_upload, content_type='application/json', compress=False, cache_control=False)
            print("Uploaded %s manifests and %s fragments (reusing %s fragments)"
                  % (len(manifests_to_upload),
                     len(fragments_to_upload),
                     len(manifests_to_upload) - len(fragments_to_upload)))


def mesh_single_component(node_id, cg, cv_path,
                          cv_mesh_dir=None, mip=3, max_err=40):
    """ Computes the mesh for a single component

    :param node_id: uint64
    :param cg: chunkedgraph instance
    :param cv_path: str
    :param cv_mesh_dir: str
    :param mip: int
    :param max_err: float
    """
    layer = cg.get_chunk_layer(node_id)

    atomic_ids = [node_id]
    if layer > 1:
        atomic_ids = cg.get_subgraph(node_id, verbose=False)

    if len(atomic_ids):
        sv_to_node_mapping = dict(zip(atomic_ids, [node_id] * len(atomic_ids)))
        chunk_mesh_task(sv_to_node_mapping, cg, cg.get_chunk_id(node_id),
                        cv_path, cv_mesh_dir=cv_mesh_dir, mip=mip,
                        max_err=max_err)
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


def run_task_bundle(settings, layer, roi):
    cgraph = ChunkedGraph(
        table_id=settings['chunkedgraph']['table_id'],
        instance_id=settings['chunkedgraph']['instance_id']
    )
    meshing = settings['meshing']
    mip = meshing.get('mip', 2)
    max_err = meshing.get('max_simplification_error', 40)
    mesh_dir = meshing.get('mesh_dir', None)

    base_chunk_span = int(cgraph.fan_out) ** max(0, layer - 2)
    chunksize = np.array(cgraph.chunk_size, dtype=np.int) * base_chunk_span

    for x in range(roi[0].start, roi[0].stop, chunksize[0]):
        for y in range(roi[1].start, roi[1].stop, chunksize[1]):
            for z in range(roi[2].start, roi[2].stop, chunksize[2]):
                chunk_id = cgraph.get_chunk_id_from_coord(layer, x, y, z)

                try:
                    chunk_mesh_task(cgraph, chunk_id, cgraph.cv_path,
                                    cv_mesh_dir=mesh_dir, mip=mip, max_err=max_err)
                except EmptyVolumeException as e:
                    print("Warning: Empty segmentation encountered: %s" % e)


if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    layer = int(sys.argv[2])
    run_task_bundle(params, layer, str_to_slice(sys.argv[3]))
