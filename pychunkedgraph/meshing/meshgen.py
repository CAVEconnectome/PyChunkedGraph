import sys
import os
import numpy as np
import json
import time
import collections
from functools import lru_cache
import datetime
import pytz
import cloudvolume
from scipy import ndimage, sparse
import networkx as nx

from multiwrapper import multiprocessing_utils as mu
from cloudvolume import Storage, EmptyVolumeException
from cloudvolume.meshservice import decode_mesh_buffer
from igneous.tasks import MeshTask

sys.path.insert(0, os.path.join(sys.path[0], '../..'))
os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"
UTC = pytz.UTC

from pychunkedgraph.backend import chunkedgraph   # noqa
from pychunkedgraph.backend.utils import serializers, column_keys  # noqa
from pychunkedgraph.meshing import meshgen_utils # noqa

@lru_cache(maxsize=None)
def get_l2_remapping(cg, chunk_id, time_stamp):
    """ Retrieves l2 node id to sv id mappping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param time_stamp: datetime object
    :return: dictionary
    """
    rr_chunk = cg.range_read_chunk(chunk_id=chunk_id,
                                   columns=column_keys.Hierarchy.Child,
                                   time_stamp=time_stamp)

    # This for-loop ensures that only the latest l2_ids are considered
    # The order by id guarantees the time order (only true for same neurons
    # but that is the case here).
    l2_remapping = {}
    all_sv_ids = set()
    for (k, row) in rr_chunk.items():
        this_sv_ids = row[0].value

        if this_sv_ids[0] in all_sv_ids:
            continue

        all_sv_ids = all_sv_ids.union(set(list(this_sv_ids)))
        l2_remapping[k] = this_sv_ids

    return l2_remapping


@lru_cache(maxsize=None)
def get_root_l2_remapping(cg, chunk_id, stop_layer, time_stamp, n_threads=4):
    """ Retrieves root to l2 node id mapping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param stop_layer: int
    :param time_stamp: datetime object
    :return: multiples
    """
    def _get_root_ids(args):
        start_id, end_id = args

        for i_id in range(start_id, end_id):
            l2_id = l2_ids[i_id]

            root_id = cg.get_root(l2_id, stop_layer=stop_layer,
                                  time_stamp=time_stamp)
            root_ids[i_id] = root_id

    l2_id_remap = get_l2_remapping(cg, chunk_id, time_stamp=time_stamp)

    l2_ids = np.array(list(l2_id_remap.keys()))

    root_ids = np.zeros(len(l2_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(l2_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(l2_ids), n_jobs + 1).astype(np.int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        mu.multithread_func(_get_root_ids, multi_args, n_threads=n_threads)

    return l2_ids, np.array(root_ids), l2_id_remap


# @lru_cache(maxsize=None)
def get_l2_overlapping_remappings(cg, chunk_id, time_stamp=None):
    """ Retrieves sv id to l2 id mapping for chunk with overlap in positive
        direction (one chunk)

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param time_stamp: datetime object
    :return: multiples
    """
    if time_stamp is None:
        time_stamp = datetime.datetime.utcnow()

    if time_stamp.tzinfo is None:
        time_stamp = UTC.localize(time_stamp)

    chunk_coords = cg.get_chunk_coordinates(chunk_id)
    chunk_layer = cg.get_chunk_layer(chunk_id)

    neigh_chunk_ids = []
    neigh_parent_chunk_ids = []

    # Collect neighboring chunks and their parent chunk ids
    # We only need to know about the parent chunk ids to figure the lowest
    # common chunk
    # Notice that the first neigh_chunk_id is equal to `chunk_id`.
    for x in range(chunk_coords[0], chunk_coords[0] + 2):
        for y in range(chunk_coords[1], chunk_coords[1] + 2):
            for z in range(chunk_coords[2], chunk_coords[2] + 2):

                # Chunk id
                neigh_chunk_id = cg.get_chunk_id(x=x, y=y, z=z,
                                                 layer=chunk_layer)
                neigh_chunk_ids.append(neigh_chunk_id)

                # Get parent chunk ids
                parent_chunk_ids = cg.get_parent_chunk_ids(neigh_chunk_id)
                neigh_parent_chunk_ids.append(parent_chunk_ids)

    # Find lowest common chunk
    # stop_layer = np.where(np.unique(neigh_parent_chunk_ids, axis=1,
    #                                 return_counts=True)[1] == 1)[0][0] + 3
    stop_layer = cg.n_layers

    # Find the parent in the lowest common chunk for each l2 id. These parent
    # ids are referred to as root ids even though they are not necessarily the
    # root id.
    neigh_l2_ids = []
    neigh_l2_id_remap = {}
    neigh_root_ids = []

    safe_l2_ids = []
    unsafe_l2_ids = []
    unsafe_root_ids = []

    # This loop is the main bottleneck
    for neigh_chunk_id in neigh_chunk_ids:
        print(neigh_chunk_id, "--------------")

        l2_ids, root_ids, l2_id_remap = \
            get_root_l2_remapping(cg, neigh_chunk_id, stop_layer,
                                  time_stamp=time_stamp)
        neigh_l2_ids.extend(l2_ids)
        neigh_l2_id_remap.update(l2_id_remap)
        neigh_root_ids.extend(root_ids)

        if neigh_chunk_id == chunk_id:
            # The first neigh_chunk_id is the one we are interested in. All l2 ids
            # that share no root id with any other l2 id are "safe", meaning that
            # we can easily obtain the complete remapping (including overlap) for these.
            # All other ones have to be resolved using the segmentation.
            u_root_ids, u_idx, c_root_ids = np.unique(neigh_root_ids,
                                                      return_counts=True,
                                                      return_index=True)

            safe_l2_ids = l2_ids[u_idx[c_root_ids == 1]]
            unsafe_l2_ids = l2_ids[~np.in1d(l2_ids, safe_l2_ids)]
            unsafe_root_ids = np.unique(root_ids[u_idx[c_root_ids != 1]])

    l2_root_dict = dict(zip(neigh_l2_ids, neigh_root_ids))
    root_l2_dict = collections.defaultdict(list)

    # Future sv id -> l2 mapping
    sv_ids = []
    l2_ids_flat = []

    # Do safe ones first
    for i_root_id in range(len(neigh_root_ids)):
        root_l2_dict[neigh_root_ids[i_root_id]].append(neigh_l2_ids[i_root_id])

    for l2_id in safe_l2_ids:
        root_id = l2_root_dict[l2_id]
        for neigh_l2_id in root_l2_dict[root_id]:
            l2_sv_ids = neigh_l2_id_remap[neigh_l2_id]
            sv_ids.extend(l2_sv_ids)
            l2_ids_flat.extend([l2_id] * len(neigh_l2_id_remap[neigh_l2_id]))

    # For the unsafe ones we can only do the in chunk svs
    # But we will map the out of chunk svs to the root id and store the
    # hierarchical information in a dictionary
    for l2_id in unsafe_l2_ids:
        sv_ids.extend(neigh_l2_id_remap[l2_id])
        l2_ids_flat.extend([l2_id] * len(neigh_l2_id_remap[l2_id]))

    unsafe_dict = collections.defaultdict(list)
    for root_id in unsafe_root_ids:
        if np.sum(~np.in1d(root_l2_dict[root_id], unsafe_l2_ids)) == 0:
            continue

        for neigh_l2_id in root_l2_dict[root_id]:
            unsafe_dict[root_id].append(neigh_l2_id)

            if neigh_l2_id in unsafe_l2_ids:
                continue

            sv_ids.extend(neigh_l2_id_remap[neigh_l2_id])
            l2_ids_flat.extend([root_id] * len(neigh_l2_id_remap[neigh_l2_id]))

    # Combine the lists for a (chunk-) global remapping
    sv_remapping = dict(zip(sv_ids, l2_ids_flat))

    return sv_remapping, unsafe_dict


def get_remapped_segmentation(cg, chunk_id, mip=2, overlap_vx=1,
                              time_stamp=None):
    def _remap(a):
        if a in sv_remapping:
            return sv_remapping[a]
        else:
            return 0

    assert mip >= cg.cv.mip

    sv_remapping, unsafe_dict = get_l2_overlapping_remappings(cg, chunk_id, time_stamp=time_stamp)

    cv = cloudvolume.CloudVolume(cg.cv.cloudpath, mip=mip)
    mip_diff = mip - cg.cv.mip

    mip_chunk_size = cg.chunk_size.astype(np.int) / np.array([2**mip_diff, 2**mip_diff, 1])
    mip_chunk_size = mip_chunk_size.astype(np.int)

    chunk_start = cg.get_chunk_coordinates(chunk_id) * mip_chunk_size
    chunk_end = chunk_start + mip_chunk_size + overlap_vx

    ws_seg = cv[chunk_start[0]: chunk_end[0],
                chunk_start[1]: chunk_end[1],
                chunk_start[2]: chunk_end[2]].squeeze()

    _remap_vec = np.vectorize(_remap)
    seg = _remap_vec(ws_seg)

    for unsafe_root_id in unsafe_dict.keys():
        bin_seg = seg == unsafe_root_id

        if np.sum(bin_seg) == 0:
            continue

        l2_edges = []
        cc_seg, n_cc = ndimage.label(bin_seg)
        for i_cc in range(1, n_cc + 1):
            bin_cc_seg = cc_seg == i_cc

            overlaps = []
            overlaps.extend(np.unique(seg[-2, :, :][bin_cc_seg[-1, :, :]]))
            overlaps.extend(np.unique(seg[:, -2, :][bin_cc_seg[:, -1, :]]))
            overlaps.extend(np.unique(seg[:, :, -2][bin_cc_seg[:, :, -1]]))
            overlaps = np.unique(overlaps)

            linked_l2_ids = overlaps[np.in1d(overlaps,
                                             unsafe_dict[unsafe_root_id])]

            if len(linked_l2_ids) == 0:
                seg[bin_cc_seg] = 0
            elif len(linked_l2_ids) == 1:
                seg[bin_cc_seg] = linked_l2_ids[0]
            else:
                seg[bin_cc_seg] = linked_l2_ids[0]

                for i_l2_id in range(len(linked_l2_ids) - 1):
                    for j_l2_id in range(i_l2_id + 1, len(linked_l2_ids)):
                        l2_edges.append([linked_l2_ids[i_l2_id],
                                         linked_l2_ids[j_l2_id]])

        if len(l2_edges) > 0:
            g = nx.Graph()
            g.add_edges_from(l2_edges)

            ccs = nx.connected_components(g)

            for cc in ccs:
                cc_ids = np.sort(list(cc))
                seg[np.in1d(seg, cc_ids[1:]).reshape(seg.shape)] = cc_ids[0]

    return seg


def get_connected(connectivity):
    u_ids, c_ids = np.unique(connectivity, return_counts=True)
    return u_ids[(c_ids % 2) == 1].astype(np.uint64)


def _get_sv_to_node_mapping_internal(cg, chunk_id, unbreakable_only):
    x, y, z = cg.get_chunk_coordinates(chunk_id)

    center_chunk_max_node_id = cg.get_node_id(
        segment_id=cg.get_segment_id_limit(chunk_id), chunk_id=chunk_id)
    xyz_plus_chunk_min_node_id = cg.get_chunk_id(
        layer=1, x=x + 1, y=y + 1, z=z + 1)

    columns = [column_keys.Connectivity.Partner,
               column_keys.Connectivity.Connected]
    if unbreakable_only:
        columns.append(column_keys.Connectivity.Affinity)

    seg_ids_center = cg.range_read_chunk(1, x, y, z, columns=columns)

    seg_ids_face_neighbors = {}
    seg_ids_face_neighbors.update(
        cg.range_read_chunk(1, x + 1, y, z, columns=columns))
    seg_ids_face_neighbors.update(
        cg.range_read_chunk(1, x, y + 1, z, columns=columns))
    seg_ids_face_neighbors.update(
        cg.range_read_chunk(1, x, y, z + 1, columns=columns))

    seg_ids_edge_neighbors = {}
    seg_ids_edge_neighbors.update(
        cg.range_read_chunk(1, x + 1, y + 1, z, columns=columns))
    seg_ids_edge_neighbors.update(
        cg.range_read_chunk(1, x, y + 1, z + 1, columns=columns))
    seg_ids_edge_neighbors.update(
        cg.range_read_chunk(1, x + 1, y, z + 1, columns=columns))

    # Retrieve all face-adjacent supervoxel
    one_hop_neighbors = {}
    for seg_id, data in seg_ids_center.items():
        data = cg.flatten_row_dict(data)
        partners = data[column_keys.Connectivity.Partner]
        connected = data[column_keys.Connectivity.Connected]
        partners = partners[connected]

        # Only keep supervoxel within the "positive" adjacent chunks
        # (and if specified only the unbreakable counterparts)
        if unbreakable_only:
            affinities = data[column_keys.Connectivity.Affinity]
            affinities = affinities[connected]

            partners = partners[(affinities == np.inf) &
                                (partners > center_chunk_max_node_id)]
        else:
            partners = partners[partners > center_chunk_max_node_id]

        one_hop_neighbors.update(dict(zip(partners, [seg_id] * len(partners))))

    # Retrieve all edge-adjacent supervoxel
    two_hop_neighbors = {}
    for seg_id, base_id in one_hop_neighbors.items():
        if seg_id not in seg_ids_face_neighbors:
            # FIXME: Those are non-face-adjacent atomic partners caused by human
            #        proofreaders. They're crazy. Watch out for them.
            continue
        data = cg.flatten_row_dict(seg_ids_face_neighbors[seg_id])
        partners = data[column_keys.Connectivity.Partner]
        connected = data[column_keys.Connectivity.Connected]
        partners = partners[connected]

        # FIXME: The partners filter also keeps some connections to within the
        # face_adjacent chunk itself and to "negative", face-adjacent chunks.
        # That's OK for now, since the filters are only there to keep the
        # number of connections low
        if unbreakable_only:
            affinities = data[column_keys.Connectivity.Affinity]
            affinities = affinities[connected]

            partners = partners[(affinities == np.inf) &
                                (partners > center_chunk_max_node_id) &
                                (partners < xyz_plus_chunk_min_node_id)]
        else:
            partners = partners[(partners > center_chunk_max_node_id) &
                                (partners < xyz_plus_chunk_min_node_id)]

        two_hop_neighbors.update(dict(zip(partners, [base_id] * len(partners))))

    # Retrieve all corner-adjacent supervoxel
    three_hop_neighbors = {}
    for seg_id, base_id in list(two_hop_neighbors.items()):
        if seg_id not in seg_ids_edge_neighbors:
            # See FIXME for edge-adjacent supervoxel - need to ignore those
            # FIXME 2: Those might also be non-face-adjacent atomic partners
            # caused by human proofreaders.
            del two_hop_neighbors[seg_id]
            continue

        data = cg.flatten_row_dict(seg_ids_edge_neighbors[seg_id])
        partners = data[column_keys.Connectivity.Partner]
        connected = data[column_keys.Connectivity.Connected]
        partners = partners[connected]

        # We are only interested in the single corner voxel, but based on the
        # neighboring supervoxels, there might be a few more - doesn't matter.
        if unbreakable_only:
            affinities = data[column_keys.Connectivity.Affinity]
            affinities = affinities[connected]

            partners = partners[(affinities == np.inf) &
                                (partners > xyz_plus_chunk_min_node_id)]
        else:
            partners = partners[partners > xyz_plus_chunk_min_node_id]

        three_hop_neighbors.update(dict(zip(partners, [base_id] * len(partners))))

    sv_to_node_mapping = {seg_id: seg_id for seg_id in seg_ids_center.keys()}
    sv_to_node_mapping.update(one_hop_neighbors)
    sv_to_node_mapping.update(two_hop_neighbors)
    sv_to_node_mapping.update(three_hop_neighbors)
    sv_to_node_mapping = {k: v for k, v in sv_to_node_mapping.items()}

    return sv_to_node_mapping


def get_sv_to_node_mapping(cg, chunk_id):
    """ Reads sv_id -> root_id mapping for a chunk from the chunkedgraph

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :return: dict
    """

    layer = cg.get_chunk_layer(chunk_id)
    assert layer <= 2

    if layer == 1:
        # Mapping a chunk containing supervoxels - need to retrieve
        # potential unbreakable counterparts for each supervoxel
        # (Look for atomic edges with infinite edge weight)
        # Also need to check the edges and the corner!
        return _get_sv_to_node_mapping_internal(cg, chunk_id, True)

    else:  # layer == 2
        # Get supervoxel mapping
        x, y, z = cg.get_chunk_coordinates(chunk_id)
        sv_to_node_mapping = _get_sv_to_node_mapping_internal(
                cg, cg.get_chunk_id(layer=1, x=x, y=y, z=z), False)

        # Update supervoxel with their parents
        seg_ids = cg.range_read_chunk(1, x, y, z, columns=column_keys.Hierarchy.Parent)

        for sv_id, base_sv_id in sv_to_node_mapping.items():
            agg_id = seg_ids[base_sv_id][0].value  # latest parent
            sv_to_node_mapping[sv_id] = agg_id

        return sv_to_node_mapping


def merge_meshes(meshes):
    vertexct = np.zeros(len(meshes) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x['num_vertices'] for x in meshes])
    vertices = np.concatenate([x['vertices'] for x in meshes])
    faces = np.concatenate([
        mesh['faces'] + vertexct[i] for i, mesh in enumerate(meshes)
    ])

    if vertexct[-1] > 0:
        # Remove duplicate vertices
        vertices, faces = np.unique(vertices[faces], return_inverse=True, axis=0)
        faces = faces.astype(np.uint32)

    return {
        'num_vertices': np.uint32(len(vertices)),
        'vertices': vertices,
        'faces': faces
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
    mesh_dir = cv_mesh_dir or cg._mesh_dir

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

        mesh_block_shape = meshgen_utils.get_mesh_block_shape(cg, layer, mip)

        chunk_offset = (cx, cy, cz) * mesh_block_shape

        task = MeshTask(
            mesh_block_shape,
            chunk_offset,
            cv_path,
            mip=mip,
            simplification_factor=999999,      # Simplify as much as possible ...
            max_simplification_error=max_err,  # ... staying below max error.
            remap_table=sv_to_node_mapping,
            generate_manifests=False,
            low_padding=0,                     # One voxel overlap to exactly line up
            high_padding=1,                    # vertex boundaries.
            mesh_dir=mesh_dir,
            cache_control='no-cache'
        )
        task.execute()

        print("Layer %d -- finished:" % layer, cx, cy, cz)

    else:
        # For each node with more than one child, create a new fragment by
        # merging the mesh fragments of the children.

        print("Retrieving children for chunk %s -- (%s, %s, %s, %s)" % (chunk_id, layer, cx, cy, cz))
        node_ids = cg.range_read_chunk(layer, cx, cy, cz, columns=column_keys.Hierarchy.Child)

        print("Collecting only nodes with more than one child: ", end="")
        # Only keep nodes with more than one child
        multi_child_nodes = {}
        for node_id, data in node_ids.items():
            children = data[0].value

            if len(children) > 1:
                multi_child_descendant = [
                    meshgen_utils.get_downstream_multi_child_node(cg, child, 2) for child in children
                ]

                multi_child_nodes[f'{node_id}:0:{meshgen_utils.get_chunk_bbox_str(cg, node_id, mip)}'] = [
                    f'{c}:0:{meshgen_utils.get_chunk_bbox_str(cg, c, mip)}' for c in multi_child_descendant
                ]
        print("%d out of %d" % (len(multi_child_nodes), len(node_ids)))
        if not multi_child_nodes:
            print("Nothing to do", cx, cy, cz)
            return

        with Storage(os.path.join(cv_path, mesh_dir)) as storage:
            i = 0
            for new_fragment_id, fragment_ids_to_fetch in multi_child_nodes.items():
                i += 1
                if i % max(1, len(multi_child_nodes) // 10) == 0:
                    print(f"{i}/{len(multi_child_nodes)}")

                fragment_contents = storage.get_files(fragment_ids_to_fetch)
                fragment_contents = {
                    x['filename']: decode_mesh_buffer(x['filename'], x['content'])
                    for x in fragment_contents
                    if x['content'] is not None and x['error'] is None
                }

                old_fragments = list(fragment_contents.values())
                if not old_fragments:
                    continue

                new_fragment = merge_meshes(old_fragments)
                new_fragment_b = b''.join([
                    new_fragment['num_vertices'].tobytes(),
                    new_fragment['vertices'].tobytes(),
                    new_fragment['faces'].tobytes()
                ])
                storage.put_file(new_fragment_id,
                                 new_fragment_b,
                                 content_type='application/octet-stream',
                                 compress=True,
                                 cache_control='no-cache')


def mesh_lvl2_previews(cg, lvl2_node_ids, cv_path=None,
                       cv_mesh_dir=None, mip=2, simplification_factor=999999,
                       max_err=40, parallel_download=8, verbose=True,
                       cache_control="no-cache", n_threads=1):

    serialized_cg_info = cg.get_serialized_info()
    del serialized_cg_info["credentials"]

    if not isinstance(lvl2_node_ids, dict):
        lvl2_node_ids = dict(zip(lvl2_node_ids, [None] * len(lvl2_node_ids)))

    mesh_dir = cv_mesh_dir or cg._mesh_dir

    multi_args = []
    for lvl2_node_id in lvl2_node_ids.keys():
        multi_args.append([serialized_cg_info, lvl2_node_id,
                           lvl2_node_ids[lvl2_node_id],
                           cv_path, mesh_dir, mip, simplification_factor,
                           max_err, parallel_download, verbose,
                           cache_control])

    # Run parallelizing
    if n_threads == 1:
        mu.multiprocess_func(_mesh_lvl2_previews_threads,
                             multi_args, n_threads=n_threads,
                             verbose=False, debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_mesh_lvl2_previews_threads,
                                multi_args, n_threads=n_threads)


def _mesh_lvl2_previews_threads(args):
    serialized_cg_info, lvl2_node_id, supervoxel_ids, \
        cv_path, cv_mesh_dir, mip, simplification_factor, \
        max_err, parallel_download, verbose, cache_control = args

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)
    mesh_lvl2_preview(cg, lvl2_node_id, supervoxel_ids=supervoxel_ids,
                      cv_path=cv_path, cv_mesh_dir=cv_mesh_dir, mip=mip,
                      simplification_factor=simplification_factor,
                      max_err=max_err, parallel_download=parallel_download,
                      verbose=verbose, cache_control=cache_control)


def mesh_lvl2_preview(cg, lvl2_node_id, supervoxel_ids=None, cv_path=None,
                      cv_mesh_dir=None, mip=2, simplification_factor=999999,
                      max_err=40, parallel_download=8, verbose=True,
                      cache_control="no-cache"):
    """ Compute a mesh for a level 2 node without hierarchy and without
        consistency beyond the chunk boundary. Useful to give the user a quick
        preview. A proper mesh hierarchy should be generated using
        `mesh_node_hierarchy()`

    :param cg: ChunkedGraph instance
    :param lvl2_node_id: int
    :param supervoxel_ids: list of np.uint64
    :param cv_path: str or None (cg._cv_path)
    :param cv_mesh_dir: str or None
    :param mip: int
    :param simplification_factor: int
    :param max_err: float
    :param parallel_download: int
    :param verbose: bool
    :param cache_control: cache_control
    """

    layer = cg.get_chunk_layer(lvl2_node_id)
    assert layer == 2

    if cv_path is None:
        cv_path = cg._cv_path

    mesh_dir = cv_mesh_dir or cg._mesh_dir

    if supervoxel_ids is None:
        supervoxel_ids = cg.get_subgraph_nodes(lvl2_node_id, verbose=verbose)

    remap_table = dict(zip(supervoxel_ids, [lvl2_node_id] * len(supervoxel_ids)))

    mesh_block_shape = meshgen_utils.get_mesh_block_shape(cg, layer, mip)

    cx, cy, cz = cg.get_chunk_coordinates(lvl2_node_id)
    chunk_offset = (cx, cy, cz) * mesh_block_shape

    task = MeshTask(
        mesh_block_shape,
        chunk_offset,
        cv_path,
        mip=mip,
        simplification_factor=simplification_factor,
        max_simplification_error=max_err,
        remap_table=remap_table,
        generate_manifests=True,
        low_padding=0,
        high_padding=0,
        mesh_dir=mesh_dir,
        parallel_download=parallel_download,
        cache_control=cache_control
    )
    if verbose:
        time_start = time.time()

    task.execute()

    if verbose:
        print("Preview Mesh for layer 2 Node ID %d: %.3fms (%d supervoxel)" %
              (lvl2_node_id, (time.time() - time_start) * 1000, len(supervoxel_ids)))
    return


def run_task_bundle(settings, layer, roi):
    cgraph = chunkedgraph.ChunkedGraph(
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
                    chunk_mesh_task(cgraph, chunk_id, cgraph._cv_path,
                                    cv_mesh_dir=mesh_dir, mip=mip, max_err=max_err)
                except EmptyVolumeException as e:
                    print("Warning: Empty segmentation encountered: %s" % e)


if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    layer = int(sys.argv[2])
    run_task_bundle(params, layer, meshgen_utils.str_to_slice(sys.argv[3]))
