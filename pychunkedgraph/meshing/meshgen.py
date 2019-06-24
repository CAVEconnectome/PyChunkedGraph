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
from cloudvolume.lib import Vec
from cloudvolume.meshservice import decode_mesh_buffer
from igneous.tasks import MeshTask
import DracoPy
import zmesh
import fastremap
import time

sys.path.insert(0, os.path.join(sys.path[0], '../..'))
os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"
UTC = pytz.UTC

from pychunkedgraph.backend import chunkedgraph   # noqa
from pychunkedgraph.backend.utils import serializers, column_keys  # noqa
from pychunkedgraph.meshing import meshgen_utils # noqa

def decode_draco_mesh_buffer(fragment):
    try:
        mesh_object = DracoPy.decode_buffer_to_mesh(fragment)
        vertices = np.array(mesh_object.points)
        faces = np.array(mesh_object.faces)
    except ValueError:
        raise ValueError("Not a valid draco mesh")

    assert len(vertices) % 3 == 0, "Draco mesh vertices not 3-D"
    num_vertices = len(vertices) // 3

    # For now, just return this dict until we figure out
    # how exactly to deal with Draco's lossiness/duplicate vertices
    return {
        'num_vertices': num_vertices,
        'vertices': vertices.reshape(num_vertices, 3),
        'faces': faces,
        'encoding_options': mesh_object.encoding_options,
        'encoding_type': 'draco'
    }


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

        # for i_id in range(start_id, end_id):
        #     l2_id = l2_ids[i_id]

        #     root_id = cg.get_root(l2_id, stop_layer=stop_layer,
        #                           time_stamp=time_stamp)
        #     root_ids[i_id] = root_id
        root_ids[start_id:end_id] = cg.get_roots(l2_ids[start_id: end_id])

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

    # return l2_ids, np.array(root_ids), l2_id_remap
    return l2_ids, root_ids, l2_id_remap


# @lru_cache(maxsize=None)
def get_l2_overlapping_remappings(cg, chunk_id, time_stamp=None, n_threads=1):
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
    neigh_parent_chunk_ids = np.array(neigh_parent_chunk_ids)
    layer_agreement = np.all((neigh_parent_chunk_ids -
                              neigh_parent_chunk_ids[0]) == 0, axis=0)
    stop_layer = np.where(layer_agreement)[0][0] + 1

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
        before_time = time.time()

        l2_ids, root_ids, l2_id_remap = \
            get_root_l2_remapping(cg, neigh_chunk_id, stop_layer,
                                  time_stamp=time_stamp, n_threads=n_threads)
        print('get_root_l2_remapping time', time.time() - before_time)
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
                              time_stamp=None, n_threads=1):
    """ Downloads + remaps ws segmentation + resolve unclear cases

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
        take the 'actual
    :param overlap_vx:
    :param time_stamp:
    :return:
    """
    def _remap(a):
        if a in sv_remapping:
            return sv_remapping[a]
        else:
            return 0

    assert mip >= cg.cv.mip

    sv_remapping, unsafe_dict = get_lx_overlapping_remappings(cg, chunk_id,
                                                              time_stamp=time_stamp,
                                                              n_threads=n_threads)

    cv = cloudvolume.CloudVolume(cg.cv.cloudpath, mip=mip)
    mip_diff = mip - cg.cv.mip

    mip_chunk_size = cg.chunk_size.astype(np.int) / np.array([2**mip_diff, 2**mip_diff, 1])
    mip_chunk_size = mip_chunk_size.astype(np.int)

    chunk_start = cg.cv.mip_voxel_offset(mip) + cg.get_chunk_coordinates(chunk_id) * mip_chunk_size
    chunk_end = chunk_start + mip_chunk_size + overlap_vx
    chunk_end = Vec.clamp(chunk_end, cg.cv.mip_voxel_offset(mip), cg.cv.mip_voxel_offset(mip) + cg.cv.mip_volume_size(mip))

    ws_seg = cv[chunk_start[0]: chunk_end[0],
                chunk_start[1]: chunk_end[1],
                chunk_start[2]: chunk_end[2]].squeeze()

    _remap_vec = np.vectorize(_remap)
    seg = _remap_vec(ws_seg).astype(np.uint64)

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


@lru_cache(maxsize=None)
def get_higher_to_lower_remapping(cg, chunk_id, time_stamp):
    """ Retrieves lx node id to sv id mappping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param time_stamp: datetime object
    :return: dictionary
    """
    def _lower_remaps(ks):
        return np.concatenate([lower_remaps[k] for k in ks])

    assert cg.get_chunk_layer(chunk_id) >= 2
    assert cg.get_chunk_layer(chunk_id) <= cg.n_layers

    print(f"\n{chunk_id} ----------------\n")

    lower_remaps = {}
    if cg.get_chunk_layer(chunk_id) > 2:
        for lower_chunk_id in cg.get_chunk_child_ids(chunk_id):
            #TODO speedup
            lower_remaps.update(get_higher_to_lower_remapping(
                cg, lower_chunk_id, time_stamp=time_stamp))

    rr_chunk = cg.range_read_chunk(chunk_id=chunk_id,
                                   columns=column_keys.Hierarchy.Child,
                                   time_stamp=time_stamp)

    # This for-loop ensures that only the latest lx_ids are considered
    # The order by id guarantees the time order (only true for same neurons
    # but that is the case here).
    lx_remapping = {}
    all_lower_ids = set()
    for k in sorted(rr_chunk.keys(), reverse=True):
        this_child_ids = rr_chunk[k][0].value

        if this_child_ids[0] in all_lower_ids:
            continue

        all_lower_ids = all_lower_ids.union(set(list(this_child_ids)))

        if cg.get_chunk_layer(chunk_id) > 2:
            try:
                lx_remapping[k] = _lower_remaps(this_child_ids)
            except KeyError:
                # KeyErrors indicate that this id is deprecated given the
                # time_stamp
                continue
        else:
            lx_remapping[k] = this_child_ids

    return lx_remapping


@lru_cache(maxsize=None)
def get_root_lx_remapping(cg, chunk_id, stop_layer, time_stamp, n_threads=1):
    """ Retrieves root to l2 node id mapping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param stop_layer: int
    :param time_stamp: datetime object
    :return: multiples
    """
    def _get_root_ids(args):
        start_id, end_id = args

        # for i_id in range(start_id, end_id):
        #     lx_id = lx_ids[i_id]

        #     root_id = cg.get_root(lx_id, stop_layer=stop_layer,
        #                           time_stamp=time_stamp)
        #     root_ids[i_id] = root_id
        root_ids[start_id:end_id] = cg.get_roots(lx_ids[start_id: end_id])

    lx_id_remap = get_higher_to_lower_remapping(cg, chunk_id, time_stamp=time_stamp)

    lx_ids = np.array(list(lx_id_remap.keys()))

    root_ids = np.zeros(len(lx_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(lx_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(lx_ids), n_jobs + 1).astype(np.int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        mu.multithread_func(_get_root_ids, multi_args, n_threads=n_threads)

    return lx_ids, np.array(root_ids), lx_id_remap


# @lru_cache(maxsize=None)
def get_lx_overlapping_remappings(cg, chunk_id, time_stamp=None, n_threads=1):
    """ Retrieves sv id to layer mapping for chunk with overlap in positive
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
    neigh_parent_chunk_ids = np.array(neigh_parent_chunk_ids)
    layer_agreement = np.all((neigh_parent_chunk_ids -
                              neigh_parent_chunk_ids[0]) == 0, axis=0)
    stop_layer = np.where(layer_agreement)[0][0] + 1 + chunk_layer
    # stop_layer = cg.n_layers

    print(f"Stop layer: {stop_layer}")

    # Find the parent in the lowest common chunk for each l2 id. These parent
    # ids are referred to as root ids even though they are not necessarily the
    # root id.
    neigh_lx_ids = []
    neigh_lx_id_remap = {}
    neigh_root_ids = []

    safe_lx_ids = []
    unsafe_lx_ids = []
    unsafe_root_ids = []

    # This loop is the main bottleneck
    for neigh_chunk_id in neigh_chunk_ids:
        print(f"Neigh: {neigh_chunk_id} --------------")

        lx_ids, root_ids, lx_id_remap = \
            get_root_lx_remapping(cg, neigh_chunk_id, stop_layer,
                                  time_stamp=time_stamp, n_threads=n_threads)
        neigh_lx_ids.extend(lx_ids)
        neigh_lx_id_remap.update(lx_id_remap)
        neigh_root_ids.extend(root_ids)

        if neigh_chunk_id == chunk_id:
            # The first neigh_chunk_id is the one we are interested in. All lx
            # ids that share no root id with any other lx id are "safe", meaning
            # that we can easily obtain the complete remapping (including
            # overlap) for these. All other ones have to be resolved using the
            # segmentation.
            u_root_ids, u_idx, c_root_ids = np.unique(neigh_root_ids,
                                                      return_counts=True,
                                                      return_index=True)

            safe_lx_ids = lx_ids[u_idx[c_root_ids == 1]]
            unsafe_lx_ids = lx_ids[~np.in1d(lx_ids, safe_lx_ids)]
            unsafe_root_ids = np.unique(root_ids[u_idx[c_root_ids != 1]])

    lx_root_dict = dict(zip(neigh_lx_ids, neigh_root_ids))
    root_lx_dict = collections.defaultdict(list)

    # Future sv id -> lx mapping
    sv_ids = []
    lx_ids_flat = []

    # Do safe ones first
    for i_root_id in range(len(neigh_root_ids)):
        root_lx_dict[neigh_root_ids[i_root_id]].append(neigh_lx_ids[i_root_id])

    for lx_id in safe_lx_ids:
        root_id = lx_root_dict[lx_id]
        for neigh_lx_id in root_lx_dict[root_id]:
            lx_sv_ids = neigh_lx_id_remap[neigh_lx_id]
            sv_ids.extend(lx_sv_ids)
            lx_ids_flat.extend([lx_id] * len(neigh_lx_id_remap[neigh_lx_id]))

    # For the unsafe ones we can only do the in chunk svs
    # But we will map the out of chunk svs to the root id and store the
    # hierarchical information in a dictionary
    for lx_id in unsafe_lx_ids:
        sv_ids.extend(neigh_lx_id_remap[lx_id])
        lx_ids_flat.extend([lx_id] * len(neigh_lx_id_remap[lx_id]))

    unsafe_dict = collections.defaultdict(list)
    for root_id in unsafe_root_ids:
        if np.sum(~np.in1d(root_lx_dict[root_id], unsafe_lx_ids)) == 0:
            continue

        for neigh_lx_id in root_lx_dict[root_id]:
            unsafe_dict[root_id].append(neigh_lx_id)

            if neigh_lx_id in unsafe_lx_ids:
                continue

            sv_ids.extend(neigh_lx_id_remap[neigh_lx_id])
            lx_ids_flat.extend([root_id] * len(neigh_lx_id_remap[neigh_lx_id]))

    # Combine the lists for a (chunk-) global remapping
    sv_remapping = dict(zip(sv_ids, lx_ids_flat))

    return sv_remapping, unsafe_dict


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


def get_meshing_necessities_from_graph(cg, chunk_id, mip):
    layer = cg.get_chunk_layer(chunk_id)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    mesh_block_shape = meshgen_utils.get_mesh_block_shape(cg, layer, mip)
    chunk_offset = (cx, cy, cz) * mesh_block_shape + cg.cv.mip_voxel_offset(mip)
    return layer, mesh_block_shape, chunk_offset


def calculate_quantization_bits_and_range(min_quantization_range, max_draco_bin_size, draco_quantization_bits=None):
    if draco_quantization_bits is None:
        draco_quantization_bits = np.ceil(np.log2(min_quantization_range / max_draco_bin_size + 1))
    num_draco_bins = 2 ** draco_quantization_bits - 1
    draco_bin_size = np.ceil(min_quantization_range / num_draco_bins)
    draco_quantization_range = draco_bin_size * num_draco_bins
    if draco_quantization_range < min_quantization_range + draco_bin_size:
        if draco_bin_size == max_draco_bin_size:
            return calculate_quantization_bits_and_range(min_quantization_range, max_draco_bin_size, draco_quantization_bits + 1)
        else:
            draco_bin_size = draco_bin_size + 1
            draco_quantization_range = draco_quantization_range + num_draco_bins
    return draco_quantization_bits, draco_quantization_range, draco_bin_size


def get_draco_encoding_settings_for_chunk(cg, chunk_id, mip=2, high_padding=1):
    """
    Calculate the proper draco encoding settings for a chunk to ensure proper stitching is possible
    on the layer above. For details about how and why we do this, please see the meshing Readme
    """
    layer, mesh_block_shape, chunk_offset = get_meshing_necessities_from_graph(cg, chunk_id, mip)
    segmentation_resolution = cg.cv.scales[mip]['resolution']
    min_quantization_range = max((mesh_block_shape + high_padding) * segmentation_resolution)
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    # max_draco_bin_size = np.floor(min(segmentation_resolution) / 2)
    draco_quantization_bits, draco_quantization_range, draco_bin_size = calculate_quantization_bits_and_range(min_quantization_range, max_draco_bin_size)
    draco_quantization_origin = chunk_offset - (chunk_offset % draco_bin_size)
    return {
        'quantization_bits': draco_quantization_bits,
        'compression_level': 1,
        'quantization_range': draco_quantization_range,
        'quantization_origin': draco_quantization_origin,
        'create_metadata': True
    }


def get_next_layer_draco_encoding_settings(cg, prev_layer_encoding_settings, next_layer_chunk_id, mip):
    old_draco_bin_size = prev_layer_encoding_settings['quantization_range'] // (2 ** prev_layer_encoding_settings['quantization_bits'] - 1)
    layer, mesh_block_shape, chunk_offset = get_meshing_necessities_from_graph(cg, next_layer_chunk_id, mip)
    segmentation_resolution = cg.cv.scales[mip]['resolution']
    min_quantization_range = max(mesh_block_shape * segmentation_resolution) + 2 * old_draco_bin_size
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    draco_quantization_bits, draco_quantization_range, draco_bin_size = calculate_quantization_bits_and_range(min_quantization_range, max_draco_bin_size)
    draco_quantization_origin = chunk_offset - old_draco_bin_size - ((chunk_offset - old_draco_bin_size) % draco_bin_size)
    return {
        'quantization_bits': draco_quantization_bits,
        'compression_level': 1,
        'quantization_range': draco_quantization_range,
        'quantization_origin': draco_quantization_origin,
        'create_metadata': True
    }


def transform_draco_vertices(mesh, encoding_settings):
    vertices = np.reshape(mesh['vertices'], (mesh['num_vertices']*3,))
    max_quantized_value = 2 ** encoding_settings['quantization_bits'] - 1
    draco_bin_size = encoding_settings['quantization_range'] / max_quantized_value
    assert np.equal(np.mod(draco_bin_size, 1), 0)
    assert np.equal(np.mod(encoding_settings['quantization_range'], 1), 0)
    assert np.equal(np.mod(encoding_settings['quantization_origin'], 1), 0).all()
    for coord in range(3):
        vertices[coord::3] -= encoding_settings['quantization_origin'][coord]
    vertices /= draco_bin_size
    vertices += 0.5
    np.floor(vertices, out=vertices)
    vertices *= draco_bin_size
    for coord in range(3):
        vertices[coord::3] += encoding_settings['quantization_origin'][coord]


def transform_draco_fragment_and_return_encoding_options(cg, fragment, layer, mip, chunk_id):
    # import ipdb
    # ipdb.set_trace()
    fragment_encoding_options = fragment['mesh']['encoding_options']
    if fragment_encoding_options is None:
        raise Error('Draco fragment has no encoding options')
    cur_encoding_settings = {
        'quantization_range': fragment_encoding_options.quantization_range,
        'quantization_bits': fragment_encoding_options.quantization_bits
    }
    node_id = fragment['node_id']
    parent_chunk_ids = cg.get_parent_chunk_ids(node_id)
    fragment_layer = cg.get_chunk_layer(node_id)
    if fragment_layer >= layer:
        raise Error(f'Node {node_id} somehow has greater or equal layer than chunk {chunk_id}')
    assert len(parent_chunk_ids) > layer - fragment_layer
    for next_layer in range(fragment_layer+1, layer+1):
        next_layer_chunk_id = parent_chunk_ids[next_layer - fragment_layer]
        next_encoding_settings = get_next_layer_draco_encoding_settings(cg, cur_encoding_settings, next_layer_chunk_id, mip)
        if next_layer < layer:
            transform_draco_vertices(fragment['mesh'], next_encoding_settings)
        cur_encoding_settings = next_encoding_settings
    return cur_encoding_settings


def merge_draco_meshes(cg, fragments):
    # TODO: change from naive/brute force merging to only merging at quantized chunk boundary
    mdata = [fragment['mesh'] for fragment in fragments]
    vertexct = np.zeros(len(mdata) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x['num_vertices'] for x in mdata])
    vertices = np.concatenate([x['vertices'] for x in mdata])
    faces = np.concatenate([
        mesh['faces'] + vertexct[i] for i, mesh in enumerate(mdata)
    ])
    if len(faces.shape) == 1:
        faces = faces.reshape(-1, 3)
    if vertexct[-1] > 0:
        vertices, faces = np.unique(vertices[faces.reshape(-1)],
                                    return_inverse=True, axis=0)
        faces = faces.reshape(-1,3).astype(np.uint32)
    # is_chunk_aligned = np.any(np.mod(verts, chunk_size) == 0, axis=1)
    # # # uniq_vertices, uniq_faces, vert_face_counts = np.unique(vertices[faces],
    # #                                                         return_inverse=True,
    # #                                                         return_counts=True,
    # #                                                         axis=0)
    # # find all vertices that have exactly 2 duplicates
    # unique_vertices, unique_inverse, counts = np.unique(verts,
    #                                                     return_inverse=True,
    #                                                     return_counts=True,
    #                                                     axis=0)
    # only_double = np.where(counts == 2)[0]
    # is_doubled = np.isin(unique_inverse, only_double)
    # # this stores whether each vertex should be merged or not
    # do_merge = np.array(is_doubled & is_chunk_aligned)

    # # setup an artificial 4th coordinate for vertex positions
    # # which will be unique in general, 
    # # but then repeated for those that are merged
    # new_vertices = np.hstack((verts, np.arange(verts.shape[0])[:, np.newaxis]))
    # new_vertices[do_merge, 3] = -1
    # fa = np.array(faces)
    # n_faces = fa.shape[0]
    # n_dim = fa.shape[1]

    # # use unique to make the artificial vertex list unique and reindex faces
    # vertices, newfaces = np.unique(new_vertices[faces.ravel(),:], return_inverse=True, axis=0)
    # faces = newfaces.reshape((n_faces, n_dim))
    # faces = faces.astype(np.uint32)

    # return vertices[:,0:3], faces
    return {
        'num_vertices': np.uint32(len(vertices)),
        'vertices': np.reshape(vertices, (len(vertices) * 3,)),
        'faces': np.reshape(faces, (len(faces) * 3,))
    }


def merge_draco_meshes_across_boundaries(cg, fragments):
    # TODO: change from naive/brute force merging to only merging at quantized chunk boundary
    mdata = [fragment['mesh'] for fragment in fragments]
    vertexct = np.zeros(len(mdata) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x['num_vertices'] for x in mdata])
    vertices = np.concatenate([x['vertices'] for x in mdata])
    faces = np.concatenate([
        mesh['faces'] + vertexct[i] for i, mesh in enumerate(mdata)
    ])
    if len(faces.shape) == 1:
        faces = faces.reshape(-1, 3)
    if vertexct[-1] > 0:
        vertices, faces = np.unique(vertices[faces.reshape(-1)],
                                    return_inverse=True, axis=0)
        faces = faces.reshape(-1,3).astype(np.uint32)


def black_out_dust_from_segmentation(seg, dust_threshold):
    seg_ids, voxel_count = np.unique(seg, return_counts=True)
    boundary = np.concatenate((seg[0,:,:], seg[-1,:,:], seg[:,0,:], seg[:,-1,:], seg[:,:,0], seg[:,:,-1]), axis=None)
    seg_ids_on_boundary = np.unique(boundary)
    dust_segids = [ sid for sid, ct in zip(seg_ids, voxel_count) if ct < int(dust_threshold) and np.isin(sid, seg_ids_on_boundary, invert=True) ]
    seg = fastremap.mask(seg, dust_segids, in_place=True)


REDIS_HOST = os.environ.get('REDIS_SERVICE_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_SERVICE_PORT', '6379')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', 'dev')
REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'

from pychunkedgraph.utils.general import redis_job
@redis_job(REDIS_URL, 'mesh_frag_test_channel')
def chunk_mesh_task_new_remapping(cg_info, chunk_id, cv_path, cv_mesh_dir=None, mip=2, max_err=320, base_layer=2, lod=0, encoding='draco', time_stamp=None, dust_threshold=None, return_frag_count=False, fragment_batch_size=None):
    cg = chunkedgraph.ChunkedGraph(**cg_info)
    mesh_dir = cv_mesh_dir or cg._mesh_dir
    start_time = time.time()
    result = []

    layer, mesh_block_shape, chunk_offset = get_meshing_necessities_from_graph(cg, chunk_id, mip)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    if layer <= 2:
        assert mip >= cg.cv.mip
        
        high_padding = 1
        result.append((chunk_id, layer, cx, cy, cz))
        print("Retrieving remap table for chunk %s -- (%s, %s, %s, %s)" % (chunk_id, layer, cx, cy, cz))
        mesher = zmesh.Mesher(cg.cv.mip_resolution(mip))
        draco_encoding_settings = get_draco_encoding_settings_for_chunk(cg, chunk_id, mip, high_padding)
        before_time = time.time()
        seg = get_remapped_segmentation(cg, chunk_id, mip, overlap_vx=high_padding, time_stamp=time_stamp)
        print('get_remapped_seg time: ', time.time() - before_time)
        if dust_threshold:
            # before_time = time.time()
            black_out_dust_from_segmentation(seg, dust_threshold)
            # print('dust removal time: ', time.time() - before_time)
        if return_frag_count:
            return np.unique(seg).shape[0]
        # print('get root cache: ', get_root_lx_remapping.cache_info())
        # print(draco_encoding_settings)
        before_time = time.time()
        mesher.mesh(seg.T)
        print('mesh time: ', time.time() - before_time)
        del seg
        simplification_time = 0
        draco_encoding_time = 0
        write_to_cloud_time = 0
        with Storage(cv_path) as storage:
            print('cv path', cv_path)
            print('mesh_dir', mesh_dir)
            print('num ids', len(mesher.ids()))
            result.append(len(mesher.ids()))
            for obj_id in mesher.ids():
                before_time = time.time()
                mesh = mesher.get_mesh(
                    obj_id,
                    simplification_factor=999999,
                    max_simplification_error=max_err
                )
                simplification_time = simplification_time + time.time() - before_time
                mesher.erase(obj_id)
                mesh.vertices[:] += chunk_offset * cg.cv.mip_resolution(mip)
                if encoding == 'draco':
                    before_time = time.time()                    
                    file_contents = DracoPy.encode_mesh_to_buffer(
                        mesh.vertices.flatten('C'), mesh.faces.flatten('C'), 
                        **draco_encoding_settings
                    )
                    draco_encoding_time = draco_encoding_time + time.time() - before_time
                    compress = False
                else:
                    file_contents = mesh.to_precomputed()
                    compress = True
                before_time = time.time()
                storage.put_file(
                    file_path=f'{mesh_dir}/{meshgen_utils.get_mesh_name(cg, obj_id, 0)}',
                    content=file_contents,
                    compress=compress,
                    cache_control='no-cache'
                )
                write_to_cloud_time = write_to_cloud_time + time.time() - before_time
        print('simplification time: ', simplification_time)
        print('draco encoding time: ', draco_encoding_time)
        print('write to cloud time: ', write_to_cloud_time)
        print('total_time: ', time.time() - start_time)
    else:
        # For each node with more than one child, create a new fragment by
        # merging the mesh fragments of the children.
        
        before_time = time.time()
        print("Retrieving children for chunk %s -- (%s, %s, %s, %s)" % (chunk_id, layer, cx, cy, cz))
        range_read = cg.range_read_chunk(layer, cx, cy, cz, columns=column_keys.Hierarchy.Child)
        print('range read chunk time', time.time() - before_time)

        print("Collecting only nodes with more than one child: ", end="")

        before_time = time.time()
        node_ids = np.array(list(range_read.keys()))
        node_rows = np.array(list(range_read.values()))
        child_fragments = np.array([fragment.value for child_fragments_for_node in node_rows for fragment in child_fragments_for_node])
        # Only keep nodes with more than one child        
        multi_child_mask = [len(fragments) > 1 for fragments in child_fragments]
        multi_child_node_ids = node_ids[multi_child_mask]
        multi_child_children_ids = child_fragments[multi_child_mask]
        multi_child_num_children = [len(children) for children in multi_child_children_ids]
        child_fragments_flat = np.array([frag for children_of_node in multi_child_children_ids for frag in children_of_node])
        multi_child_descendants = meshgen_utils.get_downstream_multi_child_nodes(cg, child_fragments_flat)
        start_index = 0
        multi_child_nodes = {}
        for i in range(len(multi_child_node_ids)):
            end_index = start_index + multi_child_num_children[i]
            descendents_for_current_node = multi_child_descendants[start_index:end_index]
            node_id = multi_child_node_ids[i]
            multi_child_nodes[f'{node_id}:0:{meshgen_utils.get_chunk_bbox_str(cg, node_id, 0)}'] = [
                f'{c}:0:{meshgen_utils.get_chunk_bbox_str(cg, c, 0)}' for c in descendents_for_current_node
            ]
            start_index = end_index
        print('new get children time', time.time() - before_time)
        print("get children time", time.time() - before_time)
        print("%d out of %d" % (len(multi_child_nodes), len(node_ids)))
        result.append((chunk_id, len(multi_child_nodes), len(node_ids)))
        if not multi_child_nodes:
            print("Nothing to do", cx, cy, cz)
            return ', '.join(str(x) for x in result)

        retrieving_time = 0
        decoding_time = 0
        transforming_time = 0
        merging_time = 0
        encoding_time = 0
        writing_time = 0
        getting_frags_time = 0
        with Storage(os.path.join(cv_path, mesh_dir)) as storage:
            before_time = time.time()
            vals = multi_child_nodes.values()
            fragment_to_fetch = [fragment for child_fragments in vals for fragment in child_fragments]
            if fragment_batch_size is None:
                files_contents = storage.get_files(fragment_to_fetch)
            else:
                files_contents = storage.get_files(fragment_to_fetch[0:fragment_batch_size])
                fragments_in_batch_processed = 0
                batches_processed = 0
                num_fragments_processed = 0
            fragment_map = {}
            for i in range(len(files_contents)):
                fragment_map[files_contents[i]['filename']] = files_contents[i]
            if fragment_batch_size is None:
                print('getting frags time', time.time() - before_time)
            else:
                extra_getting_frags_time = time.time() - before_time
            i = 0
            for new_fragment_id, fragment_ids_to_fetch in multi_child_nodes.items():
                i += 1
                if i % max(1, len(multi_child_nodes) // 10) == 0:
                    print(f"{i}/{len(multi_child_nodes)}")
                    print('retrieving_time', retrieving_time)
                    print('decoding_time', decoding_time)
                    print('transforming_time', transforming_time)
                    print('merging_time', merging_time)
                    print('encoding_time', encoding_time)
                    print('writing_time', writing_time)
                    print('batch frags time', getting_frags_time + extra_getting_frags_time)

                old_fragments = []
                before_time = time.time()
                missing_fragments = False
                getting_frags_time_for_node = 0 
                for fragment_id in fragment_ids_to_fetch:
                    if fragment_batch_size is not None:
                        fragments_in_batch_processed += 1
                        if fragments_in_batch_processed > fragment_batch_size:
                            before_get_frags_time = time.time()
                            fragments_in_batch_processed = 1
                            batches_processed += 1
                            num_fragments_processed = batches_processed * fragment_batch_size
                            files_contents = storage.get_files(fragment_to_fetch[num_fragments_processed:num_fragments_processed+fragment_batch_size])
                            fragment_map = {}
                            for j in range(len(files_contents)):
                                fragment_map[files_contents[j]['filename']] = files_contents[j]
                            getting_frags_time_for_node = time.time() - before_get_frags_time
                    fragment = fragment_map[fragment_id]
                    filename = fragment['filename']
                    end_of_node_id_index = filename.find(':')
                    if end_of_node_id_index == -1:
                        print(f'Unexpected filename {filename}. Filenames expected in format \'\{node_id}:\{lod}:\{chunk_bbox_string}\'')
                        missing_fragments = True
                    node_id_str = filename[:end_of_node_id_index]                    
                    if fragment['content'] is not None and fragment['error'] is None:
                        try:
                            old_fragments.append({
                                'mesh': decode_draco_mesh_buffer(fragment['content']),
                                'node_id': np.uint64(node_id_str)
                            })
                        except:
                            missing_fragments = True
                            result.append(f'Decoding failed for {fragment_id} in {new_fragment_id}')
                    elif cg.get_chunk_layer(np.uint64(node_id_str)) > 2:
                        missing_fragments = True
                        result.append(f'{fragment_id} missing for {new_fragment_id}')
                decoding_time = decoding_time + time.time() - before_time - getting_frags_time_for_node
                getting_frags_time += getting_frags_time_for_node

                if len(old_fragments) == 0 or missing_fragments:
                    continue

                before_time = time.time()
                draco_encoding_options = None
                for old_fragment in old_fragments:
                    if draco_encoding_options is None:
                        draco_encoding_options = transform_draco_fragment_and_return_encoding_options(cg, old_fragment, layer, mip, chunk_id)
                    else:
                        encoding_options_for_fragment = transform_draco_fragment_and_return_encoding_options(cg, old_fragment, layer, mip, chunk_id)
                        np.testing.assert_equal(draco_encoding_options['quantization_bits'], encoding_options_for_fragment['quantization_bits'])
                        np.testing.assert_equal(draco_encoding_options['quantization_range'], encoding_options_for_fragment['quantization_range'])
                        np.testing.assert_array_equal(draco_encoding_options['quantization_origin'], encoding_options_for_fragment['quantization_origin'])
                transforming_time = transforming_time + time.time() - before_time

                if len(old_fragments) > 1:
                    before_time = time.time()
                    new_fragment = merge_draco_meshes(cg, old_fragments)
                    merging_time = merging_time + time.time() - before_time
                else:
                    new_fragment = old_fragments[0]['mesh']
                    new_fragment['vertices'] = np.reshape(new_fragment['vertices'], (len(new_fragment['vertices']) * 3))

                before_time = time.time()
                new_fragment_b = DracoPy.encode_mesh_to_buffer(new_fragment['vertices'], new_fragment['faces'], **draco_encoding_options)
                encoding_time = encoding_time + time.time() - before_time
                before_time = time.time()

                storage.put_file(new_fragment_id,
                                 new_fragment_b,
                                 content_type='application/octet-stream',
                                 compress=False,
                                 cache_control='no-cache')
                writing_time = writing_time + time.time() - before_time

    print('retrieving_time', retrieving_time)
    print('decoding_time', decoding_time)
    print('transforming_time', transforming_time)
    print('merging_time', merging_time)
    print('encoding_time', encoding_time)
    print('writing_time', writing_time)
    if fragment_batch_size is not None:
        print('batch frags time', getting_frags_time + extra_getting_frags_time)
        print('num fragments processed', num_fragments_processed)
        print('batches processed', batches_processed)
    return ', '.join(str(x) for x in result)


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
