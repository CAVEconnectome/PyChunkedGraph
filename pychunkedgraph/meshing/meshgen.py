import sys
import os
import collections
import datetime
from logging import getLogger
from typing import Sequence
from functools import lru_cache

import numpy as np
import pytz
import cloudvolume
import DracoPy
import zmesh
import fastremap
from scipy import ndimage
from multiwrapper import multiprocessing_utils as mu
from cloudvolume.lib import Vec

sys.path.insert(0, os.path.join(sys.path[0], "../.."))
os.environ["TRAVIS_BRANCH"] = "IDONTKNOWWHYINEEDTHIS"
UTC = pytz.UTC

from pychunkedgraph.logging import pcg_logger
from pychunkedgraph.backend import chunkedgraph  # noqa
from pychunkedgraph.backend.utils import column_keys  # noqa
from pychunkedgraph.meshing import meshgen_utils  # noqa

# Change below to true if debugging and want to see results in stdout
PRINT_FOR_DEBUGGING = False
# Change below to false if debugging and do not need to write to cloud (warning: do not deploy w/ below set to false)
WRITING_TO_CLOUD = True


def decode_draco_mesh_buffer(fragment):
    try:
        mesh_object = DracoPy.decode_buffer_to_mesh(fragment)
        vertices = np.array(mesh_object.points)
        faces = np.array(mesh_object.faces)
    except ValueError:
        raise ValueError("Not a valid draco mesh")

    num_vertices = len(vertices)

    # For now, just return this dict until we figure out
    # how exactly to deal with Draco's lossiness/duplicate vertices
    return {
        "num_vertices": num_vertices,
        "vertices": vertices,
        "faces": faces,
        "encoding_options": mesh_object.encoding_options,
        "encoding_type": "draco",
    }


def get_remapped_segmentation(
    cg, chunk_id, mip=2, overlap_vx=1, time_stamp=None, n_threads=1
):
    """ Downloads + remaps ws segmentation + resolve unclear cases

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
    :param overlap_vx: int
    :param time_stamp:
    :return: remapped segmentation
    """
    assert mip >= cg.cv.mip

    sv_remapping, unsafe_dict = get_lx_overlapping_remappings(
        cg, chunk_id, time_stamp=time_stamp, n_threads=n_threads
    )

    cv = cloudvolume.CloudVolume(cg.cv.cloudpath, mip=mip)
    mip_diff = mip - cg.cv.mip

    mip_chunk_size = cg.chunk_size.astype(int) / np.array(
        [2 ** mip_diff, 2 ** mip_diff, 1]
    )
    mip_chunk_size = mip_chunk_size.astype(int)

    chunk_start = (
        cg.cv.mip_voxel_offset(mip)
        + cg.get_chunk_coordinates(chunk_id) * mip_chunk_size
    )
    chunk_end = chunk_start + mip_chunk_size + overlap_vx
    chunk_end = Vec.clamp(
        chunk_end,
        cg.cv.mip_voxel_offset(mip),
        cg.cv.mip_voxel_offset(mip) + cg.cv.mip_volume_size(mip),
    )

    ws_seg = cv[
        chunk_start[0] : chunk_end[0],
        chunk_start[1] : chunk_end[1],
        chunk_start[2] : chunk_end[2],
    ].squeeze()

    seg = fastremap.mask_except(ws_seg, list(sv_remapping.keys()), in_place=False)
    fastremap.remap(seg, sv_remapping, preserve_missing_labels=True, in_place=True)

    for unsafe_root_id in unsafe_dict.keys():
        bin_seg = seg == unsafe_root_id

        if np.sum(bin_seg) == 0:
            continue

        cc_seg, n_cc = ndimage.label(bin_seg)
        for i_cc in range(1, n_cc + 1):
            bin_cc_seg = cc_seg == i_cc

            overlaps = []
            overlaps.extend(np.unique(seg[-2, :, :][bin_cc_seg[-1, :, :]]))
            overlaps.extend(np.unique(seg[:, -2, :][bin_cc_seg[:, -1, :]]))
            overlaps.extend(np.unique(seg[:, :, -2][bin_cc_seg[:, :, -1]]))
            overlaps = np.unique(overlaps)

            linked_l2_ids = overlaps[np.in1d(overlaps, unsafe_dict[unsafe_root_id])]

            if len(linked_l2_ids) == 0:
                seg[bin_cc_seg] = 0
            else:
                seg[bin_cc_seg] = linked_l2_ids[0]

    return seg


# TODO: refactor (duplicated code with get_remapped_segmentation)
def get_remapped_seg_for_lvl2_nodes(
    cg,
    chunk_id: np.uint64,
    lvl2_nodes: Sequence[np.uint64],
    mip: int = 2,
    overlap_vx: int = 1,
    time_stamp=None,
    n_threads: int = 1,
):
    """ Downloads + remaps ws segmentation + resolve unclear cases, filter out all but specified lvl2_nodes

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
    :param overlap_vx: int
    :param time_stamp:
    :return: remapped segmentation
    """
    # Determine the segmentation bounding box to download given cg, chunk_id, and mip. Then download
    cv = cloudvolume.CloudVolume(cg.cv.cloudpath, mip=mip)
    mip_diff = mip - cg.cv.mip

    mip_chunk_size = cg.chunk_size.astype(int) / np.array(
        [2 ** mip_diff, 2 ** mip_diff, 1]
    )
    mip_chunk_size = mip_chunk_size.astype(int)

    chunk_start = (
        cg.cv.mip_voxel_offset(mip)
        + cg.get_chunk_coordinates(chunk_id) * mip_chunk_size
    )
    chunk_end = chunk_start + mip_chunk_size + overlap_vx
    chunk_end = Vec.clamp(
        chunk_end,
        cg.cv.mip_voxel_offset(mip),
        cg.cv.mip_voxel_offset(mip) + cg.cv.mip_volume_size(mip),
    )

    seg = cv[
        chunk_start[0] : chunk_end[0],
        chunk_start[1] : chunk_end[1],
        chunk_start[2] : chunk_end[2],
    ].squeeze()

    sv_of_lvl2_nodes = cg.get_children(lvl2_nodes)

    # Check which of the lvl2_nodes meet the chunk boundary
    node_ids_on_the_border = []
    remapping = {}
    for node, sv_list in sv_of_lvl2_nodes.items():
        node_on_the_border = False
        for sv_id in sv_list:
            remapping[sv_id] = node
            # If a node_id is on the chunk_boundary, we must check the overlap region to see if the meshes' end will be open or closed
            if (not node_on_the_border) and (
                np.isin(sv_id, seg[-2, :, :])
                or np.isin(sv_id, seg[:, -2, :])
                or np.isin(sv_id, seg[:, :, -2])
            ):
                node_on_the_border = True
                node_ids_on_the_border.append(node)

    node_ids_on_the_border = np.array(node_ids_on_the_border)
    if len(node_ids_on_the_border) > 0:
        overlap_region = np.concatenate(
            (seg[:, :, -1], seg[:, -1, :], seg[-1, :, :]), axis=None
        )
        overlap_sv_ids = np.unique(overlap_region)
        if overlap_sv_ids[0] == 0:
            overlap_sv_ids = overlap_sv_ids[1:]
        # Get the remappings for the supervoxels in the overlap region
        sv_remapping, unsafe_dict = get_lx_overlapping_remappings_for_nodes_and_svs(
            cg, chunk_id, node_ids_on_the_border, overlap_sv_ids, time_stamp, n_threads
        )
        sv_remapping.update(remapping)
        fastremap.mask_except(seg, list(sv_remapping.keys()), in_place=True)
        fastremap.remap(seg, sv_remapping, preserve_missing_labels=True, in_place=True)
        # For some supervoxel, they could map to multiple l2 nodes in the chunk, so we must perform a connected component analysis
        # to see which l2 node they are adjacent to
        for unsafe_root_id in unsafe_dict.keys():
            bin_seg = seg == unsafe_root_id

            if np.sum(bin_seg) == 0:
                continue

            cc_seg, n_cc = ndimage.label(bin_seg)
            for i_cc in range(1, n_cc + 1):
                bin_cc_seg = cc_seg == i_cc

                overlaps = []
                overlaps.extend(np.unique(seg[-2, :, :][bin_cc_seg[-1, :, :]]))
                overlaps.extend(np.unique(seg[:, -2, :][bin_cc_seg[:, -1, :]]))
                overlaps.extend(np.unique(seg[:, :, -2][bin_cc_seg[:, :, -1]]))
                overlaps = np.unique(overlaps)

                linked_l2_ids = overlaps[np.in1d(overlaps, unsafe_dict[unsafe_root_id])]

                if len(linked_l2_ids) == 0:
                    seg[bin_cc_seg] = 0
                else:
                    seg[bin_cc_seg] = linked_l2_ids[0]
    else:
        # If no nodes in our subset meet the chunk boundary we can simply retrieve the sv of the nodes in the subset
        fastremap.mask_except(seg, list(remapping.keys()), in_place=True)
        fastremap.remap(seg, remapping, preserve_missing_labels=True, in_place=True)

    return seg


@lru_cache(maxsize=32)
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

    pcg_logger.log(pcg_logger.level, f"\n{chunk_id} ----------------\n")

    lower_remaps = {}
    if cg.get_chunk_layer(chunk_id) > 2:
        for lower_chunk_id in cg.get_chunk_child_ids(chunk_id):
            # TODO speedup
            lower_remaps.update(
                get_higher_to_lower_remapping(cg, lower_chunk_id, time_stamp=time_stamp)
            )

    rr_chunk = cg.range_read_chunk(
        chunk_id=chunk_id, columns=column_keys.Hierarchy.Child, time_stamp=time_stamp
    )

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


@lru_cache(maxsize=32)
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
        root_ids[start_id:end_id] = cg.get_roots(
            lx_ids[start_id:end_id], stop_layer=stop_layer
        )

    lx_id_remap = get_higher_to_lower_remapping(cg, chunk_id, time_stamp=time_stamp)

    lx_ids = np.array(list(lx_id_remap.keys()))

    root_ids = np.zeros(len(lx_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(lx_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(lx_ids), n_jobs + 1).astype(int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        mu.multithread_func(_get_root_ids, multi_args, n_threads=n_threads)

    return lx_ids, np.array(root_ids), lx_id_remap


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
                neigh_chunk_id = cg.get_chunk_id(x=x, y=y, z=z, layer=chunk_layer)
                neigh_chunk_ids.append(neigh_chunk_id)

                # Get parent chunk ids
                parent_chunk_ids = cg.get_parent_chunk_ids(neigh_chunk_id)
                neigh_parent_chunk_ids.append(parent_chunk_ids)

    # Find lowest common chunk
    neigh_parent_chunk_ids = np.array(neigh_parent_chunk_ids)
    layer_agreement = np.all(
        (neigh_parent_chunk_ids - neigh_parent_chunk_ids[0]) == 0, axis=0
    )
    stop_layer = np.where(layer_agreement)[0][0] + 1 + chunk_layer
    # stop_layer = cg.n_layers

    # cg.logger.info(f"Stop layer: {stop_layer}")

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
        # cg.logger.info(f"Neigh: {neigh_chunk_id} --------------")

        lx_ids, root_ids, lx_id_remap = get_root_lx_remapping(
            cg, neigh_chunk_id, stop_layer, time_stamp=time_stamp, n_threads=n_threads
        )
        neigh_lx_ids.extend(lx_ids)
        neigh_lx_id_remap.update(lx_id_remap)
        neigh_root_ids.extend(root_ids)

        if neigh_chunk_id == chunk_id:
            # The first neigh_chunk_id is the one we are interested in. All lx
            # ids that share no root id with any other lx id are "safe", meaning
            # that we can easily obtain the complete remapping (including
            # overlap) for these. All other ones have to be resolved using the
            # segmentation.
            _, u_idx, c_root_ids = np.unique(
                neigh_root_ids, return_counts=True, return_index=True
            )

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


def get_root_remapping_for_nodes_and_svs(
    cg, chunk_id, node_ids, sv_ids, stop_layer, time_stamp, n_threads=1
):
    """ Retrieves root to node id mapping for specified node ids and supervoxel ids

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param node_ids: [np.uint64]
    :param stop_layer: int
    :param time_stamp: datetime object
    :return: multiples
    """

    def _get_root_ids(args):
        start_id, end_id = args

        root_ids[start_id:end_id] = cg.get_roots(
            combined_ids[start_id:end_id], stop_layer=stop_layer, time_stamp=time_stamp
        )

    rr = cg.range_read_chunk(
        chunk_id=chunk_id, columns=column_keys.Hierarchy.Parent, time_stamp=time_stamp
    )
    chunk_l2_ids = np.array(list(rr.keys()), dtype=np.uint64)
    combined_ids = np.concatenate((node_ids, sv_ids, chunk_l2_ids))

    root_ids = np.zeros(len(combined_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(combined_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(combined_ids), n_jobs + 1).astype(int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        mu.multithread_func(_get_root_ids, multi_args, n_threads=n_threads)

    sv_ids_index = len(node_ids)
    chunk_ids_index = len(node_ids) + len(sv_ids)

    return (
        root_ids[0:sv_ids_index],
        root_ids[sv_ids_index:chunk_ids_index],
        root_ids[chunk_ids_index:],
    )


def get_lx_overlapping_remappings_for_nodes_and_svs(
    cg,
    chunk_id: np.uint64,
    node_ids: Sequence[np.uint64],
    sv_ids: Sequence[np.uint64],
    time_stamp=None,
    n_threads: int = 1,
):
    """ Retrieves sv id to layer mapping for chunk with overlap in positive
        direction (one chunk)

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param node_ids: list of np.uint64
    :param sv_ids: list of np.uint64
    :param time_stamp: datetime object
    :param n_threads: int
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
                neigh_chunk_id = cg.get_chunk_id(x=x, y=y, z=z, layer=chunk_layer)
                neigh_chunk_ids.append(neigh_chunk_id)

                # Get parent chunk ids
                parent_chunk_ids = cg.get_parent_chunk_ids(neigh_chunk_id)
                neigh_parent_chunk_ids.append(parent_chunk_ids)

    # Find lowest common chunk
    neigh_parent_chunk_ids = np.array(neigh_parent_chunk_ids)
    layer_agreement = np.all(
        (neigh_parent_chunk_ids - neigh_parent_chunk_ids[0]) == 0, axis=0
    )
    stop_layer = np.where(layer_agreement)[0][0] + 1 + chunk_layer
    # stop_layer = cg.n_layers

    pcg_logger.log(pcg_logger.level, f"Stop layer: {stop_layer}")

    # Find the parent in the lowest common chunk for each node id and sv id. These parent
    # ids are referred to as root ids even though they are not necessarily the
    # root id.
    node_root_ids, sv_root_ids, chunks_root_ids = get_root_remapping_for_nodes_and_svs(
        cg, chunk_id, node_ids, sv_ids, stop_layer, time_stamp, n_threads
    )

    u_root_ids, c_root_ids = np.unique(chunks_root_ids, return_counts=True)

    # All l2 ids that share no root id with any other l2 id in the chunk are "safe", meaning
    # that we can easily obtain the complete remapping (including
    # overlap) for these. All other ones have to be resolved using the
    # segmentation.

    root_sorted_idx = np.argsort(u_root_ids)
    node_sorted_index = np.searchsorted(u_root_ids[root_sorted_idx], node_root_ids)
    node_root_counts = c_root_ids[root_sorted_idx][node_sorted_index]
    unsafe_root_ids = node_root_ids[np.where(node_root_counts > 1)]
    safe_node_ids = node_ids[~np.isin(node_root_ids, unsafe_root_ids)]

    node_to_root_dict = dict(zip(node_ids, node_root_ids))

    # Future sv id -> lx mapping
    sv_ids_to_remap = []
    node_ids_flat = []

    # Do safe ones first
    for node_id in safe_node_ids:
        root_id = node_to_root_dict[node_id]
        sv_ids_to_add = sv_ids[np.where(sv_root_ids == root_id)]
        if len(sv_ids_to_add) > 0:
            sv_ids_to_remap.extend(sv_ids_to_add)
            node_ids_flat.extend([node_id] * len(sv_ids_to_add))

    # For the unsafe roots, we will map the out of chunk svs to the root id and store the
    # hierarchical information in a dictionary
    unsafe_dict = collections.defaultdict(list)
    for root_id in unsafe_root_ids:
        sv_ids_to_add = sv_ids[np.where(sv_root_ids == root_id)]
        if len(sv_ids_to_add) > 0:
            relevant_node_ids = node_ids[np.where(node_root_ids == root_id)]
            if len(relevant_node_ids) > 0:
                unsafe_dict[root_id].extend(relevant_node_ids)
                sv_ids_to_remap.extend(sv_ids_to_add)
                node_ids_flat.extend([root_id] * len(sv_ids_to_add))

    # Combine the lists for a (chunk-) global remapping
    sv_remapping = dict(zip(sv_ids_to_remap, node_ids_flat))

    return sv_remapping, unsafe_dict


def get_meshing_necessities_from_graph(cg, chunk_id: np.uint64, mip: int):
    """ Given a chunkedgraph, chunk_id, and mip level, return the voxel dimensions of the chunk to be meshed (mesh_block_shape)
    and the chunk origin in the dataset in nm.

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :param mip: int
    """
    layer = cg.get_chunk_layer(chunk_id)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    mesh_block_shape = meshgen_utils.get_mesh_block_shape_for_mip(cg, layer, mip)
    chunk_offset = (
        (cx, cy, cz) * mesh_block_shape + cg.cv.mip_voxel_offset(mip)
    ) * cg.cv.mip_resolution(mip)
    return layer, mesh_block_shape, chunk_offset


def calculate_quantization_bits_and_range(
    min_quantization_range, max_draco_bin_size, draco_quantization_bits=None
):
    if draco_quantization_bits is None:
        draco_quantization_bits = np.ceil(
            np.log2(min_quantization_range / max_draco_bin_size + 1)
        )
    num_draco_bins = 2 ** draco_quantization_bits - 1
    draco_bin_size = np.ceil(min_quantization_range / num_draco_bins)
    draco_quantization_range = draco_bin_size * num_draco_bins
    if draco_quantization_range < min_quantization_range + draco_bin_size:
        if draco_bin_size == max_draco_bin_size:
            return calculate_quantization_bits_and_range(
                min_quantization_range, max_draco_bin_size, draco_quantization_bits + 1
            )
        else:
            draco_bin_size = draco_bin_size + 1
            draco_quantization_range = draco_quantization_range + num_draco_bins
    return draco_quantization_bits, draco_quantization_range, draco_bin_size


def get_draco_encoding_settings_for_chunk(
    cg, chunk_id: np.uint64, mip: int = 2, high_padding: int = 1
):
    """ Calculate the proper draco encoding settings for a chunk to ensure proper stitching is possible
    on the layer above. For details about how and why we do this, please see the meshing Readme

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :param mip: int
    :param high_padding: int
    """
    _, mesh_block_shape, chunk_offset = get_meshing_necessities_from_graph(
        cg, chunk_id, mip
    )
    segmentation_resolution = cg.cv.scales[mip]["resolution"]
    min_quantization_range = max(
        (mesh_block_shape + high_padding) * segmentation_resolution
    )
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    draco_quantization_bits, draco_quantization_range, draco_bin_size = calculate_quantization_bits_and_range(
        min_quantization_range, max_draco_bin_size
    )
    draco_quantization_origin = chunk_offset - (chunk_offset % draco_bin_size)
    return {
        "quantization_bits": draco_quantization_bits,
        "compression_level": 1,
        "quantization_range": draco_quantization_range,
        "quantization_origin": draco_quantization_origin,
        "create_metadata": True,
    }


def get_next_layer_draco_encoding_settings(
    cg, prev_layer_encoding_settings, next_layer_chunk_id, mip
):
    old_draco_bin_size = prev_layer_encoding_settings["quantization_range"] // (
        2 ** prev_layer_encoding_settings["quantization_bits"] - 1
    )
    _, mesh_block_shape, chunk_offset = get_meshing_necessities_from_graph(
        cg, next_layer_chunk_id, mip
    )
    segmentation_resolution = cg.cv.scales[mip]["resolution"]
    min_quantization_range = (
        max(mesh_block_shape * segmentation_resolution) + 2 * old_draco_bin_size
    )
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    draco_quantization_bits, draco_quantization_range, draco_bin_size = calculate_quantization_bits_and_range(
        min_quantization_range, max_draco_bin_size
    )
    draco_quantization_origin = (
        chunk_offset
        - old_draco_bin_size
        - ((chunk_offset - old_draco_bin_size) % draco_bin_size)
    )
    return {
        "quantization_bits": draco_quantization_bits,
        "compression_level": 1,
        "quantization_range": draco_quantization_range,
        "quantization_origin": draco_quantization_origin,
        "create_metadata": True,
    }


def transform_draco_vertices(mesh, encoding_settings):
    vertices = np.reshape(mesh["vertices"], (mesh["num_vertices"] * 3,))
    max_quantized_value = 2 ** encoding_settings["quantization_bits"] - 1
    draco_bin_size = encoding_settings["quantization_range"] / max_quantized_value
    assert np.equal(np.mod(draco_bin_size, 1), 0)
    assert np.equal(np.mod(encoding_settings["quantization_range"], 1), 0)
    assert np.equal(np.mod(encoding_settings["quantization_origin"], 1), 0).all()
    for coord in range(3):
        vertices[coord::3] -= encoding_settings["quantization_origin"][coord]
    vertices /= draco_bin_size
    vertices += 0.5
    np.floor(vertices, out=vertices)
    vertices *= draco_bin_size
    for coord in range(3):
        vertices[coord::3] += encoding_settings["quantization_origin"][coord]


def transform_draco_fragment_and_return_encoding_options(
    cg, fragment, layer, mip, chunk_id
):
    fragment_encoding_options = fragment["mesh"]["encoding_options"]
    if fragment_encoding_options is None:
        raise ValueError("Draco fragment has no encoding options")
    cur_encoding_settings = {
        "quantization_range": fragment_encoding_options.quantization_range,
        "quantization_bits": fragment_encoding_options.quantization_bits,
    }
    node_id = fragment["node_id"]
    parent_chunk_ids = cg.get_parent_chunk_ids(node_id)
    fragment_layer = cg.get_chunk_layer(node_id)
    if fragment_layer >= layer:
        raise ValueError(
            f"Node {node_id} somehow has greater or equal layer than chunk {chunk_id}"
        )
    assert len(parent_chunk_ids) > layer - fragment_layer
    for next_layer in range(fragment_layer + 1, layer + 1):
        next_layer_chunk_id = parent_chunk_ids[next_layer - fragment_layer]
        next_encoding_settings = get_next_layer_draco_encoding_settings(
            cg, cur_encoding_settings, next_layer_chunk_id, mip
        )
        if next_layer < layer:
            transform_draco_vertices(fragment["mesh"], next_encoding_settings)
        cur_encoding_settings = next_encoding_settings
    return cur_encoding_settings


def merge_draco_meshes_across_boundaries(cg, fragments, chunk_id, mip, high_padding):
    """
    Merge a list of draco mesh fragments, removing duplicate vertices that lie
    on the chunk boundary where the meshes meet.
    """
    vertexct = np.zeros(len(fragments) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x["mesh"]["num_vertices"] for x in fragments])
    vertices = np.concatenate([x["mesh"]["vertices"] for x in fragments])
    faces = np.concatenate(
        [mesh["mesh"]["faces"] + vertexct[i] for i, mesh in enumerate(fragments)]
    )
    del fragments

    if vertexct[-1] > 0:
        chunk_coords = cg.get_chunk_coordinates(chunk_id)
        coords_bottom_corner_child_chunk = chunk_coords * 2 + 1
        child_chunk_id = cg.get_chunk_id(
            None, cg.get_chunk_layer(chunk_id) - 1, *coords_bottom_corner_child_chunk
        )
        _, _, child_chunk_offset = get_meshing_necessities_from_graph(
            cg, child_chunk_id, mip
        )
        # Get the draco encoding settings for the child chunk in the "bottom corner" of the chunk_id chunk
        draco_encoding_settings_smaller_chunk = get_draco_encoding_settings_for_chunk(
            cg, child_chunk_id, mip=mip, high_padding=high_padding
        )
        draco_bin_size = draco_encoding_settings_smaller_chunk["quantization_range"] / (
            2 ** draco_encoding_settings_smaller_chunk["quantization_bits"] - 1
        )
        # Calculate which draco bin the child chunk's boundaries were placed into (for each x,y,z of boundary)
        chunk_boundary_bin_index = np.floor(
            (
                child_chunk_offset
                - draco_encoding_settings_smaller_chunk["quantization_origin"]
            )
            / draco_bin_size
            + np.float32(0.5)
        )
        # Now we can determine where the three planes of the quantized chunk boundary are
        quantized_chunk_boundary = (
            draco_encoding_settings_smaller_chunk["quantization_origin"]
            + chunk_boundary_bin_index * draco_bin_size
        )
        # Separate the vertices that are on the quantized chunk boundary from those that aren't
        are_chunk_aligned = (vertices == quantized_chunk_boundary).any(axis=1)
        vertices = np.hstack((vertices, np.arange(vertexct[-1])[:, np.newaxis]))
        chunk_aligned = vertices[are_chunk_aligned]
        not_chunk_aligned = vertices[~are_chunk_aligned]
        del vertices
        del are_chunk_aligned
        faces_remapping = {}
        # Those that are not simply pass through (simple remap)
        if len(not_chunk_aligned) > 0:
            not_chunk_aligned_remap = dict(
                zip(
                    not_chunk_aligned[:, 3].astype(np.uint32),
                    np.arange(len(not_chunk_aligned), dtype=np.uint32),
                )
            )
            faces_remapping.update(not_chunk_aligned_remap)
        # Those that are on the boundary we remove duplicates
        if len(chunk_aligned) > 0:
            unique_chunk_aligned, inverse_to_chunk_aligned = np.unique(
                chunk_aligned[:, 0:3], return_inverse=True, axis=0
            )
            chunk_aligned_remap = dict(
                zip(
                    chunk_aligned[:, 3].astype(np.uint32),
                    np.uint32(len(not_chunk_aligned))
                    + inverse_to_chunk_aligned.astype(np.uint32),
                )
            )
            faces_remapping.update(chunk_aligned_remap)
            vertices = np.concatenate((not_chunk_aligned[:, 0:3], unique_chunk_aligned))
        else:
            vertices = not_chunk_aligned[:, 0:3]
        # Remap the faces to their new vertex indices
        fastremap.remap(faces, faces_remapping, in_place=True)

    return {
        "num_vertices": np.uint32(len(vertices)),
        "vertices": vertices[:, 0:3].reshape(-1),
        "faces": faces,
    }


def black_out_dust_from_segmentation(seg, dust_threshold):
    """ Black out (set to 0) IDs in segmentation not on the segmentation border that have less voxels than dust_threshold

    :param seg: 3D segmentation (usually uint64)
    :param dust_threshold: int
    :return:
    """
    seg_ids, voxel_count = np.unique(seg, return_counts=True)
    boundary = np.concatenate(
        (
            seg[-2, :, :],
            seg[-1, :, :],
            seg[:, -2, :],
            seg[:, -1, :],
            seg[:, :, -2],
            seg[:, :, -1],
        ),
        axis=None,
    )
    seg_ids_on_boundary = np.unique(boundary)
    dust_segids = [
        sid
        for sid, ct in zip(seg_ids, voxel_count)
        if ct < int(dust_threshold) and np.isin(sid, seg_ids_on_boundary, invert=True)
    ]
    seg = fastremap.mask(seg, dust_segids, in_place=True)


def _get_timestamp_from_node_ids(cg, node_ids):
    timestamps = cg.get_node_timestamps(node_ids)
    return max(timestamps) + datetime.timedelta(milliseconds=1)


def remeshing(
    cg,
    l2_node_ids: Sequence[np.uint64],
    stop_layer: int = None,
    mesh_path: str = None,
    mip: int = 2,
    max_err: int = 320,
):
    """ Given a chunkedgraph, a list of level 2 nodes, perform remeshing and stitching up the node hierarchy (or up to the stop_layer)

    :param cg: chunkedgraph instance
    :param l2_node_ids: list of uint64
    :param stop_layer: int
    :param mesh_path: str
    :param mip: int
    :param max_err: int
    :return:
    """
    l2_chunk_dict = collections.defaultdict(set)
    # Find the chunk_ids of the l2_node_ids

    def add_nodes_to_l2_chunk_dict(ids):
        for node_id in ids:
            chunk_id = cg.get_chunk_id(node_id)
            l2_chunk_dict[chunk_id].add(node_id)

    add_nodes_to_l2_chunk_dict(l2_node_ids)
    for chunk_id, node_ids in l2_chunk_dict.items():
        if PRINT_FOR_DEBUGGING:
            pcg_logger.log(pcg_logger.level, f"remeshing {chunk_id} {node_ids}")
        l2_time_stamp = _get_timestamp_from_node_ids(cg, node_ids)
        # Remesh the l2_node_ids
        chunk_mesh_task_new_remapping(
            cg.get_serialized_info(),
            chunk_id,
            mesh_path=mesh_path,
            mip=mip,
            fragment_batch_size=20,
            node_id_subset=node_ids,
            cg=cg,
            time_stamp=l2_time_stamp
        )
    chunk_dicts = []
    max_layer = stop_layer or cg._n_layers
    for layer in range(3, max_layer + 1):
        chunk_dicts.append(collections.defaultdict(set))
    cur_chunk_dict = l2_chunk_dict
    # Find the parents of each l2_node_id up to the stop_layer, as well as their associated chunk_ids
    for layer in range(3, max_layer + 1):
        for _, node_ids in cur_chunk_dict.items():
            parent_nodes = cg.get_parents(node_ids)
            for parent_node in parent_nodes:
                chunk_layer = cg.get_chunk_layer(parent_node)
                index_in_dict_array = chunk_layer - 3
                if index_in_dict_array < len(chunk_dicts):
                    chunk_id = cg.get_chunk_id(parent_node)
                    chunk_dicts[index_in_dict_array][chunk_id].add(parent_node)
        cur_chunk_dict = chunk_dicts[layer - 3]
    for chunk_dict in chunk_dicts:
        for chunk_id, node_ids in chunk_dict.items():
            if PRINT_FOR_DEBUGGING:
                pcg_logger.log(pcg_logger.level, f"remeshing {chunk_id} {node_ids}")
            # Stitch the meshes of the parents we found in the previous loop
            chunk_mesh_task_new_remapping(
                cg.get_serialized_info(),
                chunk_id,
                mesh_path=mesh_path,
                mip=mip,
                fragment_batch_size=20,
                node_id_subset=node_ids,
                cg=cg,
            )


# TODO: refactor this bloated function
def chunk_mesh_task_new_remapping(
    cg_info,
    chunk_id,
    mesh_path,
    mip=2,
    max_err=320,
    base_layer=2,
    lod=0,
    encoding="draco",
    time_stamp=None,
    dust_threshold=None,
    return_frag_count=False,
    fragment_batch_size=None,
    node_id_subset=None,
    cg=None,
):
    from cloudfiles import CloudFiles
    if cg is None:
        cg = chunkedgraph.ChunkedGraph(**cg_info)
    mesh_path = mesh_path or cg.cv_mesh_path
    cf = CloudFiles(mesh_path)
    result = []

    layer, _, chunk_offset = get_meshing_necessities_from_graph(cg, chunk_id, mip)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    high_padding = 1
    if layer <= 2:
        assert mip >= cg.cv.mip

        result.append((chunk_id, layer, cx, cy, cz))
        pcg_logger.log(pcg_logger.level, f"Retrieving remap table for {chunk_id} - {layer}, {cx}, {cy}, {cz}")
        mesher = zmesh.Mesher(cg.cv.mip_resolution(mip))
        draco_encoding_settings = get_draco_encoding_settings_for_chunk(
            cg, chunk_id, mip, high_padding
        )
        if node_id_subset is None:
            seg = get_remapped_segmentation(
                cg, chunk_id, mip, overlap_vx=high_padding, time_stamp=time_stamp
            )
        else:
            seg = get_remapped_seg_for_lvl2_nodes(
                cg,
                chunk_id,
                node_id_subset,
                mip=mip,
                overlap_vx=high_padding,
                time_stamp=time_stamp,
            )
        if dust_threshold:
            black_out_dust_from_segmentation(seg, dust_threshold)
        if return_frag_count:
            return np.unique(seg).shape[0]
        mesher.mesh(seg.T)
        del seg

        if PRINT_FOR_DEBUGGING:
            pcg_logger.log(pcg_logger.level, f"mesh_path {mesh_path}")
            pcg_logger.log(pcg_logger.level, f"num ids {len(mesher.ids())}")
        result.append(len(mesher.ids()))
        for obj_id in mesher.ids():
            mesh = mesher.get_mesh(
                obj_id,
                simplification_factor=999999,
                max_simplification_error=max_err,
            )
            mesher.erase(obj_id)
            mesh.vertices[:] += chunk_offset
            if encoding == "draco":
                try:
                    file_contents = DracoPy.encode_mesh_to_buffer(
                        mesh.vertices.flatten("C"),
                        mesh.faces.flatten("C"),
                        **draco_encoding_settings,
                    )
                except:
                    result.append(
                        f"{obj_id} failed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
                    )
                    continue
                compress = False
            else:
                file_contents = mesh.to_precomputed()
                compress = True
            if WRITING_TO_CLOUD:
                cf.put(
                    path=f"{meshgen_utils.get_mesh_name(cg, obj_id)}",
                    content=file_contents,
                    compress=compress,
                    cache_control="no-cache",
                )
    else:
        # For each node with more than one child, create a new fragment by
        # merging the mesh fragments of the children.

        pcg_logger.log(pcg_logger.level, f"Retrieving children for {chunk_id} - {layer}, {cx}, {cy}, {cz}")
        if node_id_subset is None:
            range_read = cg.range_read_chunk(
                layer, cx, cy, cz, columns=column_keys.Hierarchy.Child
            )
        else:
            range_read = cg.read_node_id_rows(
                node_ids=node_id_subset, columns=column_keys.Hierarchy.Child
            )

        pcg_logger.log(pcg_logger.level, f"Collecting only nodes with more than one child:")
        node_ids = np.array(list(range_read.keys()))
        node_rows = np.array(list(range_read.values()))
        child_fragments = np.array(
            [
                fragment.value
                for child_fragments_for_node in node_rows
                for fragment in child_fragments_for_node
            ]
        )
        # Only keep nodes with more than one child
        multi_child_mask = [len(fragments) > 1 for fragments in child_fragments]
        multi_child_node_ids = node_ids[multi_child_mask]
        multi_child_children_ids = child_fragments[multi_child_mask]
        # Store how many children each node has, because we will retrieve all children at once
        multi_child_num_children = [
            len(children) for children in multi_child_children_ids
        ]
        child_fragments_flat = np.array(
            [
                frag
                for children_of_node in multi_child_children_ids
                for frag in children_of_node
            ]
        )
        multi_child_descendants = meshgen_utils.get_downstream_multi_child_nodes(
            cg, child_fragments_flat
        )
        start_index = 0
        multi_child_nodes = {}
        for i in range(len(multi_child_node_ids)):
            end_index = start_index + multi_child_num_children[i]
            descendents_for_current_node = multi_child_descendants[
                start_index:end_index
            ]
            node_id = multi_child_node_ids[i]
            multi_child_nodes[
                f"{node_id}:0:{meshgen_utils.get_chunk_bbox_str(cg, node_id)}"
            ] = [
                f"{c}:0:{meshgen_utils.get_chunk_bbox_str(cg, c)}"
                for c in descendents_for_current_node
            ]
            start_index = end_index
        pcg_logger.log(pcg_logger.level, f"{len(multi_child_nodes)} out of {len(node_ids)}")
        result.append((chunk_id, len(multi_child_nodes), len(node_ids)))
        if not multi_child_nodes:
            pcg_logger.log(pcg_logger.level, f"Nothing to do, {cx}, {cy}, {cz}")
            return ", ".join(str(x) for x in result)

        vals = multi_child_nodes.values()
        fragment_to_fetch = [
            fragment for child_fragments in vals for fragment in child_fragments
        ]
        if fragment_batch_size is None:
            files_contents = cf.get(fragment_to_fetch)
        else:
            files_contents = cf.get(
                fragment_to_fetch[0:fragment_batch_size]
            )
            fragments_in_batch_processed = 0
            batches_processed = 0
            num_fragments_processed = 0
        fragment_map = {}
        for i in range(len(files_contents)):
            fragment_map[files_contents[i]["path"]] = files_contents[i]
        i = 0
        for new_fragment_id, fragment_ids_to_fetch in multi_child_nodes.items():
            i += 1
            if i % max(1, len(multi_child_nodes) // 10) == 0:
                pcg_logger.log(pcg_logger.level, f"{i}/{len(multi_child_nodes)}")

            old_fragments = []
            missing_fragments = False
            for fragment_id in fragment_ids_to_fetch:
                if fragment_batch_size is not None:
                    fragments_in_batch_processed += 1
                    if fragments_in_batch_processed > fragment_batch_size:
                        fragments_in_batch_processed = 1
                        batches_processed += 1
                        num_fragments_processed = (
                            batches_processed * fragment_batch_size
                        )
                        files_contents = cf.get(
                            fragment_to_fetch[
                                num_fragments_processed : num_fragments_processed
                                + fragment_batch_size
                            ]
                        )
                        fragment_map = {}
                        for j in range(len(files_contents)):
                            fragment_map[
                                files_contents[j]["path"]
                            ] = files_contents[j]
                fragment = fragment_map[fragment_id]
                filename = fragment["path"]
                end_of_node_id_index = filename.find(":")
                if end_of_node_id_index == -1:
                    msg = f"Unexpected filename {filename}. Expected format '\{node_id}:\{lod}:\{meshgen_utils.get_chunk_bbox_str(cg, node_id)}'"
                    pcg_logger.log(pcg_logger.level, msg)
                    missing_fragments = True
                node_id_str = filename[:end_of_node_id_index]
                if fragment["content"] is not None and fragment["error"] is None:
                    try:
                        old_fragments.append(
                            {
                                "mesh": decode_draco_mesh_buffer(
                                    fragment["content"]
                                ),
                                "node_id": np.uint64(node_id_str),
                            }
                        )
                    except:
                        missing_fragments = True
                        new_fragment_str = new_fragment_id[
                            0 : new_fragment_id.find(":")
                        ]
                        result.append(
                            f"Decoding failed for {node_id_str} in {new_fragment_str}"
                        )
                elif cg.get_chunk_layer(np.uint64(node_id_str)) > 2:
                    result.append(f"{fragment_id} missing for {new_fragment_id}")

            if len(old_fragments) == 0 or missing_fragments:
                result.append(f"No meshes for {new_fragment_id}")
                continue

            draco_encoding_options = None
            for old_fragment in old_fragments:
                if draco_encoding_options is None:
                    draco_encoding_options = transform_draco_fragment_and_return_encoding_options(
                        cg, old_fragment, layer, mip, chunk_id
                    )
                else:
                    encoding_options_for_fragment = transform_draco_fragment_and_return_encoding_options(
                        cg, old_fragment, layer, mip, chunk_id
                    )
                    np.testing.assert_equal(
                        draco_encoding_options["quantization_bits"],
                        encoding_options_for_fragment["quantization_bits"],
                    )
                    np.testing.assert_equal(
                        draco_encoding_options["quantization_range"],
                        encoding_options_for_fragment["quantization_range"],
                    )
                    np.testing.assert_array_equal(
                        draco_encoding_options["quantization_origin"],
                        encoding_options_for_fragment["quantization_origin"],
                    )

            new_fragment = merge_draco_meshes_across_boundaries(
                cg, old_fragments, chunk_id, mip, high_padding
            )

            try:
                new_fragment_b = DracoPy.encode_mesh_to_buffer(
                    new_fragment["vertices"],
                    new_fragment["faces"],
                    **draco_encoding_options,
                )
            except:
                new_fragment_str = new_fragment_id[0 : new_fragment_id.find(":")]
                result.append(
                    f'Bad mesh created for {new_fragment_str}: {len(new_fragment["vertices"])} vertices, {len(new_fragment["faces"])} faces'
                )
                continue

            if WRITING_TO_CLOUD:
                cf.put(
                    path=new_fragment_id,
                    content=new_fragment_b,
                    content_type="application/octet-stream",
                    compress=False,
                    cache_control="no-cache",
                )

    if PRINT_FOR_DEBUGGING:
        pcg_logger.log(pcg_logger.level, ", ".join(str(x) for x in result))
    return ", ".join(str(x) for x in result)

