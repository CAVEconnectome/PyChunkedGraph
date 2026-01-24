# pylint: disable=invalid-name, missing-docstring, too-many-lines, wrong-import-order, import-outside-toplevel, no-member, c-extension-no-member

from typing import Sequence
import os
import numpy as np
import time
import collections
from functools import lru_cache
import datetime
import pytz
from scipy import ndimage

from multiwrapper import multiprocessing_utils as mu
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
import DracoPy
import zmesh
import fastremap

from pychunkedgraph.graph.chunkedgraph import ChunkedGraph  # noqa
from pychunkedgraph.graph import attributes  # noqa
from pychunkedgraph.meshing import meshgen_utils  # noqa
from pychunkedgraph.meshing.manifest.cache import ManifestCache


UTC = pytz.UTC

# Change below to true if debugging and want to see results in stdout
PRINT_FOR_DEBUGGING = False
# Change below to false if debugging and do not need to write to cloud (warning: do not deploy w/ below set to false)
WRITING_TO_CLOUD = True

REDIS_HOST = os.environ.get("REDIS_SERVICE_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_SERVICE_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


def decode_draco_mesh_buffer(fragment):
    try:
        mesh_object = DracoPy.decode_buffer_to_mesh(fragment)
        vertices = np.array(mesh_object.points)
        faces = np.array(mesh_object.faces)
    except ValueError as exc:
        raise ValueError("Not a valid draco mesh") from exc

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


def remap_seg_using_unsafe_dict(seg, unsafe_dict):
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

            linked_l2_ids = overlaps[np.isin(overlaps, unsafe_dict[unsafe_root_id])]

            if len(linked_l2_ids) == 0:
                seg[bin_cc_seg] = 0
            else:
                seg[bin_cc_seg] = linked_l2_ids[0]

    return seg


def get_remapped_segmentation(
    cg, chunk_id, mip=2, overlap_vx=1, time_stamp=None, n_threads=1
):
    """Downloads + remaps ws segmentation + resolve unclear cases

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
    :param overlap_vx: int
    :param time_stamp:
    :return: remapped segmentation
    """
    assert mip >= cg.meta.cv.mip

    sv_remapping, unsafe_dict = get_lx_overlapping_remappings(
        cg, chunk_id, time_stamp=time_stamp, n_threads=n_threads
    )

    ws_seg = meshgen_utils.get_ws_seg_for_chunk(cg, chunk_id, mip, overlap_vx)
    seg = fastremap.mask_except(ws_seg, list(sv_remapping.keys()), in_place=False)
    fastremap.remap(seg, sv_remapping, preserve_missing_labels=True, in_place=True)

    return remap_seg_using_unsafe_dict(seg, unsafe_dict)


def get_remapped_seg_for_lvl2_nodes(
    cg,
    chunk_id: np.uint64,
    lvl2_nodes: Sequence[np.uint64],
    mip: int = 2,
    overlap_vx: int = 1,
    time_stamp=None,
    n_threads: int = 1,
):
    """Downloads + remaps ws segmentation + resolve unclear cases,
    filter out all but specified lvl2_nodes

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
    :param overlap_vx: int
    :param time_stamp:
    :return: remapped segmentation
    """
    seg = meshgen_utils.get_ws_seg_for_chunk(cg, chunk_id, mip, overlap_vx)
    sv_of_lvl2_nodes = cg.get_children(lvl2_nodes)

    # Check which of the lvl2_nodes meet the chunk boundary
    node_ids_on_the_border = []
    remapping = {}
    for node, sv_list in sv_of_lvl2_nodes.items():
        node_on_the_border = False
        for sv_id in sv_list:
            remapping[sv_id] = node
            # If a node_id is on the chunk_boundary, we must check
            # the overlap region to see if the meshes' end will be open or closed
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
        # For some supervoxel, they could map to multiple l2 nodes in the chunk,
        # so we must perform a connected component analysis
        # to see which l2 node they are adjacent to
        return remap_seg_using_unsafe_dict(seg, unsafe_dict)
    else:
        # If no nodes in our subset meet the chunk boundary
        # we can simply retrieve the sv of the nodes in the subset
        fastremap.mask_except(seg, list(remapping.keys()), in_place=True)
        fastremap.remap(seg, remapping, preserve_missing_labels=True, in_place=True)

    return seg


@lru_cache(maxsize=None)
def get_higher_to_lower_remapping(cg, chunk_id, time_stamp):
    """Retrieves lx node id to sv id mappping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param time_stamp: datetime object
    :return: dictionary
    """

    def _lower_remaps(ks):
        return np.concatenate([lower_remaps[k] for k in ks])

    assert cg.get_chunk_layer(chunk_id) >= 2
    assert cg.get_chunk_layer(chunk_id) <= cg.meta.layer_count

    print(f"\n{chunk_id} ----------------\n")

    lower_remaps = {}
    if cg.get_chunk_layer(chunk_id) > 2:
        for lower_chunk_id in cg.get_chunk_child_ids(chunk_id):
            # TODO speedup
            lower_remaps.update(
                get_higher_to_lower_remapping(cg, lower_chunk_id, time_stamp=time_stamp)
            )

    rr_chunk = cg.range_read_chunk(
        chunk_id=chunk_id, properties=attributes.Hierarchy.Child, time_stamp=time_stamp
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


@lru_cache(maxsize=None)
def get_root_lx_remapping(cg, chunk_id, stop_layer, time_stamp, n_threads=1):
    """Retrieves root to l2 node id mapping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param stop_layer: int
    :param time_stamp: datetime object
    :return: multiples
    """

    def _get_root_ids(args):
        start_id, end_id = args
        root_ids[start_id:end_id] = cg.get_roots(
            lx_ids[start_id:end_id],
            stop_layer=stop_layer,
            fail_to_zero=True,
        )

    lx_id_remap = get_higher_to_lower_remapping(cg, chunk_id, time_stamp=time_stamp)

    lx_ids = np.array(list(lx_id_remap.keys()), dtype=np.uint64)

    root_ids = np.zeros(len(lx_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(lx_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(lx_ids), n_jobs + 1).astype(int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        mu.multithread_func(_get_root_ids, multi_args, n_threads=n_threads)

    return lx_ids, np.array(root_ids), lx_id_remap


def calculate_stop_layer(cg, chunk_id):
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
                try:
                    neigh_chunk_id = cg.get_chunk_id(x=x, y=y, z=z, layer=chunk_layer)
                    # Get parent chunk ids
                    parent_chunk_ids = cg.get_parent_chunk_ids(neigh_chunk_id)
                    neigh_chunk_ids.append(neigh_chunk_id)
                    neigh_parent_chunk_ids.append(parent_chunk_ids)
                except:
                    # cg.get_parent_chunk_id can fail if neigh_chunk_id is outside the dataset
                    # (only happens when cg.meta.bitmasks[chunk_layer+1] == log(max(x,y,z)),
                    # so only for specific datasets in which the # of chunks in the widest dimension
                    # just happens to be a power of two)
                    pass

    # Find lowest common chunk
    neigh_parent_chunk_ids = np.array(neigh_parent_chunk_ids)
    layer_agreement = np.all(
        (neigh_parent_chunk_ids - neigh_parent_chunk_ids[0]) == 0, axis=0
    )
    stop_layer = np.where(layer_agreement)[0][0] + chunk_layer

    return stop_layer, neigh_chunk_ids


# @lru_cache(maxsize=None)
def get_lx_overlapping_remappings(cg, chunk_id, time_stamp=None, n_threads=1):
    """Retrieves sv id to layer mapping for chunk with overlap in positive
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

    stop_layer, neigh_chunk_ids = calculate_stop_layer(cg, chunk_id)
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
            unsafe_lx_ids = lx_ids[~np.isin(lx_ids, safe_lx_ids)]
            unsafe_root_ids = np.unique(root_ids[u_idx[c_root_ids != 1]])

    lx_root_dict = dict(zip(neigh_lx_ids, neigh_root_ids))
    root_lx_dict = collections.defaultdict(list)

    # Future sv id -> lx mapping
    sv_ids = []
    lx_ids_flat = []

    for i_root_id in range(len(neigh_root_ids)):
        root_lx_dict[neigh_root_ids[i_root_id]].append(neigh_lx_ids[i_root_id])

    # Do safe ones first
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
        if np.sum(~np.isin(root_lx_dict[root_id], unsafe_lx_ids)) == 0:
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
    """Retrieves root to node id mapping for specified node ids and supervoxel ids

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
            combined_ids[start_id:end_id],
            stop_layer=stop_layer,
            time_stamp=time_stamp,
            fail_to_zero=True,
        )

    rr = cg.range_read_chunk(
        chunk_id=chunk_id, properties=attributes.Hierarchy.Child, time_stamp=time_stamp
    )
    chunk_sv_ids = np.unique(np.concatenate([id[0].value for id in rr.values()]))
    chunk_l2_ids = np.unique(cg.get_parents(chunk_sv_ids, time_stamp=time_stamp))
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
    """Retrieves sv id to layer mapping for chunk with overlap in positive
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

    stop_layer, _ = calculate_stop_layer(cg, chunk_id)
    print(f"Stop layer: {stop_layer}")

    # Find the parent in the lowest common chunk for each node id and sv id. These parent
    # ids are referred to as root ids even though they are not necessarily the
    # root id.
    node_root_ids, sv_root_ids, chunks_root_ids = get_root_remapping_for_nodes_and_svs(
        cg, chunk_id, node_ids, sv_ids, stop_layer, time_stamp, n_threads
    )

    u_root_ids, u_idx, c_root_ids = np.unique(
        chunks_root_ids, return_counts=True, return_index=True
    )

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
    """Given a chunkedgraph, chunk_id, and mip level, return the voxel dimensions of the chunk to be meshed (mesh_block_shape)
    and the chunk origin in the dataset in nm.

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :param mip: int
    """
    layer = cg.get_chunk_layer(chunk_id)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    mesh_block_shape = meshgen_utils.get_mesh_block_shape_for_mip(cg, layer, mip)
    voxel_resolution = cg.meta.cv.mip_resolution(mip)
    chunk_offset = (
        (cx, cy, cz) * mesh_block_shape + cg.meta.cv.mip_voxel_offset(mip)
    ) * voxel_resolution
    return layer, mesh_block_shape, chunk_offset


def calculate_quantization_bits_and_range(
    min_quantization_range, max_draco_bin_size, draco_quantization_bits=None
):
    if draco_quantization_bits is None:
        draco_quantization_bits = np.ceil(
            np.log2(min_quantization_range / max_draco_bin_size + 1)
        )
    num_draco_bins = 2**draco_quantization_bits - 1
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
    """Calculate the proper draco encoding settings for a chunk to ensure proper stitching is possible
    on the layer above. For details about how and why we do this, please see the meshing Readme

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :param mip: int
    :param high_padding: int
    """
    _, mesh_block_shape, chunk_offset = get_meshing_necessities_from_graph(
        cg, chunk_id, mip
    )
    segmentation_resolution = cg.meta.cv.mip_resolution(mip)
    min_quantization_range = max(
        (mesh_block_shape + high_padding) * segmentation_resolution
    )
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    (
        draco_quantization_bits,
        draco_quantization_range,
        draco_bin_size,
    ) = calculate_quantization_bits_and_range(
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
    segmentation_resolution = cg.meta.cv.mip_resolution(mip)
    min_quantization_range = (
        max(mesh_block_shape * segmentation_resolution) + 2 * old_draco_bin_size
    )
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    (
        draco_quantization_bits,
        draco_quantization_range,
        draco_bin_size,
    ) = calculate_quantization_bits_and_range(
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


def merge_draco_meshes_across_boundaries(
    cg, fragments, chunk_id, mip, high_padding, return_zmesh_object=False
):
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
        # Get the draco encoding settings for the
        # child chunk in the "bottom corner" of the chunk_id chunk
        draco_encoding_settings_smaller_chunk = get_draco_encoding_settings_for_chunk(
            cg, child_chunk_id, mip=mip, high_padding=high_padding
        )
        draco_bin_size = draco_encoding_settings_smaller_chunk["quantization_range"] / (
            2 ** draco_encoding_settings_smaller_chunk["quantization_bits"] - 1
        )
        # Calculate which draco bin the child chunk's boundaries
        # were placed into (for each x,y,z of boundary)
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

    if return_zmesh_object:
        return zmesh.Mesh(vertices[:, 0:3], faces.reshape(-1, 3), None)

    return {
        "num_vertices": np.uint32(len(vertices)),
        "vertices": vertices[:, 0:3].reshape(-1),
        "faces": faces,
    }


def black_out_dust_from_segmentation(seg, dust_threshold):
    """Black out (set to 0) IDs in segmentation not on the segmentation
    border that have less voxels than dust_threshold

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
    timestamps = cg.get_node_timestamps(node_ids, return_numpy=False)
    return max(timestamps) + datetime.timedelta(milliseconds=1)


def remeshing(
    cg,
    l2_node_ids: Sequence[np.uint64],
    cv_sharded_mesh_dir: str,
    cv_unsharded_mesh_path: str,
    stop_layer: int = None,
    mip: int = 2,
    max_err: int = 40,
    time_stamp: datetime.datetime or None = None,
):
    """Given a chunkedgraph, a list of level 2 nodes,
    perform remeshing and stitching up the node hierarchy (or up to the stop_layer)

    :param cg: chunkedgraph instance
    :param l2_node_ids: list of uint64
    :param stop_layer: int
    :param cv_path: str
    :param cv_mesh_dir: str
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
            print("remeshing", chunk_id, node_ids)
        try:
            l2_time_stamp = _get_timestamp_from_node_ids(cg, node_ids)
        except ValueError:
            # ignore bad/invalid messages
            return
        # Remesh the l2_node_ids
        chunk_initial_mesh_task(
            None,
            chunk_id,
            mip=mip,
            node_id_subset=node_ids,
            cg=cg,
            cv_unsharded_mesh_path=cv_unsharded_mesh_path,
            max_err=max_err,
            sharded=False,
            time_stamp=l2_time_stamp,
        )
    chunk_dicts = []
    max_layer = stop_layer or cg._n_layers
    for layer in range(3, max_layer + 1):
        chunk_dicts.append(collections.defaultdict(set))
    cur_chunk_dict = l2_chunk_dict
    # Find the parents of each l2_node_id up to the stop_layer,
    # as well as their associated chunk_ids
    for layer in range(3, max_layer + 1):
        for _, node_ids in cur_chunk_dict.items():
            parent_nodes = cg.get_parents(node_ids, time_stamp=time_stamp)
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
                print("remeshing", chunk_id, node_ids)
            # Stitch the meshes of the parents we found in the previous loop
            chunk_stitch_remeshing_task(
                None,
                chunk_id,
                mip=mip,
                fragment_batch_size=40,
                node_id_subset=node_ids,
                cg=cg,
                cv_sharded_mesh_dir=cv_sharded_mesh_dir,
                cv_unsharded_mesh_path=cv_unsharded_mesh_path,
            )


def chunk_initial_mesh_task(
    cg_name,
    chunk_id,
    cv_unsharded_mesh_path,
    mip=2,
    max_err=40,
    lod=0,
    encoding="draco",
    time_stamp=None,
    dust_threshold=None,
    return_frag_count=False,
    node_id_subset=None,
    cg=None,
    sharded=False,
    cache=True,
):
    if cg is None:
        cg = ChunkedGraph(graph_id=cg_name)
    result = []
    cache_string = "public" if cache else "no-cache"

    layer, _, chunk_offset = get_meshing_necessities_from_graph(cg, chunk_id, mip)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    high_padding = 1
    assert layer == 2
    assert mip >= cg.meta.cv.mip

    if sharded:
        cv = CloudVolume(
            f"graphene://https://localhost/segmentation/table/dummy",
            info=meshgen_utils.get_json_info(cg),
            secrets={"token": "dummy"},
        )
        sharding_info = cv.mesh.meta.info["sharding"]["2"]
        sharding_spec = ShardingSpecification.from_dict(sharding_info)
        merged_meshes = {}
        mesh_dst = os.path.join(
            cv.cloudpath, cv.mesh.meta.mesh_path, "initial", str(layer)
        )
    else:
        mesh_dst = cv_unsharded_mesh_path

    result.append((chunk_id, layer, cx, cy, cz))
    print(
        "Retrieving remap table for chunk %s -- (%s, %s, %s, %s)"
        % (chunk_id, layer, cx, cy, cz)
    )
    mesher = zmesh.Mesher(cg.meta.cv.mip_resolution(mip))
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
    mesher.mesh(seg)
    del seg
    cf = CloudFiles(mesh_dst)
    if PRINT_FOR_DEBUGGING:
        print("cv path", mesh_dst)
        print("num ids", len(mesher.ids()))
    result.append(len(mesher.ids()))
    for obj_id in mesher.ids():
        mesh = mesher.get(obj_id, reduction_factor=100, max_error=max_err)
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
            if sharded:
                merged_meshes[int(obj_id)] = file_contents
            else:
                cf.put(
                    path=f"{meshgen_utils.get_mesh_name(cg, obj_id)}",
                    content=file_contents,
                    compress=compress,
                    cache_control=cache_string,
                )
    if sharded and WRITING_TO_CLOUD:
        shard_binary = sharding_spec.synthesize_shard(merged_meshes)
        shard_filename = cv.mesh.readers[layer].get_filename(chunk_id)
        cf.put(
            shard_filename,
            shard_binary,
            content_type="application/octet-stream",
            compress=False,
            cache_control=cache_string,
        )
    if PRINT_FOR_DEBUGGING:
        print(", ".join(str(x) for x in result))
    return result


def get_multi_child_nodes(cg, chunk_id, node_id_subset=None, chunk_bbox_string=False):
    if node_id_subset is None:
        range_read = cg.range_read_chunk(
            chunk_id, properties=attributes.Hierarchy.Child
        )
    else:
        range_read = cg.client.read_nodes(
            node_ids=node_id_subset, properties=attributes.Hierarchy.Child
        )

    node_ids = np.array(list(range_read.keys()), dtype=np.uint64)
    node_rows = np.array(list(range_read.values()), dtype=object)
    child_fragments = np.array(
        [
            fragment.value
            for child_fragments_for_node in node_rows
            for fragment in child_fragments_for_node
        ], dtype=object
    )
    # Filter out node ids that do not have roots (caused by failed ingest tasks)
    root_ids = cg.get_roots(node_ids, fail_to_zero=True)
    # Only keep nodes with more than one child
    multi_child_mask = np.array(
        [len(fragments) > 1 for fragments in child_fragments], dtype=bool
    )
    root_id_mask = np.array([root_id != 0 for root_id in root_ids], dtype=bool)
    multi_child_node_ids = node_ids[multi_child_mask & root_id_mask]
    multi_child_children_ids = child_fragments[multi_child_mask & root_id_mask]
    # Store how many children each node has, because we will retrieve all children at once
    multi_child_num_children = [len(children) for children in multi_child_children_ids]
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
        descendents_for_current_node = multi_child_descendants[start_index:end_index]
        node_id = multi_child_node_ids[i]
        if chunk_bbox_string:
            multi_child_nodes[
                f"{node_id}:0:{meshgen_utils.get_chunk_bbox_str(cg, node_id)}"
            ] = [
                f"{c}:0:{meshgen_utils.get_chunk_bbox_str(cg, c)}"
                for c in descendents_for_current_node
            ]
        else:
            multi_child_nodes[multi_child_node_ids[i]] = descendents_for_current_node
        start_index = end_index

    return multi_child_nodes, multi_child_descendants


def chunk_stitch_remeshing_task(
    cg_name,
    chunk_id,
    cv_sharded_mesh_dir,
    cv_unsharded_mesh_path,
    mip=2,
    lod=0,
    fragment_batch_size=None,
    node_id_subset=None,
    cg=None,
    high_padding=1,
):
    """
    For each node with more than one child, create a new fragment by
    merging the mesh fragments of the children.
    """
    if cg is None:
        cg = ChunkedGraph(graph_id=cg_name)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    layer = cg.get_chunk_layer(chunk_id)
    result = []

    assert layer > 2

    print(
        "Retrieving children for chunk %s -- (%s, %s, %s, %s)"
        % (chunk_id, layer, cx, cy, cz)
    )

    multi_child_nodes, _ = get_multi_child_nodes(cg, chunk_id, node_id_subset, False)
    print(f"{len(multi_child_nodes)} nodes with more than one child")
    result.append((chunk_id, len(multi_child_nodes)))
    if not multi_child_nodes:
        print("Nothing to do", cx, cy, cz)
        return ", ".join(str(x) for x in result)

    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=cv_sharded_mesh_dir,
        info=meshgen_utils.get_json_info(cg),
        secrets={"token": "dummy"},
    )

    fragments_in_batch_processed = 0
    batches_processed = 0
    num_fragments_processed = 0
    fragment_to_fetch = [
        fragment
        for child_fragments in multi_child_nodes.values()
        for fragment in child_fragments
    ]
    cf = CloudFiles(cv_unsharded_mesh_path)
    if fragment_batch_size is None:
        fragment_map = cv.mesh.get_meshes_on_bypass(
            fragment_to_fetch, allow_missing=True
        )
    else:
        fragment_map = cv.mesh.get_meshes_on_bypass(
            fragment_to_fetch[0:fragment_batch_size], allow_missing=True
        )
    i = 0
    fragments_d = {}
    for new_fragment_id, fragment_ids_to_fetch in multi_child_nodes.items():
        i += 1
        if i % max(1, len(multi_child_nodes) // 10) == 0:
            print(f"{i}/{len(multi_child_nodes)}")

        old_fragments = []
        missing_fragments = False
        for fragment_id in fragment_ids_to_fetch:
            if fragment_batch_size is not None:
                fragments_in_batch_processed += 1
                if fragments_in_batch_processed > fragment_batch_size:
                    fragments_in_batch_processed = 1
                    batches_processed += 1
                    num_fragments_processed = batches_processed * fragment_batch_size
                    fragment_map = cv.mesh.get_meshes_on_bypass(
                        fragment_to_fetch[
                            num_fragments_processed : num_fragments_processed
                            + fragment_batch_size
                        ],
                        allow_missing=True,
                    )
            if fragment_id in fragment_map:
                old_frag = fragment_map[fragment_id]
                new_old_frag = {
                    "num_vertices": len(old_frag.vertices),
                    "vertices": old_frag.vertices,
                    "faces": old_frag.faces.reshape(-1),
                    "encoding_options": old_frag.encoding_options,
                    "encoding_type": "draco",
                }
                wrapper_object = {
                    "mesh": new_old_frag,
                    "node_id": np.uint64(old_frag.segid),
                }
                old_fragments.append(wrapper_object)
            elif cg.get_chunk_layer(np.uint64(fragment_id)) > 2:
                missing_fragments = True
                result.append(f"{fragment_id} missing for {new_fragment_id}")

        if len(old_fragments) == 0 or missing_fragments:
            result.append(f"No meshes for {new_fragment_id}")
            continue

        draco_encoding_options = None
        for old_fragment in old_fragments:
            if draco_encoding_options is None:
                draco_encoding_options = (
                    transform_draco_fragment_and_return_encoding_options(
                        cg, old_fragment, layer, mip, chunk_id
                    )
                )
            else:
                transform_draco_fragment_and_return_encoding_options(
                    cg, old_fragment, layer, mip, chunk_id
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
            result.append(
                f'Bad mesh created for {new_fragment_id}: {len(new_fragment["vertices"])} '
                f'vertices, {len(new_fragment["faces"])} faces'
            )
            continue

        if WRITING_TO_CLOUD:
            fragment_name = meshgen_utils.get_chunk_bbox_str(cg, new_fragment_id)
            fragment_name = f"{new_fragment_id}:0:{fragment_name}"
            fragments_d[new_fragment_id] = fragment_name
            cf.put(
                fragment_name,
                new_fragment_b,
                content_type="application/octet-stream",
                compress=False,
                cache_control="public",
            )

    manifest_cache = ManifestCache(cg.graph_id, initial=False)
    manifest_cache.set_fragments(fragments_d)

    if PRINT_FOR_DEBUGGING:
        print(", ".join(str(x) for x in result))
    return ", ".join(str(x) for x in result)


def chunk_initial_sharded_stitching_task(
    cg_name, chunk_id, mip, cg=None, high_padding=1, cache=True
):
    start_existence_check_time = time.time()
    if cg is None:
        cg = ChunkedGraph(graph_id=cg_name)

    cache_string = "public" if cache else "no-cache"

    layer = cg.get_chunk_layer(chunk_id)
    multi_child_nodes, multi_child_descendants = get_multi_child_nodes(cg, chunk_id)

    chunk_to_id_dict = collections.defaultdict(list)
    for child_node in multi_child_descendants:
        cur_chunk_id = int(cg.get_chunk_id(child_node))
        chunk_to_id_dict[cur_chunk_id].append(child_node)

    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        info=meshgen_utils.get_json_info(cg),
        secrets={"token": "dummy"},
    )
    shard_filenames = []
    shard_to_chunk_id = {}
    for cur_chunk_id in chunk_to_id_dict:
        shard_id = cv.meta.decode_chunk_position_number(cur_chunk_id)
        shard_filename = (
            str(cg.get_chunk_layer(cur_chunk_id)) + "/" + str(shard_id) + "-0.shard"
        )
        shard_to_chunk_id[shard_filename] = cur_chunk_id
        shard_filenames.append(shard_filename)
    mesh_dict = {}

    cf = CloudFiles(os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, "initial"))
    files_contents = cf.get(shard_filenames)
    for i in range(len(files_contents)):
        cur_chunk_id = shard_to_chunk_id[files_contents[i]["path"]]
        cur_layer = cg.get_chunk_layer(cur_chunk_id)
        if files_contents[i]["content"] is not None:
            disassembled_shard = cv.mesh.readers[cur_layer].disassemble_shard(
                files_contents[i]["content"]
            )
            nodes_in_chunk = chunk_to_id_dict[int(cur_chunk_id)]
            for node_in_chunk in nodes_in_chunk:
                node_in_chunk_int = int(node_in_chunk)
                if node_in_chunk_int in disassembled_shard:
                    mesh_dict[node_in_chunk_int] = disassembled_shard[node_in_chunk]
    del files_contents

    number_frags_proc = 0
    sharding_info = cv.mesh.meta.info["sharding"][str(layer)]
    sharding_spec = ShardingSpecification.from_dict(sharding_info)
    merged_meshes = {}
    biggest_frag = 0
    biggest_frag_vx_ct = 0
    bad_meshes = []
    for new_fragment_id in multi_child_nodes:
        fragment_ids_to_fetch = multi_child_nodes[new_fragment_id]
        old_fragments = []
        for frag_to_fetch in fragment_ids_to_fetch:
            try:
                old_fragments.append(
                    {
                        "mesh": decode_draco_mesh_buffer(mesh_dict[int(frag_to_fetch)]),
                        "node_id": np.uint64(frag_to_fetch),
                    }
                )
            except KeyError:
                pass
        if len(old_fragments) > 0:
            draco_encoding_options = None
            for old_fragment in old_fragments:
                if draco_encoding_options is None:
                    draco_encoding_options = (
                        transform_draco_fragment_and_return_encoding_options(
                            cg, old_fragment, layer, mip, chunk_id
                        )
                    )
                else:
                    transform_draco_fragment_and_return_encoding_options(
                        cg, old_fragment, layer, mip, chunk_id
                    )

            new_fragment = merge_draco_meshes_across_boundaries(
                cg, old_fragments, chunk_id, mip, high_padding
            )

            if len(new_fragment["vertices"]) > biggest_frag_vx_ct:
                biggest_frag = new_fragment_id
                biggest_frag_vx_ct = len(new_fragment["vertices"])

            try:
                new_fragment_b = DracoPy.encode_mesh_to_buffer(
                    new_fragment["vertices"],
                    new_fragment["faces"],
                    **draco_encoding_options,
                )
                merged_meshes[int(new_fragment_id)] = new_fragment_b
            except:
                print(f"failed to merge {new_fragment_id}")
                bad_meshes.append(new_fragment_id)
                pass
            number_frags_proc = number_frags_proc + 1
            if number_frags_proc % 1000 == 0:
                print(f"number frag proc = {number_frags_proc}")
    del mesh_dict
    shard_binary = sharding_spec.synthesize_shard(merged_meshes)
    shard_filename = cv.mesh.readers[layer].get_filename(chunk_id)
    cf = CloudFiles(
        os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, "initial", str(layer))
    )
    cf.put(
        shard_filename,
        shard_binary,
        content_type="application/octet-stream",
        compress=False,
        cache_control=cache_string,
    )
    total_time = time.time() - start_existence_check_time

    ret = {
        "chunk_id": chunk_id,
        "total_time": total_time,
        "biggest_frag": biggest_frag,
        "biggest_frag_vx_ct": biggest_frag_vx_ct,
        "number_frag": number_frags_proc,
        "bad meshes": bad_meshes,
    }
    return ret
