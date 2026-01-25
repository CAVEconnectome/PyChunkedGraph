import os
import re
import multiprocessing as mp
from time import time
from typing import List
from typing import Dict
from typing import Tuple
from typing import Sequence
from functools import lru_cache

import numpy as np
import pytz
from scipy import ndimage
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec
import DracoPy
import zmesh
import fastremap

from pychunkedgraph.graph.utils.basetypes import NODE_ID  # noqa
from pychunkedgraph.graph import attributes  # noqa
from ..graph.types import empty_1d

UTC = pytz.UTC

# Change below to true if debugging and want to see results in stdout
PRINT_FOR_DEBUGGING = False
# Change below to false if debugging and do not need to write to cloud (warning: do not deploy w/ below set to false)
WRITING_TO_CLOUD = True

REDIS_HOST = os.environ.get("REDIS_SERVICE_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_SERVICE_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


def str_to_slice(slice_str: str):
    match = re.match(r"(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)", slice_str)
    return (
        slice(int(match.group(1)), int(match.group(2))),
        slice(int(match.group(3)), int(match.group(4))),
        slice(int(match.group(5)), int(match.group(6))),
    )


def slice_to_str(slices) -> str:
    if isinstance(slices, slice):
        return "%d-%d" % (slices.start, slices.stop)
    else:
        return "_".join(map(slice_to_str, slices))


def get_chunk_bbox(cg, chunk_id: np.uint64):
    layer = cg.get_chunk_layer(chunk_id)
    chunk_block_shape = get_mesh_block_shape(cg, layer)
    bbox_start = cg.get_chunk_coordinates(chunk_id) * chunk_block_shape
    bbox_end = bbox_start + chunk_block_shape
    return tuple(slice(bbox_start[i], bbox_end[i]) for i in range(3))


def get_chunk_bbox_str(cg, chunk_id: np.uint64) -> str:
    return slice_to_str(get_chunk_bbox(cg, chunk_id))


def get_mesh_name(cg, node_id: np.uint64) -> str:
    return f"{node_id}:0:{get_chunk_bbox_str(cg, node_id)}"


@lru_cache(maxsize=None)
def get_segmentation_info(cg) -> dict:
    return cg.meta.dataset_info


def get_mesh_block_shape(cg, graphlayer: int) -> np.ndarray:
    """
    Calculate the dimensions of a segmentation block that covers
    the same region as a ChunkedGraph chunk at layer `graphlayer`.
    """
    # Segmentation is not always uniformly downsampled in all directions.
    return np.array(
        cg.meta.graph_config.CHUNK_SIZE
    ) * cg.meta.graph_config.FANOUT ** np.max([0, graphlayer - 2])


def get_mesh_block_shape_for_mip(cg, graphlayer: int, source_mip: int) -> np.ndarray:
    """
    Calculate the dimensions of a segmentation block at `source_mip` that covers
    the same region as a ChunkedGraph chunk at layer `graphlayer`.
    """
    info = get_segmentation_info(cg)

    # Segmentation is not always uniformly downsampled in all directions.
    scale_0 = info["scales"][0]
    scale_mip = info["scales"][source_mip]
    distortion = np.floor_divide(scale_mip["resolution"], scale_0["resolution"])

    graphlayer_chunksize = np.array(
        cg.meta.graph_config.CHUNK_SIZE
    ) * cg.meta.graph_config.FANOUT ** np.max([0, graphlayer - 2])

    return np.floor_divide(
        graphlayer_chunksize, distortion, dtype=int, casting="unsafe"
    )


def get_downstream_multi_child_node(cg, node_id: np.uint64, stop_layer: int = 1):
    """
    Return the first descendant of `node_id` (including itself) with more than
    one child, or the first descendant of `node_id` (including itself) on or
    below layer `stop_layer`.
    """
    layer = cg.get_chunk_layer(node_id)
    if layer <= stop_layer:
        return node_id

    children = cg.get_children(node_id)
    if len(children) > 1:
        return node_id

    if not children:
        raise ValueError(f"Node {node_id} on layer {layer} has no children.")

    return get_downstream_multi_child_node(cg, children[0], stop_layer)


def get_downstream_multi_child_nodes(
    cg, node_ids: Sequence[np.uint64], require_children=True
):
    """
    Return the first descendant of `node_ids` (including themselves) with more than
    one child, or the first descendant of `node_ids` (including themselves) on or
    below layer 2.
    """
    # FIXME: Make stop_layer configurable
    stop_layer = 2

    def recursive_helper(cur_node_ids):
        cur_node_ids, unique_to_original = np.unique(cur_node_ids, return_inverse=True)
        stop_layer_mask = np.array(
            [cg.get_chunk_layer(node_id) > stop_layer for node_id in cur_node_ids]
        )
        if np.any(stop_layer_mask):
            node_to_children_dict = cg.get_children(cur_node_ids[stop_layer_mask])
            children_array = np.array(
                list(node_to_children_dict.values()), dtype=object
            )
            only_child_mask = np.array(
                [len(children_for_node) == 1 for children_for_node in children_array]
            )
            # Extract children from object array - each filtered element is a 1-element array
            filtered_children = children_array[only_child_mask]
            only_children = (
                np.concatenate(filtered_children).astype(np.uint64)
                if filtered_children.size
                else np.array([], dtype=np.uint64)
            )
            if np.any(only_child_mask):
                temp_array = cur_node_ids[stop_layer_mask]
                temp_array[only_child_mask] = recursive_helper(only_children)
                cur_node_ids[stop_layer_mask] = temp_array
        return cur_node_ids[unique_to_original]

    return recursive_helper(node_ids)


def get_json_info(cg):
    from json import loads, dumps

    dataset_info = cg.meta.dataset_info
    dummy_app_info = {"app": {"supported_api_versions": [0, 1]}}
    info = {**dataset_info, **dummy_app_info}
    info["mesh"] = cg.meta.custom_data.get("mesh", {}).get("dir", "graphene_meshes")
    info_str = dumps(info)
    return loads(info_str)


def get_ws_seg_for_chunk(cg, chunk_id, mip, overlap_vx=1):
    cv = CloudVolume(cg.meta.cv.cloudpath, mip=mip, fill_missing=True)
    mip_diff = mip - cg.meta.cv.mip

    mip_chunk_size = np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=int) / np.array(
        [2**mip_diff, 2**mip_diff, 1]
    )
    mip_chunk_size = mip_chunk_size.astype(int)

    chunk_start = (
        cg.meta.cv.mip_voxel_offset(mip)
        + cg.get_chunk_coordinates(chunk_id) * mip_chunk_size
    )
    chunk_end = chunk_start + mip_chunk_size + overlap_vx
    chunk_end = Vec.clamp(
        chunk_end,
        cg.meta.cv.mip_voxel_offset(mip),
        cg.meta.cv.mip_voxel_offset(mip) + cg.meta.cv.mip_volume_size(mip),
    )

    ws_seg = cv[
        chunk_start[0] : chunk_end[0],
        chunk_start[1] : chunk_end[1],
        chunk_start[2] : chunk_end[2],
    ].squeeze()

    return ws_seg


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


def get_meshing_necessities_from_graph(cg, chunk_id: np.uint64, mip: int):
    """Given a chunkedgraph, chunk_id, and mip level, return the voxel dimensions of the chunk to be meshed (mesh_block_shape)
    and the chunk origin in the dataset in nm.

    :param cg: chunkedgraph instance
    :param chunk_id: uint64
    :param mip: int
    """
    layer = cg.get_chunk_layer(chunk_id)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    mesh_block_shape = get_mesh_block_shape_for_mip(cg, layer, mip)
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
    below_threshold = voxel_count < int(dust_threshold)
    not_on_boundary = ~np.isin(seg_ids, seg_ids_on_boundary)
    dust_segids = seg_ids[below_threshold & not_on_boundary]
    seg = fastremap.mask(seg, dust_segids, in_place=True)


def get_multi_child_nodes(cg, chunk_id, node_id_subset=None, chunk_bbox_string=False):
    if node_id_subset is None:
        range_read = cg.range_read_chunk(
            chunk_id, properties=attributes.Hierarchy.Child
        )
    else:
        range_read = cg.client.read_nodes(
            node_ids=node_id_subset, properties=attributes.Hierarchy.Child
        )

    node_ids = np.array(list(range_read.keys()))
    node_rows = np.array(list(range_read.values()))
    child_fragments = np.array(
        [
            fragment.value
            for child_fragments_for_node in node_rows
            for fragment in child_fragments_for_node
        ],
        dtype=object,
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
    multi_child_descendants = get_downstream_multi_child_nodes(cg, child_fragments_flat)
    start_index = 0
    multi_child_nodes = {}
    for i in range(len(multi_child_node_ids)):
        end_index = start_index + multi_child_num_children[i]
        descendents_for_current_node = multi_child_descendants[start_index:end_index]
        node_id = multi_child_node_ids[i]
        if chunk_bbox_string:
            multi_child_nodes[f"{node_id}:0:{get_chunk_bbox_str(cg, node_id)}"] = [
                f"{c}:0:{get_chunk_bbox_str(cg, c)}"
                for c in descendents_for_current_node
            ]
        else:
            multi_child_nodes[multi_child_node_ids[i]] = descendents_for_current_node
        start_index = end_index

    return multi_child_nodes, multi_child_descendants
