import re
import multiprocessing as mp
from time import time
from typing import List
from typing import Dict
from typing import Tuple
from typing import Sequence
from functools import lru_cache

import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.graph.basetypes import NODE_ID  # noqa
from ..graph.types import empty_1d


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
