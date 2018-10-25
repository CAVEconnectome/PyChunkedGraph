import re
import numpy as np

from functools import lru_cache
from cloudvolume import CloudVolume, Storage

from pychunkedgraph.backend.chunkedgraph import ChunkedGraph  # noqa


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


def get_chunk_bbox(cg: ChunkedGraph, chunk_id: np.uint64, mip: int):
    layer = cg.get_chunk_layer(chunk_id)
    chunk_block_shape = get_mesh_block_shape(cg, layer, mip)
    bbox_start = cg.get_chunk_coordinates(chunk_id) * chunk_block_shape
    bbox_end = bbox_start + chunk_block_shape
    return tuple(slice(bbox_start[i], bbox_end[i]) for i in range(3))


def get_chunk_bbox_str(cg: ChunkedGraph, chunk_id: np.uint64, mip: int) -> str:
    return slice_to_str(get_chunk_bbox(cg, chunk_id, mip))


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


def get_downstream_multi_child_node(cg: ChunkedGraph, node_id: np.uint64,
                                    stop_layer: int = 1):
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


def get_highest_child_nodes_with_meshes(cg: ChunkedGraph, node_id: np.uint64,
                                        stop_layer=1):
    test_ids = [get_downstream_multi_child_node(cg, node_id, stop_layer)]
    valid_seg_ids = []

    with Storage("%s/%s" % (cg.cv.layer_cloudpath, cg.cv.info["mesh"])) as stor:
        while len(test_ids) > 0:
            file_paths = ["%d:0" % seg_id for seg_id in test_ids]
            test_ids = []

            existence_dict = stor.files_exist(file_paths)

            for k in existence_dict:
                seg_id = np.uint64(int(k[:-2]))
                if existence_dict[k]:
                    valid_seg_ids.append(seg_id)
                else:
                    if cg.get_chunk_layer(seg_id) > stop_layer:
                        test_ids.extend(cg.get_children(seg_id))

    return valid_seg_ids