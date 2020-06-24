from typing import List
from typing import Union
from typing import Optional
from typing import Sequence
from typing import Iterable
from itertools import product

import numpy as np


def get_chunks_boundary(voxel_boundary, chunk_size) -> np.ndarray:
    """returns number of chunks in each dimension"""
    return np.ceil((voxel_boundary / chunk_size)).astype(np.int)


def normalize_bounding_box(
    meta, bounding_box: Optional[Sequence[Sequence[int]]], bb_is_coordinate: bool,
) -> Union[Sequence[Sequence[int]], None]:
    if bounding_box is None:
        return None

    if bb_is_coordinate:
        bounding_box[0] = _get_chunk_coordinates_from_vol_coordinates(
            meta,
            bounding_box[0][0],
            bounding_box[0][1],
            bounding_box[0][2],
            resolution=meta._ws_cv.resolution,
            ceil=False,
        )
        bounding_box[1] = _get_chunk_coordinates_from_vol_coordinates(
            meta,
            bounding_box[1][0],
            bounding_box[1][1],
            bounding_box[1][2],
            resolution=meta._ws_cv.resolution,
            ceil=True,
        )
        return bounding_box
    else:
        return np.array(bounding_box, dtype=np.int)


def get_chunk_layer(meta, node_or_chunk_id: np.uint64) -> int:
    """ Extract Layer from Node ID or Chunk ID
    :param node_or_chunk_id: np.uint64
    :return: int
    """
    return int(int(node_or_chunk_id) >> 64 - meta.graph_config.LAYER_ID_BITS)


def get_chunk_layers(meta, node_or_chunk_ids: Sequence[np.uint64]) -> np.ndarray:
    """ Extract Layers from Node IDs or Chunk IDs
    :param node_or_chunk_ids: np.ndarray
    :return: np.ndarray
    """
    if len(node_or_chunk_ids) == 0:
        return np.array([], dtype=np.int)
    return np.vectorize(get_chunk_layer)(meta, node_or_chunk_ids)


def get_chunk_coordinates(meta, node_or_chunk_id: np.uint64) -> np.ndarray:
    """ Extract X, Y and Z coordinate from Node ID or Chunk ID
    :param node_or_chunk_id: np.uint64
    :return: Tuple(int, int, int)
    """
    layer = get_chunk_layer(meta, node_or_chunk_id)
    bits_per_dim = meta.bitmasks[layer]

    x_offset = 64 - meta.graph_config.LAYER_ID_BITS - bits_per_dim
    y_offset = x_offset - bits_per_dim
    z_offset = y_offset - bits_per_dim

    x = int(node_or_chunk_id) >> x_offset & 2 ** bits_per_dim - 1
    y = int(node_or_chunk_id) >> y_offset & 2 ** bits_per_dim - 1
    z = int(node_or_chunk_id) >> z_offset & 2 ** bits_per_dim - 1
    return np.array([x, y, z])


def get_chunk_id(
    meta,
    node_id: Optional[np.uint64] = None,
    layer: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
) -> np.uint64:
    """ (1) Extract Chunk ID from Node ID
        (2) Build Chunk ID from Layer, X, Y and Z components
    """
    assert node_id is not None or all(v is not None for v in [layer, x, y, z])
    if node_id is not None:
        layer = get_chunk_layer(meta, node_id)
    bits_per_dim = meta.bitmasks[layer]

    if node_id is not None:
        chunk_offset = 64 - meta.graph_config.LAYER_ID_BITS - 3 * bits_per_dim
        return np.uint64((int(node_id) >> chunk_offset) << chunk_offset)
    return _compute_chunk_id(meta, layer, x, y, z)


def get_chunk_ids_from_node_ids(meta, node_ids: Iterable[np.uint64]) -> np.ndarray:
    """ Extract a list of Chunk IDs from a list of Node IDs
    :param node_ids: np.ndarray(dtype=np.uint64)
    :return: np.ndarray(dtype=np.uint64)
    """
    if len(node_ids) == 0:
        return np.array([], dtype=np.int)
    return np.vectorize(get_chunk_id)(meta, node_ids)


def _compute_chunk_id(meta, layer: int, x: int, y: int, z: int,) -> np.uint64:
    s_bits_per_dim = meta.bitmasks[layer]
    if not (
        x < 2 ** s_bits_per_dim and y < 2 ** s_bits_per_dim and z < 2 ** s_bits_per_dim
    ):
        raise ValueError(
            f"Coordinate is out of range \
            layer: {layer} bits/dim {s_bits_per_dim}. \
            [{x}, {y}, {z}]; max = {2 ** s_bits_per_dim}."
        )
    layer_offset = 64 - meta.graph_config.LAYER_ID_BITS
    x_offset = layer_offset - s_bits_per_dim
    y_offset = x_offset - s_bits_per_dim
    z_offset = y_offset - s_bits_per_dim
    return np.uint64(
        layer << layer_offset | x << x_offset | y << y_offset | z << z_offset
    )


def _get_chunk_coordinates_from_vol_coordinates(
    meta,
    x: np.int,
    y: np.int,
    z: np.int,
    resolution: Sequence[np.int],
    ceil: bool = False,
    layer: int = 1,
) -> np.ndarray:
    """Translates volume coordinates to chunk_coordinates."""
    resolution = np.array(resolution)
    scaling = np.array(meta._ws_cv.resolution / resolution, dtype=np.int)

    chunk_size = meta.graph_config.CHUNK_SIZE
    x = (x / scaling[0] - meta.voxel_bounds[0, 0]) / chunk_size[0]
    y = (y / scaling[1] - meta.voxel_bounds[1, 0]) / chunk_size[1]
    z = (z / scaling[2] - meta.voxel_bounds[2, 0]) / chunk_size[2]

    x /= meta.graph_config.FANOUT ** (max(layer - 2, 0))
    y /= meta.graph_config.FANOUT ** (max(layer - 2, 0))
    z /= meta.graph_config.FANOUT ** (max(layer - 2, 0))

    coords = np.array([x, y, z])
    if ceil:
        coords = np.ceil(coords)
    return coords.astype(np.int)


def get_bounding_children_chunks(
    chunkedgraph_meta, layer: int, chunk_coords: Sequence[int], children_layer
) -> np.ndarray:
    """Children chunk coordinates at given layer, along the boundary of a chunk"""
    chunk_coords = np.array(chunk_coords, dtype=int)
    chunks = []

    # children chunk count along one dimension
    chunks_count = chunkedgraph_meta.graph_config.FANOUT ** (layer - children_layer)
    chunk_offset = chunk_coords * chunks_count
    x1, y1, z1 = chunk_offset
    x2, y2, z2 = chunk_offset + chunks_count

    f = lambda range1, range2: product(range(*range1), range(*range2))
    from ..utils.context_managers import TimeIt

    chunks.extend([np.array([x1, d1, d2]) for d1, d2 in f((y1, y2), (z1, z2))])
    chunks.extend([np.array([x2 - 1, d1, d2]) for d1, d2 in f((y1, y2), (z1, z2))])

    chunks.extend([np.array([d1, y1, d2]) for d1, d2 in f((x1, x2), (z1, z2))])
    chunks.extend([np.array([d1, y2 - 1, d2]) for d1, d2 in f((x1, x2), (z1, z2))])

    chunks.extend([np.array([d1, d2, z1]) for d1, d2 in f((x1, x2), (y1, y2))])
    chunks.extend([np.array([d1, d2, z2 - 1]) for d1, d2 in f((x1, x2), (y1, y2))])

    chunk_bounds = chunkedgraph_meta.layer_chunk_bounds[children_layer]
    chunks = np.array(chunks, dtype=np.int)
    mask = np.all(chunks < chunk_bounds, axis=1)
    result = chunks[mask]
    return np.unique(result, axis=0) if result.size else result
