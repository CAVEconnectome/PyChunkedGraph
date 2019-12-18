from typing import Union
from typing import Optional
from typing import Sequence
from typing import Iterable

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
    return _compute_chunk_id(meta.graph_config, layer, x, y, z)


def get_chunk_ids_from_node_ids(meta, node_ids: Iterable[np.uint64]) -> np.ndarray:
    """ Extract a list of Chunk IDs from a list of Node IDs
    :param node_ids: np.ndarray(dtype=np.uint64)
    :return: np.ndarray(dtype=np.uint64)
    """
    if len(node_ids) == 0:
        return np.array([], dtype=np.int)
    return np.vectorize(get_chunk_id)(meta, node_ids)


def _compute_chunk_id(graph_config, layer: int, x: int, y: int, z: int,) -> np.uint64:
    s_bits_per_dim = graph_config.SPATIAL_BITS
    n_bits_layer_id = graph_config.LAYER_ID_BITS
    if not (
        x < 2 ** s_bits_per_dim and y < 2 ** s_bits_per_dim and z < 2 ** s_bits_per_dim
    ):
        raise ValueError(
            f"Coordinate is out of range \
            layer: {layer} bits/dim {s_bits_per_dim}. \
            [{x}, {y}, {z}]; max = {2 ** s_bits_per_dim}."
        )
    layer_offset = 64 - n_bits_layer_id
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
    """ Translates volume coordinates to chunk_coordinates
    :param x: np.int
    :param y: np.int
    :param z: np.int
    :param resolution: np.ndarray
    :param ceil bool
    :param layer: int
    :return:
    """
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
