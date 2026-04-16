# pylint: disable=invalid-name, missing-docstring

from typing import Union
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Iterable

from copy import copy
from functools import lru_cache

import numpy as np


def get_chunks_boundary(voxel_boundary, chunk_size) -> np.ndarray:
    """returns number of chunks in each dimension"""
    return np.ceil((voxel_boundary / chunk_size)).astype(int)


def normalize_bounding_box(
    meta,
    bounding_box: Optional[Sequence[Sequence[int]]],
    bbox_is_coordinate: bool,
) -> Union[Sequence[Sequence[int]], None]:
    if bounding_box is None:
        return None

    bbox = bounding_box.copy()
    if bbox_is_coordinate:
        bbox[0] = _get_chunk_coordinates_from_vol_coordinates(
            meta,
            bbox[0][0],
            bbox[0][1],
            bbox[0][2],
            resolution=meta.resolution,
            ceil=False,
        )
        bbox[1] = _get_chunk_coordinates_from_vol_coordinates(
            meta,
            bbox[1][0],
            bbox[1][1],
            bbox[1][2],
            resolution=meta.resolution,
            ceil=True,
        )
    return np.array(bbox, dtype=int)


def get_chunk_layer(meta, node_or_chunk_id: np.uint64) -> int:
    """Extract Layer from Node ID or Chunk ID"""
    return int(int(node_or_chunk_id) >> 64 - meta.graph_config.LAYER_ID_BITS)


def get_chunk_layers(meta, node_or_chunk_ids: Sequence[np.uint64]) -> np.ndarray:
    """Extract Layers from Node IDs or Chunk IDs
    :param node_or_chunk_ids: np.ndarray
    :return: np.ndarray
    """
    if len(node_or_chunk_ids) == 0:
        return np.array([], dtype=int)

    layers = np.array(node_or_chunk_ids, dtype=int)

    layers1 = layers >> (64 - meta.graph_config.LAYER_ID_BITS)
    # layers2 = np.vectorize(get_chunk_layer)(meta, node_or_chunk_ids)
    # assert np.all(layers1 == layers2)
    return layers1


def get_chunk_coordinates(meta, node_or_chunk_id: np.uint64) -> np.ndarray:
    """Extract X, Y and Z coordinate from Node ID or Chunk ID
    :param node_or_chunk_id: np.uint64
    :return: Tuple(int, int, int)
    """
    layer = get_chunk_layer(meta, node_or_chunk_id)
    bits_per_dim = meta.bitmasks[layer]

    x_offset = 64 - meta.graph_config.LAYER_ID_BITS - bits_per_dim
    y_offset = x_offset - bits_per_dim
    z_offset = y_offset - bits_per_dim

    x = int(node_or_chunk_id) >> x_offset & 2**bits_per_dim - 1
    y = int(node_or_chunk_id) >> y_offset & 2**bits_per_dim - 1
    z = int(node_or_chunk_id) >> z_offset & 2**bits_per_dim - 1
    return np.array([x, y, z])


def get_chunk_coordinates_multiple(meta, ids: np.ndarray) -> np.ndarray:
    """
    Array version of get_chunk_coordinates.
    Assumes all given IDs are in same layer.
    """
    if len(ids) == 0:
        return np.array([], dtype=int).reshape(0, 3)
    layer = get_chunk_layer(meta, ids[0])
    bits_per_dim = meta.bitmasks[layer]

    x_offset = 64 - meta.graph_config.LAYER_ID_BITS - bits_per_dim
    y_offset = x_offset - bits_per_dim
    z_offset = y_offset - bits_per_dim

    ids = np.array(ids, dtype=int)
    X = ids >> x_offset & 2**bits_per_dim - 1
    Y = ids >> y_offset & 2**bits_per_dim - 1
    Z = ids >> z_offset & 2**bits_per_dim - 1
    return np.column_stack((X, Y, Z))


def get_chunk_id(
    meta,
    node_id: Optional[np.uint64] = None,
    layer: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
) -> np.uint64:
    """(1) Extract Chunk ID from Node ID
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


def get_chunk_ids_from_coords(meta, layer: int, coords: np.ndarray):
    layer = int(layer)
    result = np.zeros(len(coords), dtype=np.uint64)
    s_bits_per_dim = meta.bitmasks[layer]

    layer_offset = 64 - meta.graph_config.LAYER_ID_BITS
    x_offset = layer_offset - s_bits_per_dim
    y_offset = x_offset - s_bits_per_dim
    z_offset = y_offset - s_bits_per_dim
    coords = np.array(coords, dtype=np.uint64)

    result |= layer << layer_offset
    result |= coords[:, 0] << x_offset
    result |= coords[:, 1] << y_offset
    result |= coords[:, 2] << z_offset
    return result


def get_chunk_ids_from_node_ids(meta, ids: Iterable[np.uint64]) -> np.ndarray:
    """Extract Chunk IDs from Node IDs"""
    if len(ids) == 0:
        return np.array([], dtype=np.uint64)

    bits_per_dims = np.array([meta.bitmasks[l] for l in get_chunk_layers(meta, ids)])
    offsets = 64 - meta.graph_config.LAYER_ID_BITS - 3 * bits_per_dims

    ids = np.array(ids, dtype=int)
    cids1 = np.array((ids >> offsets) << offsets, dtype=np.uint64)
    # cids2 = np.vectorize(get_chunk_id)(meta, ids)
    # assert np.all(cids1 == cids2)
    return cids1


def _compute_chunk_id(
    meta,
    layer: int,
    x: int,
    y: int,
    z: int,
) -> np.uint64:
    s_bits_per_dim = meta.bitmasks[layer]
    if not (x < 2**s_bits_per_dim and y < 2**s_bits_per_dim and z < 2**s_bits_per_dim):
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
    x: int,
    y: int,
    z: int,
    resolution: Sequence[int],
    ceil: bool = False,
    layer: int = 1,
) -> np.ndarray:
    """Translates volume coordinates to chunk_coordinates."""
    resolution = np.array(resolution)
    scaling = np.array(meta.resolution / resolution, dtype=int)

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
    return coords.astype(int)


@lru_cache()
def get_bounding_children_chunks(
    cg_meta, layer: int, chunk_coords: Tuple[int], children_layer, return_unique=True
) -> np.ndarray:
    """Children chunk coordinates at given layer, along the boundary of a chunk"""
    chunk_coords = np.array(chunk_coords, dtype=int)
    chunks = []

    # children chunk count along one dimension
    chunks_count = cg_meta.graph_config.FANOUT ** (layer - children_layer)
    chunk_offset = chunk_coords * chunks_count
    x1, y1, z1 = chunk_offset
    x2, y2, z2 = chunk_offset + chunks_count

    # https://stackoverflow.com/a/35608701/2683367
    f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(-1, 3)
    chunks.append(f((x1, x2 - 1), range(y1, y2), range(z1, z2)))
    chunks.append(f(range(x1, x2), (y1, y2 - 1), range(z1, z2)))
    chunks.append(f(range(x1, x2), range(y1, y2), (z1, z2 - 1)))

    chunks = np.concatenate(chunks)
    mask = np.all(chunks < cg_meta.layer_chunk_bounds[children_layer], axis=1)
    result = chunks[mask]
    if return_unique:
        return np.unique(result, axis=0) if result.size else result
    return result


@lru_cache()
def get_l2chunkids_along_boundary(cg_meta, mlayer: int, coord_a, coord_b, padding: int = 0):
    """
    Gets L2 Chunk IDs along opposing faces for larger chunks.
    If padding is enabled, more faces of L2 chunks are padded on both sides.
    This is necessary to find fake edges that can span more than 2 L2 chunks.
    """
    bounds_a = get_bounding_children_chunks(cg_meta, mlayer, tuple(coord_a), 2)
    bounds_b = get_bounding_children_chunks(cg_meta, mlayer, tuple(coord_b), 2)

    coord_a, coord_b = np.array(coord_a, dtype=int), np.array(coord_b, dtype=int)
    direction = coord_a - coord_b
    major_axis = np.argmax(np.abs(direction))

    l2chunk_count = 2 ** (mlayer - 2)
    max_coord = coord_a if direction[major_axis] > 0 else coord_b

    skip = abs(direction[major_axis]) - 1
    l2_skip = skip * l2chunk_count

    mid = max_coord[major_axis] * l2chunk_count
    face_a = mid if direction[major_axis] > 0 else (mid - l2_skip - 1)
    face_b = mid if direction[major_axis] < 0 else (mid - l2_skip - 1)

    l2chunks_a = [bounds_a[bounds_a[:, major_axis] == face_a]]
    l2chunks_b = [bounds_b[bounds_b[:, major_axis] == face_b]]

    step_a, step_b = (1, -1) if direction[major_axis] > 0 else (-1, 1)
    for _ in range(padding):
        _l2_chunks_a = copy(l2chunks_a[-1])
        _l2_chunks_b = copy(l2chunks_b[-1])
        _l2_chunks_a[:, major_axis] += step_a
        _l2_chunks_b[:, major_axis] += step_b
        l2chunks_a.append(_l2_chunks_a)
        l2chunks_b.append(_l2_chunks_b)

    l2chunks_a = np.concatenate(l2chunks_a)
    l2chunks_b = np.concatenate(l2chunks_b)

    l2chunk_ids_a = get_chunk_ids_from_coords(cg_meta, 2, l2chunks_a)
    l2chunk_ids_b = get_chunk_ids_from_coords(cg_meta, 2, l2chunks_b)
    return l2chunk_ids_a, l2chunk_ids_b


def chunks_overlapping_bbox(bbox_min, bbox_max, chunk_size) -> dict:
    """
    Find octree chunks overlapping with a bounding box in 3D
    and return a dictionary mapping chunk indices to clipped bounding boxes.
    """
    bbox_min = np.asarray(bbox_min, dtype=int)
    bbox_max = np.asarray(bbox_max, dtype=int)
    chunk_size = np.asarray(chunk_size, dtype=int)

    start_idx = np.floor_divide(bbox_min, chunk_size).astype(int)
    end_idx = np.floor_divide(bbox_max, chunk_size).astype(int)

    ix = np.arange(start_idx[0], end_idx[0] + 1)
    iy = np.arange(start_idx[1], end_idx[1] + 1)
    iz = np.arange(start_idx[2], end_idx[2] + 1)
    grid = np.stack(np.meshgrid(ix, iy, iz, indexing="ij"), axis=-1, dtype=int)
    grid = grid.reshape(-1, 3)

    chunk_min = grid * chunk_size
    chunk_max = chunk_min + chunk_size
    clipped_min = np.maximum(chunk_min, bbox_min)
    clipped_max = np.minimum(chunk_max, bbox_max)
    return {
        tuple(idx): np.stack([cmin, cmax], axis=0, dtype=int)
        for idx, cmin, cmax in zip(grid, clipped_min, clipped_max)
    }


def get_neighbors(coord, inclusive: bool = True, min_coord=None, max_coord=None):
    """
    Get all valid coordinates in the 3×3×3 cube around a given chunk,
    including the chunk itself (if inclusive=True),
    respecting bounding box constraints.
    """
    offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
    if not inclusive:
        offsets = offsets[~np.all(offsets == 0, axis=1)]

    neighbors = np.array(coord) + offsets
    if min_coord is None:
        min_coord = (0, 0, 0)
    min_coord = np.array(min_coord)
    neighbors = neighbors[(neighbors >= min_coord).all(axis=1)]

    if max_coord is not None:
        max_coord = np.array(max_coord)
        neighbors = neighbors[(neighbors <= max_coord).all(axis=1)]
    return neighbors
