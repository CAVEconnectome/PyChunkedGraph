from typing import Sequence

import numpy as np

from ..meta import ChunkedGraphMeta

def get_chunks_boundary(voxel_boundary, chunk_size) -> np.ndarray:
    """returns number of chunks in each dimension"""
    return np.ceil((voxel_boundary / chunk_size)).astype(np.int)


def compute_chunk_id(
    layer: int,
    x: int,
    y: int,
    z: int,
    s_bits_per_dim: int = 10,
    n_bits_layer_id: int = 8,
) -> np.uint64:
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


def get_chunk_coordinates_from_vol_coordinates(
    meta: ChunkedGraphMeta,
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
    # TODO pass meta
    resolution = np.array(resolution)
    scaling = np.array(self.cv.resolution / resolution, dtype=np.int)

    x = (x / scaling[0] - self.vx_vol_bounds[0, 0]) / self.chunk_size[0]
    y = (y / scaling[1] - self.vx_vol_bounds[1, 0]) / self.chunk_size[1]
    z = (z / scaling[2] - self.vx_vol_bounds[2, 0]) / self.chunk_size[2]

    x /= self.fan_out ** (max(layer - 2, 0))
    y /= self.fan_out ** (max(layer - 2, 0))
    z /= self.fan_out ** (max(layer - 2, 0))

    coords = np.array([x, y, z])
    if ceil:
        coords = np.ceil(coords)
    return coords.astype(np.int)

