import numpy as np


def get_chunks_boundary(voxel_boundary, chunk_size) -> np.ndarray:
    """returns number of chunks in each dimension"""
    return np.ceil((voxel_boundary / chunk_size)).astype(int)


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

