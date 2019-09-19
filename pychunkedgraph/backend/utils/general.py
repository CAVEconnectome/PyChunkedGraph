from typing import Sequence, Tuple

import numpy as np


def calculate_chunk_id(
    n_bits_per_dim,
    n_bits_layer_id,
    layer: int = None,
    x: int = None,
    y: int = None,
    z: int = None,
):
    if not (
        x < 2 ** n_bits_per_dim and y < 2 ** n_bits_per_dim and z < 2 ** n_bits_per_dim
    ):
        raise ValueError(
            f"Coordinate is out of range \
            layer: {layer} bits/dim {n_bits_per_dim}. \
            [{x}, {y}, {z}]; max = {2 ** n_bits_per_dim}."
        )
    layer_offset = 64 - n_bits_layer_id
    x_offset = layer_offset - n_bits_per_dim
    y_offset = x_offset - n_bits_per_dim
    z_offset = y_offset - n_bits_per_dim
    return np.uint64(
        layer << layer_offset | x << x_offset | y << y_offset | z << z_offset
    )


def get_bounding_box(
    source_coords: Sequence[Sequence[int]],
    sink_coords: Sequence[Sequence[int]],
    bb_offset: Tuple[int, int, int] = (120, 120, 12),
):
    if not source_coords:
        return None
    bb_offset = np.array(list(bb_offset))
    source_coords = np.array(source_coords)
    sink_coords = np.array(sink_coords)

    coords = np.concatenate([source_coords, sink_coords])
    bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]
    bounding_box[0] -= bb_offset
    bounding_box[1] += bb_offset
    return bounding_box
