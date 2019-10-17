from typing import Sequence
from itertools import product
import numpy as np

from ...backend import ChunkedGraphMeta


def get_touching_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta,
    layer: int,
    chunk_coords: Sequence[int],
    include_both=True,
):
    """get atomic chunks along touching faces of children chunks of a parent chunk"""
    chunk_coords = np.array(chunk_coords, dtype=int)
    touching_atomic_chunks = []

    # atomic chunk count along one dimension
    atomic_chunk_count = chunkedgraph_meta.graph_config.fanout ** (layer - 2)
    layer2_chunk_bounds = chunkedgraph_meta.layer_chunk_bounds[2]

    chunk_offset = chunk_coords * atomic_chunk_count
    mid = (atomic_chunk_count // 2) - 1

    # relevant chunks along touching planes at center
    for axis_1, axis_2 in product(*[range(atomic_chunk_count)] * 2):
        # x-y plane
        chunk_1 = chunk_offset + np.array((axis_1, axis_2, mid))
        touching_atomic_chunks.append(chunk_1)
        # x-z plane
        chunk_1 = chunk_offset + np.array((axis_1, mid, axis_2))
        touching_atomic_chunks.append(chunk_1)
        # y-z plane
        chunk_1 = chunk_offset + np.array((mid, axis_1, axis_2))
        touching_atomic_chunks.append(chunk_1)

        if include_both:
            chunk_2 = chunk_offset + np.array((axis_1, axis_2, mid + 1))
            touching_atomic_chunks.append(chunk_2)

            chunk_2 = chunk_offset + np.array((axis_1, mid + 1, axis_2))
            touching_atomic_chunks.append(chunk_2)

            chunk_2 = chunk_offset + np.array((mid + 1, axis_1, axis_2))
            touching_atomic_chunks.append(chunk_2)

    result = []
    for coords in touching_atomic_chunks:
        if np.all(np.less(coords, layer2_chunk_bounds)):
            result.append(coords)
    if result:
        return np.unique(np.array(result, dtype=int), axis=0)
    return []

