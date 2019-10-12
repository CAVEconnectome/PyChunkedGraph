from typing import Sequence
from itertools import product
import numpy as np

from ...backend import ChunkedGraphMeta
from ...backend.chunkedgraph_utils import get_children_chunk_coords


def get_touching_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta, layer: int, chunk_coords: Sequence[int]
):
    """get atomic chunks along touching faces of children chunks of a parent chunk"""
    chunk_coords = np.array(chunk_coords, dtype=int)
    touching_atomic_chunks = set()

    atomic_chunk_count = chunkedgraph_meta.graph_config.fanout ** (layer - 2)
    layer2_chunk_bounds = chunkedgraph_meta.layer_chunk_bounds[2]

    chunk_offset = chunk_coords * atomic_chunk_count
    mid = (atomic_chunk_count // 2) - 1

    # relevant chunks along touching planes at center
    for axis_1, axis_2 in product(*[range(atomic_chunk_count)] * 2):
        # x-y plane
        chunk_1 = chunk_offset + np.array((axis_1, axis_2, mid))
        chunk_2 = chunk_offset + np.array((axis_1, axis_2, mid + 1))
        touching_atomic_chunks.add(chunk_1)
        touching_atomic_chunks.add(chunk_2)

        # x-z plane
        chunk_1 = chunk_offset + np.array((axis_1, mid, axis_2))
        chunk_2 = chunk_offset + np.array((axis_1, mid + 1, axis_2))
        touching_atomic_chunks.add(chunk_1)
        touching_atomic_chunks.add(chunk_2)

        # y-z plane
        chunk_1 = chunk_offset + np.array((mid, axis_1, axis_2))
        chunk_2 = chunk_offset + np.array((mid + 1, axis_1, axis_2))
        touching_atomic_chunks.add(chunk_1)
        touching_atomic_chunks.add(chunk_2)

    result = []
    for coords in touching_atomic_chunks:
        if np.all(np.less(coords, layer2_chunk_bounds)):
            result.append(coords)

    return np.array(result, dtype=int)

