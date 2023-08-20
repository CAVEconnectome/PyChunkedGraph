# pylint: disable=invalid-name, missing-docstring

from typing import List
from typing import Sequence
from itertools import product

import numpy as np

from .utils import get_bounding_children_chunks
from ..meta import ChunkedGraphMeta


def get_touching_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta,
    layer: int,
    chunk_coords: Sequence[int],
    include_both=False,
) -> List:
    """get atomic chunk coordinates along touching faces of children chunks of a parent chunk"""
    chunk_coords = np.array(chunk_coords, dtype=int)
    touching_atomic_chunks = []

    # atomic chunk count along one dimension
    atomic_chunk_count = chunkedgraph_meta.graph_config.FANOUT ** (layer - 2)
    layer2_chunk_bounds = chunkedgraph_meta.layer_chunk_bounds[2]

    chunk_offset = chunk_coords * atomic_chunk_count
    mid = (atomic_chunk_count // 2) - 1

    # TODO (akhileshh) convert this for loop to numpy;
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

    chunks = np.array(touching_atomic_chunks, dtype=int)
    mask = np.all(chunks < layer2_chunk_bounds, axis=1)
    result = chunks[mask]
    if result.size:
        return np.unique(result, axis=0)
    return []


def get_bounding_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta, layer: int, chunk_coords: Sequence[int]
) -> List:
    """Atomic chunk coordinates along the boundary of a chunk"""
    return get_bounding_children_chunks(chunkedgraph_meta, layer, chunk_coords, 2)
