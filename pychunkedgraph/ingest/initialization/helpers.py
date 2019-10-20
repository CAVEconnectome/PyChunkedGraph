from typing import Sequence
from typing import List
from itertools import product
import numpy as np

from ...backend import ChunkedGraphMeta
from ...backend.chunkedgraph_utils import get_valid_timestamp
from ...backend.utils import basetypes


def get_touching_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta,
    layer: int,
    chunk_coords: Sequence[int],
    include_both=True,
) -> List:
    """get atomic chunk coordinates along touching faces of children chunks of a parent chunk"""
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


def get_bounding_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta, layer: int, chunk_coords: Sequence[int]
) -> List:
    """get atomic chunk coordinates along the boundary of a chunk"""
    chunk_coords = np.array(chunk_coords, dtype=int)
    atomic_chunks = []

    # atomic chunk count along one dimension
    atomic_chunk_count = chunkedgraph_meta.graph_config.fanout ** (layer - 2)
    layer2_chunk_bounds = chunkedgraph_meta.layer_chunk_bounds[2]

    chunk_offset = chunk_coords * atomic_chunk_count
    x1, y1, z1 = chunk_offset
    x2, y2, z2 = chunk_offset + atomic_chunk_count

    f = lambda range1, range2: product(range(*range1), range(*range2))

    atomic_chunks.extend([np.array([x1, d1, d2]) for d1, d2 in f((y1, y2), (z1, z2))])
    atomic_chunks.extend([np.array([x2, d1, d2]) for d1, d2 in f((y1, y2), (z1, z2))])

    atomic_chunks.extend([np.array([d1, y1, d2]) for d1, d2 in f((x1, x2), (z1, z2))])
    atomic_chunks.extend([np.array([d1, y2, d2]) for d1, d2 in f((x1, x2), (z1, z2))])

    atomic_chunks.extend([np.array([d1, d2, z1]) for d1, d2 in f((x1, x2), (y1, y2))])
    atomic_chunks.extend([np.array([d1, d2, z2]) for d1, d2 in f((x1, x2), (y1, y2))])

    result = []
    for coords in atomic_chunks:
        if np.all(np.less(coords, layer2_chunk_bounds)):
            result.append(coords)

    return np.unique(np.array(result, dtype=int), axis=0)


def get_roots(
    cg_instance,
    node_ids: Sequence[np.uint64],
    time_stamp=None,
    layer: int = None,
    n_tries: int = 1,
):
    time_stamp = get_valid_timestamp(time_stamp)
    layer = cg_instance.n_layers if not layer else min(cg_instance.n_layers, layer)

    layer_mask = np.ones(len(node_ids), dtype=np.bool)
    layer_mask[cg_instance.get_chunk_layers(node_ids) >= layer] = False
    parent_ids = np.array(node_ids, dtype=basetypes.NODE_ID)
    for _ in range(int(layer + 1)):
        filtered_ids = parent_ids[layer_mask]
        unique_ids, inverse = np.unique(filtered_ids, return_inverse=True)
        temp_parent_ids = cg_instance.get_parents(unique_ids, time_stamp=time_stamp)
        if temp_parent_ids is None:
            break
        else:
            parent_ids[layer_mask] = temp_parent_ids[inverse]
            layer_mask[cg_instance.get_chunk_layers(parent_ids) >= layer] = False
            if not np.any(cg_instance.get_chunk_layers(parent_ids) < layer):
                break

    return parent_ids
