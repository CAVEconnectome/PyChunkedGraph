from itertools import product

import numpy as np

from ..meta import ChunkedGraphMeta


def get_children_coords(
    cg_meta: ChunkedGraphMeta, layer: int, chunk_coords
) -> np.ndarray:
    chunk_coords = np.array(chunk_coords, dtype=int)
    children_layer = layer - 1
    layer_boundaries = cg_meta.layer_chunk_bounds[children_layer]
    children_coords = []

    for dcoord in product(*[range(cg_meta.graph_config.FANOUT)] * 3):
        dcoord = np.array(dcoord, dtype=int)
        child_coords = chunk_coords * cg_meta.graph_config.FANOUT + dcoord
        check_bounds = np.less(child_coords, layer_boundaries)
        if np.all(check_bounds):
            children_coords.append(child_coords)
    return children_coords
