from itertools import product
from typing import Sequence
from typing import Iterable

import numpy as np

from . import utils
from ..meta import ChunkedGraphMeta


def get_children_chunk_coords(
    meta: ChunkedGraphMeta, layer: int, chunk_coords: Sequence[int]
) -> Iterable:
    """
    Returns coordiantes of children chunks.
    Filters out chunks that are outside the boundary of the dataset.
    """
    chunk_coords = np.array(chunk_coords, dtype=int)
    children_layer = layer - 1
    layer_boundaries = meta.layer_chunk_bounds[children_layer]
    children_coords = []

    for dcoord in product(*[range(meta.graph_config.FANOUT)] * 3):
        dcoord = np.array(dcoord, dtype=int)
        child_coords = chunk_coords * meta.graph_config.FANOUT + dcoord
        check_bounds = np.less(child_coords, layer_boundaries)
        if np.all(check_bounds):
            children_coords.append(child_coords)
    return children_coords


def get_children_chunk_ids(
    meta: ChunkedGraphMeta, node_or_chunk_id: np.uint64
) -> np.ndarray:
    """Calculates the ids of the children chunks in the next lower layer."""
    coords = utils.get_chunk_coordinates(meta, node_or_chunk_id)
    layer = utils.get_chunk_layer(meta, node_or_chunk_id)

    if layer == 1:
        return np.array([])
    elif layer == 2:
        x, y, z = coords
        return np.array([utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z)])
    else:
        children_coords = get_children_chunk_coords(meta, layer, coords)
        children_chunk_ids = []
        for coords in children_coords:
            x, y, z = coords
            children_chunk_ids.append(
                utils.get_chunk_id(meta, layer=layer - 1, x=x, y=y, z=z)
            )
        return np.array(children_chunk_ids)
