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
    return np.array(children_coords)


def get_children_chunk_ids(
    meta: ChunkedGraphMeta, node_or_chunk_id: np.uint64
) -> np.ndarray:
    """Calculates the ids of the children chunks in the next lower layer."""
    x, y, z = utils.get_chunk_coordinates(meta, node_or_chunk_id)
    layer = utils.get_chunk_layer(meta, node_or_chunk_id)

    if layer == 1:
        return np.array([], dtype=np.uint64)
    elif layer == 2:
        return np.array([utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z)])
    else:
        children_coords = get_children_chunk_coords(meta, layer, (x, y, z))
        children_chunk_ids = []
        for x, y, z in children_coords:
            children_chunk_ids.append(
                utils.get_chunk_id(meta, layer=layer - 1, x=x, y=y, z=z)
            )
        return np.array(children_chunk_ids, dtype=np.uint64)


def get_parent_chunk_id(
    meta: ChunkedGraphMeta, node_or_chunk_id: np.uint64, parent_layer: int
) -> np.ndarray:
    """Parent chunk ID at given layer."""
    node_layer = utils.get_chunk_layer(meta, node_or_chunk_id)
    coord = utils.get_chunk_coordinates(meta, node_or_chunk_id)
    for _ in range(node_layer, parent_layer):
        coord = coord // meta.graph_config.FANOUT
    x, y, z = coord
    return utils.get_chunk_id(meta, layer=parent_layer, x=x, y=y, z=z)


def get_parent_chunk_id_multiple(
    meta: ChunkedGraphMeta, node_or_chunk_ids: np.ndarray
) -> np.ndarray:
    """Parent chunk IDs for multiple nodes. Assumes nodes at same layer."""

    node_layers = utils.get_chunk_layers(meta, node_or_chunk_ids)
    assert np.unique(node_layers).size == 1, np.unique(node_layers)
    parent_layer = node_layers[0] + 1
    coords = utils.get_chunk_coordinates_multiple(meta, node_or_chunk_ids)
    coords = coords // meta.graph_config.FANOUT
    return utils.get_chunk_ids_from_coords(meta, layer=parent_layer, coords=coords)


def get_parent_chunk_ids(
    meta: ChunkedGraphMeta, node_or_chunk_id: np.uint64
) -> np.ndarray:
    """Creates list of chunk parent ids (upto highest layer)."""
    parent_chunk_layers = range(
        utils.get_chunk_layer(meta, node_or_chunk_id) + 1, meta.layer_count + 1
    )
    chunk_coord = utils.get_chunk_coordinates(meta, node_or_chunk_id)
    parent_chunk_ids = [utils.get_chunk_id(meta, node_or_chunk_id)]
    for layer in parent_chunk_layers:
        chunk_coord = chunk_coord // meta.graph_config.FANOUT
        x, y, z = chunk_coord
        parent_chunk_ids.append(utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z))
    return np.array(parent_chunk_ids, dtype=np.uint64)


def get_parent_chunk_id_dict(meta: ChunkedGraphMeta, node_or_chunk_id: np.uint64):
    """
    Returns dict of {layer: parent_chunk_id}
    (Convenience function)
    """
    layer = utils.get_chunk_layer(meta, node_or_chunk_id)
    return dict(
        zip(
            range(layer, meta.layer_count + 1),
            get_parent_chunk_ids(meta, node_or_chunk_id),
        )
    )
