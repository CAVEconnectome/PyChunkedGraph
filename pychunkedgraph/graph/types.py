from typing import Iterable
from collections import namedtuple

import numpy as np

from .utils import basetypes

empty_1d = np.empty(0, dtype=basetypes.NODE_ID)
empty_2d = np.empty((0, 2), dtype=basetypes.NODE_ID)


class Node:
    def __init__(
        self,
        node_id: basetypes.NODE_ID,
        parent_id: basetypes.NODE_ID = None,
        children: Iterable = empty_1d,
    ):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children = children

    def __str__(self):
        return f"{self.node_id}:{self.parent_id}:{self.children}"

    def __repr__(self):
        return f"{self.node_id}:{self.parent_id}:{self.children}"


"""
An Agglomeration is syntactic sugar for representing
a level 2 ID and it's supervoxels and edges.
`in_edges`
    edges between supervoxels belonging to the agglomeration.
`out_edges`
    edges between supervoxels of agglomeration 
    and neighboring agglomeration.
`cross_edges_d`
    dict of cross edges {layer: cross_edges_relevant_on_that_layer}
"""
_agglomeration_fields = (
    "node_id",
    "supervoxels",
    "in_edges",
    "out_edges",
    "cross_edges_d",
)
_agglomeration_defaults = (
    None,
    np.array([], dtype=basetypes.NODE_ID),
    np.array([], dtype=basetypes.NODE_ID).reshape(-1, 2),
    np.array([], dtype=basetypes.NODE_ID).reshape(-1, 2),
    {},
)
Agglomeration = namedtuple(
    "Agglomeration", _agglomeration_fields, defaults=_agglomeration_defaults,
)
