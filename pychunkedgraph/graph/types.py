# pylint: disable=invalid-name, missing-docstring
from collections import namedtuple

import numpy as np

from pychunkedgraph.graph import basetypes

empty_1d = np.empty(0, dtype=basetypes.NODE_ID)
empty_2d = np.empty((0, 2), dtype=basetypes.NODE_ID)


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
    "cross_edges",
    "cross_edges_d",
)
_agglomeration_defaults = (
    None,
    empty_1d.copy(),
    empty_2d.copy(),
    empty_2d.copy(),
    empty_2d.copy(),
    {},
)
Agglomeration = namedtuple(
    "Agglomeration", _agglomeration_fields, defaults=_agglomeration_defaults
)
