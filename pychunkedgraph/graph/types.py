from collections import namedtuple

import numpy as np

from .utils import basetypes

"""
An Agglomeration is syntactic sugar for representing
a level 2 ID and it's supervoxels and edges.
`in_edges`
    edges between supervoxels belonging to the agglomeration.
`out_edges`
    edges between supervoxels of agglomeration 
    and neighboring agglomeration.
"""
_agglomeration_fields = ("supervoxels", "in_edges", "out_edges")
_agglomeration_defaults = (
    np.array([], dtype=basetypes.NODE_ID),
    np.array([], dtype=basetypes.NODE_ID).reshape(-1, 2),
    np.array([], dtype=basetypes.NODE_ID).reshape(-1, 2),
)
Agglomeration = namedtuple(
    "Agglomeration", _agglomeration_fields, defaults=_agglomeration_defaults,
)
