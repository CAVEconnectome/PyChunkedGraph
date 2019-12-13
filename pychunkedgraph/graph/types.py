from collections import namedtuple

import numpy as np

from .utils import basetypes

# An Agglomeration is syntactic sugar for representing
# a level 2 ID and it's supervoxels and edges.
_agglomeration_fields = ("supervoxels", "edges")
_agglomeration_defaults = (
    np.array([], dtype=basetypes.NODE_ID),
    np.array([], dtype=basetypes.NODE_ID).reshape(-1, 2),
)
Agglomeration = namedtuple(
    "Agglomeration", _agglomeration_fields, defaults=_agglomeration_defaults,
)
