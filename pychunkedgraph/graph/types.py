from collections import namedtuple

import numpy as np

from .utils import basetypes

_agglomeration_fields = ("supervoxels", "edges")
_agglomeration_defaults = (
    np.array([], dtype=basetypes.NODE_ID),
    np.array([], dtype=basetypes.NODE_ID).reshape(-1, 2),
)
Agglomeration = namedtuple(
    "Agglomeration", _agglomeration_fields, defaults=_agglomeration_defaults,
)
