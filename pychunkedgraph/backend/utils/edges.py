"""
Utils for edges
"""


import numpy as np
from basetypes import NODE_ID, EDGE_AFFINITY, EDGE_AREA

TYPES = ["in", "between", "cross"]


class Edges:
    def __init__(
        self,
        node_ids1: np.ndarray,
        node_ids2: np.ndarray,
        affinities: np.ndarray,
        areas: np.ndarray,
    ):
        self.node_ids1 = node_ids1
        self.node_ids1 = node_ids2
        self.affinities = affinities
        self.areas = areas
