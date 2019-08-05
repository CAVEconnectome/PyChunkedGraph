"""
Classes and types for edges
"""


import numpy as np

IN_CHUNK = "in"
BT_CHUNK = "between"
CX_CHUNK = "cross"
TYPES = [IN_CHUNK, BT_CHUNK, CX_CHUNK]


class Edges:
    def __init__(
        self,
        node_ids1: np.ndarray,
        node_ids2: np.ndarray,
        affinities: np.ndarray,
        areas: np.ndarray,
    ):
        self.node_ids1 = node_ids1
        self.node_ids2 = node_ids2
        self.affinities = affinities
        self.areas = areas
