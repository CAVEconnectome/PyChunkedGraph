"""
Classes and types for edges
"""

from typing import Optional

import numpy as np

IN_CHUNK = "in"
BT_CHUNK = "between"
CX_CHUNK = "cross"
TYPES = [IN_CHUNK, BT_CHUNK, CX_CHUNK]

DEFAULT_AFFINITY = np.finfo(np.float32).tiny
DEFAULT_AREA = np.finfo(np.float32).tiny


class Edges:
    def __init__(
        self,
        node_ids1: np.ndarray,
        node_ids2: np.ndarray,
        *,
        affinities: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
    ):
        assert node_ids1.size == node_ids2.size
        self.node_ids1 = node_ids1
        self.node_ids2 = node_ids2
        self._as_pairs = None

        self.affinities = np.ones(len(self.node_ids1)) * DEFAULT_AFFINITY
        if affinities is not None:
            assert node_ids1.size == affinities.size
            self.affinities = affinities

        self.areas = np.ones(len(self.node_ids1)) * DEFAULT_AREA
        if areas is not None:
            assert node_ids1.size == areas.size
            self.areas = affinities

    def __add__(self, other):
        """add two Edges instances"""
        node_ids1 = np.concatenate([self.node_ids1, other.node_ids1])
        node_ids2 = np.concatenate([self.node_ids1, other.node_ids1])
        affinities = np.concatenate([self.node_ids1, other.node_ids1])
        areas = np.concatenate([self.node_ids1, other.node_ids1])
        return Edges(node_ids1, node_ids2, affinities=affinities, areas=areas)

    def __iadd__(self, other):
        self.node_ids1 = np.concatenate([self.node_ids1, other.node_ids1])
        self.node_ids2 = np.concatenate([self.node_ids2, other.node_ids2])
        self.affinities = np.concatenate([self.affinities, other.affinities])
        self.areas = np.concatenate([self.areas, other.areas])
        return self

    def __len__(self):
        return len(self.node_ids1)

    def get_pairs(self):
        """
        return numpy array of edge pairs [[sv1, sv2] ... ]
        """
        if not self._as_pairs is None:
            return self._as_pairs
        self._as_pairs = np.vstack([self.node_ids1, self.node_ids2]).T
        return self._as_pairs
