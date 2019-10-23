"""
Classes and types for edges
"""

from typing import Optional
from collections import namedtuple

import numpy as np

from ..utils import basetypes


_edge_type_fileds = ("in_chunk", "between_chunk", "cross_chunk")
_edge_type_defaults = ("in", "between", "cross")

EdgeTypes = namedtuple("EdgeTypes", _edge_type_fileds, defaults=_edge_type_defaults)
EDGE_TYPES = EdgeTypes()

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
        self.node_ids1 = np.array(node_ids1, dtype=basetypes.NODE_ID)
        self.node_ids2 = np.array(node_ids2, dtype=basetypes.NODE_ID)
        assert self.node_ids1.size == self.node_ids2.size
        self._as_pairs = None

        self.affinities = np.ones(len(self.node_ids1)) * DEFAULT_AFFINITY
        if affinities is not None:
            assert node_ids1.size == affinities.size
            self.affinities = np.array(affinities, dtype=basetypes.EDGE_AFFINITY)

        self.areas = np.ones(len(self.node_ids1)) * DEFAULT_AREA
        if areas is not None:
            assert node_ids1.size == areas.size
            self.areas = np.array(areas, dtype=basetypes.EDGE_AREA)

    def __add__(self, other):
        """add two Edges instances"""
        node_ids1 = np.concatenate([self.node_ids1, other.node_ids1])
        node_ids2 = np.concatenate([self.node_ids2, other.node_ids2])
        affinities = np.concatenate([self.affinities, other.affinities])
        areas = np.concatenate([self.areas, other.areas])
        return Edges(node_ids1, node_ids2, affinities=affinities, areas=areas)

    def __iadd__(self, other):
        self.node_ids1 = np.concatenate([self.node_ids1, other.node_ids1])
        self.node_ids2 = np.concatenate([self.node_ids2, other.node_ids2])
        self.affinities = np.concatenate([self.affinities, other.affinities])
        self.areas = np.concatenate([self.areas, other.areas])
        return self

    def __len__(self):
        return len(self.node_ids1)

    def get_pairs(self) -> np.ndarray:
        """
        return numpy array of edge pairs [[sv1, sv2] ... ]
        """
        if not self._as_pairs is None:
            return self._as_pairs
        self._as_pairs = np.vstack([self.node_ids1, self.node_ids2]).T
        return self._as_pairs


_chunk_edges_defaults = (Edges([], []), Edges([], []), Edges([], []))
ChunkEdges = namedtuple("ChunkEdges", _edge_type_fileds, defaults=_chunk_edges_defaults)

