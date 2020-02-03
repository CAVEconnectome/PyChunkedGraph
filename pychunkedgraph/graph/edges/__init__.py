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
        chunk_split=False,
    ):
        """
        If `chunk_split` is True, the edges will have infinite affinity.
        (An edge between parts of supervoxel split due to chunk boundary).
        """
        self.node_ids1 = np.array(node_ids1, dtype=basetypes.NODE_ID)
        self.node_ids2 = np.array(node_ids2, dtype=basetypes.NODE_ID)
        assert self.node_ids1.size == self.node_ids2.size
        self._affinities = None
        self._areas = None
        self._as_pairs = None
        self._chunk_split = chunk_split

        if affinities is not None:
            self._affinities = np.array(affinities, dtype=basetypes.EDGE_AFFINITY)
            assert self.node_ids1.size == self._affinities.size

        if areas is not None:
            self._areas = np.array(areas, dtype=basetypes.EDGE_AREA)
            assert self.node_ids1.size == self._areas.size

    @property
    def affinities(self) -> np.ndarray:
        if self._affinities is not None:
            return self._affinities
        return np.ones(len(self.node_ids1)) * (
            np.inf if self._chunk_split else DEFAULT_AFFINITY
        )

    @property
    def areas(self) -> np.ndarray:
        if self._areas is not None:
            return self._affinities
        return np.ones(len(self.node_ids1)) * DEFAULT_AREA

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
        return self.node_ids1.size

    def __getitem__(self, key):
        """`key` must be a boolean numpy array."""
        try:
            return Edges(
                self.node_ids1[key],
                self.node_ids2[key],
                affinities=self.affinities[key],
                areas=self.areas[key],
            )
        except Exception as err:
            raise (err)

    def get_pairs(self) -> np.ndarray:
        """
        return numpy array of edge pairs [[sv1, sv2] ... ]
        """
        if not self._as_pairs is None:
            return self._as_pairs
        self._as_pairs = np.column_stack((self.node_ids1, self.node_ids2))
        return self._as_pairs


_chunk_edges_defaults = (Edges([], []), Edges([], []), Edges([], [], chunk_split=True))
ChunkEdges = namedtuple("ChunkEdges", _edge_type_fileds, defaults=_chunk_edges_defaults)

