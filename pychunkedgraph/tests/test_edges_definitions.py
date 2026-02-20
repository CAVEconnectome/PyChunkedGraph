"""Tests for pychunkedgraph.graph.edges.definitions"""

import pytest
import numpy as np

from pychunkedgraph.graph.edges.definitions import (
    Edges,
    EDGE_TYPES,
    DEFAULT_AFFINITY,
    DEFAULT_AREA,
)
from pychunkedgraph.graph.utils import basetypes


class TestEdgeTypes:
    def test_fields(self):
        assert EDGE_TYPES.in_chunk == "in"
        assert EDGE_TYPES.between_chunk == "between"
        assert EDGE_TYPES.cross_chunk == "cross"


class TestEdges:
    def test_creation_defaults(self):
        ids1 = np.array([1, 2], dtype=basetypes.NODE_ID)
        ids2 = np.array([3, 4], dtype=basetypes.NODE_ID)
        e = Edges(ids1, ids2)
        np.testing.assert_array_equal(e.node_ids1, ids1)
        np.testing.assert_array_equal(e.node_ids2, ids2)
        assert np.all(e.affinities == DEFAULT_AFFINITY)
        assert np.all(e.areas == DEFAULT_AREA)

    def test_creation_explicit(self):
        ids1 = np.array([1, 2], dtype=basetypes.NODE_ID)
        ids2 = np.array([3, 4], dtype=basetypes.NODE_ID)
        affs = np.array([0.5, 0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([10.0, 20.0], dtype=basetypes.EDGE_AREA)
        e = Edges(ids1, ids2, affinities=affs, areas=areas)
        np.testing.assert_array_almost_equal(e.affinities, affs)
        np.testing.assert_array_almost_equal(e.areas, areas)

    def test_creation_empty(self):
        e = Edges([], [])
        assert len(e) == 0
        pairs = e.get_pairs()
        assert pairs.shape[0] == 0

    def test_len(self):
        e = Edges([1, 2, 3], [4, 5, 6])
        assert len(e) == 3

    def test_add(self):
        e1 = Edges([1], [2], affinities=[0.5], areas=[10.0])
        e2 = Edges([3], [4], affinities=[0.9], areas=[20.0])
        e3 = e1 + e2
        assert len(e3) == 2
        np.testing.assert_array_equal(e3.node_ids1, [1, 3])
        np.testing.assert_array_equal(e3.node_ids2, [2, 4])

    def test_iadd(self):
        e1 = Edges([1], [2])
        e2 = Edges([3], [4])
        e1 += e2
        assert len(e1) == 2
        np.testing.assert_array_equal(e1.node_ids1, [1, 3])

    def test_getitem_boolean(self):
        e = Edges([1, 2, 3], [4, 5, 6], affinities=[0.1, 0.5, 0.9], areas=[1, 2, 3])
        mask = np.array([True, False, True])
        filtered = e[mask]
        assert len(filtered) == 2
        np.testing.assert_array_equal(filtered.node_ids1, [1, 3])

    def test_getitem_error(self):
        e = Edges([1, 2], [3, 4])
        with pytest.raises(Exception):
            e["invalid_key"]

    def test_get_pairs(self):
        e = Edges([1, 2], [3, 4])
        pairs = e.get_pairs()
        assert pairs.shape == (2, 2)
        np.testing.assert_array_equal(pairs[:, 0], [1, 2])
        np.testing.assert_array_equal(pairs[:, 1], [3, 4])

    def test_get_pairs_caching(self):
        e = Edges([1, 2], [3, 4])
        p1 = e.get_pairs()
        p2 = e.get_pairs()
        assert p1 is p2

    def test_size_mismatch_raises(self):
        with pytest.raises(AssertionError):
            Edges([1, 2], [3])

    def test_affinities_setter(self):
        e = Edges([1], [2])
        new_affs = np.array([0.99], dtype=basetypes.EDGE_AFFINITY)
        e.affinities = new_affs
        np.testing.assert_array_almost_equal(e.affinities, new_affs)

    def test_areas_setter(self):
        e = Edges([1], [2])
        new_areas = np.array([42.0], dtype=basetypes.EDGE_AREA)
        e.areas = new_areas
        np.testing.assert_array_almost_equal(e.areas, new_areas)
