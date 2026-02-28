"""Tests for pychunkedgraph.graph.types"""

import numpy as np

from pychunkedgraph.graph.types import empty_1d, empty_2d, Agglomeration
from pychunkedgraph.graph import basetypes


class TestEmptyArrays:
    def test_empty_1d_shape_and_dtype(self):
        assert empty_1d.shape == (0,)
        assert empty_1d.dtype == basetypes.NODE_ID

    def test_empty_2d_shape_and_dtype(self):
        assert empty_2d.shape == (0, 2)
        assert empty_2d.dtype == basetypes.NODE_ID


class TestAgglomeration:
    def test_defaults(self):
        agg = Agglomeration(node_id=np.uint64(1))
        assert agg.node_id == np.uint64(1)
        assert agg.supervoxels.shape == (0,)
        assert agg.in_edges.shape == (0, 2)
        assert agg.out_edges.shape == (0, 2)
        assert agg.cross_edges.shape == (0, 2)
        assert agg.cross_edges_d == {}

    def test_custom_fields(self):
        svs = np.array([10, 20], dtype=basetypes.NODE_ID)
        agg = Agglomeration(node_id=np.uint64(5), supervoxels=svs)
        assert agg.node_id == np.uint64(5)
        np.testing.assert_array_equal(agg.supervoxels, svs)
