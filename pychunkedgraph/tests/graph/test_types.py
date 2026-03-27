"""Tests for pychunkedgraph.graph.types"""

import numpy as np

from pychunkedgraph.graph.types import Agglomeration
from pychunkedgraph.graph import basetypes


class TestAgglomeration:
    def test_custom_fields(self):
        svs = np.array([10, 20], dtype=basetypes.NODE_ID)
        agg = Agglomeration(node_id=np.uint64(5), supervoxels=svs)
        assert agg.node_id == np.uint64(5)
        np.testing.assert_array_equal(agg.supervoxels, svs)
