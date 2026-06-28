"""Tests for pychunkedgraph.graph.utils.id_helpers"""

import numpy as np

from pychunkedgraph.graph.utils import id_helpers
from pychunkedgraph.graph.chunks import utils as chunk_utils

from ..helpers import SV, label


class TestGetSegmentIdLimit:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = label(graph, SV(seg=1))
        limit = id_helpers.get_segment_id_limit(graph.meta, node_id)
        assert limit > 0
        assert isinstance(limit, np.uint64)


class TestGetSegmentId:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = label(graph, SV(seg=42))
        seg_id = id_helpers.get_segment_id(graph.meta, node_id)
        assert seg_id == 42


class TestGetNodeId:
    def test_from_chunk_id(self, gen_graph):
        graph = gen_graph(n_layers=4)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=1, x=0, y=0, z=0)
        node_id = id_helpers.get_node_id(
            graph.meta, segment_id=np.uint64(5), chunk_id=chunk_id
        )
        assert id_helpers.get_segment_id(graph.meta, node_id) == 5
        assert chunk_utils.get_chunk_layer(graph.meta, node_id) == 1

    def test_from_components(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = id_helpers.get_node_id(
            graph.meta, segment_id=np.uint64(7), layer=2, x=1, y=2, z=3
        )
        assert id_helpers.get_segment_id(graph.meta, node_id) == 7
        assert chunk_utils.get_chunk_layer(graph.meta, node_id) == 2
        coords = chunk_utils.get_chunk_coordinates(graph.meta, node_id)
        np.testing.assert_array_equal(coords, [1, 2, 3])
