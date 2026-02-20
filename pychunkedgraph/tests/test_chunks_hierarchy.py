"""Tests for pychunkedgraph.graph.chunks.hierarchy"""

import numpy as np

from pychunkedgraph.graph.chunks import hierarchy
from pychunkedgraph.graph.chunks import utils as chunk_utils

from .helpers import to_label


class TestGetChildrenChunkCoords:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=5)
        coords = hierarchy.get_children_chunk_coords(graph.meta, 3, [0, 0, 0])
        # Layer 3 chunk at [0,0,0] has fanout=2 children: 2^3 = 8 max
        assert len(coords) > 0
        assert coords.shape[1] == 3


class TestGetChildrenChunkIds:
    def test_layer_1_returns_empty(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = to_label(graph, 1, 0, 0, 0, 1)
        result = hierarchy.get_children_chunk_ids(graph.meta, node_id)
        assert len(result) == 0

    def test_layer_2_returns_self(self, gen_graph):
        graph = gen_graph(n_layers=4)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=0, y=0, z=0)
        result = hierarchy.get_children_chunk_ids(graph.meta, chunk_id)
        assert len(result) == 1

    def test_layer_3(self, gen_graph):
        graph = gen_graph(n_layers=5)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=3, x=0, y=0, z=0)
        result = hierarchy.get_children_chunk_ids(graph.meta, chunk_id)
        assert len(result) > 0


class TestGetParentChunkId:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=5)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=0, y=0, z=0)
        parent_id = hierarchy.get_parent_chunk_id(graph.meta, chunk_id, 3)
        assert chunk_utils.get_chunk_layer(graph.meta, parent_id) == 3

    def test_parent_coords(self, gen_graph):
        graph = gen_graph(n_layers=5)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=2, y=3, z=1)
        parent_id = hierarchy.get_parent_chunk_id(graph.meta, chunk_id, 3)
        coords = chunk_utils.get_chunk_coordinates(graph.meta, parent_id)
        # With fanout=2, coords should be floor(original / 2)
        np.testing.assert_array_equal(coords, [1, 1, 0])


class TestGetParentChunkIdMultiple:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=5)
        ids = np.array(
            [
                chunk_utils.get_chunk_id(graph.meta, layer=2, x=0, y=0, z=0),
                chunk_utils.get_chunk_id(graph.meta, layer=2, x=1, y=0, z=0),
            ],
            dtype=np.uint64,
        )
        result = hierarchy.get_parent_chunk_id_multiple(graph.meta, ids)
        assert len(result) == 2
        for pid in result:
            assert chunk_utils.get_chunk_layer(graph.meta, pid) == 3


class TestGetParentChunkIds:
    def test_returns_chain(self, gen_graph):
        graph = gen_graph(n_layers=5)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=0, y=0, z=0)
        result = hierarchy.get_parent_chunk_ids(graph.meta, chunk_id)
        # Should include chunk_id + parents up to layer_count
        assert len(result) >= 2


class TestGetParentChunkIdDict:
    def test_returns_dict(self, gen_graph):
        graph = gen_graph(n_layers=5)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=0, y=0, z=0)
        result = hierarchy.get_parent_chunk_id_dict(graph.meta, chunk_id)
        assert isinstance(result, dict)
        assert 2 in result
