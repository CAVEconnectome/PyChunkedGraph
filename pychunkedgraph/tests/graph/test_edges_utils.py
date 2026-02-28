"""Tests for pychunkedgraph.graph.edges.utils"""

import numpy as np

from pychunkedgraph.graph.edges import Edges, EDGE_TYPES
from pychunkedgraph.graph.edges.utils import (
    concatenate_chunk_edges,
    concatenate_cross_edge_dicts,
    merge_cross_edge_dicts,
    get_cross_chunk_edges_layer,
)
from pychunkedgraph.graph import basetypes

from ..helpers import to_label


class TestConcatenateChunkEdges:
    def test_basic(self):
        d1 = {
            EDGE_TYPES.in_chunk: Edges([1, 2], [3, 4]),
            EDGE_TYPES.between_chunk: Edges([5], [6]),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        d2 = {
            EDGE_TYPES.in_chunk: Edges([7], [8]),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([9], [10]),
        }
        result = concatenate_chunk_edges([d1, d2])
        assert len(result[EDGE_TYPES.in_chunk]) == 3
        assert len(result[EDGE_TYPES.between_chunk]) == 1
        assert len(result[EDGE_TYPES.cross_chunk]) == 1

    def test_empty(self):
        result = concatenate_chunk_edges([])
        for edge_type in EDGE_TYPES:
            assert len(result[edge_type]) == 0


class TestConcatenateCrossEdgeDicts:
    def test_no_unique(self):
        d1 = {3: np.array([[1, 2]], dtype=basetypes.NODE_ID)}
        d2 = {3: np.array([[1, 2], [3, 4]], dtype=basetypes.NODE_ID)}
        result = concatenate_cross_edge_dicts([d1, d2], unique=False)
        assert len(result[3]) == 3  # duplicates kept

    def test_unique(self):
        d1 = {3: np.array([[1, 2]], dtype=basetypes.NODE_ID)}
        d2 = {3: np.array([[1, 2], [3, 4]], dtype=basetypes.NODE_ID)}
        result = concatenate_cross_edge_dicts([d1, d2], unique=True)
        assert len(result[3]) == 2  # duplicates removed

    def test_different_layers(self):
        d1 = {3: np.array([[1, 2]], dtype=basetypes.NODE_ID)}
        d2 = {4: np.array([[5, 6]], dtype=basetypes.NODE_ID)}
        result = concatenate_cross_edge_dicts([d1, d2])
        assert 3 in result
        assert 4 in result


class TestMergeCrossEdgeDicts:
    def test_basic(self):
        d1 = {
            np.uint64(100): {3: np.array([[1, 2]], dtype=basetypes.NODE_ID)},
        }
        d2 = {
            np.uint64(100): {3: np.array([[3, 4]], dtype=basetypes.NODE_ID)},
            np.uint64(200): {4: np.array([[5, 6]], dtype=basetypes.NODE_ID)},
        }
        result = merge_cross_edge_dicts(d1, d2)
        assert np.uint64(100) in result
        assert np.uint64(200) in result
        assert len(result[np.uint64(100)][3]) == 2


class TestGetCrossChunkEdgesLayer:
    def test_empty(self, gen_graph):
        graph = gen_graph(n_layers=4)
        result = get_cross_chunk_edges_layer(graph.meta, [])
        assert len(result) == 0

    def test_same_chunk(self, gen_graph):
        graph = gen_graph(n_layers=4)
        sv1 = to_label(graph, 1, 0, 0, 0, 1)
        sv2 = to_label(graph, 1, 0, 0, 0, 2)
        edges = np.array([[sv1, sv2]], dtype=basetypes.NODE_ID)
        result = get_cross_chunk_edges_layer(graph.meta, edges)
        assert result[0] == 1  # same chunk -> layer 1

    def test_adjacent_chunks(self, gen_graph):
        graph = gen_graph(n_layers=4)
        sv1 = to_label(graph, 1, 0, 0, 0, 1)
        sv2 = to_label(graph, 1, 1, 0, 0, 1)
        edges = np.array([[sv1, sv2]], dtype=basetypes.NODE_ID)
        result = get_cross_chunk_edges_layer(graph.meta, edges)
        assert result[0] >= 2  # different chunks -> higher layer
