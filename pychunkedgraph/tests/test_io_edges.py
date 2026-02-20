"""Tests for pychunkedgraph.io.edges using file:// protocol"""

import numpy as np
import pytest

from pychunkedgraph.io.edges import (
    serialize,
    deserialize,
    get_chunk_edges,
    put_chunk_edges,
    _parse_edges,
)
from pychunkedgraph.graph.edges import Edges, EDGE_TYPES
from pychunkedgraph.graph.utils import basetypes


class TestSerializeDeserialize:
    def test_roundtrip(self):
        ids1 = np.array([1, 2, 3], dtype=basetypes.NODE_ID)
        ids2 = np.array([4, 5, 6], dtype=basetypes.NODE_ID)
        affs = np.array([0.5, 0.6, 0.7], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([10, 20, 30], dtype=basetypes.EDGE_AREA)
        edges = Edges(ids1, ids2, affinities=affs, areas=areas)

        proto = serialize(edges)
        result = deserialize(proto)
        np.testing.assert_array_equal(result.node_ids1, ids1)
        np.testing.assert_array_equal(result.node_ids2, ids2)
        np.testing.assert_array_almost_equal(result.affinities, affs)
        np.testing.assert_array_almost_equal(result.areas, areas)

    def test_empty_edges(self):
        edges = Edges([], [])
        proto = serialize(edges)
        result = deserialize(proto)
        assert len(result) == 0


class TestParseEdges:
    def test_empty_list(self):
        result = _parse_edges([])
        assert result == []


class TestPutGetChunkEdges:
    def test_roundtrip_via_filesystem(self, tmp_path):
        edges_dir = f"file://{tmp_path}"
        chunk_coord = np.array([0, 0, 0])

        edges_d = {
            EDGE_TYPES.in_chunk: Edges(
                [1, 2],
                [3, 4],
                affinities=[0.5, 0.6],
                areas=[10, 20],
            ),
            EDGE_TYPES.between_chunk: Edges(
                [5],
                [6],
                affinities=[0.7],
                areas=[30],
            ),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }

        put_chunk_edges(edges_dir, chunk_coord, edges_d, compression_level=3)
        result = get_chunk_edges(edges_dir, [chunk_coord])

        assert EDGE_TYPES.in_chunk in result
        assert EDGE_TYPES.between_chunk in result
        assert EDGE_TYPES.cross_chunk in result
        assert len(result[EDGE_TYPES.in_chunk]) == 2
        assert len(result[EDGE_TYPES.between_chunk]) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        edges_dir = f"file://{tmp_path}"
        result = get_chunk_edges(edges_dir, [np.array([99, 99, 99])])
        for edge_type in EDGE_TYPES:
            assert len(result[edge_type]) == 0
