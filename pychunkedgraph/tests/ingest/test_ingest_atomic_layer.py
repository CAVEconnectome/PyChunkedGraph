"""Tests for pychunkedgraph.ingest.create.atomic_layer"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from pychunkedgraph.ingest.create.atomic_layer import (
    _get_chunk_nodes_and_edges,
    _get_remapping,
)
from pychunkedgraph.graph.edges import Edges, EDGE_TYPES
from pychunkedgraph.graph import basetypes


class TestGetChunkNodesAndEdges:
    def test_basic(self):
        chunk_edges_d = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1, 2], dtype=basetypes.NODE_ID),
                np.array([3, 4], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges(
                np.array([1], dtype=basetypes.NODE_ID),
                np.array([5], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        isolated = np.array([10], dtype=np.uint64)
        node_ids, edge_ids = _get_chunk_nodes_and_edges(chunk_edges_d, isolated)
        assert 10 in node_ids
        assert 1 in node_ids
        assert 3 in node_ids
        assert len(edge_ids) > 0

    def test_isolated_only(self):
        chunk_edges_d = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        isolated = np.array([10, 20], dtype=np.uint64)
        node_ids, edge_ids = _get_chunk_nodes_and_edges(chunk_edges_d, isolated)
        assert 10 in node_ids
        assert 20 in node_ids


class TestGetRemapping:
    def test_basic(self):
        chunk_edges_d = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1, 2], dtype=basetypes.NODE_ID),
                np.array([3, 4], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges(
                np.array([1], dtype=basetypes.NODE_ID),
                np.array([5], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges(
                np.array([2], dtype=basetypes.NODE_ID),
                np.array([6], dtype=basetypes.NODE_ID),
            ),
        }
        sparse_indices, remapping = _get_remapping(chunk_edges_d)
        assert EDGE_TYPES.between_chunk in remapping
        assert EDGE_TYPES.cross_chunk in remapping
