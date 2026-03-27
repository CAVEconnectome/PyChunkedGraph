"""Tests for pychunkedgraph.ingest.cluster"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pychunkedgraph.ingest.cluster import _check_edges_direction, _post_task_completion
from pychunkedgraph.graph.edges import Edges, EDGE_TYPES
from pychunkedgraph.graph import basetypes


class TestCheckEdgesDirection:
    def test_correct_direction_passes(self, gen_graph):
        """Edges with node_ids1 inside the chunk should pass."""
        cg = gen_graph(n_layers=4)
        coord = [0, 0, 0]
        chunk_id = cg.get_chunk_id(layer=1, x=0, y=0, z=0)
        node1 = cg.get_node_id(np.uint64(1), np.uint64(chunk_id))
        # node2 in a different chunk
        other_chunk_id = cg.get_chunk_id(layer=1, x=1, y=0, z=0)
        node2 = cg.get_node_id(np.uint64(1), np.uint64(other_chunk_id))

        chunk_edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges(
                np.array([node1], dtype=basetypes.NODE_ID),
                np.array([node2], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        _check_edges_direction(chunk_edges, cg, coord)  # should not raise

    def test_wrong_direction_raises(self, gen_graph):
        """Edges with node_ids1 outside the chunk should raise AssertionError."""
        cg = gen_graph(n_layers=4)
        coord = [0, 0, 0]
        chunk_id = cg.get_chunk_id(layer=1, x=0, y=0, z=0)
        node_inside = cg.get_node_id(np.uint64(1), np.uint64(chunk_id))
        other_chunk_id = cg.get_chunk_id(layer=1, x=1, y=0, z=0)
        node_outside = cg.get_node_id(np.uint64(1), np.uint64(other_chunk_id))

        chunk_edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges(
                np.array([node_outside], dtype=basetypes.NODE_ID),
                np.array([node_inside], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        with pytest.raises(AssertionError, match="all IDs must belong to same chunk"):
            _check_edges_direction(chunk_edges, cg, coord)

    def test_empty_edges_passes(self, gen_graph):
        """Empty between/cross chunk edges should not raise."""
        cg = gen_graph(n_layers=4)
        coord = [0, 0, 0]
        chunk_edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        _check_edges_direction(chunk_edges, cg, coord)  # should not raise

    def test_cross_chunk_direction(self, gen_graph):
        """Cross-chunk edges also checked — node_ids1 must be inside the chunk."""
        cg = gen_graph(n_layers=4)
        coord = [0, 0, 0]
        chunk_id = cg.get_chunk_id(layer=1, x=0, y=0, z=0)
        node1 = cg.get_node_id(np.uint64(1), np.uint64(chunk_id))
        other_chunk_id = cg.get_chunk_id(layer=1, x=1, y=0, z=0)
        node2 = cg.get_node_id(np.uint64(1), np.uint64(other_chunk_id))

        chunk_edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges(
                np.array([node1], dtype=basetypes.NODE_ID),
                np.array([node2], dtype=basetypes.NODE_ID),
            ),
        }
        _check_edges_direction(chunk_edges, cg, coord)  # should not raise

    def test_multiple_edges_one_wrong(self, gen_graph):
        """If any edge has wrong direction, assertion should fail."""
        cg = gen_graph(n_layers=4)
        coord = [0, 0, 0]
        chunk_id = cg.get_chunk_id(layer=1, x=0, y=0, z=0)
        node_inside = cg.get_node_id(np.uint64(1), np.uint64(chunk_id))
        node_inside2 = cg.get_node_id(np.uint64(2), np.uint64(chunk_id))
        other_chunk_id = cg.get_chunk_id(layer=1, x=1, y=0, z=0)
        node_outside = cg.get_node_id(np.uint64(1), np.uint64(other_chunk_id))

        chunk_edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges(
                np.array([node_inside, node_outside], dtype=basetypes.NODE_ID),
                np.array([node_outside, node_inside2], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        with pytest.raises(AssertionError):
            _check_edges_direction(chunk_edges, cg, coord)


class TestPostTaskCompletion:
    def test_marks_chunk_complete(self):
        """Should call redis sadd with correct key format."""
        imanager = MagicMock()
        _post_task_completion(imanager, layer=2, coords=np.array([1, 2, 3]))
        imanager.redis.sadd.assert_called_once_with("2c", "1_2_3")

    def test_with_split_index(self):
        """When split is provided, appends split suffix to the key."""
        imanager = MagicMock()
        _post_task_completion(imanager, layer=3, coords=np.array([0, 0, 0]), split=1)
        imanager.redis.sadd.assert_called_once_with("3c", "0_0_0_1")

    def test_without_split(self):
        """When split is None, no suffix appended."""
        imanager = MagicMock()
        _post_task_completion(imanager, layer=5, coords=np.array([4, 5, 6]))
        call_args = imanager.redis.sadd.call_args[0]
        assert call_args[0] == "5c"
        assert "_" not in call_args[1].split("_", 3)[-1] or call_args[1] == "4_5_6"
