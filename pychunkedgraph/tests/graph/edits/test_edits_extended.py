"""Tests for pychunkedgraph.graph.edits - extended coverage"""

from math import inf
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pychunkedgraph.graph.edits import CreateParentNodes, flip_ids
from pychunkedgraph.graph import basetypes
from pychunkedgraph.graph.exceptions import PostconditionError

from ...helpers import SV, build_graph


class TestFlipIds:
    def test_basic(self):
        id_map = {
            np.uint64(1): {np.uint64(10), np.uint64(11)},
            np.uint64(2): {np.uint64(20)},
        }
        result = flip_ids(id_map, [np.uint64(1), np.uint64(2)])
        assert np.uint64(10) in result
        assert np.uint64(11) in result
        assert np.uint64(20) in result

    def test_empty(self):
        id_map = {}
        result = flip_ids(id_map, [])
        assert len(result) == 0


class TestCrossEdgeParentRedistribution:
    @staticmethod
    def _create_parent_nodes(cg):
        return CreateParentNodes(
            cg,
            new_l2_ids=[],
            operation_id=134,
            time_stamp=datetime.now(UTC),
            parent_ts=datetime.now(UTC) - timedelta(seconds=1),
        )

    def test_resolved_edge_is_assigned_to_current_source_parent(self):
        parent_a = np.uint64(382805968326902400)
        parent_b = np.uint64(382805968326902401)
        child_a = np.uint64(101)
        child_b = np.uint64(102)
        destination = np.uint64(201)
        destination_parent = np.uint64(301)

        cg = MagicMock()
        cg.meta.layer_count = 5
        cg.get_chunk_layer.return_value = 3
        cg.get_children.return_value = {
            parent_a: np.array([child_a], dtype=np.uint64),
            parent_b: np.array([child_b], dtype=np.uint64),
        }
        cg.get_cross_chunk_edges.return_value = {
            child_a: {},
            child_b: {},
        }
        cg.get_chunk_layers.return_value = np.array([3, 3])
        cg.cache.cross_chunk_edges_cache = {}
        cg.client.mutate_row.return_value = "resolved-edge-mutation"

        updated_edges = {
            3: np.array([[child_a, destination]], dtype=np.uint64),
        }
        edge_nodes = np.array([child_a, destination], dtype=np.uint64)
        resolved_parents = np.array(
            [parent_b, destination_parent], dtype=np.uint64
        )

        create_parents = self._create_parent_nodes(cg)
        with (
            patch(
                "pychunkedgraph.graph.edits.get_latest_edges_wrapper",
                return_value=(updated_edges, edge_nodes),
            ),
            patch(
                "pychunkedgraph.graph.edits.get_new_nodes",
                return_value=resolved_parents,
            ),
        ):
            entries = create_parents._update_cross_edge_cache_batched(
                [parent_a, parent_b]
            )

        assert entries == ["resolved-edge-mutation"]
        assert cg.cache.cross_chunk_edges_cache[parent_a] == {}
        np.testing.assert_array_equal(
            cg.cache.cross_chunk_edges_cache[parent_b][3],
            np.array([[parent_b, destination_parent]], dtype=np.uint64),
        )

    def test_source_parent_outside_edit_batch_fails_before_mutation(self):
        parent_a = np.uint64(400)
        parent_b = np.uint64(401)
        child_a = np.uint64(101)
        destination = np.uint64(201)
        outside_parent = np.uint64(499)

        cg = MagicMock()
        cg.meta.layer_count = 5
        cg.get_chunk_layer.return_value = 3
        cg.get_children.return_value = {
            parent_a: np.array([child_a], dtype=np.uint64),
            parent_b: np.array([], dtype=np.uint64),
        }
        cg.get_cross_chunk_edges.return_value = {child_a: {}}
        cg.get_chunk_layers.return_value = np.array([3, 3])
        cg.cache.cross_chunk_edges_cache = {}

        updated_edges = {
            3: np.array([[child_a, destination]], dtype=np.uint64),
        }
        edge_nodes = np.array([child_a, destination], dtype=np.uint64)
        resolved_parents = np.array(
            [outside_parent, np.uint64(301)], dtype=np.uint64
        )

        create_parents = self._create_parent_nodes(cg)
        with (
            patch(
                "pychunkedgraph.graph.edits.get_latest_edges_wrapper",
                return_value=(updated_edges, edge_nodes),
            ),
            patch(
                "pychunkedgraph.graph.edits.get_new_nodes",
                return_value=resolved_parents,
            ),
            pytest.raises(PostconditionError, match="parent resolution"),
        ):
            create_parents._update_cross_edge_cache_batched([parent_a, parent_b])

        cg.client.mutate_row.assert_not_called()
        assert cg.cache.cross_chunk_edges_cache == {}

    @pytest.mark.parametrize(
        "resolved_parents",
        [
            np.array([0, 301], dtype=np.uint64),
            np.array([400, 0], dtype=np.uint64),
        ],
    )
    def test_zero_parent_fails_before_mutation(self, resolved_parents):
        parent = np.uint64(400)
        child = np.uint64(101)
        destination = np.uint64(201)

        cg = MagicMock()
        cg.meta.layer_count = 5
        cg.get_chunk_layer.return_value = 3
        cg.get_children.return_value = {
            parent: np.array([child], dtype=np.uint64),
        }
        cg.get_cross_chunk_edges.return_value = {child: {}}
        cg.cache.cross_chunk_edges_cache = {}

        updated_edges = {
            3: np.array([[child, destination]], dtype=np.uint64),
        }
        edge_nodes = np.array([child, destination], dtype=np.uint64)

        create_parents = self._create_parent_nodes(cg)
        with (
            patch(
                "pychunkedgraph.graph.edits.get_latest_edges_wrapper",
                return_value=(updated_edges, edge_nodes),
            ),
            patch(
                "pychunkedgraph.graph.edits.get_new_nodes",
                return_value=resolved_parents,
            ),
            pytest.raises(PostconditionError, match="parent resolution"),
        ):
            create_parents._update_cross_edge_cache_batched([parent])

        cg.get_chunk_layers.assert_not_called()
        cg.client.mutate_row.assert_not_called()
        assert cg.cache.cross_chunk_edges_cache == {}

    def test_destination_parent_at_wrong_layer_fails_before_mutation(self):
        parent = np.uint64(400)
        child = np.uint64(101)
        destination = np.uint64(201)
        destination_parent = np.uint64(301)

        cg = MagicMock()
        cg.meta.layer_count = 5
        cg.get_chunk_layer.return_value = 3
        cg.get_children.return_value = {
            parent: np.array([child], dtype=np.uint64),
        }
        cg.get_cross_chunk_edges.return_value = {child: {}}
        cg.get_chunk_layers.return_value = np.array([3, 4])
        cg.cache.cross_chunk_edges_cache = {}

        updated_edges = {
            3: np.array([[child, destination]], dtype=np.uint64),
        }
        edge_nodes = np.array([child, destination], dtype=np.uint64)
        resolved_parents = np.array(
            [parent, destination_parent], dtype=np.uint64
        )

        create_parents = self._create_parent_nodes(cg)
        with (
            patch(
                "pychunkedgraph.graph.edits.get_latest_edges_wrapper",
                return_value=(updated_edges, edge_nodes),
            ),
            patch(
                "pychunkedgraph.graph.edits.get_new_nodes",
                return_value=resolved_parents,
            ),
            pytest.raises(PostconditionError, match="parent resolution"),
        ):
            create_parents._update_cross_edge_cache_batched([parent])

        cg.client.mutate_row.assert_not_called()
        assert cg.cache.cross_chunk_edges_cache == {}


class TestInitOldHierarchy:
    def test_basic(self, gen_graph):
        from pychunkedgraph.graph.edits import _init_old_hierarchy

        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
            edges=[("a0", "a1", 0.5)],
        )

        l2_parent = cg.get_parent(sv["a0"])
        result = _init_old_hierarchy(cg, np.array([l2_parent], dtype=np.uint64))
        assert l2_parent in result
        assert 2 in result[l2_parent]
