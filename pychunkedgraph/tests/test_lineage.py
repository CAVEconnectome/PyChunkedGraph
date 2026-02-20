"""Tests for pychunkedgraph.graph.lineage"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest
from networkx import DiGraph

from pychunkedgraph.graph.lineage import (
    get_latest_root_id,
    get_future_root_ids,
    get_past_root_ids,
    get_root_id_history,
    lineage_graph,
    get_previous_root_ids,
    _get_node_properties,
)
from pychunkedgraph.graph import attributes

from .helpers import create_chunk, to_label
from ..ingest.create.parent_layer import add_parent_chunk


class TestLineage:
    def _build_and_merge(self, gen_graph):
        """Build a graph with 2 isolated SVs, then merge them."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )

        old_root_0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root_1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))

        # Merge
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, old_root_0, old_root_1, new_root

    def test_get_latest_root_id_current(self, gen_graph):
        graph, _, _, new_root = self._build_and_merge(gen_graph)
        latest = get_latest_root_id(graph, new_root)
        assert new_root in latest

    def test_get_latest_root_id_after_edit(self, gen_graph):
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        latest = get_latest_root_id(graph, old_root_0)
        assert new_root in latest

    def test_get_future_root_ids(self, gen_graph):
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        future = get_future_root_ids(graph, old_root_0)
        assert new_root in future

    def test_get_past_root_ids(self, gen_graph):
        graph, old_root_0, old_root_1, new_root = self._build_and_merge(gen_graph)
        past = get_past_root_ids(graph, new_root)
        assert old_root_0 in past or old_root_1 in past

    def test_get_root_id_history(self, gen_graph):
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        history = get_root_id_history(graph, old_root_0)
        assert len(history) >= 2
        assert old_root_0 in history
        assert new_root in history

    def test_lineage_graph(self, gen_graph):
        """lineage_graph should return a DiGraph with nodes for old and new roots."""
        graph, old_root_0, old_root_1, new_root = self._build_and_merge(gen_graph)
        lg = lineage_graph(graph, [new_root])
        assert isinstance(lg, DiGraph)
        # The lineage graph should contain the new root
        assert new_root in lg.nodes
        # Should have at least 2 nodes (old root(s) + new root)
        assert len(lg.nodes) >= 2
        # Should have edges connecting old roots to the new root
        assert lg.number_of_edges() > 0

    def test_lineage_graph_with_timestamps(self, gen_graph):
        """lineage_graph should respect timestamp boundaries."""
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        # Build lineage graph with a past timestamp that includes the merge
        past = datetime.now(UTC) - timedelta(days=20)
        future = datetime.now(UTC) + timedelta(days=1)
        lg = lineage_graph(
            graph, [new_root], timestamp_past=past, timestamp_future=future
        )
        assert isinstance(lg, DiGraph)
        assert new_root in lg.nodes

    def test_lineage_graph_single_node_id(self, gen_graph):
        """lineage_graph should accept a single integer node_id."""
        graph, _, _, new_root = self._build_and_merge(gen_graph)
        lg = lineage_graph(graph, int(new_root))
        assert isinstance(lg, DiGraph)
        assert new_root in lg.nodes

    def test_get_previous_root_ids(self, gen_graph):
        """After a merge, get_previous_root_ids of the new root should include the old roots."""
        graph, old_root_0, old_root_1, new_root = self._build_and_merge(gen_graph)
        result = get_previous_root_ids(graph, [new_root])
        assert isinstance(result, dict)
        assert new_root in result
        previous = result[new_root]
        # The previous roots of the merged node should include the old roots
        assert old_root_0 in previous or old_root_1 in previous

    def test_get_node_properties(self, gen_graph):
        """_get_node_properties should extract timestamp and operation_id from a node entry."""
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        # Read the new root node with all properties
        node_entry = graph.client.read_node(new_root)
        assert node_entry is not None

        # _get_node_properties expects a dict with at least Hierarchy.Child
        props = _get_node_properties(node_entry)
        assert isinstance(props, dict)
        # Should have a 'timestamp' key with a float value (epoch seconds)
        assert "timestamp" in props
        assert isinstance(props["timestamp"], float)
        assert props["timestamp"] > 0

    def test_get_node_properties_with_operation_id(self, gen_graph):
        """Nodes created by edits should have an operation_id in their properties."""
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        # The old root should have NewParent and OperationID set after the merge
        node_entry = graph.client.read_node(old_root_0)
        props = _get_node_properties(node_entry)
        assert "timestamp" in props
        # Old roots involved in an edit should have operation_id
        if attributes.OperationLogs.OperationID in node_entry:
            assert "operation_id" in props


class TestGetFutureRootIdsLatest:
    """Test get_future_root_ids with different time_stamp values."""

    def _build_graph_with_two_merges(self, gen_graph):
        """Build a graph with 3 SVs, do 2 merges:
        First merge SV0+SV1 -> root_A
        Then merge root_A+SV2 -> root_B
        """
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)
        from .helpers import create_chunk, to_label

        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
                to_label(graph, 1, 0, 0, 0, 2),
            ],
            edges=[],
            timestamp=fake_ts,
        )

        old_root_0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root_1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))
        old_root_2 = graph.get_root(to_label(graph, 1, 0, 0, 0, 2))

        # First merge: SV0 + SV1
        result1 = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        mid_root = result1.new_root_ids[0]

        # Second merge: merged root + SV2
        result2 = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 2)],
            affinities=[0.3],
        )
        final_root = result2.new_root_ids[0]

        return graph, old_root_0, old_root_1, old_root_2, mid_root, final_root

    def test_future_root_ids_finds_chain(self, gen_graph):
        """get_future_root_ids from original root should find mid and final roots."""
        graph, old_root_0, _, _, mid_root, final_root = (
            self._build_graph_with_two_merges(gen_graph)
        )
        future = get_future_root_ids(graph, old_root_0)
        # Should find at least the mid root and final root
        assert len(future) >= 1
        # The final root should be reachable
        assert mid_root in future or final_root in future

    def test_future_root_ids_with_past_timestamp(self, gen_graph):
        """Using a very old timestamp should find nothing (no future roots before that time)."""
        graph, old_root_0, _, _, _, _ = self._build_graph_with_two_merges(gen_graph)
        very_old = datetime.now(UTC) - timedelta(days=20)
        future = get_future_root_ids(graph, old_root_0, time_stamp=very_old)
        # With a very old timestamp, no future roots should be found since
        # all edits happened after that time
        assert len(future) == 0

    def test_future_root_ids_current_root_returns_empty(self, gen_graph):
        """For the latest root, get_future_root_ids should return empty."""
        graph, _, _, _, _, final_root = self._build_graph_with_two_merges(gen_graph)
        future = get_future_root_ids(graph, final_root)
        assert len(future) == 0


class TestGetPastRootIdsTimestamps:
    """Test get_past_root_ids with different time_stamp values."""

    def _build_and_merge(self, gen_graph):
        """Build a graph with 2 isolated SVs, then merge them."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)
        from .helpers import create_chunk, to_label

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )

        old_root_0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root_1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))

        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, old_root_0, old_root_1, new_root

    def test_past_root_ids_of_merged_root(self, gen_graph):
        """get_past_root_ids of the merged root should find old roots."""
        graph, old_root_0, old_root_1, new_root = self._build_and_merge(gen_graph)
        past = get_past_root_ids(graph, new_root)
        assert old_root_0 in past or old_root_1 in past

    def test_past_root_ids_with_future_timestamp(self, gen_graph):
        """Using a far-future timestamp should find nothing (no past roots after that time)."""
        graph, _, _, new_root = self._build_and_merge(gen_graph)
        far_future = datetime.now(UTC) + timedelta(days=365)
        past = get_past_root_ids(graph, new_root, time_stamp=far_future)
        # With a far-future timestamp, the condition row_time_stamp > time_stamp
        # will be False, so no past roots should be found
        assert len(past) == 0

    def test_past_root_ids_original_root_empty(self, gen_graph):
        """An original root with no prior edits should have no past root ids."""
        graph, old_root_0, _, _ = self._build_and_merge(gen_graph)
        past = get_past_root_ids(graph, old_root_0)
        # The original root has no former parents, so past should be empty
        assert len(past) == 0


class TestGetRootIdHistory:
    """Test get_root_id_history returns full history."""

    def _build_and_merge(self, gen_graph):
        """Build a graph with 2 isolated SVs, then merge them."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)
        from .helpers import create_chunk, to_label

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )

        old_root_0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root_1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))

        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, old_root_0, old_root_1, new_root

    def test_history_after_merge(self, gen_graph):
        """After merge, get_root_id_history should contain past and current root."""
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        history = get_root_id_history(graph, old_root_0)
        assert isinstance(history, np.ndarray)
        # Should contain the queried root itself
        assert old_root_0 in history
        # Should contain the new root
        assert new_root in history
        assert len(history) >= 2

    def test_history_from_new_root(self, gen_graph):
        """get_root_id_history from the new root should include old roots."""
        graph, old_root_0, old_root_1, new_root = self._build_and_merge(gen_graph)
        history = get_root_id_history(graph, new_root)
        assert isinstance(history, np.ndarray)
        assert new_root in history
        # At least one old root should appear in the history
        assert old_root_0 in history or old_root_1 in history

    def test_history_with_timestamps(self, gen_graph):
        """get_root_id_history with restrictive timestamps may limit results."""
        graph, old_root_0, _, new_root = self._build_and_merge(gen_graph)
        # Very narrow time window: only current root
        far_future = datetime.now(UTC) + timedelta(days=365)
        very_old = datetime.now(UTC) - timedelta(days=365)
        history = get_root_id_history(
            graph,
            new_root,
            time_stamp_past=far_future,
            time_stamp_future=very_old,
        )
        assert isinstance(history, np.ndarray)
        # At minimum, the queried root itself should be in the history
        assert new_root in history


class TestGetRootIdHistoryDetailed:
    """Detailed tests for get_root_id_history covering all branches."""

    def _build_graph_with_two_merges(self, gen_graph):
        """Build a graph with 3 SVs, do 2 merges:
        First merge SV0+SV1 -> root_A
        Then merge root_A+SV2 -> root_B
        """
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
                to_label(graph, 1, 0, 0, 0, 2),
            ],
            edges=[],
            timestamp=fake_ts,
        )

        old_root_0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root_1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))
        old_root_2 = graph.get_root(to_label(graph, 1, 0, 0, 0, 2))

        # First merge: SV0 + SV1
        result1 = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        mid_root = result1.new_root_ids[0]

        # Second merge: merged root + SV2
        result2 = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 2)],
            affinities=[0.3],
        )
        final_root = result2.new_root_ids[0]

        return graph, old_root_0, old_root_1, old_root_2, mid_root, final_root

    def test_history_contains_all_roots_from_old(self, gen_graph):
        """get_root_id_history from original root should contain all related roots."""
        graph, old_root_0, _, _, mid_root, final_root = (
            self._build_graph_with_two_merges(gen_graph)
        )
        history = get_root_id_history(graph, old_root_0)
        assert isinstance(history, np.ndarray)
        # Should contain the queried root itself
        assert old_root_0 in history
        # Should contain mid_root (first merge)
        assert mid_root in history
        # Should contain final_root (second merge)
        assert final_root in history

    def test_history_from_mid_root(self, gen_graph):
        """get_root_id_history from mid root should include both past and future."""
        graph, old_root_0, old_root_1, _, mid_root, final_root = (
            self._build_graph_with_two_merges(gen_graph)
        )
        history = get_root_id_history(graph, mid_root)
        assert isinstance(history, np.ndarray)
        assert mid_root in history
        # Should include past roots
        assert old_root_0 in history or old_root_1 in history
        # Should include future root
        assert final_root in history

    def test_history_from_final_root(self, gen_graph):
        """get_root_id_history from final root should include all past roots."""
        graph, old_root_0, old_root_1, old_root_2, mid_root, final_root = (
            self._build_graph_with_two_merges(gen_graph)
        )
        history = get_root_id_history(graph, final_root)
        assert isinstance(history, np.ndarray)
        assert final_root in history
        # Should include the mid root
        assert mid_root in history
        # Should include at least one of the original roots
        assert old_root_0 in history or old_root_1 in history or old_root_2 in history

    def test_history_with_narrow_past_timestamp(self, gen_graph):
        """get_root_id_history with a very recent past timestamp excludes old roots."""
        graph, old_root_0, _, _, mid_root, final_root = (
            self._build_graph_with_two_merges(gen_graph)
        )
        # Use a very recent past timestamp to exclude past roots
        recent = datetime.now(UTC) + timedelta(days=365)
        history = get_root_id_history(
            graph,
            mid_root,
            time_stamp_past=recent,
        )
        assert isinstance(history, np.ndarray)
        # Should contain the root itself
        assert mid_root in history
        # Should still contain future roots (timestamp_future defaults to max)
        assert final_root in history

    def test_history_with_narrow_future_timestamp(self, gen_graph):
        """get_root_id_history with a very old future timestamp excludes future roots."""
        graph, old_root_0, _, _, mid_root, final_root = (
            self._build_graph_with_two_merges(gen_graph)
        )
        # Use a very old future timestamp to exclude future roots
        very_old = datetime.now(UTC) - timedelta(days=365)
        history = get_root_id_history(
            graph,
            mid_root,
            time_stamp_future=very_old,
        )
        assert isinstance(history, np.ndarray)
        # Should contain the root itself
        assert mid_root in history
        # Should contain past roots (timestamp_past defaults to min)
        assert old_root_0 in history
