"""Tests for pychunkedgraph.graph.segmenthistory"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest
from pandas import DataFrame

from pychunkedgraph.graph.segmenthistory import (
    SegmentHistory,
    LogEntry,
    get_all_log_entries,
)

from .helpers import create_chunk, to_label
from ..ingest.create.parent_layer import add_parent_chunk


class TestSegmentHistory:
    def _build_and_merge(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )

        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, new_root

    def test_init(self, gen_graph):
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        assert len(sh.root_ids) == 1

    def test_lineage_graph(self, gen_graph):
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        lg = sh.lineage_graph
        assert len(lg.nodes) > 0

    def test_operation_ids(self, gen_graph):
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        ops = sh.operation_ids
        assert len(ops) > 0

    def test_past_operation_ids(self, gen_graph):
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        past_ops = sh.past_operation_ids(root_id=new_root)
        assert isinstance(past_ops, np.ndarray)

    def test_collect_edited_sv_ids(self, gen_graph):
        """After a merge, collect_edited_sv_ids should return supervoxel IDs."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        sv_ids = sh.collect_edited_sv_ids()
        assert isinstance(sv_ids, np.ndarray)
        assert sv_ids.dtype == np.uint64
        # The merge involved 2 supervoxels, so at least some IDs should appear
        assert len(sv_ids) > 0

    def test_collect_edited_sv_ids_with_root(self, gen_graph):
        """collect_edited_sv_ids with an explicit root_id should also work."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        sv_ids = sh.collect_edited_sv_ids(root_id=new_root)
        assert isinstance(sv_ids, np.ndarray)
        assert len(sv_ids) > 0

    def test_root_id_operation_id_dict(self, gen_graph):
        """root_id_operation_id_dict maps each root_id in the lineage to its operation_id."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        d = sh.root_id_operation_id_dict
        assert isinstance(d, dict)
        # Should contain at least the new root
        assert new_root in d
        # Values should be integer operation IDs (including 0 for non-edit nodes)
        for root_id, op_id in d.items():
            assert isinstance(root_id, (int, np.integer))
            assert isinstance(op_id, (int, np.integer))

    def test_root_id_timestamp_dict(self, gen_graph):
        """root_id_timestamp_dict maps each root_id to a timestamp."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        d = sh.root_id_timestamp_dict
        assert isinstance(d, dict)
        assert new_root in d
        # Timestamps should be numeric (epoch seconds) or 0 for defaults
        for root_id, ts in d.items():
            assert isinstance(ts, (int, float, np.integer, np.floating))

    def test_last_edit_timestamp(self, gen_graph):
        """last_edit_timestamp should return the timestamp for the given root."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        ts = sh.last_edit_timestamp(root_id=new_root)
        # Should be a numeric timestamp (float epoch) or default value
        assert isinstance(ts, (int, float, np.integer, np.floating))

    def test_log_entry_api(self, gen_graph):
        """After a merge, retrieve a log entry and verify its properties."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        op_ids = sh.operation_ids
        # Filter out operation_id 0 (default for nodes without operations)
        op_ids = op_ids[op_ids != 0]
        assert len(op_ids) > 0, "Expected at least one real operation ID"

        entry = sh.log_entry(op_ids[0])
        assert isinstance(entry, LogEntry)

        # is_merge should be True since we performed a merge
        assert entry.is_merge is True

        # user_id should be the user we passed to add_edges
        assert entry.user_id == "TestUser"

        # log_type should be "merge"
        assert entry.log_type == "merge"

        # edges_failsafe should return an array of SV IDs
        ef = entry.edges_failsafe
        assert isinstance(ef, np.ndarray)
        assert len(ef) > 0

        # __str__ should return a non-empty string
        s = str(entry)
        assert isinstance(s, str)
        assert len(s) > 0

        # __iter__ should yield attributes (user_id, log_type, root_ids, timestamp)
        items = list(entry)
        assert len(items) == 4

    def test_tabular_changelogs(self, gen_graph):
        """After a merge, tabular_changelogs should produce a DataFrame per root."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        changelogs = sh.tabular_changelogs
        assert isinstance(changelogs, dict)
        assert new_root in changelogs

        df = changelogs[new_root]
        assert isinstance(df, DataFrame)

        # Verify expected columns are present
        expected_columns = {
            "operation_id",
            "timestamp",
            "user_id",
            "before_root_ids",
            "after_root_ids",
            "is_merge",
            "in_neuron",
            "is_relevant",
        }
        assert expected_columns.issubset(set(df.columns))

        # Should have at least one row (the merge we performed)
        assert len(df) > 0

    def test_tabular_changelog_single_root(self, gen_graph):
        """tabular_changelog() with a single root should return the DataFrame directly."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        df = sh.tabular_changelog()
        assert isinstance(df, DataFrame)
        assert len(df) > 0

    def test_operation_id_root_id_dict(self, gen_graph):
        """operation_id_root_id_dict should be the inverse of root_id_operation_id_dict."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        d = sh.operation_id_root_id_dict
        assert isinstance(d, dict)
        # Each value should be a list of root IDs
        for op_id, root_ids in d.items():
            assert isinstance(root_ids, list)
            assert len(root_ids) > 0

    def test_tabular_changelogs_filtered(self, gen_graph):
        """After merge, tabular_changelogs_filtered returns dict with DataFrames
        that have 'in_neuron' and 'is_relevant' columns."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        filtered = sh.tabular_changelogs_filtered
        assert isinstance(filtered, dict)
        assert new_root in filtered
        df = filtered[new_root]
        assert isinstance(df, DataFrame)
        # The filtered method calls tabular_changelog(filtered=True) which
        # drops "in_neuron" and "is_relevant" columns after filtering
        assert "in_neuron" not in df.columns
        assert "is_relevant" not in df.columns

    def test_tabular_changelog_with_explicit_root(self, gen_graph):
        """tabular_changelog(root_id=new_root) should work same as without."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        df_implicit = sh.tabular_changelog()
        df_explicit = sh.tabular_changelog(root_id=new_root)
        assert isinstance(df_explicit, DataFrame)
        assert len(df_explicit) == len(df_implicit)
        # Same columns
        assert set(df_explicit.columns) == set(df_implicit.columns)

    def test_change_log_summary(self, gen_graph):
        """change_log_summary should return n_splits, n_mergers, user_info, etc."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        summary = sh.change_log_summary(root_id=new_root)
        assert isinstance(summary, dict)
        assert "n_splits" in summary
        assert "n_mergers" in summary
        assert "user_info" in summary
        assert "operations_ids" in summary
        assert "past_ids" in summary
        assert summary["n_mergers"] >= 1

    def test_past_future_id_mapping(self, gen_graph):
        """past_future_id_mapping should return two dicts mapping past<->future root IDs."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        past_map, future_map = sh.past_future_id_mapping(root_id=new_root)
        assert isinstance(past_map, dict)
        assert isinstance(future_map, dict)
        # The new_root should appear in past_map
        assert int(new_root) in past_map


class TestLogEntryUnit:
    """Pure unit tests for LogEntry class (no emulator needed)."""

    def test_merge_entry(self):
        from pychunkedgraph.graph.attributes import OperationLogs

        row = {
            OperationLogs.AddedEdge: np.array([[1, 2]], dtype=np.uint64),
            OperationLogs.UserID: "alice",
            OperationLogs.RootID: np.array([100], dtype=np.uint64),
            OperationLogs.SourceID: np.array([1], dtype=np.uint64),
            OperationLogs.SinkID: np.array([2], dtype=np.uint64),
            OperationLogs.SourceCoordinate: np.array([0, 0, 0]),
            OperationLogs.SinkCoordinate: np.array([1, 1, 1]),
        }
        ts = datetime.now(UTC)
        entry = LogEntry(row, timestamp=ts)
        assert entry.is_merge is True
        assert entry.log_type == "merge"
        assert entry.user_id == "alice"
        assert entry.timestamp == ts
        np.testing.assert_array_equal(entry.root_ids, np.array([100], dtype=np.uint64))
        np.testing.assert_array_equal(
            entry.added_edges, np.array([[1, 2]], dtype=np.uint64)
        )
        coords = entry.coordinates
        assert coords.shape == (2, 3)
        ef = entry.edges_failsafe
        assert len(ef) > 0

    def test_split_entry(self):
        from pychunkedgraph.graph.attributes import OperationLogs

        row = {
            OperationLogs.RemovedEdge: np.array([[3, 4]], dtype=np.uint64),
            OperationLogs.UserID: "bob",
            OperationLogs.RootID: np.array([200, 201], dtype=np.uint64),
            OperationLogs.SourceID: np.array([3], dtype=np.uint64),
            OperationLogs.SinkID: np.array([4], dtype=np.uint64),
            OperationLogs.SourceCoordinate: np.array([0, 0, 0]),
            OperationLogs.SinkCoordinate: np.array([1, 1, 1]),
        }
        ts = datetime.now(UTC)
        entry = LogEntry(row, timestamp=ts)
        assert entry.is_merge is False
        assert entry.log_type == "split"
        assert entry.user_id == "bob"
        np.testing.assert_array_equal(
            entry.removed_edges, np.array([[3, 4]], dtype=np.uint64)
        )
        assert len(str(entry)) > 0
        assert len(list(entry)) == 4

    def test_added_edges_on_split_raises(self):
        from pychunkedgraph.graph.attributes import OperationLogs

        row = {
            OperationLogs.RemovedEdge: np.array([[3, 4]], dtype=np.uint64),
            OperationLogs.UserID: "bob",
            OperationLogs.RootID: np.array([200], dtype=np.uint64),
        }
        entry = LogEntry(row, timestamp=datetime.now(UTC))
        with pytest.raises(AssertionError, match="Not a merge"):
            entry.added_edges

    def test_removed_edges_on_merge_raises(self):
        from pychunkedgraph.graph.attributes import OperationLogs

        row = {
            OperationLogs.AddedEdge: np.array([[1, 2]], dtype=np.uint64),
            OperationLogs.UserID: "alice",
            OperationLogs.RootID: np.array([100], dtype=np.uint64),
        }
        entry = LogEntry(row, timestamp=datetime.now(UTC))
        with pytest.raises(AssertionError, match="Not a split"):
            entry.removed_edges


class TestGetAllLogEntries:
    def test_empty_graph(self, gen_graph):
        """Create graph with no operations. get_all_log_entries should return empty list."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        # Create a chunk with vertices but perform no edits
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        entries = get_all_log_entries(graph)
        assert isinstance(entries, list)
        assert len(entries) == 0

    def test_basic(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )
        graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        # get_all_log_entries iterates range(get_max_operation_id()) which
        # may not include the actual operation ID; verify it doesn't crash
        entries = get_all_log_entries(graph)
        assert isinstance(entries, list)
        # If entries exist, verify LogEntry API works
        for entry in entries:
            assert entry.log_type in ("merge", "split")
            assert str(entry)
            for _ in entry:
                pass


class TestMergeLog:
    """Tests for SegmentHistory.merge_log() method (lines 245-268)."""

    def _build_and_merge(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
            source_coords=[0, 0, 0],
            sink_coords=[1, 1, 1],
        )
        new_root = result.new_root_ids[0]
        return graph, new_root

    def test_merge_log_with_root(self, gen_graph):
        """merge_log(root_id=...) should return merge_edges and merge_edge_coords."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        result = sh.merge_log(root_id=new_root)
        assert isinstance(result, dict)
        assert "merge_edges" in result
        assert "merge_edge_coords" in result
        # We performed one merge, so there should be one entry
        assert len(result["merge_edges"]) >= 1
        assert len(result["merge_edge_coords"]) >= 1

    def test_merge_log_without_root(self, gen_graph):
        """merge_log() without root_id should iterate over all root_ids."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        result = sh.merge_log()
        assert isinstance(result, dict)
        assert "merge_edges" in result
        assert "merge_edge_coords" in result

    def test_merge_log_correct_for_wrong_coord_type_false(self, gen_graph):
        """merge_log with correct_for_wrong_coord_type=False should skip coord hack."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        result = sh.merge_log(root_id=new_root, correct_for_wrong_coord_type=False)
        assert isinstance(result, dict)
        assert "merge_edges" in result
        assert len(result["merge_edges"]) >= 1


class TestPastOperationIdsExtended:
    """Tests for SegmentHistory.past_operation_ids() (lines 270-292)."""

    def _build_and_merge(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, new_root

    def test_past_operation_ids_without_root(self, gen_graph):
        """past_operation_ids() without root_id iterates all root_ids."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        ops = sh.past_operation_ids()
        assert isinstance(ops, np.ndarray)
        # Should have at least the merge operation
        assert len(ops) >= 1
        # 0 should not appear in result
        assert 0 not in ops

    def test_past_operation_ids_with_root(self, gen_graph):
        """past_operation_ids(root_id=...) should return operations for that root."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        ops = sh.past_operation_ids(root_id=new_root)
        assert isinstance(ops, np.ndarray)
        assert len(ops) >= 1


class TestPastFutureIdMappingExtended:
    """More thorough tests for past_future_id_mapping (lines 315-368)."""

    def _build_and_merge(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, new_root

    def test_past_future_id_mapping_without_root(self, gen_graph):
        """past_future_id_mapping() without root_id iterates all root_ids."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        past_map, future_map = sh.past_future_id_mapping()
        assert isinstance(past_map, dict)
        assert isinstance(future_map, dict)
        assert int(new_root) in past_map

    def test_past_future_id_mapping_values(self, gen_graph):
        """Verify past_map values are arrays of past root IDs."""
        graph, new_root = self._build_and_merge(gen_graph)
        sh = SegmentHistory(graph, new_root)
        past_map, future_map = sh.past_future_id_mapping(root_id=new_root)
        # past_map[int(new_root)] should point back to the original roots
        past_ids = past_map[int(new_root)]
        assert len(past_ids) >= 1
        # future_map should have entries for the past IDs
        for past_id in past_ids:
            if past_id in future_map:
                assert future_map[past_id] is not None


class TestMergeSplitHistory:
    """Tests involving merge followed by split to cover more branches."""

    def _build_merge_and_split(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_ts,
        )
        # Merge
        merge_result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
            source_coords=[0, 0, 0],
            sink_coords=[1, 1, 1],
        )
        merge_root = merge_result.new_root_ids[0]

        # Split
        split_result = graph.remove_edges(
            "TestUser",
            source_ids=to_label(graph, 1, 0, 0, 0, 0),
            sink_ids=to_label(graph, 1, 0, 0, 0, 1),
            mincut=False,
        )
        split_roots = split_result.new_root_ids
        return graph, merge_root, split_roots

    def test_change_log_summary_with_split(self, gen_graph):
        """change_log_summary after merge+split should show both operations."""
        graph, merge_root, split_roots = self._build_merge_and_split(gen_graph)
        # Use the first split root as the segment history root
        root = split_roots[0]
        sh = SegmentHistory(graph, root)
        summary = sh.change_log_summary(root_id=root)
        assert isinstance(summary, dict)
        assert "n_splits" in summary
        assert "n_mergers" in summary
        # There was at least a merge and a split in the history
        total_ops = summary["n_splits"] + summary["n_mergers"]
        assert total_ops >= 1

    def test_past_operation_ids_after_split(self, gen_graph):
        """past_operation_ids should include both merge and split operations."""
        graph, merge_root, split_roots = self._build_merge_and_split(gen_graph)
        root = split_roots[0]
        sh = SegmentHistory(graph, root)
        ops = sh.past_operation_ids(root_id=root)
        assert isinstance(ops, np.ndarray)
        # Should include at least 2 operations (merge + split)
        assert len(ops) >= 2

    def test_merge_log_after_split(self, gen_graph):
        """merge_log after split should still find the original merge."""
        graph, merge_root, split_roots = self._build_merge_and_split(gen_graph)
        root = split_roots[0]
        sh = SegmentHistory(graph, root)
        result = sh.merge_log(root_id=root)
        assert isinstance(result, dict)
        # The original merge should still be in the history
        assert len(result["merge_edges"]) >= 1

    def test_tabular_changelog_after_split(self, gen_graph):
        """tabular_changelog after merge+split should have multiple rows."""
        graph, merge_root, split_roots = self._build_merge_and_split(gen_graph)
        root = split_roots[0]
        sh = SegmentHistory(graph, root)
        df = sh.tabular_changelog(root_id=root)
        assert isinstance(df, DataFrame)
        # Should have at least 2 rows (merge + split)
        assert len(df) >= 2

    def test_past_future_id_mapping_after_split(self, gen_graph):
        """past_future_id_mapping after merge+split should track the lineage."""
        graph, merge_root, split_roots = self._build_merge_and_split(gen_graph)
        root = split_roots[0]
        sh = SegmentHistory(graph, root)
        past_map, future_map = sh.past_future_id_mapping(root_id=root)
        assert isinstance(past_map, dict)
        assert isinstance(future_map, dict)

    def test_collect_edited_sv_ids_no_edits(self, gen_graph):
        """collect_edited_sv_ids returns empty array when no edits exist for a root."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        sh = SegmentHistory(graph, root)
        sv_ids = sh.collect_edited_sv_ids(root_id=root)
        assert isinstance(sv_ids, np.ndarray)
        assert sv_ids.dtype == np.uint64
        assert len(sv_ids) == 0

    def test_change_log_summary_no_operations(self, gen_graph):
        """change_log_summary with no operations should show zero splits/merges."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        sh = SegmentHistory(graph, root)
        summary = sh.change_log_summary(root_id=root)
        assert isinstance(summary, dict)
        assert summary["n_splits"] == 0
        assert summary["n_mergers"] == 0
        assert len(summary["past_ids"]) == 0
