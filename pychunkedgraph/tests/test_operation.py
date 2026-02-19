"""Integration tests for GraphEditOperation and its subclasses.

Tests operation type identification from log records, operation inversion,
and undo/redo chain resolution — all using real graph operations through
the BigTable emulator.
"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from .helpers import create_chunk, to_label
from ..graph import attributes
from ..graph.operation import (
    GraphEditOperation,
    MergeOperation,
    SplitOperation,
    RedoOperation,
    UndoOperation,
)
from ..ingest.create.parent_layer import add_parent_chunk


class TestOperationFromLogRecord:
    """Test that GraphEditOperation.from_log_record correctly identifies operation types."""

    @pytest.fixture()
    def merged_graph(self, gen_graph):
        """Build a simple 2-chunk graph and perform a merge, returning (cg, operation_id)."""
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Split first to get two separate roots
        split_result = cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )

        # Now merge them back
        merge_result = cg.add_edges(
            "test_user",
            atomic_edges=[[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0)]],
            source_coords=[0, 0, 0],
            sink_coords=[0, 0, 0],
        )
        return cg, merge_result.operation_id, split_result.operation_id

    @pytest.mark.timeout(30)
    def test_merge_log_record_type(self, merged_graph):
        """MergeOperation should be correctly identified from a real merge log record."""
        cg, merge_op_id, _ = merged_graph
        log_record, _ = cg.client.read_log_entry(merge_op_id)
        op_type = GraphEditOperation.get_log_record_type(log_record)
        assert op_type is MergeOperation

    @pytest.mark.timeout(30)
    def test_split_log_record_type(self, merged_graph):
        """SplitOperation should be correctly identified from a real split log record."""
        cg, _, split_op_id = merged_graph
        log_record, _ = cg.client.read_log_entry(split_op_id)
        op_type = GraphEditOperation.get_log_record_type(log_record)
        assert op_type is SplitOperation

    @pytest.mark.timeout(30)
    def test_merge_from_log_record(self, merged_graph):
        """from_log_record should return a MergeOperation for a real merge log."""
        cg, merge_op_id, _ = merged_graph
        log_record, _ = cg.client.read_log_entry(merge_op_id)
        graph_op = GraphEditOperation.from_log_record(cg, log_record)
        assert isinstance(graph_op, MergeOperation)

    @pytest.mark.timeout(30)
    def test_split_from_log_record(self, merged_graph):
        """from_log_record should return a SplitOperation for a real split log."""
        cg, _, split_op_id = merged_graph
        log_record, _ = cg.client.read_log_entry(split_op_id)
        graph_op = GraphEditOperation.from_log_record(cg, log_record)
        assert isinstance(graph_op, SplitOperation)

    @pytest.mark.timeout(30)
    def test_unknown_log_record_fails(self, gen_graph):
        """TypeError when encountering a log record with no recognizable operation columns."""
        cg = gen_graph(n_layers=3)
        fake_record = {attributes.OperationLogs.UserID: "test_user"}
        with pytest.raises(TypeError):
            GraphEditOperation.from_log_record(cg, fake_record)


class TestOperationInversion:
    """Test that operation inversion produces the correct inverse type and edges."""

    @pytest.fixture()
    def split_and_merge_ops(self, gen_graph):
        """Build graph, split, merge — return (cg, merge_op_id, split_op_id)."""
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        split_result = cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )
        merge_result = cg.add_edges(
            "test_user",
            atomic_edges=[[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0)]],
            source_coords=[0, 0, 0],
            sink_coords=[0, 0, 0],
        )
        return cg, merge_result.operation_id, split_result.operation_id

    @pytest.mark.timeout(30)
    def test_invert_merge_produces_split(self, split_and_merge_ops):
        """Inverse of a MergeOperation should be a SplitOperation with matching edges."""
        cg, merge_op_id, _ = split_and_merge_ops
        log_record, _ = cg.client.read_log_entry(merge_op_id)
        merge_op = GraphEditOperation.from_log_record(cg, log_record)
        inverted = merge_op.invert()
        assert isinstance(inverted, SplitOperation)
        assert np.all(np.equal(merge_op.added_edges, inverted.removed_edges))

    @pytest.mark.timeout(30)
    def test_invert_split_produces_merge(self, split_and_merge_ops):
        """Inverse of a SplitOperation should be a MergeOperation with matching edges."""
        cg, _, split_op_id = split_and_merge_ops
        log_record, _ = cg.client.read_log_entry(split_op_id)
        split_op = GraphEditOperation.from_log_record(cg, log_record)
        inverted = split_op.invert()
        assert isinstance(inverted, MergeOperation)
        assert np.all(np.equal(split_op.removed_edges, inverted.added_edges))


class TestUndoRedoChainResolution:
    """Test undo/redo chain resolution through real graph operations."""

    @pytest.fixture()
    def graph_with_undo(self, gen_graph):
        """Build graph, perform split, then undo — return (cg, split_op_id, undo_result)."""
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Split
        split_result = cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )
        # Undo the split (= merge)
        undo_result = cg.undo_operation("test_user", split_result.operation_id)
        return cg, split_result.operation_id, undo_result

    @pytest.mark.timeout(30)
    def test_undo_log_record_type(self, graph_with_undo):
        """Undo operation log record should be identified as UndoOperation."""
        cg, _, undo_result = graph_with_undo
        log_record, _ = cg.client.read_log_entry(undo_result.operation_id)
        op_type = GraphEditOperation.get_log_record_type(log_record)
        assert op_type is UndoOperation

    @pytest.mark.timeout(30)
    def test_undo_from_log_resolves_correctly(self, graph_with_undo):
        """from_log_record on an undo record should resolve the chain to an UndoOperation."""
        cg, split_op_id, undo_result = graph_with_undo
        log_record, _ = cg.client.read_log_entry(undo_result.operation_id)
        resolved_op = GraphEditOperation.from_log_record(cg, log_record)
        # Undo of a split -> UndoOperation whose inverse is a MergeOperation
        assert isinstance(resolved_op, UndoOperation)

    @pytest.mark.timeout(30)
    def test_redo_after_undo(self, graph_with_undo):
        """Redo of the original split (after undo) should produce a RedoOperation log."""
        cg, split_op_id, undo_result = graph_with_undo

        # Redo the original split (which was undone)
        redo_result = cg.redo_operation("test_user", split_op_id)
        assert redo_result.operation_id is not None
        redo_log, _ = cg.client.read_log_entry(redo_result.operation_id)
        resolved_op = GraphEditOperation.from_log_record(cg, redo_log)
        assert isinstance(resolved_op, RedoOperation)

    @pytest.mark.timeout(30)
    def test_undo_redo_chain_prevention(self, graph_with_undo):
        """Direct UndoOperation/RedoOperation on undo/redo targets should raise ValueError."""
        cg, _, undo_result = graph_with_undo

        # Direct UndoOperation on an undo record should fail
        with pytest.raises(ValueError):
            UndoOperation(
                cg,
                user_id="test_user",
                superseded_operation_id=undo_result.operation_id,
                multicut_as_split=True,
            )

        # Direct RedoOperation on an undo record should also fail
        with pytest.raises(ValueError):
            RedoOperation(
                cg,
                user_id="test_user",
                superseded_operation_id=undo_result.operation_id,
                multicut_as_split=True,
            )
