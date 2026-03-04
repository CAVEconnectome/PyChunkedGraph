"""Integration tests for GraphEditOperation and its subclasses.

Tests operation type identification from log records, operation inversion,
undo/redo chain resolution, ID validation, and execute error handling
-- all using real graph operations through the BigTable emulator.
"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from ..helpers import create_chunk, to_label
from ...graph import attributes
from ...graph.operation import (
    GraphEditOperation,
    MergeOperation,
    MulticutOperation,
    SplitOperation,
    RedoOperation,
    UndoOperation,
)
from ...graph.exceptions import PreconditionError, PostconditionError
from ...ingest.create.parent_layer import add_parent_chunk


def _build_two_sv_disconnected(gen_graph):
    """2-layer graph, two disconnected SVs in the same chunk."""
    cg = gen_graph(n_layers=2, atomic_chunk_bounds=np.array([1, 1, 1]))
    ts = datetime.now(UTC) - timedelta(days=10)
    create_chunk(
        cg,
        vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
        edges=[],
        timestamp=ts,
    )
    return cg, ts


def _build_two_sv_connected(gen_graph):
    """2-layer graph, two connected SVs in the same chunk."""
    cg = gen_graph(n_layers=2, atomic_chunk_bounds=np.array([1, 1, 1]))
    ts = datetime.now(UTC) - timedelta(days=10)
    create_chunk(
        cg,
        vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
        edges=[
            (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
        ],
        timestamp=ts,
    )
    return cg, ts


def _build_cross_chunk(gen_graph):
    """4-layer graph with cross-chunk edges suitable for MulticutOperation."""
    cg = gen_graph(n_layers=4)
    ts = datetime.now(UTC) - timedelta(days=10)
    sv0 = to_label(cg, 1, 0, 0, 0, 0)
    sv1 = to_label(cg, 1, 0, 0, 0, 1)
    create_chunk(
        cg,
        vertices=[sv0, sv1],
        edges=[
            (sv0, sv1, 0.5),
            (sv0, to_label(cg, 1, 1, 0, 0, 0), inf),
        ],
        timestamp=ts,
    )
    create_chunk(
        cg,
        vertices=[to_label(cg, 1, 1, 0, 0, 0)],
        edges=[(to_label(cg, 1, 1, 0, 0, 0), sv0, inf)],
        timestamp=ts,
    )
    add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)
    add_parent_chunk(cg, 3, [1, 0, 0], n_threads=1)
    add_parent_chunk(cg, 4, [0, 0, 0], n_threads=1)
    return cg, ts, sv0, sv1


# ===========================================================================
# Existing tests (log record types, inversion, undo/redo chain)
# ===========================================================================
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
        """Build graph, split, merge -- return (cg, merge_op_id, split_op_id)."""
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
        """Build graph, perform split, then undo -- return (cg, split_op_id, undo_result)."""
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


# ===========================================================================
# NEW: Multicut log record type identification (lines 151-153)
# ===========================================================================
class TestGetLogRecordTypeMulticut:
    """Synthetic tests for MulticutOperation identification in get_log_record_type."""

    def test_bbox_only_is_multicut(self):
        """BoundingBoxOffset with no RemovedEdge -> MulticutOperation (line 152-153)."""
        log = {attributes.OperationLogs.BoundingBoxOffset: np.array([10, 10, 10])}
        assert GraphEditOperation.get_log_record_type(log) is MulticutOperation

    def test_removed_edge_with_bbox_multicut_as_split_true(self):
        """RemovedEdge + BoundingBoxOffset + multicut_as_split=True -> SplitOperation (line 150)."""
        log = {
            attributes.OperationLogs.RemovedEdge: np.array([[1, 2]], dtype=np.uint64),
            attributes.OperationLogs.BoundingBoxOffset: np.array([10, 10, 10]),
        }
        assert (
            GraphEditOperation.get_log_record_type(log, multicut_as_split=True)
            is SplitOperation
        )

    def test_removed_edge_with_bbox_multicut_as_split_false(self):
        """RemovedEdge + BoundingBoxOffset + multicut_as_split=False -> MulticutOperation (line 151)."""
        log = {
            attributes.OperationLogs.RemovedEdge: np.array([[1, 2]], dtype=np.uint64),
            attributes.OperationLogs.BoundingBoxOffset: np.array([10, 10, 10]),
        }
        assert (
            GraphEditOperation.get_log_record_type(log, multicut_as_split=False)
            is MulticutOperation
        )

    def test_undo_log_record(self):
        """UndoOperationID in log -> UndoOperation."""
        log = {attributes.OperationLogs.UndoOperationID: np.uint64(42)}
        assert GraphEditOperation.get_log_record_type(log) is UndoOperation

    def test_redo_log_record(self):
        """RedoOperationID in log -> RedoOperation."""
        log = {attributes.OperationLogs.RedoOperationID: np.uint64(42)}
        assert GraphEditOperation.get_log_record_type(log) is RedoOperation

    def test_empty_log_raises_type_error(self):
        """Empty log record should raise TypeError (line 154)."""
        with pytest.raises(TypeError, match="Could not determine"):
            GraphEditOperation.get_log_record_type({})


# ===========================================================================
# NEW: from_log_record MulticutOperation path (lines 235-251)
# ===========================================================================
class TestFromLogRecordMulticutPath:
    """Test from_log_record for the MulticutOperation path with multicut_as_split=False."""

    @pytest.mark.timeout(60)
    def test_multicut_from_log_record(self, gen_graph):
        """A multicut operation's log, read back with multicut_as_split=False,
        should be reconstructed as MulticutOperation (lines 235-249)."""
        cg, _, sv0, sv1 = _build_cross_chunk(gen_graph)
        source_coords = [[0, 0, 0]]
        sink_coords = [[512, 0, 0]]
        try:
            mc_result = cg.remove_edges(
                "test_user",
                source_ids=sv0,
                sink_ids=sv1,
                source_coords=source_coords,
                sink_coords=sink_coords,
                mincut=True,
            )
        except (PreconditionError, PostconditionError):
            pytest.skip("Multicut not feasible in this small test graph")

        log, _ = cg.client.read_log_entry(mc_result.operation_id)
        op = GraphEditOperation.from_log_record(cg, log, multicut_as_split=False)
        assert isinstance(op, MulticutOperation)

        # With default multicut_as_split=True -> SplitOperation
        op2 = GraphEditOperation.from_log_record(cg, log, multicut_as_split=True)
        assert isinstance(op2, SplitOperation)


# ===========================================================================
# NEW: from_operation_id (lines 278-281)
# ===========================================================================
class TestFromOperationId:
    """Test GraphEditOperation.from_operation_id round-trip."""

    @pytest.mark.timeout(30)
    def test_from_operation_id_merge(self, gen_graph):
        """from_operation_id should reconstruct a MergeOperation."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])
        op = GraphEditOperation.from_operation_id(cg, result.operation_id)
        assert isinstance(op, MergeOperation)
        # privileged_mode defaults to False
        assert op.privileged_mode is False

    @pytest.mark.timeout(30)
    def test_from_operation_id_privileged(self, gen_graph):
        """from_operation_id with privileged_mode=True should propagate the flag (line 280)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])
        op = GraphEditOperation.from_operation_id(
            cg, result.operation_id, privileged_mode=True
        )
        assert op.privileged_mode is True

    @pytest.mark.timeout(30)
    def test_from_operation_id_split(self, gen_graph):
        """from_operation_id should reconstruct a SplitOperation."""
        cg, _ = _build_two_sv_connected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )
        op = GraphEditOperation.from_operation_id(cg, result.operation_id)
        assert isinstance(op, SplitOperation)


# ===========================================================================
# NEW: MulticutOperation.invert() (line 974-981)
# ===========================================================================
class TestMulticutInversion:
    """Test MulticutOperation.invert() returns a MergeOperation."""

    @pytest.mark.timeout(30)
    def test_multicut_invert(self, gen_graph):
        """MulticutOperation.invert() -> MergeOperation with removed_edges as added_edges."""
        cg, _, sv0, sv1 = _build_cross_chunk(gen_graph)
        mc_op = MulticutOperation(
            cg,
            user_id="test_user",
            source_ids=[sv0],
            sink_ids=[sv1],
            source_coords=[[0, 0, 0]],
            sink_coords=[[512, 0, 0]],
            bbox_offset=[240, 240, 24],
            removed_edges=np.array([[sv0, sv1]], dtype=np.uint64),
        )
        inverted = mc_op.invert()
        assert isinstance(inverted, MergeOperation)
        np.testing.assert_array_equal(inverted.added_edges, mc_op.removed_edges)


# ===========================================================================
# NEW: ID validation -- self-loops and overlapping IDs (lines 593-596, 732-733, 871-875)
# ===========================================================================
class TestIDValidation:
    """Test PreconditionError on self-loops and overlapping IDs."""

    @pytest.mark.timeout(30)
    def test_merge_self_loop_raises(self, gen_graph):
        """added_edges where source == sink should raise PreconditionError (line 596)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        with pytest.raises(PreconditionError, match="self-loop"):
            MergeOperation(
                cg,
                user_id="test_user",
                added_edges=[[sv0, sv0]],
                source_coords=None,
                sink_coords=None,
            )

    @pytest.mark.timeout(30)
    def test_split_self_loop_raises(self, gen_graph):
        """removed_edges where source == sink should raise PreconditionError (line 733)."""
        cg, _ = _build_two_sv_connected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        with pytest.raises(PreconditionError, match="self-loop"):
            SplitOperation(
                cg,
                user_id="test_user",
                removed_edges=[[sv0, sv0]],
                source_coords=None,
                sink_coords=None,
            )

    @pytest.mark.timeout(30)
    def test_multicut_overlapping_ids_raises(self, gen_graph):
        """source_ids overlapping sink_ids should raise PreconditionError (line 872)."""
        cg, _, sv0, sv1 = _build_cross_chunk(gen_graph)
        with pytest.raises(PreconditionError, match="both sink and source"):
            MulticutOperation(
                cg,
                user_id="test_user",
                source_ids=[sv0, sv1],
                sink_ids=[sv1],
                source_coords=[[0, 0, 0], [1, 0, 0]],
                sink_coords=[[1, 0, 0]],
                bbox_offset=[240, 240, 24],
            )


# ===========================================================================
# NEW: Empty coords / affinities normalization (lines 82, 86, 593)
# ===========================================================================
class TestEmptyCoordsAffinities:
    """Empty source/sink coords and affinities should be normalized to None."""

    @pytest.mark.timeout(30)
    def test_empty_source_coords_becomes_none(self, gen_graph):
        """source_coords with size 0 should be stored as None (line 82)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        op = MergeOperation(
            cg,
            user_id="test_user",
            added_edges=[[sv0, sv1]],
            source_coords=np.array([], dtype=np.int64).reshape(0, 3),
            sink_coords=np.array([], dtype=np.int64).reshape(0, 3),
        )
        assert op.source_coords is None
        assert op.sink_coords is None

    @pytest.mark.timeout(30)
    def test_empty_affinities_becomes_none(self, gen_graph):
        """affinities with size 0 should be stored as None (line 593)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        op = MergeOperation(
            cg,
            user_id="test_user",
            added_edges=[[sv0, sv1]],
            source_coords=None,
            sink_coords=None,
            affinities=np.array([], dtype=np.float32),
        )
        assert op.affinities is None


# ===========================================================================
# NEW: Merge / Split preconditions via execute (lines 618, 765)
# ===========================================================================
class TestEditPreconditions:
    """Test precondition errors raised during _apply."""

    @pytest.mark.timeout(30)
    def test_merge_same_segment_raises(self, gen_graph):
        """Merging SVs already in the same root raises PreconditionError (line 618)."""
        cg, _ = _build_two_sv_connected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        with pytest.raises(PreconditionError, match="different objects"):
            cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

    @pytest.mark.timeout(30)
    def test_split_different_roots_raises(self, gen_graph):
        """Splitting SVs from different roots raises PreconditionError (line 765)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        with pytest.raises(PreconditionError, match="same object"):
            cg.remove_edges("test_user", source_ids=sv0, sink_ids=sv1, mincut=False)


# ===========================================================================
# NEW: Undo / Redo via actual operations (lines 1160-1175, 1245-1259, etc.)
# ===========================================================================
class TestUndoRedoExecute:
    """End-to-end undo/redo tests that verify graph state after execute."""

    def _build_connected_cross_chunk(self, gen_graph):
        """Build a 3-layer graph with between-chunk edge -- suitable for split+undo."""
        cg = gen_graph(n_layers=3)
        ts = datetime.now(UTC) - timedelta(days=10)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 1, 0, 0, 0)
        create_chunk(
            cg,
            vertices=[sv0],
            edges=[(sv0, sv1, 0.5)],
            timestamp=ts,
        )
        create_chunk(
            cg,
            vertices=[sv1],
            edges=[(sv1, sv0, 0.5)],
            timestamp=ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=ts, n_threads=1)
        return cg, sv0, sv1

    @pytest.mark.timeout(60)
    def test_undo_split_restores_root(self, gen_graph):
        """After split + undo, the SVs should share a root again (lines 1160-1175)."""
        cg, sv0, sv1 = self._build_connected_cross_chunk(gen_graph)
        assert cg.get_root(sv0) == cg.get_root(sv1)

        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )
        assert cg.get_root(sv0) != cg.get_root(sv1)

        undo_result = cg.undo_operation("test_user", split_result.operation_id)
        assert cg.get_root(sv0) == cg.get_root(sv1)

    @pytest.mark.timeout(60)
    def test_redo_split_after_undo(self, gen_graph):
        """After split + undo, redo the split directly (lines 1036-1043, 1094-1106)."""
        cg, sv0, sv1 = self._build_connected_cross_chunk(gen_graph)

        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )
        assert cg.get_root(sv0) != cg.get_root(sv1)

        undo_result = cg.undo_operation("test_user", split_result.operation_id)
        assert cg.get_root(sv0) == cg.get_root(sv1)

        # Redo the original split directly
        redo_result = cg.redo_operation("test_user", split_result.operation_id)
        # The redo should succeed and re-apply the split
        assert redo_result.operation_id is not None

    @pytest.mark.timeout(60)
    def test_undo_of_undo_resolves_to_redo(self, gen_graph):
        """Undoing an undo should resolve to a RedoOperation (lines 102-108)."""
        cg, sv0, sv1 = self._build_connected_cross_chunk(gen_graph)
        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )
        undo_result = cg.undo_operation("test_user", split_result.operation_id)

        op = GraphEditOperation.undo_operation(
            cg, user_id="test_user", operation_id=undo_result.operation_id
        )
        assert isinstance(op, RedoOperation)


# ===========================================================================
# NEW: UndoOperation / RedoOperation .invert() (lines 1087, 1228)
# ===========================================================================
class TestUndoRedoInvert:
    """Test invert() on UndoOperation and RedoOperation."""

    @pytest.mark.timeout(60)
    def test_undo_invert_is_redo(self, gen_graph):
        """UndoOperation.invert() -> RedoOperation (line 1228)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        merge_result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        undo_op = GraphEditOperation.undo_operation(
            cg, user_id="test_user", operation_id=merge_result.operation_id
        )
        assert isinstance(undo_op, UndoOperation)
        inverted = undo_op.invert()
        assert isinstance(inverted, RedoOperation)
        assert inverted.superseded_operation_id == undo_op.superseded_operation_id

    @pytest.mark.timeout(60)
    def test_redo_invert_is_undo(self, gen_graph):
        """RedoOperation.invert() -> UndoOperation (line 1087)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        merge_result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        redo_op = GraphEditOperation.redo_operation(
            cg, user_id="test_user", operation_id=merge_result.operation_id
        )
        assert isinstance(redo_op, RedoOperation)
        inverted = redo_op.invert()
        assert isinstance(inverted, UndoOperation)
        assert inverted.superseded_operation_id == redo_op.superseded_operation_id


# ===========================================================================
# NEW: UndoOperation / RedoOperation edge attributes (lines 1040-1043, 1172-1175)
# ===========================================================================
class TestUndoRedoEdgeAttributes:
    """Verify that undo/redo operations carry the correct edge attributes."""

    @pytest.mark.timeout(60)
    def test_undo_merge_has_removed_edges(self, gen_graph):
        """Undoing a merge -> inverse is SplitOp -> undo should have removed_edges (line 1175)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        merge_result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        undo_op = GraphEditOperation.undo_operation(
            cg, user_id="test_user", operation_id=merge_result.operation_id
        )
        assert hasattr(undo_op, "removed_edges")
        assert undo_op.removed_edges.shape[1] == 2

    @pytest.mark.timeout(60)
    def test_undo_split_has_added_edges(self, gen_graph):
        """Undoing a split -> inverse is MergeOp -> undo should have added_edges (line 1173)."""
        cg, _ = _build_two_sv_connected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )

        undo_op = GraphEditOperation.undo_operation(
            cg, user_id="test_user", operation_id=split_result.operation_id
        )
        assert hasattr(undo_op, "added_edges")
        assert undo_op.added_edges.shape[1] == 2

    @pytest.mark.timeout(60)
    def test_redo_merge_has_added_edges(self, gen_graph):
        """RedoOperation for a merge should have added_edges (line 1040-1041)."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        merge_result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        redo_op = GraphEditOperation.redo_operation(
            cg, user_id="test_user", operation_id=merge_result.operation_id
        )
        assert isinstance(redo_op, RedoOperation)
        assert hasattr(redo_op, "added_edges")
        assert redo_op.added_edges.shape[1] == 2

    @pytest.mark.timeout(60)
    def test_redo_split_has_removed_edges(self, gen_graph):
        """RedoOperation for a split should have removed_edges (line 1042-1043)."""
        cg, _ = _build_two_sv_connected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)
        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )

        redo_op = GraphEditOperation.redo_operation(
            cg, user_id="test_user", operation_id=split_result.operation_id
        )
        assert isinstance(redo_op, RedoOperation)
        assert hasattr(redo_op, "removed_edges")
        assert redo_op.removed_edges.shape[1] == 2


# ===========================================================================
# NEW: Undo / Redo log record type from actual operations
# ===========================================================================
class TestUndoRedoLogRecordTypes:
    """Verify that actual undo/redo operations produce correct log record types."""

    def _build_and_split(self, gen_graph):
        """Build a cross-chunk graph and split it -- suitable for undo/redo."""
        cg = gen_graph(n_layers=3)
        ts = datetime.now(UTC) - timedelta(days=10)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 1, 0, 0, 0)
        create_chunk(
            cg,
            vertices=[sv0],
            edges=[(sv0, sv1, 0.5)],
            timestamp=ts,
        )
        create_chunk(
            cg,
            vertices=[sv1],
            edges=[(sv1, sv0, 0.5)],
            timestamp=ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=ts, n_threads=1)
        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )
        return cg, sv0, sv1, split_result

    @pytest.mark.timeout(60)
    def test_undo_log_type(self, gen_graph):
        """Undo operation log should be identified as UndoOperation."""
        cg, sv0, sv1, split_result = self._build_and_split(gen_graph)
        undo_result = cg.undo_operation("test_user", split_result.operation_id)

        log, _ = cg.client.read_log_entry(undo_result.operation_id)
        assert GraphEditOperation.get_log_record_type(log) is UndoOperation

    @pytest.mark.timeout(60)
    def test_redo_log_type(self, gen_graph):
        """Redo operation log should be identified as RedoOperation."""
        cg, sv0, sv1, split_result = self._build_and_split(gen_graph)
        undo_result = cg.undo_operation("test_user", split_result.operation_id)

        # Redo the split that was just undone
        redo_result = cg.redo_operation("test_user", split_result.operation_id)
        assert redo_result.operation_id is not None

        log, _ = cg.client.read_log_entry(redo_result.operation_id)
        assert GraphEditOperation.get_log_record_type(log) is RedoOperation


# ===========================================================================
# NEW: execute() error handling -- PreconditionError clears cache (lines 436, 460-462)
# ===========================================================================
class TestExecuteErrorHandling:
    """Test that execute() clears cache on PreconditionError/PostconditionError."""

    @pytest.mark.timeout(30)
    def test_execute_precondition_error_clears_cache(self, gen_graph):
        """Trigger PreconditionError during merge (same-segment merge) and verify cache is cleared."""
        cg, _ = _build_two_sv_connected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)

        # Merging already-connected SVs raises PreconditionError
        with pytest.raises(PreconditionError, match="different objects"):
            cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        # After the error, the graph cache should have been cleared (set to None)
        assert cg.cache is None

    @pytest.mark.timeout(30)
    def test_execute_postcondition_error_clears_cache(self, gen_graph):
        """PostconditionError during execute should also clear cache (lines 463-465)."""
        from unittest.mock import patch

        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)

        # Mock _apply to raise PostconditionError
        with patch.object(
            MergeOperation,
            "_apply",
            side_effect=PostconditionError("test postcondition error"),
        ):
            with pytest.raises(PostconditionError, match="test postcondition error"):
                cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        # Cache should have been cleared
        assert cg.cache is None

    @pytest.mark.timeout(30)
    def test_execute_assertion_error_clears_cache(self, gen_graph):
        """AssertionError/RuntimeError during execute should also clear cache (lines 466-468)."""
        from unittest.mock import patch

        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)

        # Mock _apply to raise RuntimeError
        with patch.object(
            MergeOperation, "_apply", side_effect=RuntimeError("test runtime error")
        ):
            with pytest.raises(RuntimeError, match="test runtime error"):
                cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])

        assert cg.cache is None


# ===========================================================================
# NEW: UndoOperation.execute() edge validation (lines 1245-1267)
# ===========================================================================
class TestUndoEdgeValidation:
    """Test UndoOperation.execute() edge validation logic."""

    def _build_connected_cross_chunk(self, gen_graph):
        """Build a 3-layer graph with between-chunk edge suitable for split+undo."""
        cg = gen_graph(n_layers=3)
        ts = datetime.now(UTC) - timedelta(days=10)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 1, 0, 0, 0)
        create_chunk(
            cg,
            vertices=[sv0],
            edges=[(sv0, sv1, 0.5)],
            timestamp=ts,
        )
        create_chunk(
            cg,
            vertices=[sv1],
            edges=[(sv1, sv0, 0.5)],
            timestamp=ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=ts, n_threads=1)
        return cg, sv0, sv1

    @pytest.mark.timeout(60)
    def test_undo_split_restores_edges(self, gen_graph):
        """After undo of a split, edges should be active again."""
        cg, sv0, sv1 = self._build_connected_cross_chunk(gen_graph)

        # Verify initially connected
        assert cg.get_root(sv0) == cg.get_root(sv1)

        # Split
        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )
        assert cg.get_root(sv0) != cg.get_root(sv1)

        # Undo the split
        undo_result = cg.undo_operation("test_user", split_result.operation_id)
        assert undo_result.operation_id is not None

        # Edges should be active again -- the SVs share a root
        assert cg.get_root(sv0) == cg.get_root(sv1)

    @pytest.mark.timeout(60)
    def test_undo_merge_via_undo_operation_class(self, gen_graph):
        """UndoOperation on a merge constructs with inverse being SplitOperation."""
        cg, _ = _build_two_sv_disconnected(gen_graph)
        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)

        # Merge
        merge_result = cg.add_edges("test_user", [sv0, sv1], affinities=[0.3])
        assert cg.get_root(sv0) == cg.get_root(sv1)

        # Build the UndoOperation manually to inspect its structure
        undo_op = GraphEditOperation.undo_operation(
            cg, user_id="test_user", operation_id=merge_result.operation_id
        )
        assert isinstance(undo_op, UndoOperation)
        # The inverse of a merge is a split, so removed_edges should be set
        assert hasattr(undo_op, "removed_edges")
        assert undo_op.removed_edges.shape[1] == 2

    @pytest.mark.timeout(60)
    def test_undo_noop_when_split_already_undone(self, gen_graph):
        """UndoOperation.execute() with edges already active returns early (lines 1253-1258)."""
        cg, sv0, sv1 = self._build_connected_cross_chunk(gen_graph)

        # Split
        split_result = cg.remove_edges(
            "test_user", source_ids=sv0, sink_ids=sv1, mincut=False
        )

        # First undo
        undo_result1 = cg.undo_operation("test_user", split_result.operation_id)
        assert cg.get_root(sv0) == cg.get_root(sv1)

        # Second undo of the same split -- the inverse is a MergeOp
        # and since those edges are already active, it should return early
        # (lines 1253-1258: if np.all(a): early return with empty Result)
        undo_result2 = cg.undo_operation("test_user", split_result.operation_id)
        # The early return path returns a Result with operation_id=None and empty arrays
        assert undo_result2.operation_id is None
        assert len(undo_result2.new_root_ids) == 0
