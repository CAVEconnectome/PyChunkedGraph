from collections import namedtuple

import numpy as np
import pytest

from pychunkedgraph.backend.graphoperation import (
    GraphEditOperation,
    MergeOperation,
    MulticutOperation,
    RedoOperation,
    SplitOperation,
    UndoOperation,
)
from pychunkedgraph.backend.utils import column_keys


class FakeLogRecords:
    Record = namedtuple("graph_op", ("id", "record"))

    _records = [
        {  # 0: Merge with coordinates
            column_keys.OperationLogs.AddedEdge: np.array([[1, 2]], dtype=np.uint64),
            column_keys.OperationLogs.SinkCoordinate: np.array([[1, 2, 3]]),
            column_keys.OperationLogs.SourceCoordinate: np.array([[4, 5, 6]]),
            column_keys.OperationLogs.UserID: "42",
        },
        {  # 1: Multicut with coordinates
            column_keys.OperationLogs.BoundingBoxOffset: np.array([240, 240, 24]),
            column_keys.OperationLogs.RemovedEdge: np.array(
                [[1, 3], [4, 1], [1, 5]], dtype=np.uint64
            ),
            column_keys.OperationLogs.SinkCoordinate: np.array([[1, 2, 3]]),
            column_keys.OperationLogs.SinkID: np.array([1], dtype=np.uint64),
            column_keys.OperationLogs.SourceCoordinate: np.array([[4, 5, 6]]),
            column_keys.OperationLogs.SourceID: np.array([2], dtype=np.uint64),
            column_keys.OperationLogs.UserID: "42",
        },
        {  # 2: Split with coordinates
            column_keys.OperationLogs.RemovedEdge: np.array(
                [[1, 3], [4, 1], [1, 5]], dtype=np.uint64
            ),
            column_keys.OperationLogs.SinkCoordinate: np.array([[1, 2, 3]]),
            column_keys.OperationLogs.SinkID: np.array([1], dtype=np.uint64),
            column_keys.OperationLogs.SourceCoordinate: np.array([[4, 5, 6]]),
            column_keys.OperationLogs.SourceID: np.array([2], dtype=np.uint64),
            column_keys.OperationLogs.UserID: "42",
        },
        {  # 3: Undo of records[0]
            column_keys.OperationLogs.UndoOperationID: np.uint64(0),
            column_keys.OperationLogs.UserID: "42",
        },
        {  # 4: Redo of records[0]
            column_keys.OperationLogs.RedoOperationID: np.uint64(0),
            column_keys.OperationLogs.UserID: "42",
        },
        {  # 5: Unknown record
            column_keys.OperationLogs.UserID: "42",
        },
    ]

    MERGE = Record(id=np.uint64(0), record=_records[0])
    MULTICUT = Record(id=np.uint64(1), record=_records[1])
    SPLIT = Record(id=np.uint64(2), record=_records[2])
    UNDO = Record(id=np.uint64(3), record=_records[3])
    REDO = Record(id=np.uint64(4), record=_records[4])
    UNKNOWN = Record(id=np.uint64(5), record=_records[5])

    @classmethod
    def get(cls, idx: int):
        try:
            return cls._records[idx], "2020-01-01 00:00:00.000000+00:00"
        except IndexError as err:
            raise KeyError(err)  # Bigtable would throw KeyError instead


@pytest.fixture(scope="function")
def cg(mocker):
    graph = mocker.MagicMock()
    graph.get_chunk_layer = mocker.MagicMock(return_value=1)
    graph.read_log_row = mocker.MagicMock(side_effect=FakeLogRecords.get)
    return graph


def test_read_from_log_merge(mocker, cg):
    """MergeOperation should be correctly identified by an existing AddedEdge column.
        Coordinates are optional."""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.MERGE.record)
    assert isinstance(graph_operation, MergeOperation)


def test_read_from_log_multicut(mocker, cg):
    """MulticutOperation should be correctly identified by a Sink/Source ID and
        BoundingBoxOffset column. Unless requested as SplitOperation..."""
    graph_operation = GraphEditOperation.from_log_record(
        cg, FakeLogRecords.MULTICUT.record, multicut_as_split=False
    )
    assert isinstance(graph_operation, MulticutOperation)

    graph_operation = GraphEditOperation.from_log_record(
        cg, FakeLogRecords.MULTICUT.record, multicut_as_split=True
    )
    assert isinstance(graph_operation, SplitOperation)


def test_read_from_log_split(mocker, cg):
    """SplitOperation should be correctly identified by the lack of a
        BoundingBoxOffset column."""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.SPLIT.record)
    assert isinstance(graph_operation, SplitOperation)


def test_read_from_log_undo(mocker, cg):
    """UndoOperation should be correctly identified by the UndoOperationID."""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.UNDO.record)
    assert isinstance(graph_operation, UndoOperation)


def test_read_from_log_redo(mocker, cg):
    """RedoOperation should be correctly identified by the RedoOperationID."""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.REDO.record)
    assert isinstance(graph_operation, RedoOperation)


def test_read_from_log_undo_undo(mocker, cg):
    """Undo[Undo[Merge]] -> Redo[Merge]"""
    fake_log_record = {
        column_keys.OperationLogs.UndoOperationID: np.uint64(FakeLogRecords.UNDO.id),
        column_keys.OperationLogs.UserID: "42",
    }

    graph_operation = GraphEditOperation.from_log_record(cg, fake_log_record)
    assert isinstance(graph_operation, RedoOperation)
    assert isinstance(graph_operation.superseded_operation, MergeOperation)


def test_read_from_log_undo_redo(mocker, cg):
    """Undo[Redo[Merge]] -> Undo[Merge]"""
    fake_log_record = {
        column_keys.OperationLogs.UndoOperationID: np.uint64(FakeLogRecords.REDO.id),
        column_keys.OperationLogs.UserID: "42",
    }

    graph_operation = GraphEditOperation.from_log_record(cg, fake_log_record)
    assert isinstance(graph_operation, UndoOperation)
    assert isinstance(graph_operation.inverse_superseded_operation, SplitOperation)


def test_read_from_log_redo_undo(mocker, cg):
    """Redo[Undo[Merge]] -> Undo[Merge]"""
    fake_log_record = {
        column_keys.OperationLogs.RedoOperationID: np.uint64(FakeLogRecords.UNDO.id),
        column_keys.OperationLogs.UserID: "42",
    }

    graph_operation = GraphEditOperation.from_log_record(cg, fake_log_record)
    assert isinstance(graph_operation, UndoOperation)
    assert isinstance(graph_operation.inverse_superseded_operation, SplitOperation)


def test_read_from_log_redo_redo(mocker, cg):
    """Redo[Redo[Merge]] -> Redo[Merge]"""
    fake_log_record = {
        column_keys.OperationLogs.RedoOperationID: np.uint64(FakeLogRecords.REDO.id),
        column_keys.OperationLogs.UserID: "42",
    }

    graph_operation = GraphEditOperation.from_log_record(cg, fake_log_record)
    assert isinstance(graph_operation, RedoOperation)
    assert isinstance(graph_operation.superseded_operation, MergeOperation)


def test_invert_merge(mocker, cg):
    """Inverse of Merge is a Split"""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.MERGE.record)
    inverted_graph_operation = graph_operation.invert()
    assert isinstance(inverted_graph_operation, SplitOperation)
    assert np.all(np.equal(graph_operation.added_edges, inverted_graph_operation.removed_edges))


@pytest.mark.skip(reason="Can't test right now - would require recalculting the Multicut")
def test_invert_multicut(mocker, cg):
    """Inverse of a Multicut is a Merge"""


def test_invert_split(mocker, cg):
    """Inverse of Split is a Merge"""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.SPLIT.record)
    inverted_graph_operation = graph_operation.invert()
    assert isinstance(inverted_graph_operation, MergeOperation)
    assert np.all(np.equal(graph_operation.removed_edges, inverted_graph_operation.added_edges))


def test_invert_undo(mocker, cg):
    """Inverse of Undo[x] is Redo[x]"""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.UNDO.record)
    inverted_graph_operation = graph_operation.invert()
    assert isinstance(inverted_graph_operation, RedoOperation)
    assert (
        graph_operation.superseded_operation_id == inverted_graph_operation.superseded_operation_id
    )


def test_invert_redo(mocker, cg):
    """Inverse of Redo[x] is Undo[x]"""
    graph_operation = GraphEditOperation.from_log_record(cg, FakeLogRecords.REDO.record)
    inverted_graph_operation = graph_operation.invert()
    assert (
        graph_operation.superseded_operation_id == inverted_graph_operation.superseded_operation_id
    )


def test_undo_redo_chain_fails(mocker, cg):
    """Prevent creation of Undo/Redo chains"""
    with pytest.raises(ValueError):
        UndoOperation(
            cg,
            user_id="DAU",
            superseded_operation_id=FakeLogRecords.UNDO.id,
            multicut_as_split=False,
        )
    with pytest.raises(ValueError):
        UndoOperation(
            cg,
            user_id="DAU",
            superseded_operation_id=FakeLogRecords.REDO.id,
            multicut_as_split=False,
        )
    with pytest.raises(ValueError):
        RedoOperation(
            cg,
            user_id="DAU",
            superseded_operation_id=FakeLogRecords.UNDO.id,
            multicut_as_split=False,
        )
    with pytest.raises(ValueError):
        UndoOperation(
            cg,
            user_id="DAU",
            superseded_operation_id=FakeLogRecords.REDO.id,
            multicut_as_split=False,
        )

def test_unknown_log_record_fails(cg, mocker):
    """TypeError when encountering unknown log row"""
    with pytest.raises(TypeError):
        GraphEditOperation.from_log_record(cg, FakeLogRecords.UNKNOWN.record)
