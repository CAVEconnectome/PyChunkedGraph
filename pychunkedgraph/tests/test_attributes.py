"""Tests for pychunkedgraph.graph.attributes"""

import numpy as np
import pytest

from pychunkedgraph.graph.attributes import (
    _Attribute,
    _AttributeArray,
    Concurrency,
    Connectivity,
    Hierarchy,
    GraphMeta,
    GraphVersion,
    OperationLogs,
    from_key,
)
from pychunkedgraph.graph.utils import basetypes


class TestAttribute:
    def test_serialize_deserialize_numpy(self):
        attr = Hierarchy.Child
        arr = np.array([1, 2, 3], dtype=basetypes.NODE_ID)
        data = attr.serialize(arr)
        result = attr.deserialize(data)
        np.testing.assert_array_equal(result, arr)

    def test_serialize_deserialize_string(self):
        attr = OperationLogs.UserID
        data = attr.serialize("test_user")
        assert attr.deserialize(data) == "test_user"

    def test_basetype(self):
        assert Hierarchy.Child.basetype == basetypes.NODE_ID.type
        assert OperationLogs.UserID.basetype == str

    def test_index(self):
        attr = Connectivity.CrossChunkEdge[5]
        assert attr.index == 5

    def test_family_id(self):
        assert Hierarchy.Child.family_id == "0"
        assert Concurrency.Counter.family_id == "1"
        assert OperationLogs.UserID.family_id == "2"


class TestAttributeArray:
    def test_getitem(self):
        attr = Connectivity.AtomicCrossChunkEdge[3]
        assert isinstance(attr, _Attribute)
        assert attr.key == b"atomic_cross_edges_3"

    def test_pattern(self):
        assert Connectivity.CrossChunkEdge.pattern == b"cross_edges_%d"

    def test_serialize_deserialize(self):
        arr = np.array([[1, 2], [3, 4]], dtype=basetypes.NODE_ID)
        data = Connectivity.CrossChunkEdge.serialize(arr)
        result = Connectivity.CrossChunkEdge.deserialize(data)
        np.testing.assert_array_equal(result, arr)


class TestFromKey:
    def test_valid_key(self):
        result = from_key("0", b"children")
        assert result is Hierarchy.Child

    def test_invalid_key_raises(self):
        with pytest.raises(KeyError, match="Unknown key"):
            from_key("99", b"nonexistent")


class TestOperationLogs:
    def test_all_returns_list(self):
        result = OperationLogs.all()
        assert isinstance(result, list)
        assert len(result) == 16
        assert OperationLogs.OperationID in result
        assert OperationLogs.UserID in result
        assert OperationLogs.RootID in result
        assert OperationLogs.AddedEdge in result

    def test_status_codes(self):
        assert OperationLogs.StatusCodes.SUCCESS.value == 0
        assert OperationLogs.StatusCodes.CREATED.value == 1
        assert OperationLogs.StatusCodes.EXCEPTION.value == 2
        assert OperationLogs.StatusCodes.WRITE_STARTED.value == 3
        assert OperationLogs.StatusCodes.WRITE_FAILED.value == 4
