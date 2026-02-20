"""Tests for pychunkedgraph.graph.utils.serializers"""

import numpy as np

from pychunkedgraph.graph.utils.serializers import (
    _Serializer,
    NumPyArray,
    NumPyValue,
    String,
    JSON,
    Pickle,
    UInt64String,
    pad_node_id,
    serialize_uint64,
    deserialize_uint64,
    serialize_uint64s_to_regex,
    serialize_key,
    deserialize_key,
)
from pychunkedgraph.graph.utils import basetypes


class TestNumPyArray:
    def test_roundtrip(self):
        s = NumPyArray(dtype=basetypes.NODE_ID)
        arr = np.array([1, 2, 3], dtype=basetypes.NODE_ID)
        data = s.serialize(arr)
        result = s.deserialize(data)
        np.testing.assert_array_equal(result, arr)

    def test_with_shape(self):
        s = NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2))
        arr = np.array([[1, 2], [3, 4]], dtype=basetypes.NODE_ID)
        data = s.serialize(arr)
        result = s.deserialize(data)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, arr)

    def test_with_compression(self):
        s = NumPyArray(dtype=basetypes.NODE_ID, compression_level=3)
        arr = np.array([1, 2, 3, 4, 5], dtype=basetypes.NODE_ID)
        data = s.serialize(arr)
        result = s.deserialize(data)
        np.testing.assert_array_equal(result, arr)

    def test_basetype(self):
        s = NumPyArray(dtype=basetypes.NODE_ID)
        assert s.basetype == basetypes.NODE_ID.type


class TestNumPyValue:
    def test_roundtrip(self):
        s = NumPyValue(dtype=basetypes.NODE_ID)
        val = np.uint64(42)
        data = s.serialize(val)
        result = s.deserialize(data)
        assert result == val


class TestString:
    def test_roundtrip(self):
        s = String()
        data = s.serialize("hello")
        assert s.deserialize(data) == "hello"


class TestJSON:
    def test_roundtrip(self):
        s = JSON()
        obj = {"key": "value", "nested": [1, 2, 3]}
        data = s.serialize(obj)
        assert s.deserialize(data) == obj


class TestPickle:
    def test_roundtrip(self):
        s = Pickle()
        obj = {"complex": [1, 2], "nested": {"a": True}}
        data = s.serialize(obj)
        assert s.deserialize(data) == obj


class TestUInt64String:
    def test_roundtrip(self):
        s = UInt64String()
        val = np.uint64(12345)
        data = s.serialize(val)
        result = s.deserialize(data)
        assert result == val


class TestPadNodeId:
    def test_padding(self):
        result = pad_node_id(np.uint64(42))
        assert len(result) == 20
        assert result == "00000000000000000042"

    def test_large_id(self):
        result = pad_node_id(np.uint64(12345678901234567890))
        assert len(result) == 20


class TestSerializeUint64:
    def test_default(self):
        result = serialize_uint64(np.uint64(42))
        assert isinstance(result, bytes)
        assert b"00000000000000000042" in result

    def test_counter(self):
        result = serialize_uint64(np.uint64(42), counter=True)
        assert result.startswith(b"i")

    def test_fake_edges(self):
        result = serialize_uint64(np.uint64(42), fake_edges=True)
        assert result.startswith(b"f")


class TestDeserializeUint64:
    def test_default(self):
        serialized = serialize_uint64(np.uint64(42))
        result = deserialize_uint64(serialized)
        assert result == np.uint64(42)

    def test_fake_edges(self):
        serialized = serialize_uint64(np.uint64(42), fake_edges=True)
        result = deserialize_uint64(serialized, fake_edges=True)
        assert result == np.uint64(42)


class TestSerializeUint64sToRegex:
    def test_multiple_ids(self):
        ids = [np.uint64(1), np.uint64(2)]
        result = serialize_uint64s_to_regex(ids)
        assert isinstance(result, bytes)
        assert b"|" in result


class TestSerializeKey:
    def test_roundtrip(self):
        key = "test_key_123"
        serialized = serialize_key(key)
        assert isinstance(serialized, bytes)
        assert deserialize_key(serialized) == key
