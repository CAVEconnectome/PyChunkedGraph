from typing import Any, Iterable
import json
import pickle

import numpy as np
import zstandard as zstd


class _Serializer:
    def __init__(self, serializer, deserializer, basetype=Any, compression_level=None):
        self._serializer = serializer
        self._deserializer = deserializer
        self._basetype = basetype
        self._compression_level = compression_level

    def serialize(self, obj):
        content = self._serializer(obj)
        if self._compression_level:
            return zstd.ZstdCompressor(level=self._compression_level).compress(content)
        return content

    def deserialize(self, obj):
        if self._compression_level:
            obj = zstd.ZstdDecompressor().decompressobj().decompress(obj)
        return self._deserializer(obj)

    @property
    def basetype(self):
        return self._basetype


class NumPyArray(_Serializer):
    @staticmethod
    def _deserialize(val, dtype, shape=None, order=None):
        data = np.frombuffer(val, dtype=dtype)
        if shape is not None:
            return data.reshape(shape, order=order)
        if order is not None:
            return data.reshape(data.shape, order=order)
        return data

    def __init__(self, dtype, shape=None, order=None, compression_level=None):
        def _serialize_array(x):
            # NumPy 2.x: use view() instead of newbyteorder()
            arr = np.asarray(x, dtype=dtype)
            target_dtype = arr.dtype.newbyteorder(dtype.byteorder)
            return arr.view(target_dtype).tobytes()
        
        super().__init__(
            serializer=_serialize_array,
            deserializer=lambda x: NumPyArray._deserialize(
                x, dtype, shape=shape, order=order
            ),
            basetype=dtype.type,
            compression_level=compression_level,
        )


class NumPyValue(_Serializer):
    def __init__(self, dtype):
        def _serialize_scalar(x):
            # NumPy 2.x: use view() instead of newbyteorder()
            arr = np.asarray(x, dtype=dtype)
            target_dtype = arr.dtype.newbyteorder(dtype.byteorder)
            return arr.view(target_dtype).tobytes()
        
        super().__init__(
            serializer=_serialize_scalar,
            deserializer=lambda x: np.frombuffer(x, dtype=dtype)[0],
            basetype=dtype.type,
        )


class String(_Serializer):
    def __init__(self, encoding="utf-8"):
        super().__init__(
            serializer=lambda x: x.encode(encoding),
            deserializer=lambda x: x.decode(),
            basetype=str,
        )


class JSON(_Serializer):
    def __init__(self):
        super().__init__(
            serializer=lambda x: json.dumps(x).encode("utf-8"),
            deserializer=lambda x: json.loads(x.decode()),
            basetype=str,
        )


class Pickle(_Serializer):
    def __init__(self):
        super().__init__(
            serializer=lambda x: pickle.dumps(x),
            deserializer=lambda x: pickle.loads(x),
            basetype=str,
        )


class UInt64String(_Serializer):
    def __init__(self):
        super().__init__(
            serializer=serialize_uint64,
            deserializer=deserialize_uint64,
            basetype=np.uint64,
        )


def pad_node_id(node_id: np.uint64) -> str:
    """ Pad node id to 20 digits

    :param node_id: int
    :return: str
    """
    return "%.20d" % node_id


def serialize_uint64(node_id: np.uint64, counter=False, fake_edges=False) -> bytes:
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    if counter:
        return serialize_key("i%s" % pad_node_id(node_id))  # type: ignore
    if fake_edges:
        return serialize_key("f%s" % pad_node_id(node_id))  # type: ignore
    return serialize_key(pad_node_id(node_id))  # type: ignore


def serialize_uint64s_to_regex(node_ids: Iterable[np.uint64]) -> bytes:
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    node_id_str = "".join(["%s|" % pad_node_id(node_id) for node_id in node_ids])[:-1]
    return serialize_key(node_id_str)  # type: ignore


def deserialize_uint64(node_id: bytes, fake_edges=False) -> np.uint64:
    """ De-serializes a node id from a BigTable row

    :param node_id: bytes
    :return: np.uint64
    """
    if fake_edges:
        return np.uint64(node_id[1:].decode())  # type: ignore
    return np.uint64(node_id.decode())  # type: ignore


def serialize_key(key: str) -> bytes:
    """ Serializes a key to be ingested by a bigtable table row

    :param key: str
    :return: bytes
    """
    return key.encode("utf-8")


def deserialize_key(key: bytes) -> str:
    """ Deserializes a row key

    :param key: bytes
    :return: str
    """
    return key.decode()
