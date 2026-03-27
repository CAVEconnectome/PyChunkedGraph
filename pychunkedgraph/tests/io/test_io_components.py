"""Tests for pychunkedgraph.io.components using file:// protocol"""

import numpy as np
import pytest

from pychunkedgraph.io.components import (
    serialize,
    deserialize,
    put_chunk_components,
    get_chunk_components,
)
from pychunkedgraph.graph import basetypes


class TestSerializeDeserialize:
    def test_roundtrip(self):
        components = [
            {np.uint64(1), np.uint64(2), np.uint64(3)},
            {np.uint64(4), np.uint64(5)},
        ]
        proto = serialize(components)
        result = deserialize(proto)
        # Each supervoxel should map to its component index
        assert result[np.uint64(1)] == result[np.uint64(2)] == result[np.uint64(3)]
        assert result[np.uint64(4)] == result[np.uint64(5)]
        assert result[np.uint64(1)] != result[np.uint64(4)]

    def test_empty_components(self):
        # serialize([]) raises ValueError because np.concatenate
        # is called on an empty list; this matches production behavior
        # where empty components are never serialized
        with pytest.raises(ValueError):
            serialize([])


class TestPutGetChunkComponents:
    def test_roundtrip_via_filesystem(self, tmp_path):
        components_dir = f"file://{tmp_path}"
        chunk_coord = np.array([1, 2, 3])

        components = [
            {np.uint64(10), np.uint64(20)},
            {np.uint64(30)},
        ]
        put_chunk_components(components_dir, components, chunk_coord)
        result = get_chunk_components(components_dir, chunk_coord)

        assert np.uint64(10) in result
        assert np.uint64(20) in result
        assert np.uint64(30) in result
        assert result[np.uint64(10)] == result[np.uint64(20)]
        assert result[np.uint64(10)] != result[np.uint64(30)]

    def test_missing_file_returns_empty(self, tmp_path):
        components_dir = f"file://{tmp_path}"
        result = get_chunk_components(components_dir, np.array([99, 99, 99]))
        assert result == {}
