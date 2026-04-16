"""Tests for pychunkedgraph.ingest.ran_agglomeration - selected unit tests"""

from binascii import crc32

import numpy as np
import pytest

from pychunkedgraph.ingest.ran_agglomeration import (
    _crc_check,
    _get_cont_chunk_coords,
    define_active_edges,
    get_active_edges,
)
from pychunkedgraph.graph.edges import Edges, EDGE_TYPES
from pychunkedgraph.graph import basetypes


class TestCrcCheck:
    def test_valid(self):
        payload = b"test data here"
        crc = np.array([crc32(payload)], dtype=np.uint32).tobytes()
        full = payload + crc
        _crc_check(full)  # should not raise

    def test_invalid(self):
        payload = b"test data here"
        bad_crc = np.array([12345], dtype=np.uint32).tobytes()
        full = payload + bad_crc
        with pytest.raises(AssertionError):
            _crc_check(full)


class TestDefineActiveEdges:
    def test_basic(self):
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1, 2], dtype=basetypes.NODE_ID),
                np.array([3, 4], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        # Both sv1 and sv2 map to same agg ID -> active
        mapping = {1: 0, 2: 0, 3: 0, 4: 0}
        active, isolated = define_active_edges(edges, mapping)
        assert np.all(active[EDGE_TYPES.in_chunk])

    def test_unmapped_edges(self):
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1], dtype=basetypes.NODE_ID),
                np.array([2], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        # sv1 not in mapping -> isolated
        mapping = {2: 0}
        active, isolated = define_active_edges(edges, mapping)
        assert not active[EDGE_TYPES.in_chunk][0]
        assert 1 in isolated


class TestGetActiveEdges:
    def test_basic(self):
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1, 2], dtype=basetypes.NODE_ID),
                np.array([3, 4], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        mapping = {1: 0, 2: 0, 3: 0, 4: 0}
        chunk_edges, pseudo_isolated = get_active_edges(edges, mapping)
        for et in EDGE_TYPES:
            assert et in chunk_edges
        assert len(pseudo_isolated) > 0


class TestGetContChunkCoords:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)

        class FakeIM:
            cg_meta = graph.meta

        coord_a = np.array([1, 0, 0])
        coord_b = np.array([0, 0, 0])
        result = _get_cont_chunk_coords(FakeIM(), coord_a, coord_b)
        assert isinstance(result, list)

    def test_returns_only_valid_coords(self, gen_graph):
        """All returned coords should not be out of bounds."""
        graph = gen_graph(n_layers=4)

        class FakeIM:
            cg_meta = graph.meta

        coord_a = np.array([1, 0, 0])
        coord_b = np.array([0, 0, 0])
        result = _get_cont_chunk_coords(FakeIM(), coord_a, coord_b)
        for coord in result:
            assert not graph.meta.is_out_of_bounds(coord)

    def test_symmetric_direction(self, gen_graph):
        """Swapping coord_a and coord_b should yield the same set of neighboring coords."""
        graph = gen_graph(n_layers=4)

        class FakeIM:
            cg_meta = graph.meta

        coord_a = np.array([1, 0, 0])
        coord_b = np.array([0, 0, 0])
        result_ab = _get_cont_chunk_coords(FakeIM(), coord_a, coord_b)
        result_ba = _get_cont_chunk_coords(FakeIM(), coord_b, coord_a)

        # Convert to sets of tuples for comparison
        set_ab = {tuple(c) for c in result_ab}
        set_ba = {tuple(c) for c in result_ba}
        assert set_ab == set_ba

    def test_non_adjacent_raises(self, gen_graph):
        """Non-adjacent chunks (differing in more than one dim) should raise AssertionError."""
        graph = gen_graph(n_layers=4)

        class FakeIM:
            cg_meta = graph.meta

        coord_a = np.array([1, 1, 0])
        coord_b = np.array([0, 0, 0])
        with pytest.raises(AssertionError):
            _get_cont_chunk_coords(FakeIM(), coord_a, coord_b)

    def test_y_dim_adjacency(self, gen_graph):
        """Test adjacency along y dimension."""
        graph = gen_graph(n_layers=4)

        class FakeIM:
            cg_meta = graph.meta

        coord_a = np.array([0, 1, 0])
        coord_b = np.array([0, 0, 0])
        result = _get_cont_chunk_coords(FakeIM(), coord_a, coord_b)
        assert isinstance(result, list)
        # All returned coords should differ from chunk_coord_l along y
        for coord in result:
            assert not graph.meta.is_out_of_bounds(coord)


class TestParseEdgePayloads:
    def test_empty_payloads(self):
        """Empty list of payloads should return empty result."""
        from pychunkedgraph.ingest.ran_agglomeration import _parse_edge_payloads

        result = _parse_edge_payloads(
            [], edge_dtype=[("sv1", np.uint64), ("sv2", np.uint64)]
        )
        assert result == []

    def test_none_content_skipped(self):
        """Payloads with None content should be skipped."""
        from pychunkedgraph.ingest.ran_agglomeration import _parse_edge_payloads

        payloads = [{"content": None}]
        result = _parse_edge_payloads(
            payloads, edge_dtype=[("sv1", np.uint64), ("sv2", np.uint64)]
        )
        assert result == []

    def test_valid_payload(self):
        """A valid payload with correct CRC should be parsed."""
        from pychunkedgraph.ingest.ran_agglomeration import _parse_edge_payloads

        dtype = [("sv1", np.uint64), ("sv2", np.uint64)]
        data = np.array([(1, 2), (3, 4)], dtype=dtype)
        raw = data.tobytes()
        crc_val = np.array([crc32(raw)], dtype=np.uint32).tobytes()
        content = raw + crc_val

        payloads = [{"content": content}]
        result = _parse_edge_payloads(payloads, edge_dtype=dtype)
        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]["sv1"] == 1
        assert result[0][1]["sv2"] == 4

    def test_bad_crc_raises(self):
        """Payload with bad CRC should raise AssertionError."""
        from pychunkedgraph.ingest.ran_agglomeration import _parse_edge_payloads

        dtype = [("sv1", np.uint64), ("sv2", np.uint64)]
        data = np.array([(1, 2)], dtype=dtype)
        raw = data.tobytes()
        bad_crc = np.array([99999], dtype=np.uint32).tobytes()
        content = raw + bad_crc

        payloads = [{"content": content}]
        with pytest.raises(AssertionError):
            _parse_edge_payloads(payloads, edge_dtype=dtype)


class TestDefineActiveEdgesExtended:
    def test_both_unmapped(self):
        """When both endpoints are unmapped, edge should be inactive and both isolated."""
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([10, 20], dtype=basetypes.NODE_ID),
                np.array([30, 40], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        mapping = {}  # No IDs in mapping
        active, isolated = define_active_edges(edges, mapping)
        # All edges should be inactive
        assert not np.any(active[EDGE_TYPES.in_chunk])
        # All unmapped IDs should appear in isolated
        for sv_id in [10, 20, 30, 40]:
            assert sv_id in isolated

    def test_different_agg_ids(self):
        """Edges where sv1 and sv2 map to different agg IDs should be inactive."""
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1], dtype=basetypes.NODE_ID),
                np.array([2], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        mapping = {1: 100, 2: 200}  # Different agg IDs
        active, isolated = define_active_edges(edges, mapping)
        assert not active[EDGE_TYPES.in_chunk][0]

    def test_empty_edges(self):
        """Empty edge arrays should produce empty active arrays."""
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([], dtype=basetypes.NODE_ID),
                np.array([], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        mapping = {}
        active, isolated = define_active_edges(edges, mapping)
        assert len(active[EDGE_TYPES.in_chunk]) == 0

    def test_between_chunk_edges_active(self):
        """Between-chunk edges should also be classified."""
        edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges(
                np.array([1, 2], dtype=basetypes.NODE_ID),
                np.array([3, 4], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges([], []),
        }
        # 1->3 same agg, 2->4 different agg
        mapping = {1: 0, 3: 0, 2: 1, 4: 2}
        active, isolated = define_active_edges(edges, mapping)
        assert active[EDGE_TYPES.between_chunk][0]  # same agg
        assert not active[EDGE_TYPES.between_chunk][1]  # different agg


class TestGetActiveEdgesExtended:
    def test_cross_chunk_always_active(self):
        """Cross-chunk edges should always be kept active regardless of mapping."""
        edges = {
            EDGE_TYPES.in_chunk: Edges([], []),
            EDGE_TYPES.between_chunk: Edges([], []),
            EDGE_TYPES.cross_chunk: Edges(
                np.array([1, 2], dtype=basetypes.NODE_ID),
                np.array([3, 4], dtype=basetypes.NODE_ID),
                affinities=np.array([float("inf"), float("inf")]),
                areas=np.array([1.0, 1.0]),
            ),
        }
        mapping = {}  # Empty mapping - but cross_chunk should still be active
        chunk_edges, pseudo_isolated = get_active_edges(edges, mapping)
        assert len(chunk_edges[EDGE_TYPES.cross_chunk].node_ids1) == 2

    def test_pseudo_isolated_includes_all_node_ids(self):
        """pseudo_isolated should include all node_ids from all edge types."""
        edges = {
            EDGE_TYPES.in_chunk: Edges(
                np.array([1], dtype=basetypes.NODE_ID),
                np.array([2], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.between_chunk: Edges(
                np.array([3], dtype=basetypes.NODE_ID),
                np.array([4], dtype=basetypes.NODE_ID),
            ),
            EDGE_TYPES.cross_chunk: Edges(
                np.array([5], dtype=basetypes.NODE_ID),
                np.array([6], dtype=basetypes.NODE_ID),
                affinities=np.array([float("inf")]),
                areas=np.array([1.0]),
            ),
        }
        mapping = {1: 0, 2: 0, 3: 0, 4: 0}
        chunk_edges, pseudo_isolated = get_active_edges(edges, mapping)
        # Should include node_ids1 from all types and node_ids2 from in_chunk
        for sv_id in [1, 2, 3, 5]:
            assert sv_id in pseudo_isolated


class TestGetIndex:
    """Tests for _get_index which reads sharded index data from CloudFiles."""

    def test_inchunk_index(self):
        """Test _get_index with inchunk_or_agg=True uses single-u8 chunkid dtype."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import (
            CRC_LEN,
            HEADER_LEN,
            VERSION_LEN,
            _get_index,
        )

        # Create fake index data with inchunk dtype
        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array([(100, 20, 50)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        # Build a fake header: version (4 bytes) + idx_offset (8 bytes) + idx_length (8 bytes) = 20 bytes
        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        cf = MagicMock()
        # First call returns headers, second call returns index data
        cf.get.side_effect = [
            [{"path": "test.data", "content": header_content}],
            [{"path": "test.data", "content": index_with_crc}],
        ]

        result = _get_index(cf, ["test.data"], inchunk_or_agg=True)
        assert "test.data" in result
        assert result["test.data"][0]["chunkid"] == 100
        assert result["test.data"][0]["offset"] == 20
        assert result["test.data"][0]["size"] == 50

    def test_between_chunk_index(self):
        """Test _get_index with inchunk_or_agg=False uses 2-u8 chunkid dtype."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import (
            CRC_LEN,
            HEADER_LEN,
            VERSION_LEN,
            _get_index,
        )

        # Between-chunk index uses ("chunkid", "2u8") -> two uint64 values
        dt = np.dtype([("chunkid", "2u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array([((200, 300), 40, 60)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        cf = MagicMock()
        cf.get.side_effect = [
            [{"path": "between.data", "content": header_content}],
            [{"path": "between.data", "content": index_with_crc}],
        ]

        result = _get_index(cf, ["between.data"], inchunk_or_agg=False)
        assert "between.data" in result
        assert result["between.data"][0]["chunkid"][0] == 200
        assert result["between.data"][0]["chunkid"][1] == 300
        assert result["between.data"][0]["offset"] == 40
        assert result["between.data"][0]["size"] == 60

    def test_none_content_skipped(self):
        """When header content is None, that file should be skipped in the index."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import _get_index

        cf = MagicMock()
        # Header returns None content for the file
        cf.get.side_effect = [
            [{"path": "missing.data", "content": None}],
            [],  # No index_infos to fetch
        ]

        result = _get_index(cf, ["missing.data"], inchunk_or_agg=True)
        assert result == {}

    def test_multiple_files(self):
        """Test _get_index with multiple filenames, one valid and one missing."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _get_index,
        )

        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array([(500, 100, 200)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        cf = MagicMock()
        cf.get.side_effect = [
            [
                {"path": "valid.data", "content": header_content},
                {"path": "invalid.data", "content": None},
            ],
            [{"path": "valid.data", "content": index_with_crc}],
        ]

        result = _get_index(cf, ["valid.data", "invalid.data"], inchunk_or_agg=True)
        assert "valid.data" in result
        assert "invalid.data" not in result

    def test_multiple_index_entries(self):
        """Test _get_index with multiple entries in a single file index."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _get_index,
        )

        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array(
            [(100, 20, 50), (200, 70, 80), (300, 150, 30)], dtype=dt
        )
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        cf = MagicMock()
        cf.get.side_effect = [
            [{"path": "multi.data", "content": header_content}],
            [{"path": "multi.data", "content": index_with_crc}],
        ]

        result = _get_index(cf, ["multi.data"], inchunk_or_agg=True)
        assert "multi.data" in result
        assert len(result["multi.data"]) == 3
        assert result["multi.data"][0]["chunkid"] == 100
        assert result["multi.data"][1]["chunkid"] == 200
        assert result["multi.data"][2]["chunkid"] == 300


class TestReadInChunkFiles:
    """Tests for _read_in_chunk_files which reads edge data for a specific chunk."""

    def test_basic_read(self):
        """Mock CloudFiles to test full read flow for in-chunk files."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_in_chunk_files,
        )

        chunk_id = np.uint64(100)
        edge_dtype = [("sv1", np.uint64), ("sv2", np.uint64)]

        # Build index: one entry for our chunk_id
        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
        edge_data = np.array([(10, 20)], dtype=edge_dtype)
        edge_bytes = edge_data.tobytes()
        edge_crc = np.array([crc32(edge_bytes)], dtype=np.uint32).tobytes()
        edge_payload = edge_bytes + edge_crc

        data_offset = np.uint64(HEADER_LEN)
        data_size = np.uint64(len(edge_payload))

        index_entries = np.array([(chunk_id, data_offset, data_size)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(data_offset + data_size)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            # 1st call: headers
            [{"path": "in_chunk_0_0_0_0.data", "content": header_content}],
            # 2nd call: index data
            [{"path": "in_chunk_0_0_0_0.data", "content": index_with_crc}],
            # 3rd call: edge payloads
            [{"path": "in_chunk_0_0_0_0.data", "content": edge_payload}],
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_in_chunk_files(
                chunk_id, "gs://fake/path", ["in_chunk_0_0_0_0.data"], edge_dtype
            )

        assert len(result) == 1
        assert result[0][0]["sv1"] == 10
        assert result[0][0]["sv2"] == 20

    def test_no_matching_chunk(self):
        """When the index has no entry matching the requested chunk_id, no payloads are fetched."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_in_chunk_files,
        )

        chunk_id = np.uint64(999)
        edge_dtype = [("sv1", np.uint64), ("sv2", np.uint64)]

        # Index entry for a *different* chunk_id
        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array([(100, 20, 50)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            [{"path": "in_chunk_0_0_0_0.data", "content": header_content}],
            [{"path": "in_chunk_0_0_0_0.data", "content": index_with_crc}],
            [],  # No payloads fetched
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_in_chunk_files(
                chunk_id, "gs://fake/path", ["in_chunk_0_0_0_0.data"], edge_dtype
            )

        assert result == []


class TestReadBetweenOrFakeChunkFiles:
    """Tests for _read_between_or_fake_chunk_files which reads between-chunk edge data."""

    def _make_between_index_and_header(self, entries_list):
        """Helper to create between-chunk index data and header.

        entries_list: list of (chunkid0, chunkid1, offset, size) tuples
        """
        from pychunkedgraph.ingest.ran_agglomeration import HEADER_LEN

        dt = np.dtype([("chunkid", "2u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array(
            [((c0, c1), off, sz) for c0, c1, off, sz in entries_list], dtype=dt
        )
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )
        return header_content, index_with_crc

    def test_basic_between_chunk_read(self):
        """Test reading between-chunk files with matching chunk pair."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_between_or_fake_chunk_files,
        )

        chunk_id = np.uint64(100)
        adjacent_id = np.uint64(200)
        edge_dtype = [("sv1", np.uint64), ("sv2", np.uint64)]

        # Create edge payload
        edge_data = np.array([(10, 20)], dtype=edge_dtype)
        edge_bytes = edge_data.tobytes()
        edge_crc = np.array([crc32(edge_bytes)], dtype=np.uint32).tobytes()
        edge_payload = edge_bytes + edge_crc

        data_offset = np.uint64(HEADER_LEN)
        data_size = np.uint64(len(edge_payload))

        header_content, index_with_crc = self._make_between_index_and_header(
            [(100, 200, int(data_offset), int(data_size))]
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            # headers
            [{"path": "between.data", "content": header_content}],
            # index data
            [{"path": "between.data", "content": index_with_crc}],
            # chunk_finfos payloads (forward direction)
            [{"path": "between.data", "content": edge_payload}],
            # adj_chunk_finfos payloads (reverse direction) - empty
            [],
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_between_or_fake_chunk_files(
                chunk_id, adjacent_id, "gs://fake/path", ["between.data"], edge_dtype
            )

        assert len(result) == 1
        assert result[0][0]["sv1"] == 10
        assert result[0][0]["sv2"] == 20

    def test_reverse_direction(self):
        """Test reading from the adjacent->chunk direction (swapped columns in result dtype)."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_between_or_fake_chunk_files,
        )

        chunk_id = np.uint64(100)
        adjacent_id = np.uint64(200)
        edge_dtype = [("sv1", np.uint64), ("sv2", np.uint64)]

        # Edge payload for the *reverse* direction (adjacent_id, chunk_id)
        # When reading reverse direction, the dtype columns are swapped: (sv2, sv1)
        rev_edge_dtype = [("sv2", np.uint64), ("sv1", np.uint64)]
        edge_data = np.array([(30, 40)], dtype=rev_edge_dtype)
        edge_bytes = edge_data.tobytes()
        edge_crc = np.array([crc32(edge_bytes)], dtype=np.uint32).tobytes()
        edge_payload = edge_bytes + edge_crc

        data_offset = np.uint64(HEADER_LEN)
        data_size = np.uint64(len(edge_payload))

        # Index entry: (adjacent_id, chunk_id) => reverse direction
        header_content, index_with_crc = self._make_between_index_and_header(
            [(200, 100, int(data_offset), int(data_size))]
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            # headers
            [{"path": "between.data", "content": header_content}],
            # index
            [{"path": "between.data", "content": index_with_crc}],
            # chunk_finfos (forward) - empty
            [],
            # adj_chunk_finfos (reverse)
            [{"path": "between.data", "content": edge_payload}],
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_between_or_fake_chunk_files(
                chunk_id, adjacent_id, "gs://fake/path", ["between.data"], edge_dtype
            )

        # Result comes from adj_result which used the swapped dtype
        assert len(result) == 1
        assert result[0][0]["sv2"] == 30
        assert result[0][0]["sv1"] == 40

    def test_no_matching_pairs(self):
        """When no chunk pair matches, should return empty list."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_between_or_fake_chunk_files,
        )

        chunk_id = np.uint64(100)
        adjacent_id = np.uint64(200)
        edge_dtype = [("sv1", np.uint64), ("sv2", np.uint64)]

        # Index entry for a totally different pair
        header_content, index_with_crc = self._make_between_index_and_header(
            [(999, 888, 20, 50)]
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            [{"path": "between.data", "content": header_content}],
            [{"path": "between.data", "content": index_with_crc}],
            [],  # No forward payloads
            [],  # No reverse payloads
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_between_or_fake_chunk_files(
                chunk_id, adjacent_id, "gs://fake/path", ["between.data"], edge_dtype
            )

        assert result == []


class TestReadAggFiles:
    """Tests for _read_agg_files which reads agglomeration remap data."""

    def test_basic_agg_read(self):
        """Test reading agglomeration files returns edge list."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            CRC_LEN,
            HEADER_LEN,
            _read_agg_files,
        )

        chunk_id = np.uint64(42)

        # Index entry for our chunk
        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])

        # Build edge data: pairs of node IDs
        edges = np.array([[10, 20], [30, 40]], dtype=basetypes.NODE_ID)
        edge_bytes = edges.tobytes()
        edge_crc = np.array([crc32(edge_bytes)], dtype=np.uint32).tobytes()
        edge_payload = edge_bytes + edge_crc

        data_offset = np.uint64(HEADER_LEN)
        data_size = np.uint64(len(edge_payload))

        index_entries = np.array([(chunk_id, data_offset, data_size)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(data_offset + data_size)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            # headers
            [{"path": "done_0_0_0_0.data", "content": header_content}],
            # index
            [{"path": "done_0_0_0_0.data", "content": index_with_crc}],
            # payloads
            [{"path": "done_0_0_0_0.data", "content": edge_payload}],
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_agg_files(
                ["done_0_0_0_0.data"], [chunk_id], "gs://fake/remap/"
            )

        # Result is a list starting with empty_2d, plus our edge data
        assert len(result) >= 2  # empty_2d + our edges
        # The last element should be our 2x2 edge array
        combined = np.concatenate(result)
        assert combined.shape[1] == 2
        assert len(combined) == 2

    def test_missing_file_skipped(self):
        """When a filename is not in files_index (KeyError), it should be skipped."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_agg_files,
        )

        # No valid headers -> empty index
        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            # headers: all None
            [{"path": "done_0_0_0_0.data", "content": None}],
            [],  # empty index_infos
            [],  # empty payloads
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_agg_files(
                ["done_0_0_0_0.data"], [np.uint64(42)], "gs://fake/remap/"
            )

        # Should only contain the initial empty_2d
        assert len(result) == 1
        assert result[0].shape == (0, 2)

    def test_none_payload_skipped(self):
        """When a payload content is None, it should be skipped."""
        from unittest.mock import MagicMock, patch

        from pychunkedgraph.ingest.ran_agglomeration import (
            HEADER_LEN,
            _read_agg_files,
        )

        chunk_id = np.uint64(42)
        dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
        index_entries = np.array([(chunk_id, 20, 50)], dtype=dt)
        index_bytes = index_entries.tobytes()
        index_crc = np.array([crc32(index_bytes)], dtype=np.uint32).tobytes()
        index_with_crc = index_bytes + index_crc

        idx_offset = np.uint64(HEADER_LEN)
        idx_length = np.uint64(len(index_with_crc))
        version = b"SO01"
        header_content = (
            version
            + np.array([idx_offset], dtype=np.uint64).tobytes()
            + np.array([idx_length], dtype=np.uint64).tobytes()
        )

        mock_cf = MagicMock()
        mock_cf.get.side_effect = [
            [{"path": "done_0_0_0_0.data", "content": header_content}],
            [{"path": "done_0_0_0_0.data", "content": index_with_crc}],
            [{"path": "done_0_0_0_0.data", "content": None}],  # None payload
        ]

        with patch(
            "pychunkedgraph.ingest.ran_agglomeration.CloudFiles",
            return_value=mock_cf,
        ):
            result = _read_agg_files(
                ["done_0_0_0_0.data"], [chunk_id], "gs://fake/remap/"
            )

        # Should only contain the initial empty_2d (None content was skipped)
        assert len(result) == 1
        assert result[0].shape == (0, 2)


class TestReadRawEdgeData:
    """Tests for read_raw_edge_data which orchestrates edge collection and writing."""

    from unittest.mock import patch, MagicMock

    @patch("pychunkedgraph.ingest.ran_agglomeration._collect_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.postprocess_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_edges")
    def test_basic(self, mock_put, mock_postprocess, mock_collect):
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import read_raw_edge_data

        # Setup mock return values
        edge_dict = {}
        for et in EDGE_TYPES:
            edge_dict[et] = {
                "sv1": np.array([1, 2], dtype=np.uint64),
                "sv2": np.array([3, 4], dtype=np.uint64),
                "aff": np.array([0.5, 0.6]),
                "area": np.array([10, 20]),
            }
        # cross_chunk doesn't have aff/area in the read path (they get inf/ones)
        edge_dict[EDGE_TYPES.cross_chunk] = {
            "sv1": np.array([5], dtype=np.uint64),
            "sv2": np.array([6], dtype=np.uint64),
        }
        mock_collect.return_value = edge_dict
        mock_postprocess.return_value = edge_dict

        imanager = MagicMock()
        imanager.cg_meta.data_source.EDGES = "gs://fake/edges"

        result = read_raw_edge_data(imanager, [0, 0, 0])
        assert EDGE_TYPES.in_chunk in result
        assert EDGE_TYPES.between_chunk in result
        assert EDGE_TYPES.cross_chunk in result
        # in_chunk should have 2 edges
        assert len(result[EDGE_TYPES.in_chunk].node_ids1) == 2
        # cross_chunk should have 1 edge with inf affinity
        assert len(result[EDGE_TYPES.cross_chunk].node_ids1) == 1
        assert np.isinf(result[EDGE_TYPES.cross_chunk].affinities[0])
        # put_chunk_edges should have been called since there are edges
        mock_put.assert_called_once()

    @patch("pychunkedgraph.ingest.ran_agglomeration._collect_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.postprocess_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_edges")
    def test_no_edges(self, mock_put, mock_postprocess, mock_collect):
        """When all edge types are empty, put_chunk_edges should not be called."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import read_raw_edge_data

        edge_dict = {et: {} for et in EDGE_TYPES}
        mock_collect.return_value = edge_dict
        mock_postprocess.return_value = edge_dict

        imanager = MagicMock()
        imanager.cg_meta.data_source.EDGES = "gs://fake/edges"

        result = read_raw_edge_data(imanager, [0, 0, 0])
        # All edge types should be empty Edges objects
        for et in EDGE_TYPES:
            assert len(result[et].node_ids1) == 0
        mock_put.assert_not_called()

    @patch("pychunkedgraph.ingest.ran_agglomeration._collect_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.postprocess_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_edges")
    def test_edges_but_no_storage_path(self, mock_put, mock_postprocess, mock_collect):
        """When EDGES path is empty/falsy, put_chunk_edges should not be called."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import read_raw_edge_data

        edge_dict = {}
        for et in EDGE_TYPES:
            edge_dict[et] = {
                "sv1": np.array([1], dtype=np.uint64),
                "sv2": np.array([2], dtype=np.uint64),
                "aff": np.array([0.5]),
                "area": np.array([10]),
            }
        mock_collect.return_value = edge_dict
        mock_postprocess.return_value = edge_dict

        imanager = MagicMock()
        imanager.cg_meta.data_source.EDGES = ""  # empty string = falsy

        result = read_raw_edge_data(imanager, [0, 0, 0])
        assert EDGE_TYPES.in_chunk in result
        mock_put.assert_not_called()

    @patch("pychunkedgraph.ingest.ran_agglomeration._collect_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.postprocess_edge_data")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_edges")
    def test_partial_edges(self, mock_put, mock_postprocess, mock_collect):
        """Only in_chunk has edges, others empty."""
        from unittest.mock import MagicMock

        from pychunkedgraph.ingest.ran_agglomeration import read_raw_edge_data

        edge_dict = {
            EDGE_TYPES.in_chunk: {
                "sv1": np.array([1, 2], dtype=np.uint64),
                "sv2": np.array([3, 4], dtype=np.uint64),
                "aff": np.array([0.5, 0.6]),
                "area": np.array([10, 20]),
            },
            EDGE_TYPES.between_chunk: {},
            EDGE_TYPES.cross_chunk: {},
        }
        mock_collect.return_value = edge_dict
        mock_postprocess.return_value = edge_dict

        imanager = MagicMock()
        imanager.cg_meta.data_source.EDGES = "gs://fake/edges"

        result = read_raw_edge_data(imanager, [0, 0, 0])
        assert len(result[EDGE_TYPES.in_chunk].node_ids1) == 2
        assert len(result[EDGE_TYPES.between_chunk].node_ids1) == 0
        assert len(result[EDGE_TYPES.cross_chunk].node_ids1) == 0
        # Should still write because in_chunk has edges
        mock_put.assert_called_once()


class TestReadRawAgglomerationData:
    """Tests for read_raw_agglomeration_data which reads agg remap files."""

    from unittest.mock import patch, MagicMock

    @patch("pychunkedgraph.ingest.ran_agglomeration._read_agg_files")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_components")
    def test_basic(self, mock_put_components, mock_read_agg, gen_graph):
        from pychunkedgraph.ingest.ran_agglomeration import read_raw_agglomeration_data
        from unittest.mock import MagicMock

        graph = gen_graph(n_layers=4)
        imanager = MagicMock()
        imanager.cg_meta = graph.meta
        imanager.config.AGGLOMERATION = "gs://fake/agg"

        # Return edge pairs that form connected components
        mock_read_agg.return_value = [np.array([[1, 2], [2, 3]], dtype=np.uint64)]

        mapping = read_raw_agglomeration_data(imanager, np.array([0, 0, 0]))
        assert isinstance(mapping, dict)
        # 1, 2, 3 should all map to the same component
        assert mapping[1] == mapping[2] == mapping[3]
        # put_chunk_components should have been called
        mock_put_components.assert_called_once()

    @patch("pychunkedgraph.ingest.ran_agglomeration._read_agg_files")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_components")
    def test_multiple_components(self, mock_put_components, mock_read_agg, gen_graph):
        from pychunkedgraph.ingest.ran_agglomeration import read_raw_agglomeration_data
        from unittest.mock import MagicMock

        graph = gen_graph(n_layers=4)
        imanager = MagicMock()
        imanager.cg_meta = graph.meta
        imanager.config.AGGLOMERATION = "gs://fake/agg"

        # Two separate components: {1,2} and {3,4}
        mock_read_agg.return_value = [np.array([[1, 2], [3, 4]], dtype=np.uint64)]

        mapping = read_raw_agglomeration_data(imanager, np.array([0, 0, 0]))
        assert isinstance(mapping, dict)
        assert mapping[1] == mapping[2]
        assert mapping[3] == mapping[4]
        # The two components should have different IDs
        assert mapping[1] != mapping[3]

    @patch("pychunkedgraph.ingest.ran_agglomeration._read_agg_files")
    @patch("pychunkedgraph.ingest.ran_agglomeration.put_chunk_components")
    def test_no_components_path(self, mock_put_components, mock_read_agg, gen_graph):
        """When COMPONENTS is None (falsy), put_chunk_components should not be called."""
        from pychunkedgraph.ingest.ran_agglomeration import read_raw_agglomeration_data
        from unittest.mock import MagicMock

        graph = gen_graph(n_layers=4)
        # Replace the data_source with one that has COMPONENTS=None
        original_ds = graph.meta.data_source
        graph.meta._data_source = original_ds._replace(COMPONENTS=None)

        imanager = MagicMock()
        imanager.cg_meta = graph.meta
        imanager.config.AGGLOMERATION = "gs://fake/agg"

        mock_read_agg.return_value = [np.array([[1, 2]], dtype=np.uint64)]

        mapping = read_raw_agglomeration_data(imanager, np.array([0, 0, 0]))
        assert isinstance(mapping, dict)
        mock_put_components.assert_not_called()

        # Restore original data_source
        graph.meta._data_source = original_ds
