"""Tests for the chunk-edge read path.

get_chunk_edges fetches and decompresses chunk edge files in batches and, when given
the queried object's supervoxels, filters each chunk as it is parsed. Both behaviors
exist to bound peak memory, so what matters is that neither changes the result: the
output must be identical to reading everything at once and filtering at the end.
"""

import numpy as np
import pytest
import zstandard as zstd

from ..io import edges as io_edges
from ..io.protobuf.chunkEdges_pb2 import ChunkEdgesMsg
from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.utils import basetypes


def _edges(pairs):
    a = np.array([p[0] for p in pairs], dtype=basetypes.NODE_ID)
    b = np.array([p[1] for p in pairs], dtype=basetypes.NODE_ID)
    return Edges(
        a,
        b,
        affinities=np.arange(len(pairs), dtype=basetypes.EDGE_AFFINITY) + 1,
        areas=np.arange(len(pairs), dtype=basetypes.EDGE_AREA) + 2,
    )


def _blob(in_chunk, between, cross):
    msg = ChunkEdgesMsg()
    msg.in_chunk.CopyFrom(io_edges.serialize(in_chunk))
    msg.between_chunk.CopyFrom(io_edges.serialize(between))
    msg.cross_chunk.CopyFrom(io_edges.serialize(cross))
    return zstd.ZstdCompressor().compress(msg.SerializeToString())


class FakeCloudFiles:
    """Records each get() so batching is observable."""

    def __init__(self, blobs):
        self.blobs = blobs
        self.calls = []

    def __call__(self, *args, **kwargs):
        return self

    def get(self, fnames, raw=False):
        self.calls.append(list(fnames))
        return [{"content": self.blobs.get(f)} for f in fnames]


@pytest.fixture
def chunks(monkeypatch):
    """10 chunks; each has one edge touching sv 5 and two that do not."""
    blobs, coords = {}, []
    for i in range(10):
        base = 1000 * (i + 1)
        blobs[f"edges_{i}_0_0.proto.zst"] = _blob(
            _edges([(5, base), (base + 1, base + 2)]),
            _edges([(base + 3, 5)]),
            _edges([(base + 4, base + 5)]),
        )
        coords.append(np.array([i, 0, 0]))
    fake = FakeCloudFiles(blobs)
    monkeypatch.setattr(io_edges, "CloudFiles", fake)
    return fake, coords


def _all_pairs(result):
    return sorted(
        tuple(p) for t in EDGE_TYPES for p in result[t].get_pairs().tolist()
    )


class TestBatching:
    @pytest.mark.parametrize("batch_size", [1, 3, 7, 64, 1000])
    def test_result_is_independent_of_batch_size(self, chunks, batch_size):
        fake, coords = chunks
        expected = _all_pairs(io_edges.get_chunk_edges("gs://x", coords, batch_size=1000))

        got = _all_pairs(io_edges.get_chunk_edges("gs://x", coords, batch_size=batch_size))

        assert got == expected

    @pytest.mark.parametrize("batch_size,expected_calls", [(1, 10), (3, 4), (5, 2), (64, 1)])
    def test_fetches_in_batches(self, chunks, batch_size, expected_calls):
        fake, coords = chunks
        fake.calls.clear()

        io_edges.get_chunk_edges("gs://x", coords, batch_size=batch_size)

        assert len(fake.calls) == expected_calls
        assert max(len(c) for c in fake.calls) <= batch_size
        assert sum(len(c) for c in fake.calls) == len(coords)

    def test_every_file_requested_exactly_once(self, chunks):
        fake, coords = chunks
        fake.calls.clear()

        io_edges.get_chunk_edges("gs://x", coords, batch_size=3)

        requested = [f for call in fake.calls for f in call]
        assert sorted(requested) == sorted(fake.blobs)

    def test_missing_chunk_files_are_skipped(self, chunks):
        fake, coords = chunks
        fake.blobs["edges_2_0_0.proto.zst"] = None

        result = io_edges.get_chunk_edges("gs://x", coords, batch_size=3)

        assert all((3000, 3001) != p for p in _all_pairs(result))


class TestFiltering:
    def test_filter_matches_filtering_after_the_fact(self, chunks):
        fake, coords = chunks
        svs = np.array([5], dtype=basetypes.NODE_ID)

        filtered = io_edges.get_chunk_edges("gs://x", coords, supervoxels=svs)
        unfiltered = io_edges.get_chunk_edges("gs://x", coords)
        expected = {
            t: unfiltered[t].filter_touching(np.unique(svs)) for t in EDGE_TYPES
        }

        for t in EDGE_TYPES:
            np.testing.assert_array_equal(filtered[t].node_ids1, expected[t].node_ids1)
            np.testing.assert_array_equal(filtered[t].node_ids2, expected[t].node_ids2)
            np.testing.assert_array_equal(filtered[t].affinities, expected[t].affinities)
            np.testing.assert_array_equal(filtered[t].areas, expected[t].areas)

    def test_filter_keeps_only_touching_edges(self, chunks):
        fake, coords = chunks
        svs = np.array([5], dtype=basetypes.NODE_ID)

        result = io_edges.get_chunk_edges("gs://x", coords, supervoxels=svs)

        pairs = _all_pairs(result)
        assert len(pairs) == 20  # one in_chunk + one between_chunk per chunk
        assert all(5 in p for p in pairs)

    @pytest.mark.parametrize("batch_size", [1, 3, 64])
    def test_filter_is_independent_of_batch_size(self, chunks, batch_size):
        fake, coords = chunks
        svs = np.array([5], dtype=basetypes.NODE_ID)

        got = _all_pairs(
            io_edges.get_chunk_edges("gs://x", coords, supervoxels=svs, batch_size=batch_size)
        )

        assert got == _all_pairs(
            io_edges.get_chunk_edges("gs://x", coords, supervoxels=svs, batch_size=1000)
        )

    def test_no_supervoxels_reads_everything(self, chunks):
        fake, coords = chunks

        result = io_edges.get_chunk_edges("gs://x", coords)

        assert len(_all_pairs(result)) == 40  # 4 edges x 10 chunks

    def test_unsorted_and_duplicated_supervoxels_are_normalized(self, chunks):
        fake, coords = chunks
        messy = np.array([5, 5, 5], dtype=basetypes.NODE_ID)

        got = _all_pairs(io_edges.get_chunk_edges("gs://x", coords, supervoxels=messy))

        assert got == _all_pairs(
            io_edges.get_chunk_edges(
                "gs://x", coords, supervoxels=np.array([5], dtype=basetypes.NODE_ID)
            )
        )


class TestDecompression:
    """_decompress replaced multi_decompress_to_buffer (removed in zstandard 0.23).

    A thread pool matches its throughput because decompression releases the GIL, but
    ZstdDecompressor is not thread-safe, so correctness under threads is what these pin.
    """

    def _blobs(self, n=32, size=200_000):
        rng = np.random.default_rng(3)
        cctx = zstd.ZstdCompressor(level=3)
        raw = [rng.integers(0, 255, size=size, dtype=np.uint8).tobytes() for _ in range(n)]
        return raw, [cctx.compress(r) for r in raw]

    @pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
    def test_matches_input_for_any_thread_count(self, n_threads):
        raw, blobs = self._blobs()

        out = io_edges._decompress(blobs, n_threads)

        assert [bytes(o) for o in out] == raw

    def test_threaded_matches_serial(self):
        """Sharing one ZstdDecompressor across threads corrupts silently; this catches it."""
        raw, blobs = self._blobs()

        assert [bytes(o) for o in io_edges._decompress(blobs, 4)] == [
            bytes(o) for o in io_edges._decompress(blobs, 1)
        ]

    def test_preserves_order(self):
        raw, blobs = self._blobs(n=16)

        out = [bytes(o) for o in io_edges._decompress(blobs, 4)]

        assert out == raw  # pool.map must not reorder

    @pytest.mark.parametrize("n_threads", [1, 4])
    def test_empty_and_single(self, n_threads):
        raw, blobs = self._blobs(n=1)

        assert io_edges._decompress([], n_threads) == []
        assert [bytes(o) for o in io_edges._decompress(blobs, n_threads)] == raw

    @pytest.mark.parametrize("n_threads", [1, 4])
    def test_frames_without_content_size(self, n_threads):
        """dctx.decompress needs the size in the header; decompressobj covers frames without it."""
        rng = np.random.default_rng(4)
        raw = [rng.integers(0, 255, size=50_000, dtype=np.uint8).tobytes() for _ in range(4)]
        cctx = zstd.ZstdCompressor(level=3, write_content_size=False)
        blobs = [cctx.compress(r) for r in raw]

        assert [bytes(o) for o in io_edges._decompress(blobs, n_threads)] == raw
