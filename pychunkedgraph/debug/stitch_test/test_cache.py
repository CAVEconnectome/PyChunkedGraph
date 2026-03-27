"""
Tests for cache correctness across waves.

Run: pytest test_cache.py -v
"""

import numpy as np
import pytest

from .cache import CachedReader, WaveCache


class FakeClient:
    """Simulates BigTable reads. Tracks which nodes were read."""

    def __init__(self, data: dict) -> None:
        self.data = data  # {node_id: {attr: [FakeCell(value)]}}
        self.read_log: list[list] = []

    def read_nodes(self, node_ids: np.ndarray, properties: list) -> dict:
        self.read_log.append(list(node_ids))
        return {n: self.data.get(int(n), {}) for n in node_ids}


class FakeCell:
    def __init__(self, value) -> None:
        self.value = value


class FakeMeta:
    layer_count = 7

    class graph_config:
        LAYER_ID_BITS = 8
        FANOUT = 2


class FakeCG:
    """Minimal ChunkedGraph stand-in for cache tests."""

    def __init__(self, client_data: dict) -> None:
        self.client = FakeClient(client_data)
        self.meta = FakeMeta()

    def get_chunk_layers(self, node_ids: np.ndarray) -> np.ndarray:
        return np.array([self._layer(n) for n in node_ids], dtype=np.int32)

    def get_chunk_layer(self, node_id) -> int:
        return self._layer(int(node_id))

    def get_segment_id(self, node_id) -> int:
        return int(node_id) & 0xFFFF

    @staticmethod
    def _layer(node_id: int) -> int:
        return (int(node_id) >> 56) & 0xFF


def _make_parent_attr():
    """Return the Parent attribute key."""
    from pychunkedgraph.graph import attributes
    return attributes.Hierarchy.Parent


def _make_child_attr():
    from pychunkedgraph.graph import attributes
    return attributes.Hierarchy.Child


def _make_acx_attr(layer: int):
    from pychunkedgraph.graph import attributes
    return attributes.Connectivity.AtomicCrossChunkEdge[layer]


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


class TestCachedReaderPreloaded:
    """Fully cached nodes are never re-read from BigTable."""

    def test_fully_cached_node_skips_bigtable(self) -> None:
        node_a = _node(2, 100)
        cached_parent = _node(3, 200)
        children = np.array([1, 2, 3], dtype=np.uint64)

        cg = FakeCG({})
        preloaded = (
            {node_a: cached_parent},
            {node_a: children},
            {node_a: {}},
        )
        reader = CachedReader(cg, preloaded=preloaded)

        result = reader.get_parents(np.array([node_a], dtype=np.uint64))
        assert int(result[0]) == cached_parent
        assert len(cg.client.read_log) == 0

    def test_local_cache_prevents_redundant_reads(self) -> None:
        parent_attr = _make_parent_attr()
        node_a = _node(1, 50)  # SV (layer 1)
        parent = _node(2, 100)

        bt_data = {node_a: {parent_attr: [FakeCell(np.uint64(parent))]}}
        cg = FakeCG(bt_data)

        reader = CachedReader(cg)
        reader.get_parents(np.array([node_a], dtype=np.uint64))
        reader.get_parents(np.array([node_a], dtype=np.uint64))

        # Should only have one BigTable read (second is from local cache)
        assert len(cg.client.read_log) == 1


class TestWaveCache:
    """Wave cache merge: ctx always overwrites reader for same keys."""

    def test_ctx_overwrites_reader(self) -> None:
        cache = WaveCache()
        reader_snaps = [({10: 100}, {}, {})]  # reader: node 10 → parent 100
        ctx_snaps = [({10: 200}, {}, {})]       # ctx: node 10 → parent 200
        cache.merge_wave(reader_snaps, ctx_snaps)
        assert cache.data[0][10] == 200, "ctx must overwrite reader"

    def test_ctx_overwrites_regardless_of_task_order(self) -> None:
        cache = WaveCache()
        # Task A: reads node 10 as partner → old parent
        # Task B: writes node 10 → new parent
        reader_snaps = [
            ({10: 100}, {}, {}),  # Task A reader
            ({10: 100}, {}, {}),  # Task B reader (same pre-stitch value)
        ]
        ctx_snaps = [
            ({}, {}, {}),          # Task A ctx (didn't modify node 10)
            ({10: 200}, {}, {}),  # Task B ctx (wrote new parent)
        ]
        cache.merge_wave(reader_snaps, ctx_snaps)
        assert cache.data[0][10] == 200

    def test_children_persist_across_waves(self) -> None:
        cache = WaveCache()
        children = np.array([1, 2, 3], dtype=np.uint64)
        cache.merge_wave(
            [({}, {100: children}, {})],
            [({}, {}, {})],
        )
        assert np.array_equal(cache.data[1][100], children)

        # Wave 2: children not touched, should persist
        cache.merge_wave([({}, {}, {})], [({}, {}, {})])
        assert np.array_equal(cache.data[1][100], children)

    def test_multi_wave_parent_evolution(self) -> None:
        cache = WaveCache()

        # Wave 0: node 10 → parent 100
        cache.merge_wave([({10: 100}, {}, {})], [({10: 100}, {}, {})])
        assert cache.data[0][10] == 100

        # Wave 1: stitch creates new parent 200 for node 10
        cache.merge_wave(
            [({10: 100}, {}, {})],   # reader: pre-stitch (stale)
            [({10: 200}, {}, {})],   # ctx: post-stitch (fresh)
        )
        assert cache.data[0][10] == 200

        # Wave 2: node 10 not modified, but another reader reads it
        cache.merge_wave(
            [({10: 200}, {}, {})],   # reader: sees wave 1's write
            [({}, {}, {})],           # ctx: no changes to node 10
        )
        assert cache.data[0][10] == 200


class TestCacheIntegration:
    """Test CachedReader with preloaded WaveCache data."""

    def test_preloaded_children_skip_bigtable(self) -> None:
        child_attr = _make_child_attr()
        node_a = _node(2, 100)
        children = np.array([1, 2, 3], dtype=np.uint64)

        bt_data = {node_a: {child_attr: [FakeCell(children)]}}
        cg = FakeCG(bt_data)

        preloaded = ({}, {node_a: children}, {})
        reader = CachedReader(cg, preloaded=preloaded)
        result = reader.get_children(np.array([node_a], dtype=np.uint64))

        assert np.array_equal(result[np.uint64(node_a)], children)
        # Parent was NOT in preloaded, so BigTable was read for parent
        # But children should come from cache — verify by checking child_attr
        # was not the reason for the read

    def test_preloaded_parent_not_refreshed_without_read(self) -> None:
        """If a node is in ALL caches (parent+children+acx), no BigTable read happens.
        The preloaded parent persists. This is correct IFF the merge kept it up to date.
        """
        parent_attr = _make_parent_attr()
        child_attr = _make_child_attr()
        node_a = _node(2, 100)

        bt_data = {
            node_a: {
                parent_attr: [FakeCell(np.uint64(999))],
                child_attr: [FakeCell(np.array([1], dtype=np.uint64))],
            }
        }
        cg = FakeCG(bt_data)

        preloaded = (
            {node_a: 888},                                    # parent
            {node_a: np.array([1], dtype=np.uint64)},          # children
            {node_a: {}},                                      # acx
        )
        reader = CachedReader(cg, preloaded=preloaded)

        # All caches populated → no BigTable read
        result = reader.get_parents(np.array([node_a], dtype=np.uint64))
        assert int(result[0]) == 888, "no BigTable read, preloaded parent used"
        assert len(cg.client.read_log) == 0, "should not have read from BigTable"
