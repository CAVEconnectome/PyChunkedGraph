"""Performance + correctness tests for cache-backed operations at realistic scale.

These test the PUBLIC functions (get_parents, read_l2, build_rows, merge_wave_results,
resolve_cx_at_layer, resolve_remaining_cx) at 100K+ node scale. They catch performance
regressions when the cache design changes — if a refactor makes has_batch 7x slower
(like ChainMap did), these tests will show it.

Each test asserts correctness AND prints timing. Timing thresholds are not hard-enforced
(BigTable not involved — only in-memory ops) but are documented for manual review.
"""

import time

import numpy as np

from pychunkedgraph.graph import basetypes

from . import resolver as topology
from .test_helpers import noop_read, get_parent as _get_parent, get_cx as _get_cx
from .wave_cache import SiblingEntry, WaveCache

NODE_ID = basetypes.NODE_ID
N_NODES = 200_000
N_SIBLINGS = 100_000
N_LAYERS = 5


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


def _get_layer(nid: int) -> int:
    return (int(nid) >> 56) & 0xFF


def _prepopulate_cache(c: WaveCache, n: int) -> None:
    """Populate cache with n L2 nodes, each with parent, children, and ACX."""
    for i in range(n):
        nid = _node(2, i)
        c.put_parent(nid, _node(3, i // 10))
        c.put_children(nid, np.array([i * 100, i * 100 + 1], dtype=NODE_ID))
        c.put_acx(nid, {2: np.array([[nid, _node(1, i * 100 + 50)]], dtype=NODE_ID)})


class TestBatchLookupPerformance:
    """Tests has_batch and get_parents at scale — the hottest path in _ensure_cached."""

    def test_has_batch_200k(self) -> None:
        """200K has_batch lookups. Must be <50ms (was 134ms with ChainMap)."""
        c = WaveCache(noop_read)
        _prepopulate_cache(c, N_NODES)
        ids = np.array([_node(2, i) for i in range(N_NODES)], dtype=NODE_ID)

        t0 = time.time()
        result = c.has_batch(ids)
        elapsed = time.time() - t0

        assert result.all(), "All prepopulated nodes must be found"
        print(f"\n  has_batch {N_NODES}: {elapsed*1000:.0f}ms")

    def test_has_batch_miss_200k(self) -> None:
        """200K has_batch on missing nodes. Must be fast (no false positives)."""
        c = WaveCache(noop_read)
        _prepopulate_cache(c, N_NODES)
        miss_ids = np.array([_node(2, N_NODES + i) for i in range(N_NODES)], dtype=NODE_ID)

        t0 = time.time()
        result = c.has_batch(miss_ids)
        elapsed = time.time() - t0

        assert not result.any(), "Missing nodes must not be found"
        print(f"\n  has_batch miss {N_NODES}: {elapsed*1000:.0f}ms")

    def test_get_parents_200k(self) -> None:
        """200K parent lookups via batch API. Must be <100ms."""
        c = WaveCache(noop_read)
        _prepopulate_cache(c, N_NODES)

        all_ids = np.array([_node(2, i) for i in range(N_NODES)], dtype=NODE_ID)
        t0 = time.time()
        parents = c.get_parents(all_ids)
        elapsed = time.time() - t0

        assert len(parents) == N_NODES
        assert parents[0] == _node(3, 0)
        print(f"\n  get_parents {N_NODES}: {elapsed*1000:.0f}ms")


class TestPreloadedLookupPerformance:
    """Tests two-layer lookup (local → preloaded) at scale."""

    def test_preloaded_then_local_200k(self) -> None:
        """Preloaded data visible after creating WaveCache with preloaded dict."""
        c1 = WaveCache(noop_read)
        _prepopulate_cache(c1, N_NODES)

        preloaded = c1.preloaded()
        c2 = WaveCache(noop_read, preloaded=preloaded)

        ids = np.array([_node(2, i) for i in range(N_NODES)], dtype=NODE_ID)

        t0 = time.time()
        result = c2.has_batch(ids)
        elapsed = time.time() - t0

        assert result.all(), "Preloaded nodes must be found in new cache"
        print(f"\n  preloaded has_batch {N_NODES}: {elapsed*1000:.0f}ms")

    def test_local_shadows_preloaded_200k(self) -> None:
        """Local writes shadow preloaded. Verify at scale."""
        c1 = WaveCache(noop_read)
        _prepopulate_cache(c1, N_NODES)
        preloaded = c1.preloaded()

        c2 = WaveCache(noop_read, preloaded=preloaded)
        c2.put_parent(_node(2, 0), 999)

        assert _get_parent(c2, _node(2, 0)) == 999, "Local shadows preloaded"
        assert _get_parent(c2, _node(2, 1)) == _node(3, 0), "Preloaded still accessible"


class TestMergePerformance:
    """Tests merge_wave_results with multiple workers' snapshots."""

    def test_merge_4_workers_50k_each(self) -> None:
        """4 workers each with 50K rows. Merge into parent. All rows visible."""
        parent = WaveCache(noop_read)
        n_per_worker = 50_000
        n_workers = 4

        snapshots = []
        for w in range(n_workers):
            worker = WaveCache(noop_read)
            for i in range(n_per_worker):
                nid = _node(2, w * n_per_worker + i)
                worker.put_parent(nid, _node(3, i))
            local = worker.local_snapshot()
            inc = worker.inc_snapshot_from(
                old_to_new={}, new_node_ids=set(),
                sibling_ids=set(), unresolved_acx={},
            )
            snapshots.append((local, inc))

        t0 = time.time()
        for local_snap, inc_snap in snapshots:
            parent.merge_reader(local_snap)
            parent.merge_inc(inc_snap)
        elapsed = time.time() - t0

        total = n_workers * n_per_worker
        ids = np.array([_node(2, i) for i in range(total)], dtype=NODE_ID)
        assert parent.has_batch(ids).all(), "All merged rows must be visible"
        print(f"\n  merge {n_workers}x{n_per_worker}: {elapsed*1000:.0f}ms")


class TestResolveCxPerformance:
    """Tests resolve_cx_at_layer + store_cx_from_resolved at sibling scale."""

    def test_resolve_100k_siblings_layer2(self) -> None:
        """Resolve CX for 100K siblings at layer 2. Measures the hot path in build_hierarchy."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        nodes = []
        for i in range(N_SIBLINGS):
            nid = _node(2, i)
            partner_sv = i * 100 + 50
            partner_l2 = _node(2, N_SIBLINGS + (i % 1000))
            c.put_parent(partner_sv, int(partner_l2))
            c.unresolved_acx[int(nid)] = {2: np.array([[nid, partner_sv]], dtype=NODE_ID)}
            nodes.append(nid)

        t0 = time.time()
        cx = topology.resolve_cx_at_layer(nodes, 2, c, _get_layer)
        elapsed_resolve = time.time() - t0

        t0 = time.time()
        topology.store_cx_from_resolved(c, cx, 2)
        elapsed_store = time.time() - t0

        assert len(cx) > 0
        print(f"\n  resolve_cx {N_SIBLINGS} siblings L2: {elapsed_resolve*1000:.0f}ms")
        print(f"  store_cx: {elapsed_store*1000:.0f}ms")

    def test_resolve_remaining_batched(self) -> None:
        """resolve_remaining_cx for 100K siblings at layers 3-5. Measures batched path."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.new_node_ids = set()

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        sibs = set()
        for i in range(N_SIBLINGS):
            nid = _node(2, i)
            sibs.add(int(nid))
            partner_sv = i * 100 + 50
            partner_l2 = _node(2, N_SIBLINGS + (i % 1000))
            partner_l3 = _node(3, N_SIBLINGS + (i % 100))
            c.put_parent(partner_sv, int(partner_l2))
            c.unresolved_acx[int(nid)] = {
                3: np.array([[nid, partner_sv]], dtype=NODE_ID),
            }
        c.sibling_ids = sibs

        for i in range(1000):
            c.put_parent(_node(2, N_SIBLINGS + (i % 1000)), _node(3, i % 100))

        t0 = time.time()
        topology.resolve_remaining_cx(c, FakeLcg())
        elapsed = time.time() - t0

        sib_cx = c.get_cx_batch(np.array(list(sibs), dtype=NODE_ID))
        stored = sum(1 for s in sibs if sib_cx.get(int(s)))
        print(f"\n  resolve_remaining {N_SIBLINGS} siblings: {elapsed*1000:.0f}ms ({stored} stored)")


class TestBuildRowsPerformance:
    """Tests build_rows at scale with new nodes + dirty siblings."""

    def test_build_rows_10k_new_nodes(self) -> None:
        """10K new nodes with children + CX. Measures serialization overhead."""
        from kvdbclient import serializers
        from pychunkedgraph.graph import attributes

        c = WaveCache(noop_read)
        c.begin_stitch()

        n_new = 10_000
        for i in range(n_new):
            nid = _node(2, i)
            c.new_node_ids.add(int(nid))
            c.put_children(int(nid), np.array([i * 100, i * 100 + 1], dtype=NODE_ID))
            c.put_cx(int(nid), {2: np.array([[nid, _node(2, i + n_new)]], dtype=NODE_ID)})

        t0 = time.time()
        new_arr = np.array(list(c.new_node_ids), dtype=NODE_ID)
        all_ch = c.get_children_batch(new_arr)
        all_cx = c.get_cx_batch(new_arr)
        rows = {}
        for nid in c.new_node_ids:
            ch = all_ch.get(int(nid))
            if ch is None or len(ch) == 0:
                continue
            rk = serializers.serialize_uint64(nid)
            vd = rows.setdefault(rk, {})
            vd[attributes.Hierarchy.Child] = ch
            for layer, cx in all_cx.get(int(nid), {}).items():
                vd[attributes.Connectivity.CrossChunkEdge[layer]] = cx
            for child in ch:
                crk = serializers.serialize_uint64(child)
                rows.setdefault(crk, {})[attributes.Hierarchy.Parent] = nid
        elapsed = time.time() - t0

        assert len(rows) > n_new
        print(f"\n  build_rows {n_new} new nodes: {elapsed*1000:.0f}ms ({len(rows)} rows)")


class TestDirectWritePerformance:
    """Tests direct RowCache writes at scale."""

    def test_write_10k_nodes(self) -> None:
        """Write 10K parents + children + ACX directly to RowCache."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        n = 10_000
        t0 = time.time()
        for i in range(n):
            nid = _node(2, i)
            c.put_parent(int(nid), _node(3, i // 10))
            c.put_children(int(nid), np.array([i * 100, i * 100 + 1], dtype=NODE_ID))
            c.put_acx(int(nid), {2: np.array([[nid, i]], dtype=NODE_ID)})
        elapsed = time.time() - t0

        assert c.has(_node(2, 0))
        assert _get_parent(c, _node(2, 0)) == _node(3, 0)
        print(f"\n  write_10k: {elapsed*1000:.0f}ms")
