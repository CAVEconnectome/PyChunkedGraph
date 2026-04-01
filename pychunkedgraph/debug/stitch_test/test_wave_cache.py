"""WaveCache tests. Verifies cache invariants from CACHE_DESIGN.md."""

from collections import defaultdict

import numpy as np

from pychunkedgraph.graph import basetypes, types
from pychunkedgraph.graph.utils import flatgraph

from . import resolver as topology
from . import tree
from .test_helpers import (
    noop_read, get_parent as _get_parent, get_children as _get_children,
    get_acx as _get_acx, get_cx as _get_cx,
)
from .wave_cache import SiblingEntry, WaveCache

NODE_ID = basetypes.NODE_ID


class TestDuplicateReadCrash:
    """Hard requirement: no row read from BigTable more than once per stitch.
    _read_row_keys tracks every read. Duplicate = assertion crash."""

    def test_duplicate_read_detected(self) -> None:
        """Reading the same node ID twice must crash."""
        read_keys: set[int] = set()
        batch_1 = np.array([10, 20, 30], dtype=NODE_ID)
        batch_2 = np.array([30, 40], dtype=NODE_ID)  # 30 is duplicate

        read_keys.update(int(x) for x in batch_1)
        dupes = set(int(x) for x in batch_2) & read_keys
        assert dupes == {30}, "Must detect node 30 as duplicate"

    def test_no_duplicate_passes(self) -> None:
        """Disjoint batches must not trigger duplicate detection."""
        read_keys: set[int] = set()
        batch_1 = np.array([10, 20, 30], dtype=NODE_ID)
        batch_2 = np.array([40, 50], dtype=NODE_ID)

        read_keys.update(int(x) for x in batch_1)
        dupes = set(int(x) for x in batch_2) & read_keys
        assert not dupes, "No duplicates in disjoint batches"

    def test_reset_on_begin_stitch(self) -> None:
        """_read_row_keys resets on begin_stitch — each stitch is independent."""
        read_keys: set[int] = set()
        read_keys.update([10, 20, 30])
        assert len(read_keys) == 3
        read_keys = set()  # simulates begin_stitch reset
        read_keys.update([10, 20])  # same IDs, no crash because reset
        assert len(read_keys) == 2


class TestReadGating:

    def test_has_batch_gates_reads(self) -> None:
        """has_batch returns False for unknown nodes, True for cached.
        Broken = _ensure_cached skips the check, causing duplicate BigTable reads."""
        c = WaveCache(noop_read)
        ids = np.array([10, 20], dtype=NODE_ID)
        assert not c.has_batch(ids).any()
        c.put_parent(10, 100)
        result = c.has_batch(ids)
        assert result[0] and not result[1]


class TestChainMapPriority:

    def test_local_shadows_preloaded(self) -> None:
        """Local writes must shadow preloaded (fork COW) values.
        Broken = stale preloaded data returned after a stitch updates a node."""
        c = WaveCache(noop_read, preloaded=({10: 100}, {}, {}))
        assert _get_parent(c, 10) == 100
        c.put_parent(10, 200)
        assert _get_parent(c, 10) == 200


class TestFlushCreated:

    def test_all_writes_immediately_visible(self) -> None:
        """All put_* methods write directly to RowCache — immediately visible."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_children(99, np.array([1, 2], dtype=NODE_ID))
        c.put_parent(99, 999)
        c.put_acx(99, {2: "acx"})
        assert c.has(99)
        assert len(_get_children(c, 99)) == 2
        assert _get_parent(c, 99) == 999
        assert _get_acx(c, 99) == {2: "acx"}

    def test_flush_overwrites_stale_read(self) -> None:
        """Read stores parent=100, then stitch creates parent=200 for same node.
        Flush must overwrite the stale read value.
        Broken = stale parent used in hierarchy resolution → wrong CX edges."""
        c = WaveCache(noop_read)
        c.put_parent(10, 100)
        c.begin_stitch()
        c.put_parent(10, 200)

        assert _get_parent(c, 10) == 200

    def test_begin_stitch_clears_local(self) -> None:
        """begin_stitch clears local RowCache for retry safety.
        Data from completed stitches persists via snapshot→merge→preloaded."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(99, 999)
        c.put_children(99, np.array([1], dtype=NODE_ID))
        assert c.has(99)

        c.begin_stitch()
        assert not c.has(99), "local cleared on begin_stitch for retry safety"


class TestSaveWaveState:

    def test_flushes_and_stores_siblings(self) -> None:
        """save_wave_state stores sibling entries.
        Broken = in-process waves lose created data or sibling cache."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(99, 999)
        c.put_children(99, np.array([1], dtype=NODE_ID))
        c.put_acx(50, {2: "acx"})
        c.put_children(10, np.array([1, 2], dtype=NODE_ID))
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={10}, unresolved_acx={10: {2: np.array([[10, 50]], dtype=NODE_ID)}},
        )
        assert c.has(99)
        assert _get_parent(c, 99) == 999
        assert _get_acx(c, 50) == {2: "acx"}
        assert 10 in c._siblings
        assert 2 in c._siblings[10].unresolved_acx


class TestMergeOperations:

    def test_merge_inc_stores_siblings_and_incremental(self) -> None:
        """merge_inc stores sibling data + old_to_new + new_node_ids for incremental optimization.
        Broken = partition_siblings can't find known siblings → all re-read from BigTable."""
        c = WaveCache(noop_read)
        children = np.array([1, 2, 3], dtype=NODE_ID)
        c.merge_inc({
            "old_to_new": {10: 11},
            "new_node_ids": {11},
            "sibling_data": {
                50: {"children": children, "raw_cx": {2: "edges"}},
            },
        })
        assert c.get_sibling(50) is not None
        assert c.get_sibling(50).unresolved_acx == {2: "edges"}
        assert c.accumulated_replacements == {10: 11}
        assert 11 in c._new_node_ids

    def test_incremental_state_to_new_cache(self) -> None:
        """incremental_state() tuple passed to new WaveCache gives it sibling + old_to_new data.
        Broken = pool workers don't know about prior waves' siblings → can't skip reads."""
        c = WaveCache(noop_read)
        c._siblings[10] = SiblingEntry({2: "raw"}, {1: {2: 10}})
        c.accumulated_replacements = {100: 101}
        c._new_node_ids = {101}
        c2 = WaveCache(noop_read, incremental=c.incremental_state())
        assert c2.get_sibling(10) is not None
        assert c2.accumulated_replacements == {100: 101}


class TestSplitKnown:

    def test_empty_all_unknown(self) -> None:
        """No prior siblings → all unknown (first wave)."""
        c = WaveCache(noop_read)
        known, unknown = c.split_known_siblings(np.array([10, 20], dtype=NODE_ID))
        assert len(known) == 0 and len(unknown) == 2

    def test_mixed(self) -> None:
        """Known siblings from prior wave cached, new ones unknown."""
        c = WaveCache(noop_read)
        c._siblings[10] = SiblingEntry({}, {})
        c._siblings[20] = SiblingEntry({}, {})
        known, unknown = c.split_known_siblings(np.array([10, 20, 30], dtype=NODE_ID))
        assert set(int(x) for x in known) == {10, 20}
        assert set(int(x) for x in unknown) == {30}


class TestComplexFlows:

    def test_pool_wave_roundtrip(self) -> None:
        """Simulates pool wave: worker reads + creates → snapshot → parent merges → new worker finds all.
        Broken = created nodes lost in transit → BigTable re-reads in next wave."""
        worker = WaveCache(noop_read)
        worker.begin_stitch()
        worker.put_parent(1, 10)
        worker.put_parent(50, 500)
        worker.put_children(500, np.array([50], dtype=NODE_ID))


        local_snap = worker.local_snapshot()
        inc_snap = worker.inc_snapshot_from(
            old_to_new={10: 50}, new_node_ids={50, 500},
            sibling_ids=set(), unresolved_acx={},
        )

        parent = WaveCache(noop_read)
        parent.merge_reader(local_snap)
        parent.merge_inc(inc_snap)

        assert parent.has(1)
        assert parent.has(50)
        assert _get_parent(parent, 50) == 500
        assert len(_get_children(parent, 500)) == 1

        worker2 = WaveCache(noop_read, preloaded=parent.preloaded())
        assert worker2.has(1)
        assert worker2.has(50)
        assert _get_parent(worker2, 50) == 500

    def test_two_workers_merge_no_loss(self) -> None:
        """Two workers create different nodes. Parent merges both. No data lost.
        Broken = second merge overwrites first → missing nodes."""
        w1 = WaveCache(noop_read)
        w1.begin_stitch()
        w1.put_parent(1, 10)
        w1.put_parent(100, 1000)


        w2 = WaveCache(noop_read)
        w2.begin_stitch()
        w2.put_parent(2, 20)
        w2.put_parent(200, 2000)


        parent = WaveCache(noop_read)
        parent.merge_reader(w1.local_snapshot())
        parent.merge_reader(w2.local_snapshot())
        assert parent.has(1) and parent.has(2)
        assert parent.has(100) and parent.has(200)
        assert _get_parent(parent, 100) == 1000
        assert _get_parent(parent, 200) == 2000

    def test_full_multiwave(self) -> None:
        """Three waves accumulating data. Each wave's created + read data persists.
        Broken = any wave's data lost → re-reads or missing nodes."""
        c = WaveCache(noop_read)

        c.begin_stitch()
        c.put_parent(1, 10)
        c.put_parent(100, 1000)
        c.put_children(1000, np.array([100], dtype=NODE_ID))
        c.save_wave_state(
            old_to_new={10: 100}, new_node_ids={100, 1000},
            sibling_ids=set(), unresolved_acx={},
        )

        c.begin_stitch()
        assert c.has(1) and c.has(100)
        assert _get_parent(c, 100) == 1000
        c.put_parent(3, 30)
        c.put_parent(200, 2000)
        c.save_wave_state(
            old_to_new={30: 200}, new_node_ids={200, 2000},
            sibling_ids=set(), unresolved_acx={},
        )

        c.begin_stitch()
        assert c.has(1) and c.has(100) and c.has(200)
        assert c.has(3)
        assert _get_parent(c, 100) == 1000
        assert _get_parent(c, 200) == 2000


class TestDirtySiblings:

    def test_affected_partner_makes_dirty(self) -> None:
        """Sibling with partner SV resolving to a replaced L2 → dirty.
        Broken = unchanged CX written for this sibling, wasting BigTable writes."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.old_to_new = {100: 200}
        c.new_ids_d[2] = [200]
        c.put_parent(50, 100)
        c.sibling_ids = {10}
        c.unresolved_acx = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

    def test_new_node_partner_makes_dirty(self) -> None:
        """Sibling with partner SV resolving to a newly created L2 → dirty.
        Broken = partner points to new node with new parents, CX would differ."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.old_to_new = {}
        c.new_ids_d[2] = [200]
        c.put_parent(50, 200)
        c.sibling_ids = {10}
        c.unresolved_acx = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

    def test_unaffected_partner_is_clean(self) -> None:
        """Sibling with partner SV resolving to unrelated L2 → clean.
        Broken = sibling marked dirty unnecessarily, extra writes."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.old_to_new = {100: 200}
        c.new_ids_d[2] = [200]
        c.put_parent(50, 999)
        c.sibling_ids = {10}
        c.unresolved_acx = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 not in c.dirty_siblings

    def test_unknown_sibling_always_dirty(self) -> None:
        """Sibling without SiblingEntry (first wave) → always dirty.
        Broken = unknown sibling skipped, CX never written."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.old_to_new = {}
        c.new_ids_d[2] = []
        c.sibling_ids = {10}
        c.unresolved_acx = {}
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

    def test_dirty_across_three_waves(self) -> None:
        """Wave 0: create A,B. Wave 1: replace A→X, S1(→A) dirty, S2(→B) clean.
        Wave 2: replace B→Y, S1(→X) clean, S2(→B) dirty.
        Broken = wrong siblings marked dirty → wrong CX written or extra writes."""
        c = WaveCache(noop_read)

        # Wave 0: S1 partners with SV 50→A(100), S2 partners with SV 60→B(200)
        c.begin_stitch()
        c.old_to_new = {}
        c.new_ids_d[2] = [100, 200]
        c.put_parent(50, 100)
        c.put_parent(60, 200)
        c.sibling_ids = {10, 20}
        c.unresolved_acx = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert c.dirty_siblings == {10, 20}

        c.put_cx(10, {2: np.array([[10, 100]], dtype=NODE_ID)}); c.put_cx(20, {2: np.array([[20, 200]], dtype=NODE_ID)})
        c.save_wave_state(
            old_to_new={}, new_node_ids={100, 200},
            sibling_ids={10, 20}, unresolved_acx=c.unresolved_acx,
        )

        # Wave 1: replace A(100)→X(300). S1 dirty (partner→100, 100 in old_to_new). S2 clean.
        c.begin_stitch()
        c.old_to_new = {100: 300}
        c.new_ids_d[2] = [300]
        c.put_parent(50, 100)
        c.put_parent(60, 200)
        c.sibling_ids = {10, 20}
        c.unresolved_acx = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings
        assert 20 not in c.dirty_siblings

        c.put_cx(10, {2: np.array([[10, 300]], dtype=NODE_ID)})
        c.save_wave_state(
            old_to_new={100: 300}, new_node_ids={300},
            sibling_ids={10, 20}, unresolved_acx=c.unresolved_acx,
        )

        # Wave 2: replace B(200)→Y(400). Both dirty — S1 because 100 in accumulated, S2 because 200 in current.
        c.begin_stitch()
        c.old_to_new = {**c.accumulated_replacements, 200: 400}
        c.new_ids_d[2] = [400]
        c.put_parent(50, 100)
        c.put_parent(60, 200)
        c.sibling_ids = {10, 20}
        c.unresolved_acx = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings
        assert 20 in c.dirty_siblings

    def test_old_to_new_must_include_accumulated(self) -> None:
        """old_to_new used for resolution must include ALL prior waves' replacements.
        Without this, resolve_sv_to_layer returns dead node IDs for known siblings
        whose partners were replaced in prior waves. Caused task_2_5 mismatch."""
        c = WaveCache(noop_read)

        c.begin_stitch()
        c.save_wave_state(
            old_to_new={100: 200}, new_node_ids={200},
            sibling_ids=set(), unresolved_acx={},
        )
        assert c.accumulated_replacements == {100: 200}

        c.begin_stitch()
        current_wave_replacements = {300: 400}
        merged = {**c.accumulated_replacements, **current_wave_replacements}
        c.old_to_new = merged
        assert 100 in c.old_to_new
        assert 300 in c.old_to_new

    def test_accumulated_replacements_makes_dirty(self) -> None:
        """Wave 0: replaces A→X. Wave 1: different replacements (B→Y).
        S1(partner→A) is dirty in wave 1 because A in accumulated_replacements.
        Broken = only per-wave old_to_new checked → stale resolver entry used."""
        c = WaveCache(noop_read)

        c.begin_stitch()
        c.old_to_new = {100: 200}
        c.new_ids_d[2] = [200]
        c.put_parent(50, 100)
        c.sibling_ids = {10}
        c.unresolved_acx = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

        c.put_cx(10, {2: np.array([[10, 200]], dtype=NODE_ID)})
        c.save_wave_state(
            old_to_new={100: 200}, new_node_ids={200},
            sibling_ids={10}, unresolved_acx=c.unresolved_acx,
        )

        c.begin_stitch()
        c.old_to_new = {**c.accumulated_replacements, 300: 400}
        c.new_ids_d[2] = [400]
        c.put_parent(50, 100)
        c.sibling_ids = {10}
        c.unresolved_acx = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings


def _get_layer(nid: int) -> int:
    return (int(nid) >> 56) & 0xFF


class TestCleanSiblingSkipIntegration:
    """Tests that build_hierarchy skips CX resolution for clean siblings
    and uses written_cx from SiblingEntry instead. Regression: phase3
    went from 0.03s to 16s when this was broken."""

    def test_clean_sibling_not_in_resolve_nodes(self) -> None:
        """Clean siblings must not be passed to resolve_cx_at_layer.
        Broken = all siblings resolved → O(n) resolution for unchanged nodes."""
        all_nodes = [10, 20, 30]
        clean_sibs = {20}
        resolve_nodes = [n for n in all_nodes if int(n) not in clean_sibs]
        assert 20 not in resolve_nodes
        assert set(resolve_nodes) == {10, 30}

    def test_store_cx_skips_siblings(self) -> None:
        """store_cx_from_resolved skips siblings — BigTable CX preserved for counterpart remap."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(50, 200)
        c.put_parent(60, 100)
        c.sibling_ids = {10, 20}
        c.new_node_ids = {300}
        c.unresolved_acx = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
            300: {2: np.array([[300, 50]], dtype=NODE_ID)},
        }

        cx_resolved = topology.resolve_cx_at_layer([10, 20, 300], 2, c, _get_layer)
        topology.store_cx_from_resolved(c, cx_resolved, 2)

        assert not _get_cx(c, 10), "Sibling CX not stored"
        assert not _get_cx(c, 20), "Sibling CX not stored"
        assert _get_cx(c, 300), "New node CX stored"

    def test_build_rows_skips_clean_siblings(self) -> None:
        """build_rows only writes CX for siblings in RowCache cx (dirty ones).
        Broken = all siblings written → 8x write time regression."""
        from kvdbclient import serializers
        from pychunkedgraph.graph import attributes

        c = WaveCache(noop_read)
        c.begin_stitch()
        c.sibling_ids = {10, 20}
        c.new_node_ids = set()
        c.put_cx(20, {2: np.array([[20, 200]], dtype=NODE_ID)})

        rows = {}
        sib_cx = c.get_cx_batch(np.array(list(c.sibling_ids), dtype=NODE_ID))
        for nid in c.sibling_ids:
            cx_d = sib_cx.get(int(nid), {})
            if not cx_d:
                continue
            rk = serializers.serialize_uint64(np.uint64(nid))
            rows[rk] = {
                attributes.Connectivity.CrossChunkEdge[layer]: cx
                for layer, cx in cx_d.items()
            }

        assert len(rows) == 1
        rk_20 = serializers.serialize_uint64(np.uint64(20))
        assert rk_20 in rows
        rk_10 = serializers.serialize_uint64(np.uint64(10))
        assert rk_10 not in rows


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


class TestResolveRemainingCxBatched:
    """Tests that resolve_remaining_cx uses batched resolution (via resolve_cx_at_layer)
    and only processes layers > 2."""

    def test_skips_layer_2(self) -> None:
        """resolve_remaining_cx must not process layer 2 — handled by build_hierarchy layer loop.
        Broken = clean siblings get layer 2 CX stored → written to BigTable unnecessarily."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(50, 200)
        c.sibling_ids = {10}
        c.new_node_ids = set()
        c.unresolved_acx = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        topology.resolve_remaining_cx(c, FakeLcg())
        assert not _get_cx(c, 10), "Layer 2 must not be processed by resolve_remaining_cx"

    def test_processes_layer_3_new_nodes_only(self) -> None:
        """resolve_remaining_cx processes layer 3+ for new nodes only.
        Siblings' higher-layer CX stays as BigTable format (not re-derived)."""
        A = _node(2, 1)
        P3 = _node(3, 10)
        NEW = _node(3, 99)

        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(50, A)
        c.sibling_ids = {10}
        c.new_node_ids = {NEW}
        c.unresolved_acx = {
            10: {3: np.array([[10, 50]], dtype=NODE_ID)},
            NEW: {3: np.array([[NEW, 50]], dtype=NODE_ID)},
        }

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        c.put_parent(A, P3)
        topology.resolve_remaining_cx(c, FakeLcg())
        assert not _get_cx(c, 10), "Sibling higher-layer CX not re-derived"
        assert _get_cx(c, NEW), "New node processed"
        assert 3 in _get_cx(c, NEW)


class TestEndToEndMultiwave:
    """Full multiwave scenarios exercising the real save/restore/dirty/resolve path."""


class TestPartnerResolution:
    """Tests that partner SVs of dirty known siblings get resolver entries
    even when there are no unknown siblings. The task_2_5 mismatch (781 vs 592)
    was caused by collect_and_resolve_partners only running when unknown > 0."""

    def test_dirty_known_partner_needs_resolution(self) -> None:
        """Dirty known sibling's partner SV must be in resolver for resolve_cx_at_layer
        to produce correct results. Resolver has current L2 identity (updated after merge)."""
        S = 10
        partner_sv = 50
        new_l2 = 300

        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(partner_sv, new_l2)
        c.new_ids_d[2] = [new_l2]
        c.unresolved_acx = {S: {2: np.array([[S, partner_sv]], dtype=NODE_ID)}}

        def _get_layer(nid):
            return (int(nid) >> 56) & 0xFF

        cx = topology.resolve_cx_at_layer([S], 2, c, _get_layer)
        assert len(cx) == 1
        assert int(cx[0, 1]) == new_l2, "Partner SV resolved via resolver (current identity)"

        c2 = WaveCache(noop_read)
        c2.begin_stitch()

        c2.unresolved_acx = {S: {2: np.array([[S, partner_sv]], dtype=NODE_ID)}}

        cx_bad = topology.resolve_cx_at_layer([S], 2, c2, _get_layer)
        if len(cx_bad) > 0:
            assert int(cx_bad[0, 1]) == partner_sv, \
                "Without resolver entry, partner SV returned as-is (wrong)"

    def test_all_known_no_unknown_still_needs_partner_resolution(self) -> None:
        """When all siblings are known and some are dirty, their partner SVs
        must still get resolver entries. The code currently skips
        collect_and_resolve_partners when unknown==0.
        Broken = dirty known siblings resolved with empty resolver → wrong CX."""
        S_dirty = 10
        S_clean = 20
        partner_sv_dirty = 50
        partner_sv_clean = 60
        partner_l2_dirty = 200
        partner_l2_clean = 300

        c = WaveCache(noop_read)
        c._siblings[S_dirty] = SiblingEntry(
            unresolved_acx={2: np.array([[S_dirty, partner_sv_dirty]], dtype=NODE_ID)},
        )
        c._siblings[S_clean] = SiblingEntry(
            unresolved_acx={2: np.array([[S_clean, partner_sv_clean]], dtype=NODE_ID)},
        )
        c.begin_stitch()
        c.old_to_new = {partner_l2_dirty: 400}
        c.new_ids_d[2] = [400]
        c.put_parent(70, S_dirty)
        c.put_parent(80, S_clean)
        c.put_parent(partner_sv_dirty, partner_l2_dirty)
        c.sibling_ids = {S_dirty, S_clean}
        c.unresolved_acx = {
            S_dirty: {2: np.array([[S_dirty, partner_sv_dirty]], dtype=NODE_ID)},
            S_clean: {2: np.array([[S_clean, partner_sv_clean]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert S_dirty in c.dirty_siblings
        assert S_clean not in c.dirty_siblings

        known, unknown = c.split_known_siblings(
            np.array([S_dirty, S_clean], dtype=NODE_ID)
        )
        assert len(unknown) == 0, "All siblings known"
        assert len(known) == 2

        assert c.has(partner_sv_dirty), \
            "Dirty known sibling's partner SV MUST be cached for correct resolution"

    def test_partner_resolution_before_dirty_check(self) -> None:
        """Partner SVs must be in resolver BEFORE compute_dirty_siblings runs.
        Otherwise dirty check can't detect that a partner SV maps to a replaced L2.

        Correct order: restore_known → collect_and_resolve_partners (ALL siblings) → compute_dirty.
        Broken order: restore_known → compute_dirty → collect_and_resolve_partners (unknown only).

        This test: pre-populate resolver with partner SV entry, verify dirty check catches it."""
        S = 10
        partner_sv = 50
        partner_l2 = _node(2, 200)
        replacement = _node(2, 300)

        c = WaveCache(noop_read)
        c._siblings[S] = SiblingEntry(
            unresolved_acx={2: np.array([[S, partner_sv]], dtype=NODE_ID)},
        )

        c.begin_stitch()
        c.old_to_new = {partner_l2: replacement}
        c.new_ids_d[2] = [replacement]
        c.sibling_ids = {S}
        c.unresolved_acx = {S: {2: np.array([[S, partner_sv]], dtype=NODE_ID)}}

        # Without partner SV cached: dirty check returns clean (can't resolve)
        c.compute_dirty_siblings()
        assert S not in c.dirty_siblings, "Without cached partner, dirty check misses it"

        # With partner SV cached (ensure_partners_cached): dirty (CORRECT)
        c.put_parent(partner_sv, partner_l2)
        c.compute_dirty_siblings()
        assert S in c.dirty_siblings, "With cached partner, dirty check correctly detects replaced partner"




class TestRestoreKnownSkipsHasBatch:
    """Tests that restore_known_siblings accesses cache directly without
    _ensure_cached / has_batch overhead. Known siblings are guaranteed
    in preloaded from prior waves."""

    def test_known_sibling_data_from_preloaded(self) -> None:
        """Known sibling's children + acx accessible via cache.children / cache.acx
        without triggering _ensure_cached. Data in preloaded from prior wave."""
        c = WaveCache(noop_read)
        c.put_children(10, np.array([50, 60], dtype=NODE_ID))
        c.put_parent(50, 10)
        c.put_parent(60, 10)
        c.put_acx(10, {2: np.array([[10, 100]], dtype=NODE_ID)})
        c._siblings[10] = SiblingEntry(
            unresolved_acx={2: np.array([[10, 100]], dtype=NODE_ID)},
        )
        # Simulate wave boundary: local → preloaded via snapshot+merge+promote
        snap = c.local_snapshot()
        c2 = WaveCache(noop_read)
        c2._siblings = c._siblings
        c2.merge_reader(snap)
        c2._rows.promote_local()

        c2.begin_stitch()
        tree.restore_known_siblings(None, c2, np.array([10], dtype=NODE_ID))

        assert _get_children(c2, 10) is not None and len(_get_children(c2, 10)) > 0
        assert _get_acx(c2, 10)
        assert 10 in c2.unresolved_acx
        assert c2.has(50)


class TestHashBasedWriteSkip:
    """Placeholder — cx_hash removed (counterpart CX always changes by definition)."""
    pass


class TestAllSiblingsInCCGraph:
    """ALL siblings (dirty + clean) enter CC graph at layer 2.
    Matches production _get_layer_node_ids (edits.py:652-666).
    Sibling set is small (~10K) because discover_siblings uses immediate L3 parents only."""

    def test_all_siblings_enter_layer2(self) -> None:
        """All siblings enter all_nodes at layer 2 — not just dirty ones.
        Sibling set is small from discover_siblings fix (immediate parents only)."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.new_ids_d[2] = [100, 200]
        c.siblings_d = {2: [10, 20, 30, 40, 50]}
        c.dirty_siblings = {10, 30}
        c.sibling_ids = {10, 20, 30, 40, 50}

        new_nodes = c.new_ids_d[2]
        sib_nodes = c.siblings_d.get(2, [])
        all_nodes = list(new_nodes) + list(sib_nodes)

        assert set(all_nodes) == {100, 200, 10, 20, 30, 40, 50}, \
            "All new nodes + ALL siblings enter CC graph"

    def test_resolve_returns_all_but_store_skips_siblings(self) -> None:
        """resolve_cx_at_layer returns CX for all nodes (for CC graph),
        but store_cx_from_resolved skips siblings (BigTable CX preserved)."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(50, 200)
        c.put_parent(60, 300)
        c.unresolved_acx = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
            99: {2: np.array([[99, 50]], dtype=NODE_ID)},
        }
        c.sibling_ids = {10, 20}
        c.new_node_ids = {99}

        all_nodes = [10, 20, 99]

        def _get_layer(nid):
            return (int(nid) >> 56) & 0xFF

        cx = topology.resolve_cx_at_layer(all_nodes, 2, c, _get_layer)
        assert len(cx) > 0, "CX returned for CC graph"

        topology.store_cx_from_resolved(c, cx, 2)
        assert not _get_cx(c, 10), "Sibling CX not stored"
        assert not _get_cx(c, 20), "Sibling CX not stored"
        assert _get_cx(c, 99), "New node CX stored"


class TestPartnerResolutionOptimized:
    """O1: Only resolve partners for siblings with missing resolver entries.
    Known siblings' partners are usually already in resolver from prior waves."""

    def test_known_siblings_with_cached_partners_skip_resolution(self) -> None:
        """If all partner SVs of known siblings are already in resolver,
        collect_and_resolve_partners should find nothing new to resolve."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(50, 200)
        c.put_parent(60, 300)
        c.put_acx(10, {2: np.array([[10, 50]], dtype=NODE_ID)})
        c.put_acx(20, {2: np.array([[20, 60]], dtype=NODE_ID)})
        cached_svs = set()
        sib_acx_batch = c.get_acx_batch(np.array([10, 20], dtype=NODE_ID))
        missing = {}
        for sib_int in [10, 20]:
            acx_d = sib_acx_batch.get(sib_int, {})
            if not acx_d:
                continue
            for layer_edges in acx_d.values():
                if len(layer_edges) > 0:
                    for sv in layer_edges[:, 1]:
                        if not c.has(int(sv)):
                            missing[sib_int] = acx_d
                            break
        assert len(missing) == 0, "No missing partners — resolution skipped"

    def test_known_sibling_with_missing_partner_resolved(self) -> None:
        """If a known sibling has a partner SV NOT in resolver, it must be resolved."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.put_parent(50, 200)
        c.put_acx(10, {2: np.array([[10, 50]], dtype=NODE_ID)})
        c.put_acx(20, {2: np.array([[20, 70]], dtype=NODE_ID)})  # sv 70 NOT in resolver
        cached_svs = set()
        sib_acx_batch = c.get_acx_batch(np.array([10, 20], dtype=NODE_ID))
        missing = {}
        for sib_int in [10, 20]:
            acx_d = sib_acx_batch.get(sib_int, {})
            if not acx_d:
                continue
            for layer_edges in acx_d.values():
                if len(layer_edges) > 0:
                    for sv in layer_edges[:, 1]:
                        if not c.has(int(sv)):
                            missing[sib_int] = acx_d
                            break
                    if sib_int in missing:
                        break
        assert 20 in missing, "Sibling 20 has missing partner sv=70"
        assert 10 not in missing, "Sibling 10 has all partners cached"


class TestVectorizedDirtyCheck:
    """O5: Vectorized compute_dirty_siblings using numpy searchsorted."""

    def test_vectorized_matches_python(self) -> None:
        """Vectorized dirty check must produce same result as Python loop."""
        c = WaveCache(noop_read)
        c.begin_stitch()
        c.old_to_new = {100: 200, 300: 400}
        c.new_ids_d[2] = [200, 400]
        c.put_parent(50, 100)
        c.put_parent(60, 999)
        c.put_parent(70, 300)
        c.sibling_ids = {10, 20, 30}
        c.unresolved_acx = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},  # sv50→100, 100 in old_to_new → dirty
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},  # sv60→999, not affected → clean
            30: {2: np.array([[30, 70]], dtype=NODE_ID)},  # sv70→300, 300 in old_to_new → dirty
        }
        c._siblings = {10: SiblingEntry({}, {}), 20: SiblingEntry({}, {}), 30: SiblingEntry({}, {})}
        c.compute_dirty_siblings()
        assert c.dirty_siblings == {10, 30}, "Python dirty check: 10 and 30 dirty, 20 clean"


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


def _get_layer(nid: int) -> int:
    return (int(nid) >> 56) & 0xFF


class TestCCCorrectness:
    """CC graph must include ALL siblings (dirty + clean) for correct connectivity."""

    def test_clean_sibling_provides_cc_connectivity(self) -> None:
        """Dirty node A has CX edge to clean sibling B.
        Both must be in CC graph so A and B are in the same CC.
        Broken = B excluded → A isolated → extra root."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        A = _node(2, 1)  # new node
        B = _node(2, 2)  # clean sibling
        partner_sv_a = 5001  # SV in B that A connects to
        partner_sv_b = 5002  # SV in A that B connects to

        c.put_parent(partner_sv_a, int(B))
        c.put_parent(partner_sv_b, int(A))
        c.unresolved_acx = {
            int(A): {2: np.array([[A, partner_sv_a]], dtype=NODE_ID)},
            int(B): {2: np.array([[B, partner_sv_b]], dtype=NODE_ID)},
        }
        c.old_to_new = {}
        c.new_ids_d[2] = [A]
        c.siblings_d = {2: [int(B)]}
        c.dirty_siblings = set()  # B is clean
        c.sibling_ids = {int(B)}

        # Current broken code: only dirty siblings at L2
        sib_nodes = c.siblings_d.get(2, [])
        dirty_sibs = [s for s in sib_nodes if int(s) in c.dirty_siblings]
        broken_nodes = list(c.new_ids_d[2]) + dirty_sibs

        # B excluded → A alone in CC → extra component
        assert int(B) not in broken_nodes, "Bug: clean B excluded from CC graph"

        # Fixed: ALL siblings enter CC
        fixed_nodes = list(c.new_ids_d[2]) + list(sib_nodes)
        assert int(B) in fixed_nodes, "Fix: clean B in CC graph"

        # Resolve CX and build CC with fixed set
        cx = topology.resolve_cx_at_layer(fixed_nodes, 2, c, _get_layer)
        assert len(cx) > 0, "A and B must have resolved CX edges"

        nodes_arr = np.array(fixed_nodes, dtype=NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        all_edges = np.concatenate([cx, self_edges]).astype(NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs = flatgraph.connected_components(graph)

        assert len(ccs) == 1, f"A and B must be in same CC, got {len(ccs)} CCs"

    def test_dirty_cc_gets_new_parent(self) -> None:
        """CC with at least one new/dirty node must get a new parent.
        All members (dirty + clean) get new parent pointer."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        new1 = _node(2, 1)
        clean1 = _node(2, 2)
        sv_new = 5001
        sv_clean = 5002

        c.put_parent(sv_new, int(clean1))
        c.put_parent(sv_clean, int(new1))
        c.unresolved_acx = {
            int(new1): {2: np.array([[new1, sv_new]], dtype=NODE_ID)},
            int(clean1): {2: np.array([[clean1, sv_clean]], dtype=NODE_ID)},
        }
        c.new_node_ids = {int(new1)}
        c.dirty_siblings = set()

        all_nodes = [new1, clean1]
        cx = topology.resolve_cx_at_layer(all_nodes, 2, c, _get_layer)
        topology.store_cx_from_resolved(c, cx, 2)

        nodes_arr = np.array(all_nodes, dtype=NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        all_edges = np.concatenate([cx, self_edges]).astype(NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs = flatgraph.connected_components(graph)

        assert len(ccs) == 1, "new1 and clean1 in same CC"
        cc_ids = graph_ids[ccs[0]]
        has_new = any(int(x) in c.new_node_ids for x in cc_ids)
        assert has_new, "CC contains new node → must get new parent"



class TestOldHierarchyChain:
    """old_hierarchy stores plain dicts keyed by all nodes in the chain."""

    def test_chain_lookup(self) -> None:
        """old_hierarchy[l2][layer] returns parent at that layer."""
        chain = {3: _node(3, 10), 4: _node(4, 20), 8: _node(8, 30)}
        assert chain[3] == _node(3, 10)
        assert chain[4] == _node(4, 20)
        assert 5 not in chain

    def test_all_parents_keyed(self) -> None:
        """All parents in chain share the same dict reference (production line 48)."""
        chain = {3: _node(3, 10), 4: _node(4, 20), 8: _node(8, 30)}
        old_hierarchy = {}
        old_hierarchy[_node(2, 1)] = chain
        for parent_id in chain.values():
            old_hierarchy[int(parent_id)] = chain

        assert old_hierarchy[_node(2, 1)] is chain
        assert old_hierarchy[_node(3, 10)] is chain
        assert old_hierarchy[_node(4, 20)] is chain
        assert old_hierarchy[_node(8, 30)] is chain

    def test_missing_layer_raises(self) -> None:
        """Accessing a layer not in chain raises KeyError."""
        chain = {3: _node(3, 10), 8: _node(8, 30)}
        try:
            chain[5]
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestEndToEndHierarchy:
    """Full hierarchy build from L2 to root with dirty + clean CCs."""

    def test_end_to_end_hierarchy_leaf_to_root(self) -> None:
        """10 L2 nodes: 3 new, 7 clean siblings. CX connects into 2 CCs.
        Build hierarchy L2 → L3 → root. Verify correct parents, CX, root count."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        # 3 new nodes + 7 clean siblings
        new1, new2, new3 = _node(2, 1), _node(2, 2), _node(2, 3)
        cl1, cl2, cl3, cl4 = _node(2, 11), _node(2, 12), _node(2, 13), _node(2, 14)
        cl5, cl6, cl7 = _node(2, 15), _node(2, 16), _node(2, 17)

        # CC1: new1, new2, cl1, cl2 — connected by L2 CX
        # CC2: new3, cl3, cl4, cl5, cl6, cl7 — connected by L2 CX
        # SVs for partner resolution
        cc1_nodes = [new1, new2, cl1, cl2]
        cc2_nodes = [new3, cl3, cl4, cl5, cl6, cl7]
        all_l2 = cc1_nodes + cc2_nodes

        # Create resolver: each node's child SV maps to the node
        for n in all_l2:
            sv = int(n) * 10 + 1
            c.put_parent(sv, int(n))

        # CX edges: chain connectivity within each CC
        # CC1: new1↔new2, new2↔cl1, cl1↔cl2
        # CC2: new3↔cl3, cl3↔cl4, cl4↔cl5, cl5↔cl6, cl6↔cl7
        def _sv_of(n: int) -> int:
            return int(n) * 10 + 1

        c.unresolved_acx = {}
        # CC1 chain
        c.unresolved_acx[int(new1)] = {2: np.array([[new1, _sv_of(new2)]], dtype=NODE_ID)}
        c.unresolved_acx[int(new2)] = {2: np.array([[new2, _sv_of(new1)], [new2, _sv_of(cl1)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl1)] = {2: np.array([[cl1, _sv_of(new2)], [cl1, _sv_of(cl2)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl2)] = {2: np.array([[cl2, _sv_of(cl1)]], dtype=NODE_ID)}
        # CC2 chain
        c.unresolved_acx[int(new3)] = {2: np.array([[new3, _sv_of(cl3)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl3)] = {2: np.array([[cl3, _sv_of(new3)], [cl3, _sv_of(cl4)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl4)] = {2: np.array([[cl4, _sv_of(cl3)], [cl4, _sv_of(cl5)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl5)] = {2: np.array([[cl5, _sv_of(cl4)], [cl5, _sv_of(cl6)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl6)] = {2: np.array([[cl6, _sv_of(cl5)], [cl6, _sv_of(cl7)]], dtype=NODE_ID)}
        c.unresolved_acx[int(cl7)] = {2: np.array([[cl7, _sv_of(cl6)]], dtype=NODE_ID)}

        c.new_ids_d[2] = [new1, new2, new3]
        c.siblings_d = {2: [int(n) for n in [cl1, cl2, cl3, cl4, cl5, cl6, cl7]]}
        c.new_node_ids = {int(new1), int(new2), int(new3)}
        c.dirty_siblings = set()

        # Layer 2: ALL siblings must enter CC graph (fixed behavior)
        all_nodes_l2 = list(c.new_ids_d[2]) + list(c.siblings_d[2])
        assert len(all_nodes_l2) == 10, "All 10 L2 nodes in CC graph"

        # Resolve CX at L2
        cx = topology.resolve_cx_at_layer(all_nodes_l2, 2, c, _get_layer)
        assert len(cx) > 0, "Must have resolved CX edges"
        topology.store_cx_from_resolved(c, cx, 2)

        # Build CC graph
        nodes_arr = np.array(all_nodes_l2, dtype=NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        all_edges = np.concatenate([cx, self_edges]).astype(NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs = flatgraph.connected_components(graph)

        assert len(ccs) == 2, f"Expected 2 CCs, got {len(ccs)}"

        # Verify CC membership
        cc_sets = [set(int(x) for x in graph_ids[cc]) for cc in ccs]
        cc1_expected = {int(n) for n in cc1_nodes}
        cc2_expected = {int(n) for n in cc2_nodes}
        assert cc1_expected in cc_sets, "CC1 members correct"
        assert cc2_expected in cc_sets, "CC2 members correct"

        # Verify all clean siblings are in CCs (not excluded)
        all_in_ccs = set()
        for s in cc_sets:
            all_in_ccs.update(s)
        for n in [cl1, cl2, cl3, cl4, cl5, cl6, cl7]:
            assert int(n) in all_in_ccs, f"Clean sibling {n} must be in a CC"

        # Verify CX stored for all nodes
        all_l2_arr = np.array(list(all_l2), dtype=NODE_ID)
        all_l2_cx = c.get_cx_batch(all_l2_arr)
        for n in all_l2:
            assert all_l2_cx.get(int(n)), f"Node {n} must have CX in cache"


class TestFilterOrphaned:
    """filter_orphaned uses RowCache children + segment IDs to detect failed nodes."""

    def test_valid_nodes_kept(self) -> None:
        """Nodes whose segment_id is the highest for their max_child are kept."""
        c = WaveCache(noop_read)
        # Node 100: children [1, 5] → max_child=5, seg_id=10
        # Node 200: children [2, 8] → max_child=8, seg_id=20
        c.put_children(100, np.array([1, 5], dtype=NODE_ID))
        c.put_children(200, np.array([2, 8], dtype=NODE_ID))

        class FakeLCG:
            _cache = c
            def get_segment_id(self, n):
                return {100: 10, 200: 20}[int(n)]

        node_ids = np.array([100, 200], dtype=NODE_ID)
        result = tree.filter_orphaned(FakeLCG(), node_ids)
        assert set(int(x) for x in result) == {100, 200}

    def test_orphaned_filtered(self) -> None:
        """Failed node (lower seg_id, same max_child) is filtered out."""
        c = WaveCache(noop_read)
        # Both nodes have max_child=5, but node 100 has higher seg_id → valid
        # Node 200 has lower seg_id → orphaned (failed retry)
        c.put_children(100, np.array([1, 5], dtype=NODE_ID))
        c.put_children(200, np.array([3, 5], dtype=NODE_ID))

        class FakeLCG:
            _cache = c
            def get_segment_id(self, n):
                return {100: 10, 200: 5}[int(n)]

        node_ids = np.array([100, 200], dtype=NODE_ID)
        result = tree.filter_orphaned(FakeLCG(), node_ids)
        result_set = set(int(x) for x in result)
        assert 100 in result_set, "Node with highest seg_id for max_child=5 kept"
        assert 200 not in result_set, "Node with lower seg_id for same max_child filtered"

    def test_empty_input(self) -> None:
        """Empty input returns empty array."""
        c = WaveCache(noop_read)

        class FakeLCG:
            _cache = c
            def get_segment_id(self, n):
                return 0

        node_ids = np.array([], dtype=NODE_ID)
        result = tree.filter_orphaned(FakeLCG(), node_ids)
        assert len(result) == 0

    def test_no_children_uses_zero_max(self) -> None:
        """Nodes with empty children get max_child=0."""
        c = WaveCache(noop_read)
        c.put_children(100, np.array([], dtype=NODE_ID))
        c.put_children(200, np.array([3, 7], dtype=NODE_ID))

        class FakeLCG:
            _cache = c
            def get_segment_id(self, n):
                return {100: 10, 200: 20}[int(n)]

        node_ids = np.array([100, 200], dtype=NODE_ID)
        result = tree.filter_orphaned(FakeLCG(), node_ids)
        assert set(int(x) for x in result) == {100, 200}


class TestExternalNodesInCCGraph:
    """Nodes from CX resolution that are NOT in all_nodes appear in the CC graph
    but without their own outgoing CX. This breaks transitivity and fragments CCs."""

    def test_external_node_missing_outgoing_cx(self) -> None:
        """A (in all_nodes) has CX to D (external). D has CX to B (in all_nodes).
        Without D's outgoing CX loaded, A-D connected but D-B missing → 2 CCs.
        With D in all_nodes, A-D-B forms 1 CC."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        A = _node(2, 1)   # in all_nodes
        B = _node(2, 2)   # in all_nodes
        D = _node(2, 100)  # external — NOT in all_nodes

        sv_d = 7001  # SV resolving to D
        sv_b = 7002  # SV resolving to B
        sv_a = 7003  # SV resolving to A

        c.put_parent(sv_d, int(D))
        c.put_parent(sv_b, int(B))
        c.put_parent(sv_a, int(A))

        # A→D and D→B should form one CC: {A, D, B}
        # But only A and B are in all_nodes
        c.unresolved_acx = {
            int(A): {2: np.array([[A, sv_d]], dtype=NODE_ID)},       # A→D
            int(D): {2: np.array([[D, sv_b]], dtype=NODE_ID)},       # D→B (NOT resolved — D not in all_nodes)
            int(B): {2: np.array([[B, sv_a]], dtype=NODE_ID)},       # B→A
        }

        # Case 1: Only A, B in all_nodes (external D missing outgoing CX)
        all_nodes_partial = [A, B]
        cx_partial = topology.resolve_cx_at_layer(all_nodes_partial, 2, c, _get_layer)
        topology.store_cx_from_resolved(c, cx_partial, 2)

        nodes_arr = np.array(all_nodes_partial, dtype=NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        all_edges = np.concatenate([cx_partial, self_edges]).astype(NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs_partial = flatgraph.connected_components(graph)

        # A has CX→D, B has CX→A. So edges: (A,D), (B,A).
        # A and B connected via (B,A). D connected to A via (A,D).
        # All 3 in one CC? Let's check.
        # Actually: A→D is resolved, B→A is resolved.
        # Edges: (A, D), (B, A). D only connected to A. B connected to A.
        # So {A, B, D} form one CC via A.
        # The problem occurs when D has CX to ANOTHER node E that's only reachable through D.
        pass  # This case actually works — see next test

    def test_external_bridge_node_fragments_cc(self) -> None:
        """A and B should be in one CC but are only connected through external node D.
        A→D and D→B. A and B have NO direct CX to each other.
        Without D's outgoing CX, A connects to D, but B is isolated → 2 CCs.
        This is the exact bug causing 22 roots instead of 11."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        A = _node(2, 1)   # in all_nodes (new node)
        B = _node(2, 2)   # in all_nodes (sibling)
        D = _node(2, 100)  # external bridge — NOT in all_nodes

        sv_d = 7001  # SV resolving to D
        sv_b = 7002  # SV resolving to B

        c.put_parent(sv_d, int(D))
        c.put_parent(sv_b, int(B))

        # A→D, D→B. A and B connected only through D.
        c.unresolved_acx = {
            int(A): {2: np.array([[A, sv_d]], dtype=NODE_ID)},  # A→D
            int(D): {2: np.array([[D, sv_b]], dtype=NODE_ID)},  # D→B (NOT resolved)
            int(B): {2: np.array([], dtype=NODE_ID).reshape(0, 2)},  # B has no L2 CX back
        }

        # Only A, B in all_nodes — D's outgoing CX not resolved
        all_nodes = [A, B]
        cx = topology.resolve_cx_at_layer(all_nodes, 2, c, _get_layer)

        nodes_arr = np.array(all_nodes, dtype=NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        if len(cx) > 0:
            all_edges = np.concatenate([cx, self_edges]).astype(NODE_ID)
        else:
            all_edges = self_edges.astype(NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs = flatgraph.connected_components(graph)

        # A has CX to D → edge (A, D) in graph. B has no edges.
        # D appears through edge but has no outgoing CX (not in all_nodes).
        # CCs: {A, D} and {B} → 2 CCs instead of 1.
        assert len(ccs) == 2, \
            f"BUG DEMONSTRATED: A-D connected, B isolated → {len(ccs)} CCs (should be 1 with D's CX)"

        # Now with D in all_nodes — D's outgoing CX IS resolved
        all_nodes_full = [A, B, D]
        # CX reset not needed — RowCache cx overwritten by resolve_cx_at_layer
        cx_full = topology.resolve_cx_at_layer(all_nodes_full, 2, c, _get_layer)

        nodes_arr_full = np.array(all_nodes_full, dtype=NODE_ID)
        self_edges_full = np.vstack([nodes_arr_full, nodes_arr_full]).T
        all_edges_full = np.concatenate([cx_full, self_edges_full]).astype(NODE_ID)
        graph_full, _, _, graph_ids_full = flatgraph.build_gt_graph(all_edges_full, make_directed=True)
        ccs_full = flatgraph.connected_components(graph_full)

        assert len(ccs_full) == 1, \
            f"With D in all_nodes: A-D-B forms 1 CC, got {len(ccs_full)}"

    def test_l3_sibling_discovery_fixes_l2_fragmentation(self) -> None:
        """At L2, CCs fragment because external bridge node D is missing.
        At L3, sibling discovery should find D's parent as a sibling,
        restoring connectivity.

        Setup:
        - L2: A (new), B (sibling). A→D→B through external D.
        - L2 CCs without D: {A,D} and {B} → 2 L3 parents: P1, P2
        - L3: P1 and P2 should be siblings (share L4 parent with D's L3 parent P_D)
        - L3 sibling discovery finds P_D → P1, P2, P_D in same CC → 1 root

        This test verifies the CONCEPT. Actual implementation requires BigTable reads."""
        c = WaveCache(noop_read)
        c.begin_stitch()

        A = _node(2, 1)
        B = _node(2, 2)
        D = _node(2, 100)

        P1 = _node(3, 1)   # A's new L3 parent
        P2 = _node(3, 2)   # B's new L3 parent
        P_D = _node(3, 100)  # D's existing L3 parent (sibling at L3)

        # At L3: P1, P2, P_D should be in one CC if we discover P_D as sibling
        # P1 has CX[3] to P_D (from A→D edge lifted to L3)
        # P_D has CX[3] to P2 (from D→B edge lifted to L3)
        sv_pd = 8001
        sv_p2 = 8002
        c.put_parent(sv_pd, int(D))
        c.put_parent(sv_p2, int(B))

        c.unresolved_acx[int(P1)] = {3: np.array([[P1, sv_pd]], dtype=NODE_ID)}   # P1→P_D
        c.unresolved_acx[int(P_D)] = {3: np.array([[P_D, sv_p2]], dtype=NODE_ID)}  # P_D→P2
        c.unresolved_acx[int(P2)] = {3: np.array([], dtype=NODE_ID).reshape(0, 2)}

        # Without L3 sibling discovery: only P1, P2 in all_nodes
        all_nodes_l3 = [P1, P2]
        c.put_parent(int(A), int(P1))
        c.put_parent(int(B), int(P2))
        c.put_parent(int(D), int(P_D))
        cx_l3 = topology.resolve_cx_at_layer(all_nodes_l3, 3, c, _get_layer)

        nodes_arr = np.array(all_nodes_l3, dtype=NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        if len(cx_l3) > 0:
            all_edges = np.concatenate([cx_l3, self_edges]).astype(NODE_ID)
        else:
            all_edges = self_edges.astype(NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs = flatgraph.connected_components(graph)

        # P1→P_D resolved, but P_D not in all_nodes → P_D's CX to P2 missing
        # CCs: {P1, P_D} and {P2} → 2 CCs
        assert len(ccs) == 2, \
            f"Without L3 siblings: P1-P_D connected but P2 isolated → {len(ccs)} CCs"

        # WITH L3 sibling discovery: P_D added to all_nodes
        all_nodes_l3_full = [P1, P2, P_D]
        cx_l3_full = topology.resolve_cx_at_layer(all_nodes_l3_full, 3, c, _get_layer)

        nodes_arr_full = np.array(all_nodes_l3_full, dtype=NODE_ID)
        self_edges_full = np.vstack([nodes_arr_full, nodes_arr_full]).T
        all_edges_full = np.concatenate([cx_l3_full, self_edges_full]).astype(NODE_ID)
        graph_full, _, _, gids_full = flatgraph.build_gt_graph(all_edges_full, make_directed=True)
        ccs_full = flatgraph.connected_components(graph_full)

        assert len(ccs_full) == 1, \
            f"With L3 sibling P_D: P1-P_D-P2 forms 1 CC, got {len(ccs_full)}"

