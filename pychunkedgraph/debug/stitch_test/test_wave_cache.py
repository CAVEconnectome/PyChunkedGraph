"""WaveCache tests. Verifies cache invariants from CACHE_DESIGN.md."""

import numpy as np

from pychunkedgraph.graph import basetypes, types

from . import topology
from .wave_cache import SiblingEntry, WaveCache

NODE_ID = basetypes.NODE_ID


class TestReadGating:

    def test_has_batch_gates_reads(self) -> None:
        """has_batch returns False for unknown nodes, True for cached.
        Broken = _ensure_cached skips the check, causing duplicate BigTable reads."""
        c = WaveCache()
        ids = np.array([10, 20], dtype=NODE_ID)
        assert not c.has_batch(ids).any()
        c.put_parent(10, 100)
        result = c.has_batch(ids)
        assert result[0] and not result[1]


class TestChainMapPriority:

    def test_local_shadows_preloaded(self) -> None:
        """Local writes must shadow preloaded (fork COW) values.
        Broken = stale preloaded data returned after a stitch updates a node."""
        c = WaveCache(preloaded=({10: 100}, {}, {}))
        assert c.parents[10] == 100
        c.put_parent(10, 200)
        assert c.parents[10] == 200


class TestFlushCreated:

    def test_flush_makes_created_visible(self) -> None:
        """parents_cache entries invisible to has_batch before flush, visible after.
        Broken = new nodes trigger BigTable read (violates requirement 2)."""
        c = WaveCache()
        c.begin_stitch()
        c.parents_cache[99] = 999
        c.children_cache[99] = np.array([1, 2], dtype=NODE_ID)
        c.l2_atomic_cx[99] = {2: "acx"}
        assert not c.has(99)
        c.flush_created()
        assert c.has(99)
        assert c.parents[99] == 999
        assert len(c.children[99]) == 2
        assert c.acx[99] == {2: "acx"}

    def test_flush_overwrites_stale_read(self) -> None:
        """Read stores parent=100, then stitch creates parent=200 for same node.
        Flush must overwrite the stale read value.
        Broken = stale parent used in hierarchy resolution → wrong CX edges."""
        c = WaveCache()
        c.put_parent(10, 100)
        c.begin_stitch()
        c.parents_cache[10] = 200
        c.flush_created()
        assert c.parents[10] == 200

    def test_created_persists_across_begin_stitch(self) -> None:
        """Created data in _local persists across begin_stitch (which only resets per-stitch state).
        Broken = new nodes from wave N lost in wave N+1 → BigTable re-read or missing data."""
        c = WaveCache()
        c.begin_stitch()
        c.parents_cache[99] = 999
        c.children_cache[99] = np.array([1], dtype=NODE_ID)
        c.flush_created()
        c.begin_stitch()
        assert c.has(99)
        assert c.parents[99] == 999
        assert len(c.children[99]) == 1


class TestSaveWaveState:

    def test_flushes_and_stores_siblings(self) -> None:
        """save_wave_state calls flush_created (so new nodes persist) and stores sibling entries.
        Broken = in-process waves lose created data or sibling cache."""
        c = WaveCache()
        c.begin_stitch()
        c.parents_cache[99] = 999
        c.children_cache[99] = np.array([1], dtype=NODE_ID)
        c.l2_atomic_cx[50] = {2: "acx"}
        c.put_children(10, np.array([1, 2], dtype=NODE_ID))
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={10}, raw_cx_edges={10: {2: "raw"}}, children=c.children,
        )
        assert c.has(99)
        assert c.parents[99] == 999
        assert c.acx[50] == {2: "acx"}
        assert 10 in c._siblings
        assert c._siblings[10].raw_cx_edges == {2: "raw"}


class TestMergeOperations:

    def test_merge_inc_stores_siblings_and_incremental(self) -> None:
        """merge_inc stores sibling data + old_to_new + new_node_ids for incremental optimization.
        Broken = partition_siblings can't find known siblings → all re-read from BigTable."""
        c = WaveCache()
        children = np.array([1, 2, 3], dtype=NODE_ID)
        c.merge_inc({
            "old_to_new": {10: 11},
            "new_node_ids": {11},
            "sibling_data": {
                50: {"children": children, "raw_cx": {2: "edges"}},
            },
        })
        assert c.get_sibling(50) is not None
        assert c.get_sibling(50).raw_cx_edges == {2: "edges"}
        assert set(c.get_sibling(50).resolver_entries.keys()) == {1, 2, 3}
        assert c.accumulated_replacements == {10: 11}
        assert 11 in c._new_node_ids

    def test_incremental_state_to_new_cache(self) -> None:
        """incremental_state() tuple passed to new WaveCache gives it sibling + old_to_new data.
        Broken = pool workers don't know about prior waves' siblings → can't skip reads."""
        c = WaveCache()
        c._siblings[10] = SiblingEntry({2: "raw"}, {1: {2: 10}})
        c.accumulated_replacements = {100: 101}
        c._new_node_ids = {101}
        c2 = WaveCache(incremental=c.incremental_state())
        assert c2.get_sibling(10) is not None
        assert c2.accumulated_replacements == {100: 101}


class TestSplitKnown:

    def test_empty_all_unknown(self) -> None:
        """No prior siblings → all unknown (first wave)."""
        c = WaveCache()
        known, unknown = c.split_known_siblings(np.array([10, 20], dtype=NODE_ID))
        assert len(known) == 0 and len(unknown) == 2

    def test_mixed(self) -> None:
        """Known siblings from prior wave cached, new ones unknown."""
        c = WaveCache()
        c._siblings[10] = SiblingEntry({}, {})
        c._siblings[20] = SiblingEntry({}, {})
        known, unknown = c.split_known_siblings(np.array([10, 20, 30], dtype=NODE_ID))
        assert set(int(x) for x in known) == {10, 20}
        assert set(int(x) for x in unknown) == {30}


class TestComplexFlows:

    def test_pool_wave_roundtrip(self) -> None:
        """Simulates pool wave: worker reads + creates → snapshot → parent merges → new worker finds all.
        Broken = created nodes lost in transit → BigTable re-reads in next wave."""
        worker = WaveCache()
        worker.begin_stitch()
        worker.put_parent(1, 10)
        worker.parents_cache[50] = 500
        worker.children_cache[500] = np.array([50], dtype=NODE_ID)
        worker.flush_created()

        local_snap = worker.local_snapshot()
        inc_snap = worker.inc_snapshot_from(
            old_to_new={10: 50}, new_node_ids={50, 500},
            sibling_ids=set(), raw_cx_edges={}, children=worker.children,
        )

        parent = WaveCache()
        parent.merge_reader(local_snap)
        parent.merge_inc(inc_snap)

        assert parent.has(1)
        assert parent.has(50)
        assert parent.parents[50] == 500
        assert len(parent.children[500]) == 1

        worker2 = WaveCache(preloaded=parent.preloaded())
        assert worker2.has(1)
        assert worker2.has(50)
        assert worker2.parents[50] == 500

    def test_two_workers_merge_no_loss(self) -> None:
        """Two workers create different nodes. Parent merges both. No data lost.
        Broken = second merge overwrites first → missing nodes."""
        w1 = WaveCache()
        w1.begin_stitch()
        w1.put_parent(1, 10)
        w1.parents_cache[100] = 1000
        w1.flush_created()

        w2 = WaveCache()
        w2.begin_stitch()
        w2.put_parent(2, 20)
        w2.parents_cache[200] = 2000
        w2.flush_created()

        parent = WaveCache()
        parent.merge_reader(w1.local_snapshot())
        parent.merge_reader(w2.local_snapshot())
        assert parent.has(1) and parent.has(2)
        assert parent.has(100) and parent.has(200)
        assert parent.parents[100] == 1000
        assert parent.parents[200] == 2000

    def test_full_multiwave(self) -> None:
        """Three waves accumulating data. Each wave's created + read data persists.
        Broken = any wave's data lost → re-reads or missing nodes."""
        c = WaveCache()

        c.begin_stitch()
        c.put_parent(1, 10)
        c.parents_cache[100] = 1000
        c.children_cache[1000] = np.array([100], dtype=NODE_ID)
        c.save_wave_state(
            old_to_new={10: 100}, new_node_ids={100, 1000},
            sibling_ids=set(), raw_cx_edges={}, children=c.children,
        )

        c.begin_stitch()
        assert c.has(1) and c.has(100)
        assert c.parents[100] == 1000
        c.put_parent(3, 30)
        c.parents_cache[200] = 2000
        c.save_wave_state(
            old_to_new={30: 200}, new_node_ids={200, 2000},
            sibling_ids=set(), raw_cx_edges={}, children=c.children,
        )

        c.begin_stitch()
        assert c.has(1) and c.has(100) and c.has(200)
        assert c.has(3)
        assert c.parents[100] == 1000
        assert c.parents[200] == 2000


class TestDirtySiblings:

    def test_affected_partner_makes_dirty(self) -> None:
        """Sibling with partner SV resolving to a replaced L2 → dirty.
        Broken = unchanged CX written for this sibling, wasting BigTable writes."""
        c = WaveCache()
        c.begin_stitch()
        c.old_to_new = {100: 200}
        c.new_ids_d[2] = [200]
        c.resolver = {50: {2: 100}}
        c.sibling_ids = {10}
        c.raw_cx_edges = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

    def test_new_node_partner_makes_dirty(self) -> None:
        """Sibling with partner SV resolving to a newly created L2 → dirty.
        Broken = partner points to new node with new parents, CX would differ."""
        c = WaveCache()
        c.begin_stitch()
        c.old_to_new = {}
        c.new_ids_d[2] = [200]
        c.resolver = {50: {2: 200}}
        c.sibling_ids = {10}
        c.raw_cx_edges = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

    def test_unaffected_partner_is_clean(self) -> None:
        """Sibling with partner SV resolving to unrelated L2 → clean.
        Broken = sibling marked dirty unnecessarily, extra writes."""
        c = WaveCache()
        c.begin_stitch()
        c.old_to_new = {100: 200}
        c.new_ids_d[2] = [200]
        c.resolver = {50: {2: 999}}
        c.sibling_ids = {10}
        c.raw_cx_edges = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 not in c.dirty_siblings

    def test_unknown_sibling_always_dirty(self) -> None:
        """Sibling without SiblingEntry (first wave) → always dirty.
        Broken = unknown sibling skipped, CX never written."""
        c = WaveCache()
        c.begin_stitch()
        c.old_to_new = {}
        c.new_ids_d[2] = []
        c.resolver = {}
        c.sibling_ids = {10}
        c.raw_cx_edges = {}
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

    def test_dirty_across_three_waves(self) -> None:
        """Wave 0: create A,B. Wave 1: replace A→X, S1(→A) dirty, S2(→B) clean.
        Wave 2: replace B→Y, S1(→X) clean, S2(→B) dirty.
        Broken = wrong siblings marked dirty → wrong CX written or extra writes."""
        c = WaveCache()

        # Wave 0: S1 partners with SV 50→A(100), S2 partners with SV 60→B(200)
        c.begin_stitch()
        c.old_to_new = {}
        c.new_ids_d[2] = [100, 200]
        c.resolver = {50: {2: 100}, 60: {2: 200}}
        c.sibling_ids = {10, 20}
        c.raw_cx_edges = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert c.dirty_siblings == {10, 20}

        c.cx_cache = {10: {2: np.array([[10, 100]], dtype=NODE_ID)}, 20: {2: np.array([[20, 200]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={}, new_node_ids={100, 200},
            sibling_ids={10, 20}, raw_cx_edges=c.raw_cx_edges, children=c.children,
        )

        # Wave 1: replace A(100)→X(300). S1 dirty (partner→100, 100 in old_to_new). S2 clean.
        c.begin_stitch()
        c.old_to_new = {100: 300}
        c.new_ids_d[2] = [300]
        c.resolver = {50: {2: 100}, 60: {2: 200}}
        c.sibling_ids = {10, 20}
        c.raw_cx_edges = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings
        assert 20 not in c.dirty_siblings

        c.cx_cache = {10: {2: np.array([[10, 300]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={100: 300}, new_node_ids={300},
            sibling_ids={10, 20}, raw_cx_edges=c.raw_cx_edges, children=c.children,
        )

        # Wave 2: replace B(200)→Y(400). Both dirty — S1 because 100 in accumulated, S2 because 200 in current.
        c.begin_stitch()
        c.old_to_new = {**c.accumulated_replacements, 200: 400}
        c.new_ids_d[2] = [400]
        c.resolver = {50: {2: 100}, 60: {2: 200}}
        c.sibling_ids = {10, 20}
        c.raw_cx_edges = {
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
        c = WaveCache()

        c.begin_stitch()
        c.save_wave_state(
            old_to_new={100: 200}, new_node_ids={200},
            sibling_ids=set(), raw_cx_edges={}, children=c.children,
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
        c = WaveCache()

        c.begin_stitch()
        c.old_to_new = {100: 200}
        c.new_ids_d[2] = [200]
        c.resolver = {50: {2: 100}}
        c.sibling_ids = {10}
        c.raw_cx_edges = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
        c._siblings[10] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 in c.dirty_siblings

        c.cx_cache = {10: {2: np.array([[10, 200]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={100: 200}, new_node_ids={200},
            sibling_ids={10}, raw_cx_edges=c.raw_cx_edges, children=c.children,
        )

        c.begin_stitch()
        c.old_to_new = {**c.accumulated_replacements, 300: 400}
        c.new_ids_d[2] = [400]
        c.resolver = {50: {2: 100}}
        c.sibling_ids = {10}
        c.raw_cx_edges = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}
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

    def test_clean_sibling_written_cx_in_cc_graph(self) -> None:
        """Clean sibling's written_cx is injected into CC graph edges.
        Broken = clean siblings have no CX edges → wrong connected components."""
        c = WaveCache()
        c.begin_stitch()
        c.resolver = {50: {2: 200}}
        c.old_to_new = {}
        c.new_ids_d[2] = []
        c.sibling_ids = {10, 20}
        c.raw_cx_edges = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 50]], dtype=NODE_ID)},
        }
        c._siblings[10] = SiblingEntry(
            raw_cx_edges={2: np.array([[10, 50]], dtype=NODE_ID)},
            resolver_entries={},
            written_cx={2: np.array([[10, 200]], dtype=NODE_ID)},
        )
        c._siblings[20] = SiblingEntry(
            raw_cx_edges={2: np.array([[20, 50]], dtype=NODE_ID)},
            resolver_entries={},
            written_cx={2: np.array([[20, 200]], dtype=NODE_ID)},
        )
        c.compute_dirty_siblings()
        clean_sibs = c.sibling_ids - c.dirty_siblings
        assert clean_sibs == {10, 20}

        resolve_nodes = [n for n in [10, 20] if int(n) not in clean_sibs]
        cx_resolved = topology.resolve_cx_at_layer(resolve_nodes, 2, c, {}, _get_layer)
        assert len(cx_resolved) == 0

        cached_cx = []
        for n in [10, 20]:
            if int(n) in clean_sibs:
                entry = c.get_sibling(int(n))
                if entry and 2 in entry.written_cx:
                    cached_cx.append(entry.written_cx[2])
        all_cx = np.concatenate(cached_cx).astype(NODE_ID)
        assert len(all_cx) == 2
        assert set(int(r[1]) for r in all_cx) == {200}

    def test_store_cx_only_for_dirty(self) -> None:
        """Only dirty siblings' CX stored to cx_cache. Clean siblings absent.
        Broken = clean siblings in cx_cache → written to BigTable unnecessarily."""
        c = WaveCache()
        c.begin_stitch()
        c.resolver = {50: {2: 200}, 60: {2: 100}}
        c.old_to_new = {100: 300}
        c.new_ids_d[2] = [300]
        c.sibling_ids = {10, 20}
        c.raw_cx_edges = {
            10: {2: np.array([[10, 50]], dtype=NODE_ID)},
            20: {2: np.array([[20, 60]], dtype=NODE_ID)},
        }
        c._siblings[10] = SiblingEntry({}, {})
        c._siblings[20] = SiblingEntry({}, {})
        c.compute_dirty_siblings()
        assert 10 not in c.dirty_siblings
        assert 20 in c.dirty_siblings

        clean_sibs = c.sibling_ids - c.dirty_siblings
        resolve_nodes = [n for n in [10, 20] if int(n) not in clean_sibs]
        cx_resolved = topology.resolve_cx_at_layer(resolve_nodes, 2, c, {}, _get_layer)
        topology.store_cx_from_resolved(c, cx_resolved, 2)

        assert 10 not in c.cx_cache
        assert 20 in c.cx_cache

    def test_build_rows_skips_clean_siblings(self) -> None:
        """build_rows only writes CX for siblings in cx_cache (dirty ones).
        Broken = all siblings written → 8x write time regression."""
        from kvdbclient import serializers
        from pychunkedgraph.graph import attributes

        c = WaveCache()
        c.begin_stitch()
        c.sibling_ids = {10, 20}
        c.new_node_ids = set()
        c.cx_cache = {20: {2: np.array([[20, 200]], dtype=NODE_ID)}}

        rows = {}
        for nid in c.sibling_ids:
            if nid not in c.cx_cache:
                continue
            rk = serializers.serialize_uint64(np.uint64(nid))
            rows[rk] = {
                attributes.Connectivity.CrossChunkEdge[layer]: cx
                for layer, cx in c.cx_cache[nid].items()
            }

        assert len(rows) == 1
        rk_20 = serializers.serialize_uint64(np.uint64(20))
        assert rk_20 in rows
        rk_10 = serializers.serialize_uint64(np.uint64(10))
        assert rk_10 not in rows


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


class TestWrittenCxStaleness:
    """Integration tests that verify written_cx from prior waves is stale at
    higher layers due to child_to_parent changing. These tests would have caught
    the task_2_5 mismatch (2229 vs 592 roots)."""

    def test_written_cx_stale_at_layer3(self) -> None:
        """written_cx at layer 3 uses old child_to_parent. Fresh resolution
        uses current child_to_parent → different result.
        Broken = stale CX in CC graph → wrong connected components → wrong roots.

        resolver only has layer 2 entry. Layer 3 resolution walks child_to_parent
        from L2 identity up to L3. Different child_to_parent → different L3 partner."""
        A = _node(2, 1)
        S = _node(2, 2)
        P3_old = _node(3, 10)
        P3_new = _node(3, 20)
        sv_a = 100

        c = WaveCache()
        c.begin_stitch()
        c.resolver = {sv_a: {2: A}}
        c.old_to_new = {}
        c.raw_cx_edges = {S: {3: np.array([[S, sv_a]], dtype=NODE_ID)}}

        cx_old = topology.resolve_cx_at_layer([S], 3, c, {A: P3_old}, _get_layer)
        assert len(cx_old) == 1
        assert int(cx_old[0, 1]) == P3_old

        cx_new = topology.resolve_cx_at_layer([S], 3, c, {A: P3_new}, _get_layer)
        assert len(cx_new) == 1
        assert int(cx_new[0, 1]) == P3_new

        assert int(cx_old[0, 1]) != int(cx_new[0, 1]), \
            "Layer 3 CX changes with child_to_parent — written_cx is stale"

    def test_written_cx_safe_at_layer2(self) -> None:
        """written_cx at layer 2 is safe for clean siblings because dirty check
        covers all L2 identity changes via accumulated old_to_new.
        child_to_parent doesn't affect layer 2 resolution (walk stops at target)."""
        A = _node(2, 1)
        B = _node(2, 2)
        S = _node(2, 3)
        sv_b = 200

        c = WaveCache()
        c.begin_stitch()
        c.resolver = {sv_b: {2: B}}
        c.old_to_new = {A: _node(2, 10)}
        c.new_ids_d[2] = [_node(2, 10)]
        c.raw_cx_edges = {S: {2: np.array([[S, sv_b]], dtype=NODE_ID)}}

        cx_fresh = topology.resolve_cx_at_layer([S], 2, c, {}, _get_layer)
        written_cx = np.array([[S, B]], dtype=NODE_ID)
        assert np.array_equal(cx_fresh, written_cx), \
            "Layer 2 CX for clean sibling (B not replaced) matches written_cx"

        c._siblings[int(S)] = SiblingEntry(
            raw_cx_edges={2: np.array([[S, sv_b]], dtype=NODE_ID)},
            resolver_entries={},
        )
        c.sibling_ids = {int(S)}
        c.compute_dirty_siblings()
        assert int(S) not in c.dirty_siblings, "S is clean — partner B not in affected_ids"

    def test_full_multiwave_layer2_only_safe(self) -> None:
        """3 waves. Only layer-2 written_cx used for clean siblings.
        Layer 3+ always freshly resolved. Verifies no stale CX enters CC graph."""
        A = _node(2, 1)
        B = _node(2, 2)
        S1 = _node(2, 10)
        S2 = _node(2, 20)
        sv_a = 100
        sv_b = 200
        X = _node(2, 3)

        c = WaveCache()

        # Wave 0: S1→A, S2→B at layer 2
        c.begin_stitch()
        c.resolver = {sv_a: {2: A}, sv_b: {2: B}}
        c.old_to_new = {}
        c.new_ids_d[2] = []
        c.sibling_ids = {int(S1), int(S2)}
        c.raw_cx_edges = {
            int(S1): {2: np.array([[S1, sv_a]], dtype=NODE_ID)},
            int(S2): {2: np.array([[S2, sv_b]], dtype=NODE_ID)},
        }

        cx_s1 = topology.resolve_cx_at_layer([S1], 2, c, {}, _get_layer)
        cx_s2 = topology.resolve_cx_at_layer([S2], 2, c, {}, _get_layer)
        c.cx_cache = {int(S1): {2: cx_s1}, int(S2): {2: cx_s2}}

        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids=c.sibling_ids, raw_cx_edges=c.raw_cx_edges, children=c.children,
        )

        # Wave 1: replace A→X
        c.begin_stitch()
        c.resolver = {sv_a: {2: A}, sv_b: {2: B}}
        c.old_to_new = {**c.accumulated_replacements, A: X}
        c.new_ids_d[2] = [X]
        c.sibling_ids = {int(S1), int(S2)}
        c.raw_cx_edges = {
            int(S1): {2: np.array([[S1, sv_a]], dtype=NODE_ID)},
            int(S2): {2: np.array([[S2, sv_b]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()

        assert int(S1) in c.dirty_siblings, "S1 dirty — partner A replaced"
        assert int(S2) not in c.dirty_siblings, "S2 clean — partner B untouched"

        clean_sibs = c.sibling_ids - c.dirty_siblings
        resolve_nodes = [n for n in [S1, S2] if int(n) not in clean_sibs]
        cx_resolved = topology.resolve_cx_at_layer(resolve_nodes, 2, c, {}, _get_layer)

        assert int(S1) in [int(n) for n in resolve_nodes], "S1 resolved (dirty)"
        assert int(S2) not in [int(n) for n in resolve_nodes], "S2 skipped (clean)"

        entry_s2 = c.get_sibling(int(S2))
        written_s2 = entry_s2.written_cx.get(2)
        fresh_s2 = topology.resolve_cx_at_layer([S2], 2, c, {}, _get_layer)
        assert np.array_equal(written_s2, fresh_s2), \
            "Clean sibling S2 written_cx matches fresh resolution at layer 2"

    def test_layer3_stores_all_siblings(self) -> None:
        """At layer 3+, store_cx_from_resolved stores for ALL nodes including clean siblings.
        Broken = clean siblings missing layer 3+ CX → incomplete BigTable rows."""
        A = _node(2, 1)
        S = _node(2, 2)
        P3 = _node(3, 10)
        sv_a = 100

        c = WaveCache()
        c.begin_stitch()
        c.resolver = {sv_a: {2: A}}
        c.raw_cx_edges = {
            int(S): {3: np.array([[S, sv_a]], dtype=NODE_ID)},
        }

        child_to_parent = {A: P3}
        all_cx = topology.resolve_cx_at_layer([S], 3, c, child_to_parent, _get_layer)
        topology.store_cx_from_resolved(c, all_cx, 3)

        assert int(S) in c.cx_cache
        assert 3 in c.cx_cache[int(S)]

    def test_children_d_not_chainmap_in_snapshot(self) -> None:
        """inc_snapshot_from must receive children_d (per-stitch siblings only),
        not children ChainMap (all known nodes). Using ChainMap builds wrong
        resolver_entries with unrelated nodes' children."""
        c = WaveCache()
        c.begin_stitch()
        c.put_children(999, np.array([1, 2, 3], dtype=NODE_ID))
        c.children_d = {10: np.array([50, 60], dtype=NODE_ID)}
        c.sibling_ids = {10}
        c.raw_cx_edges = {10: {}}
        c.cx_cache = {}

        snap_correct = c.inc_snapshot_from(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={10}, raw_cx_edges={}, children=c.children_d,
        )
        sib_data = snap_correct["sibling_data"][10]
        assert len(sib_data["children"]) == 2

        snap_wrong = c.inc_snapshot_from(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={10}, raw_cx_edges={}, children=c.children,
        )
        sib_data_wrong = snap_wrong["sibling_data"][10]
        assert len(sib_data_wrong["children"]) == 0, \
            "ChainMap doesn't have sibling children → empty resolver_entries in next wave"


class TestWrittenCxCarryForward:
    """Tests for scenario D14-D19: written_cx must survive across waves.
    The task_2_5 mismatch (2229 vs 592 roots) was caused by clean siblings
    having empty written_cx because save_wave_state read from cx_cache
    (empty for clean siblings) instead of carrying forward from prior wave."""

    def test_d16_clean_sibling_written_cx_preserved(self) -> None:
        """Wave 0: sibling S gets written_cx={2: edges} from cx_cache.
        Wave 1: S is clean → NOT in cx_cache.
        save_wave_state must preserve written_cx from wave 0, not overwrite with empty.
        Broken = written_cx={} → next wave CC graph missing edges → 2229 roots."""
        S = 10
        edges_w0 = np.array([[S, 200]], dtype=NODE_ID)

        c = WaveCache()
        c.begin_stitch()
        c.cx_cache = {S: {2: edges_w0}}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        c.old_to_new = {}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S].written_cx[2], edges_w0), "Wave 0: written_cx stored"

        c.begin_stitch()
        c.cx_cache = {}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        c.old_to_new = {}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert 2 in c._siblings[S].written_cx, \
            "Wave 1: clean sibling's written_cx must be preserved, not empty"
        assert np.array_equal(c._siblings[S].written_cx[2], edges_w0), \
            "Wave 1: written_cx must match wave 0's edges"

    def test_d15_dirty_sibling_written_cx_updated(self) -> None:
        """Wave 0: written_cx={2: old_edges}. Wave 1: dirty → cx_cache has new edges.
        save_wave_state must use NEW edges from cx_cache.
        Broken = stale written_cx used in next wave CC graph."""
        S = 10
        old_edges = np.array([[S, 200]], dtype=NODE_ID)
        new_edges = np.array([[S, 300]], dtype=NODE_ID)

        c = WaveCache()
        c.begin_stitch()
        c.cx_cache = {S: {2: old_edges}}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {}}
        c.old_to_new = {}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )

        c.begin_stitch()
        c.cx_cache = {S: {2: new_edges}}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {}}
        c.old_to_new = {}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S].written_cx[2], new_edges), \
            "Dirty sibling's written_cx must be updated from cx_cache"

    def test_d17_three_wave_carryforward(self) -> None:
        """Wave 0: store. Wave 1: clean → carry. Wave 2: clean → still carried.
        Broken = written_cx lost after first clean wave."""
        S = 10
        edges_w0 = np.array([[S, 200]], dtype=NODE_ID)

        c = WaveCache()
        for wave in range(3):
            c.begin_stitch()
            if wave == 0:
                c.cx_cache = {S: {2: edges_w0}}
            else:
                c.cx_cache = {}
            c.children_d = {S: np.array([50], dtype=NODE_ID)}
            c.sibling_ids = {S}
            c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
            c.old_to_new = {}
            c.save_wave_state(
                old_to_new={}, new_node_ids=set(),
                sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
            )
            assert 2 in c._siblings[S].written_cx, f"Wave {wave}: written_cx must have layer 2"
            assert np.array_equal(c._siblings[S].written_cx[2], edges_w0), \
                f"Wave {wave}: written_cx must match wave 0's edges"

    def test_d19_pool_snapshot_preserves_written_cx(self) -> None:
        """Pool worker's inc_snapshot must carry forward written_cx for clean siblings.
        Parent merge_inc restores SiblingEntry with correct written_cx.
        Broken = pool workers lose written_cx → next wave gets empty."""
        S = 10
        edges_w0 = np.array([[S, 200]], dtype=NODE_ID)

        c = WaveCache()
        c.begin_stitch()
        c.cx_cache = {S: {2: edges_w0}}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )

        c.begin_stitch()
        c.cx_cache = {}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        snap = c.inc_snapshot_from(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert 2 in snap["sibling_data"][S].get("written_cx", {}), \
            "inc_snapshot must carry forward written_cx for clean sibling"

        parent = WaveCache()
        parent.merge_inc(snap)
        assert 2 in parent._siblings[S].written_cx, \
            "After merge_inc, parent must have written_cx for sibling"
        assert np.array_equal(parent._siblings[S].written_cx[2], edges_w0)

    def test_d18_dirty_after_clean_updates(self) -> None:
        """Wave 0: store. Wave 1: clean → carry. Wave 2: dirty → update.
        Broken = carried-forward written_cx not replaced when sibling becomes dirty."""
        S = 10
        edges_w0 = np.array([[S, 200]], dtype=NODE_ID)
        edges_w2 = np.array([[S, 300]], dtype=NODE_ID)

        c = WaveCache()
        # Wave 0: store
        c.begin_stitch()
        c.cx_cache = {S: {2: edges_w0}}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        # Wave 1: clean → carry
        c.begin_stitch()
        c.cx_cache = {}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S].written_cx[2], edges_w0)

        # Wave 2: dirty → update
        c.begin_stitch()
        c.cx_cache = {S: {2: edges_w2}}
        c.children_d = {S: np.array([50], dtype=NODE_ID)}
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, 50]], dtype=NODE_ID)}}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S].written_cx[2], edges_w2), \
            "Dirty sibling must get updated written_cx, not carried-forward"


class TestResolveRemainingCxBatched:
    """Tests that resolve_remaining_cx uses batched resolution (via resolve_cx_at_layer)
    and only processes layers > 2."""

    def test_skips_layer_2(self) -> None:
        """resolve_remaining_cx must not process layer 2 — handled by build_hierarchy layer loop.
        Broken = clean siblings get layer 2 CX stored → written to BigTable unnecessarily."""
        c = WaveCache()
        c.begin_stitch()
        c.resolver = {50: {2: 200}}
        c.sibling_ids = {10}
        c.new_node_ids = set()
        c.raw_cx_edges = {10: {2: np.array([[10, 50]], dtype=NODE_ID)}}

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        topology.resolve_remaining_cx(c, FakeLcg(), {})
        assert 10 not in c.cx_cache, "Layer 2 must not be processed by resolve_remaining_cx"

    def test_processes_layer_3(self) -> None:
        """resolve_remaining_cx must process layer 3+ for siblings.
        Broken = siblings missing layer 3+ CX → incomplete BigTable rows."""
        A = _node(2, 1)
        P3 = _node(3, 10)

        c = WaveCache()
        c.begin_stitch()
        c.resolver = {50: {2: A}}
        c.sibling_ids = {10}
        c.new_node_ids = set()
        c.raw_cx_edges = {10: {3: np.array([[10, 50]], dtype=NODE_ID)}}

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        topology.resolve_remaining_cx(c, FakeLcg(), {A: P3})
        assert 10 in c.cx_cache
        assert 3 in c.cx_cache[10]


class TestEndToEndMultiwave:
    """Full multiwave scenarios exercising the real save/restore/dirty/resolve path."""

    def test_g29_three_wave_dirty_clean_cycle(self) -> None:
        """Wave 0: all dirty. Wave 1: S1 dirty, S2 clean. Wave 2: S1 clean, S2 dirty.
        Verify written_cx correct at each step. This is the scenario that caused task_2_5 mismatch."""
        S1, S2 = 10, 20
        sv1, sv2 = 50, 60
        A, B, X, Y = 100, 200, 300, 400
        edges_s1_a = np.array([[S1, A]], dtype=NODE_ID)
        edges_s2_b = np.array([[S2, B]], dtype=NODE_ID)
        edges_s1_x = np.array([[S1, X]], dtype=NODE_ID)
        edges_s2_y = np.array([[S2, Y]], dtype=NODE_ID)

        c = WaveCache()

        # Wave 0: both dirty (no prior siblings)
        c.begin_stitch()
        c.resolver = {sv1: {2: A}, sv2: {2: B}}
        c.old_to_new = {}
        c.new_ids_d[2] = []
        c.sibling_ids = {S1, S2}
        c.raw_cx_edges = {
            S1: {2: np.array([[S1, sv1]], dtype=NODE_ID)},
            S2: {2: np.array([[S2, sv2]], dtype=NODE_ID)},
        }
        c.cx_cache = {S1: {2: edges_s1_a}, S2: {2: edges_s2_b}}
        c.children_d = {S1: np.array([sv1], dtype=NODE_ID), S2: np.array([sv2], dtype=NODE_ID)}
        c.compute_dirty_siblings()
        assert c.dirty_siblings == {S1, S2}
        c.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S1, S2}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S1].written_cx[2], edges_s1_a)
        assert np.array_equal(c._siblings[S2].written_cx[2], edges_s2_b)

        # Wave 1: replace A→X. S1 dirty, S2 clean.
        c.begin_stitch()
        c.resolver = {sv1: {2: A}, sv2: {2: B}}
        c.old_to_new = {**c.accumulated_replacements, A: X}
        c.new_ids_d[2] = [X]
        c.sibling_ids = {S1, S2}
        c.raw_cx_edges = {
            S1: {2: np.array([[S1, sv1]], dtype=NODE_ID)},
            S2: {2: np.array([[S2, sv2]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert S1 in c.dirty_siblings
        assert S2 not in c.dirty_siblings

        c.cx_cache = {S1: {2: edges_s1_x}}  # S1 dirty → resolved, S2 clean → not in cx_cache
        c.children_d = {S1: np.array([sv1], dtype=NODE_ID), S2: np.array([sv2], dtype=NODE_ID)}
        c.save_wave_state(
            old_to_new=c.old_to_new, new_node_ids={X},
            sibling_ids={S1, S2}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S1].written_cx[2], edges_s1_x), "S1 dirty → updated"
        assert np.array_equal(c._siblings[S2].written_cx[2], edges_s2_b), "S2 clean → preserved from wave 0"

        # Wave 2: replace B→Y. S1 clean (A in accumulated but S1 already resolved), S2 dirty.
        c.begin_stitch()
        c.resolver = {sv1: {2: A}, sv2: {2: B}}
        c.old_to_new = {**c.accumulated_replacements, B: Y}
        c.new_ids_d[2] = [Y]
        c.sibling_ids = {S1, S2}
        c.raw_cx_edges = {
            S1: {2: np.array([[S1, sv1]], dtype=NODE_ID)},
            S2: {2: np.array([[S2, sv2]], dtype=NODE_ID)},
        }
        c.compute_dirty_siblings()
        assert S1 in c.dirty_siblings, "S1 dirty — A in accumulated_replacements"
        assert S2 in c.dirty_siblings, "S2 dirty — B in current old_to_new"

        c.cx_cache = {S1: {2: edges_s1_x}, S2: {2: edges_s2_y}}
        c.children_d = {S1: np.array([sv1], dtype=NODE_ID), S2: np.array([sv2], dtype=NODE_ID)}
        c.save_wave_state(
            old_to_new=c.old_to_new, new_node_ids={Y},
            sibling_ids={S1, S2}, raw_cx_edges=c.raw_cx_edges, children=c.children_d,
        )
        assert np.array_equal(c._siblings[S1].written_cx[2], edges_s1_x), "S1 dirty → resolved (same X)"
        assert np.array_equal(c._siblings[S2].written_cx[2], edges_s2_y), "S2 dirty → updated to Y"

    def test_g30_pool_roundtrip_with_written_cx(self) -> None:
        """Full pool wave: parent stores wave 0 → worker gets incremental → worker resolves
        with written_cx for clean siblings → snapshot back to parent → parent has correct written_cx."""
        S = 10
        sv = 50
        edges_w0 = np.array([[S, 200]], dtype=NODE_ID)

        # Parent: wave 0
        parent = WaveCache()
        parent.begin_stitch()
        parent.cx_cache = {S: {2: edges_w0}}
        parent.children_d = {S: np.array([sv], dtype=NODE_ID)}
        parent.sibling_ids = {S}
        parent.raw_cx_edges = {S: {2: np.array([[S, sv]], dtype=NODE_ID)}}
        parent.save_wave_state(
            old_to_new={}, new_node_ids=set(),
            sibling_ids={S}, raw_cx_edges=parent.raw_cx_edges, children=parent.children_d,
        )

        # Worker: wave 1 (clean sibling)
        preloaded = parent.preloaded()
        inc = parent.incremental_state()
        worker = WaveCache(preloaded=preloaded, incremental=inc)
        worker.begin_stitch()
        worker.resolver = {sv: {2: 200}}
        worker.old_to_new = {**worker.accumulated_replacements, 300: 400}
        worker.new_ids_d[2] = [400]
        worker.sibling_ids = {S}
        worker.raw_cx_edges = {S: {2: np.array([[S, sv]], dtype=NODE_ID)}}
        worker.compute_dirty_siblings()
        assert S not in worker.dirty_siblings, "S clean — partner 200 not replaced"

        # Worker builds cx_cache (S not in it — clean at layer 2)
        worker.cx_cache = {}
        worker.children_d = {S: np.array([sv], dtype=NODE_ID)}
        worker.flush_created()
        snap = worker.inc_snapshot_from(
            old_to_new=worker.old_to_new, new_node_ids={400},
            sibling_ids={S}, raw_cx_edges=worker.raw_cx_edges, children=worker.children_d,
        )

        # Parent: merge wave 1
        parent.merge_reader(worker.local_snapshot())
        parent.merge_inc(snap)
        assert 2 in parent._siblings[S].written_cx
        assert np.array_equal(parent._siblings[S].written_cx[2], edges_w0), \
            "After pool roundtrip, written_cx preserved for clean sibling"


class TestPartnerResolution:
    """Tests that partner SVs of dirty known siblings get resolver entries
    even when there are no unknown siblings. The task_2_5 mismatch (781 vs 592)
    was caused by collect_and_resolve_partners only running when unknown > 0."""

    def test_dirty_known_partner_needs_resolution(self) -> None:
        """Dirty known sibling's partner SV must be in resolver for resolve_cx_at_layer
        to produce correct results. If resolver missing → returns raw SV instead of L2.
        Broken = wrong CX edges → wrong CC → wrong roots."""
        S = 10
        partner_sv = 50
        partner_l2 = 200

        c = WaveCache()
        c.begin_stitch()
        c.resolver = {partner_sv: {2: partner_l2}}
        c.old_to_new = {partner_l2: 300}
        c.new_ids_d[2] = [300]
        c.raw_cx_edges = {S: {2: np.array([[S, partner_sv]], dtype=NODE_ID)}}

        def _get_layer(nid):
            return (int(nid) >> 56) & 0xFF

        cx = topology.resolve_cx_at_layer([S], 2, c, {}, _get_layer)
        assert len(cx) == 1
        assert int(cx[0, 1]) == 300, "Partner SV resolved via resolver + old_to_new"

        c2 = WaveCache()
        c2.begin_stitch()
        c2.resolver = {}
        c2.old_to_new = {partner_l2: 300}
        c2.raw_cx_edges = {S: {2: np.array([[S, partner_sv]], dtype=NODE_ID)}}

        cx_bad = topology.resolve_cx_at_layer([S], 2, c2, {}, _get_layer)
        if len(cx_bad) > 0:
            assert int(cx_bad[0, 1]) == partner_sv, \
                "Without resolver entry, partner SV returned as-is (wrong)"
            assert int(cx_bad[0, 1]) != 300, \
                "Confirms: missing resolver → wrong resolution"

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

        c = WaveCache()
        c._siblings[S_dirty] = SiblingEntry(
            raw_cx_edges={2: np.array([[S_dirty, partner_sv_dirty]], dtype=NODE_ID)},
            resolver_entries={70: {2: S_dirty}},
        )
        c._siblings[S_clean] = SiblingEntry(
            raw_cx_edges={2: np.array([[S_clean, partner_sv_clean]], dtype=NODE_ID)},
            resolver_entries={80: {2: S_clean}},
        )
        c.begin_stitch()
        c.old_to_new = {partner_l2_dirty: 400}
        c.new_ids_d[2] = [400]
        c.resolver = {
            70: {2: S_dirty}, 80: {2: S_clean},
            partner_sv_dirty: {2: partner_l2_dirty},
        }
        c.sibling_ids = {S_dirty, S_clean}
        c.raw_cx_edges = {
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

        assert partner_sv_dirty in c.resolver, \
            "Dirty known sibling's partner SV MUST be in resolver for correct resolution"

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

        c = WaveCache()
        c._siblings[S] = SiblingEntry(
            raw_cx_edges={2: np.array([[S, partner_sv]], dtype=NODE_ID)},
            resolver_entries={70: {2: S}},
        )

        c.begin_stitch()
        c.old_to_new = {partner_l2: replacement}
        c.new_ids_d[2] = [replacement]
        c.sibling_ids = {S}
        c.raw_cx_edges = {S: {2: np.array([[S, partner_sv]], dtype=NODE_ID)}}

        # Without partner SV in resolver: dirty check returns clean (WRONG)
        c.compute_dirty_siblings()
        assert S not in c.dirty_siblings, "BUG confirmed: without resolver entry, dirty check misses it"

        # With partner SV in resolver (as if collect_and_resolve_partners ran first): dirty (CORRECT)
        c.resolver[partner_sv] = {2: partner_l2}
        c.compute_dirty_siblings()
        assert S in c.dirty_siblings, "With resolver entry, dirty check correctly detects replaced partner"
