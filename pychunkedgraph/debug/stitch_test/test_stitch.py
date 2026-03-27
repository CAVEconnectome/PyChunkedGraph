"""Tests for tree, topology, and stitch operations. No BigTable needed."""

import numpy as np
from pychunkedgraph.graph import basetypes, types

from . import topology
from . import tree
from .wave_cache import WaveCache

NODE_ID = basetypes.NODE_ID


def _get_layer(nid: int) -> int:
    return (int(nid) >> 56) & 0xFF


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


def _cache(**kw) -> WaveCache:
    c = WaveCache()
    c.begin_stitch()
    c.resolver = kw.get("resolver", {})
    c.old_to_new = kw.get("old_to_new", {})
    c.raw_cx_edges = kw.get("raw_cx_edges", {})
    return c


class TestResolveSvToLayer:

    def test_external_sv_full_chain(self) -> None:
        c = _cache(resolver={100: {2: 200, 3: 300, 4: 400}})
        assert tree.resolve_sv_to_layer(100, 3, c, {}, _get_layer) == 300

    def test_external_sv_best_below_target(self) -> None:
        c = _cache(resolver={100: {2: 200, 4: 400}})
        assert tree.resolve_sv_to_layer(100, 3, c, {}, _get_layer) == 200

    def test_own_sv_with_old_to_new(self) -> None:
        c = _cache(resolver={100: {2: 200}}, old_to_new={200: 201})
        assert tree.resolve_sv_to_layer(100, 2, c, {}, _get_layer) == 201

    def test_own_sv_walks_child_to_parent(self) -> None:
        l2, l3 = _node(2, 10), _node(3, 20)
        c = _cache(resolver={100: {2: l2}})
        assert tree.resolve_sv_to_layer(100, 3, c, {l2: l3}, _get_layer) == l3

    def test_walk_stops_at_target_layer(self) -> None:
        l2, l3, l4 = _node(2, 10), _node(3, 20), _node(4, 30)
        c = _cache(resolver={100: {2: l2}})
        assert tree.resolve_sv_to_layer(100, 3, c, {l2: l3, l3: l4}, _get_layer) == l3

    def test_unknown_sv_returns_self(self) -> None:
        c = _cache()
        assert tree.resolve_sv_to_layer(999, 2, c, {}, _get_layer) == 999


class TestResolveCxAtLayer:

    def test_basic_resolution(self) -> None:
        c = _cache(
            raw_cx_edges={10: {2: np.array([[10, 100]], dtype=NODE_ID)}},
            resolver={100: {2: 200}},
        )
        result = topology.resolve_cx_at_layer([10], 2, c, {}, _get_layer)
        assert len(result) == 1
        assert int(result[0, 0]) == 10
        assert int(result[0, 1]) == 200

    def test_self_loops_filtered(self) -> None:
        c = _cache(
            raw_cx_edges={10: {2: np.array([[10, 100]], dtype=NODE_ID)}},
            resolver={100: {2: 10}},
        )
        result = topology.resolve_cx_at_layer([10], 2, c, {}, _get_layer)
        assert len(result) == 0

    def test_duplicates_removed(self) -> None:
        c = _cache(
            raw_cx_edges={10: {2: np.array([[10, 100], [10, 101]], dtype=NODE_ID)}},
            resolver={100: {2: 200}, 101: {2: 200}},
        )
        result = topology.resolve_cx_at_layer([10], 2, c, {}, _get_layer)
        assert len(result) == 1

    def test_empty_edges(self) -> None:
        c = _cache()
        result = topology.resolve_cx_at_layer([10], 2, c, {}, _get_layer)
        assert len(result) == 0

    def test_multiple_nodes(self) -> None:
        c = _cache(
            raw_cx_edges={
                10: {2: np.array([[10, 100]], dtype=NODE_ID)},
                20: {2: np.array([[20, 200]], dtype=NODE_ID)},
            },
            resolver={100: {2: 300}, 200: {2: 400}},
        )
        result = topology.resolve_cx_at_layer([10, 20], 2, c, {}, _get_layer)
        assert len(result) == 2
        pairs = set((int(r[0]), int(r[1])) for r in result)
        assert (10, 300) in pairs
        assert (20, 400) in pairs


class TestAcxToCx:

    def test_remaps_col0(self) -> None:
        acx = {2: np.array([[50, 100], [60, 200]], dtype=NODE_ID)}
        cx = topology.acx_to_cx(acx, 999)
        assert 2 in cx
        assert np.all(cx[2][:, 0] == 999)
        assert set(int(x) for x in cx[2][:, 1]) == {100, 200}

    def test_empty_layer_skipped(self) -> None:
        acx = {2: np.array([], dtype=NODE_ID).reshape(0, 2), 3: np.array([[1, 2]], dtype=NODE_ID)}
        cx = topology.acx_to_cx(acx, 999)
        assert 2 not in cx
        assert 3 in cx

    def test_does_not_modify_original(self) -> None:
        original = np.array([[50, 100]], dtype=NODE_ID)
        topology.acx_to_cx({2: original}, 999)
        assert int(original[0, 0]) == 50


class TestStoreCxAndPropagate:

    def test_store_cx_splits_correctly(self) -> None:
        c = _cache()
        cx_edges = np.array([[10, 100], [10, 200], [20, 300], [20, 400], [30, 500]], dtype=NODE_ID)
        topology.store_cx_from_resolved(c, cx_edges, 2)
        assert len(c.cx_cache) == 3
        assert len(c.cx_cache[10][2]) == 2
        assert len(c.cx_cache[20][2]) == 2
        assert len(c.cx_cache[30][2]) == 1

    def test_store_cx_empty(self) -> None:
        c = _cache()
        topology.store_cx_from_resolved(c, types.empty_2d, 2)
        assert len(c.cx_cache) == 0

    def test_consistency(self) -> None:
        c = _cache(
            raw_cx_edges={10: {
                2: np.array([[10, 100]], dtype=NODE_ID),
                3: np.array([[10, 100]], dtype=NODE_ID),
            }},
            resolver={100: {2: 200, 3: 300}},
        )
        resolved_l2 = topology.resolve_cx_at_layer([10], 2, c, {}, _get_layer)
        topology.store_cx_from_resolved(c, resolved_l2, 2)
        assert 10 in c.cx_cache
        assert int(c.cx_cache[10][2][0, 1]) == 200
        assert int(c.raw_cx_edges[10][3][0, 1]) == 100

    def test_store_cx_multiple_layers(self) -> None:
        c = _cache()
        topology.store_cx_from_resolved(c, np.array([[10, 100]], dtype=NODE_ID), 2)
        topology.store_cx_from_resolved(c, np.array([[10, 200]], dtype=NODE_ID), 3)
        assert set(c.cx_cache[10].keys()) == {2, 3}
        assert int(c.cx_cache[10][2][0, 1]) == 100
        assert int(c.cx_cache[10][3][0, 1]) == 200


class TestResolveRemainingCx:

    def test_fills_missing_layers(self) -> None:
        c = _cache(
            raw_cx_edges={10: {
                2: np.array([[10, 100]], dtype=NODE_ID),
                3: np.array([[10, 100]], dtype=NODE_ID),
            }},
            resolver={100: {2: 200, 3: 300}},
        )
        c.cx_cache[10] = {2: np.array([[10, 200]], dtype=NODE_ID)}
        c.new_node_ids = {10}
        topology.resolve_remaining_cx(c, None, {})
        assert 3 in c.cx_cache[10]
        assert int(c.cx_cache[10][3][0, 1]) == 300

    def test_skips_already_resolved(self) -> None:
        existing = np.array([[10, 999]], dtype=NODE_ID)
        c = _cache(
            raw_cx_edges={10: {2: np.array([[10, 100]], dtype=NODE_ID)}},
            resolver={100: {2: 200}},
        )
        c.cx_cache[10] = {2: existing}
        c.new_node_ids = {10}
        topology.resolve_remaining_cx(c, None, {})
        assert int(c.cx_cache[10][2][0, 1]) == 999

    def test_filters_self_loops(self) -> None:
        c = _cache(
            raw_cx_edges={10: {3: np.array([[10, 100]], dtype=NODE_ID)}},
            resolver={100: {3: 10}},
        )
        c.new_node_ids = {10}
        topology.resolve_remaining_cx(c, None, {})
        assert 3 not in c.cx_cache.get(10, {})

    def test_includes_siblings_with_raw_cx(self) -> None:
        """resolve_remaining_cx fills layer 3+ CX for siblings (layer 2 handled by layer loop)."""
        l2a = _node(2, 1)
        l3a = _node(3, 1)
        c = _cache(
            raw_cx_edges={50: {3: np.array([[50, 100]], dtype=NODE_ID)}},
            resolver={100: {2: l2a}},
        )
        c.sibling_ids = {50}
        c.new_node_ids = set()

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        topology.resolve_remaining_cx(c, FakeLcg(), {l2a: l3a})
        assert 3 in c.cx_cache[50]

    def test_uses_child_to_parent_walk(self) -> None:
        l2a, l3a = _node(2, 1), _node(3, 1)
        c = _cache(
            raw_cx_edges={10: {3: np.array([[10, 100]], dtype=NODE_ID)}},
            resolver={100: {2: l2a}},
        )
        c.new_node_ids = {10}

        class FakeLcg:
            def get_chunk_layer(self, nid):
                return (int(nid) >> 56) & 0xFF

        topology.resolve_remaining_cx(c, FakeLcg(), {l2a: l3a})
        assert 3 in c.cx_cache[10]
        assert int(c.cx_cache[10][3][0, 1]) == l3a


class TestResolveSvToLayerEdgeCases:

    def test_old_to_new_then_child_to_parent(self) -> None:
        l2_old, l2_new = _node(2, 10), _node(2, 11)
        l3 = _node(3, 20)
        c = _cache(resolver={100: {2: l2_old}}, old_to_new={l2_old: l2_new})
        result = tree.resolve_sv_to_layer(100, 3, c, {l2_new: l3}, _get_layer)
        assert result == l3

    def test_resolver_picks_highest_below_target(self) -> None:
        c = _cache(resolver={100: {2: 200, 3: 300, 5: 500}})
        assert tree.resolve_sv_to_layer(100, 4, c, {}, _get_layer) == 300

    def test_walk_chain_multiple_hops(self) -> None:
        l2, l3, l4 = _node(2, 1), _node(3, 1), _node(4, 1)
        c = _cache(resolver={100: {2: l2}})
        result = tree.resolve_sv_to_layer(100, 4, c, {l2: l3, l3: l4}, _get_layer)
        assert result == l4


