"""Tests for tree, topology, and stitch operations. No BigTable needed."""

import numpy as np
from pychunkedgraph.graph import basetypes, types

from . import resolver as topology
from . import tree
from .test_helpers import get_cx as _get_cx, get_parent as _get_parent, make_cache as _cache

NODE_ID = basetypes.NODE_ID


def _get_layer(nid: int) -> int:
    return (int(nid) >> 56) & 0xFF


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


class _FakeLcg:
    def get_chunk_layer(self, nid):
        return (int(nid) >> 56) & 0xFF


class TestResolveSvToLayer:

    def test_full_chain(self) -> None:
        sv, l2, l3, l4 = 100, _node(2, 1), _node(3, 1), _node(4, 1)
        c = _cache()
        c.put_parent(sv, l2)
        c.put_parent(l2, l3)
        c.put_parent(l3, l4)
        assert tree.resolve_sv_to_layer(sv, 3, c, _get_layer) == l3

    def test_best_below_target(self) -> None:
        sv, l2, l4 = 100, _node(2, 1), _node(4, 1)
        c = _cache()
        c.put_parent(sv, l2)
        c.put_parent(l2, l4)
        assert tree.resolve_sv_to_layer(sv, 3, c, _get_layer) == l2

    def test_sv_to_l2(self) -> None:
        sv, l2 = 100, _node(2, 1)
        c = _cache()
        c.put_parent(sv, l2)
        assert tree.resolve_sv_to_layer(sv, 2, c, _get_layer) == l2

    def test_walks_cache_parents(self) -> None:
        l2, l3 = _node(2, 10), _node(3, 20)
        c = _cache()
        c.put_parent(100, l2)
        c.put_parent(l2, l3)
        assert tree.resolve_sv_to_layer(100, 3, c, _get_layer) == l3

    def test_walk_stops_at_target_layer(self) -> None:
        l2, l3, l4 = _node(2, 10), _node(3, 20), _node(4, 30)
        c = _cache()
        c.put_parent(100, l2)
        c.put_parent(l2, l3)
        c.put_parent(l3, l4)
        assert tree.resolve_sv_to_layer(100, 3, c, _get_layer) == l3

    def test_unknown_sv_returns_self(self) -> None:
        c = _cache()
        assert tree.resolve_sv_to_layer(999, 2, c, _get_layer) == 999


class TestResolveCxAtLayer:

    def test_basic_resolution(self) -> None:
        n1, sv, l2 = _node(2, 10), 100, _node(2, 20)
        c = _cache(
            unresolved_acx={n1: {2: np.array([[n1, sv]], dtype=NODE_ID)}},
        )
        c.put_parent(sv, l2)
        result = topology.resolve_cx_at_layer([n1], 2, c, _get_layer)
        assert len(result) == 1
        assert int(result[0, 0]) == n1
        assert int(result[0, 1]) == l2

    def test_self_loops_filtered(self) -> None:
        n1, sv = _node(2, 10), 100
        c = _cache(
            unresolved_acx={n1: {2: np.array([[n1, sv]], dtype=NODE_ID)}},
        )
        c.put_parent(sv, n1)
        result = topology.resolve_cx_at_layer([n1], 2, c, _get_layer)
        assert len(result) == 0

    def test_duplicates_removed(self) -> None:
        n1, sv1, sv2, l2 = _node(2, 10), 100, 101, _node(2, 20)
        c = _cache(
            unresolved_acx={n1: {2: np.array([[n1, sv1], [n1, sv2]], dtype=NODE_ID)}},
        )
        c.put_parent(sv1, l2)
        c.put_parent(sv2, l2)
        result = topology.resolve_cx_at_layer([n1], 2, c, _get_layer)
        assert len(result) == 1

    def test_empty_edges(self) -> None:
        c = _cache()
        result = topology.resolve_cx_at_layer([_node(2, 10)], 2, c, _get_layer)
        assert len(result) == 0

    def test_multiple_nodes(self) -> None:
        n1, n2 = _node(2, 10), _node(2, 20)
        sv1, sv2 = 100, 200
        l2a, l2b = _node(2, 30), _node(2, 40)
        c = _cache(
            unresolved_acx={
                n1: {2: np.array([[n1, sv1]], dtype=NODE_ID)},
                n2: {2: np.array([[n2, sv2]], dtype=NODE_ID)},
            },
        )
        c.put_parent(sv1, l2a)
        c.put_parent(sv2, l2b)
        result = topology.resolve_cx_at_layer([n1, n2], 2, c, _get_layer)
        assert len(result) == 2
        pairs = set((int(r[0]), int(r[1])) for r in result)
        assert (n1, l2a) in pairs
        assert (n2, l2b) in pairs


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
        assert _get_cx(c, 10) and _get_cx(c, 20) and _get_cx(c, 30)
        assert len(_get_cx(c, 10)[2]) == 2
        assert len(_get_cx(c, 20)[2]) == 2
        assert len(_get_cx(c, 30)[2]) == 1

    def test_store_cx_empty(self) -> None:
        c = _cache()
        topology.store_cx_from_resolved(c, types.empty_2d, 2)
        assert not _get_cx(c, 10)

    def test_consistency(self) -> None:
        n1, sv, l2, l3 = _node(2, 10), 100, _node(2, 20), _node(3, 1)
        c = _cache(
            unresolved_acx={n1: {
                2: np.array([[n1, sv]], dtype=NODE_ID),
                3: np.array([[n1, sv]], dtype=NODE_ID),
            }},
        )
        c.put_parent(sv, l2)
        c.put_parent(l2, l3)
        resolved_l2 = topology.resolve_cx_at_layer([n1], 2, c, _get_layer)
        topology.store_cx_from_resolved(c, resolved_l2, 2)
        assert _get_cx(c, n1)
        assert int(_get_cx(c, n1)[2][0, 1]) == l2
        assert int(c.unresolved_acx[n1][3][0, 1]) == sv

    def test_store_cx_multiple_layers(self) -> None:
        c = _cache()
        topology.store_cx_from_resolved(c, np.array([[10, 100]], dtype=NODE_ID), 2)
        topology.store_cx_from_resolved(c, np.array([[10, 200]], dtype=NODE_ID), 3)
        assert set(_get_cx(c, 10).keys()) == {2, 3}
        assert int(_get_cx(c, 10)[2][0, 1]) == 100
        assert int(_get_cx(c, 10)[3][0, 1]) == 200


class TestResolveRemainingCx:

    def test_fills_missing_layers(self) -> None:
        n3 = _node(3, 10)
        l2 = _node(2, 1)
        l3 = _node(3, 1)
        c = _cache(
            unresolved_acx={n3: {
                3: np.array([[n3, 100]], dtype=NODE_ID),
            }},
        )
        c.put_parent(100, l2)
        c.put_parent(l2, l3)
        c.new_node_ids = {n3}
        topology.resolve_remaining_cx(c, _FakeLcg())
        assert 3 in _get_cx(c, n3)
        assert int(_get_cx(c, n3)[3][0, 1]) == l3

    def test_skips_already_resolved(self) -> None:
        n3 = _node(3, 10)
        l3_target = _node(3, 2)
        existing = np.array([[n3, 999]], dtype=NODE_ID)
        c = _cache(
            unresolved_acx={n3: {3: np.array([[n3, 100]], dtype=NODE_ID)}},
        )
        c.put_parent(100, l3_target)
        c.put_cx(n3, {3: existing})
        c.new_node_ids = {n3}
        topology.resolve_remaining_cx(c, _FakeLcg())
        assert int(_get_cx(c, n3)[3][0, 1]) == 999

    def test_filters_self_loops(self) -> None:
        n3 = _node(3, 10)
        c = _cache(
            unresolved_acx={n3: {3: np.array([[n3, 100]], dtype=NODE_ID)}},
        )
        c.put_parent(100, n3)
        c.new_node_ids = {n3}
        topology.resolve_remaining_cx(c, _FakeLcg())
        assert 3 not in _get_cx(c, n3)

    def test_includes_new_nodes_with_raw_cx(self) -> None:
        n3 = _node(3, 50)
        l2a = _node(2, 1)
        l3a = _node(3, 1)
        c = _cache(
            unresolved_acx={n3: {3: np.array([[n3, 100]], dtype=NODE_ID)}},
        )
        c.new_node_ids = {n3}
        c.put_parent(100, l2a)
        c.put_parent(l2a, l3a)
        topology.resolve_remaining_cx(c, _FakeLcg())
        assert 3 in _get_cx(c, n3)

    def test_uses_cache_parents_walk(self) -> None:
        n3 = _node(3, 10)
        l2a, l3a = _node(2, 1), _node(3, 1)
        c = _cache(
            unresolved_acx={n3: {3: np.array([[n3, 100]], dtype=NODE_ID)}},
        )
        c.new_node_ids = {n3}
        c.put_parent(100, l2a)
        c.put_parent(l2a, l3a)
        topology.resolve_remaining_cx(c, _FakeLcg())
        assert 3 in _get_cx(c, n3)
        assert int(_get_cx(c, n3)[3][0, 1]) == l3a


class TestResolveSvToLayerEdgeCases:

    def test_sv_to_l3_via_cache(self) -> None:
        l2_new = _node(2, 11)
        l3 = _node(3, 20)
        c = _cache()
        c.put_parent(100, l2_new)
        c.put_parent(l2_new, l3)
        result = tree.resolve_sv_to_layer(100, 3, c, _get_layer)
        assert result == l3

    def test_stops_at_highest_below_target(self) -> None:
        sv, l2, l3, l5 = 100, _node(2, 1), _node(3, 1), _node(5, 1)
        c = _cache()
        c.put_parent(sv, l2)
        c.put_parent(l2, l3)
        c.put_parent(l3, l5)
        assert tree.resolve_sv_to_layer(sv, 4, c, _get_layer) == l3

    def test_walk_chain_multiple_hops(self) -> None:
        l2, l3, l4 = _node(2, 1), _node(3, 1), _node(4, 1)
        c = _cache()
        c.put_parent(100, l2)
        c.put_parent(l2, l3)
        c.put_parent(l3, l4)
        result = tree.resolve_sv_to_layer(100, 4, c, _get_layer)
        assert result == l4


