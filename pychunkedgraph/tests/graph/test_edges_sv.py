"""Comprehensive tests for pychunkedgraph.graph.edges_sv — edge routing after SV split."""

import numpy as np
import pytest

from pychunkedgraph.graph import basetypes
from pychunkedgraph.graph.exceptions import PostconditionError
from pychunkedgraph.graph.edges_sv import (
    _get_new_edges,
    _match_by_label,
    _match_by_proximity,
    _match_inf_unsplit,
    _match_partner,
    _expand_partners,
    validate_split_edges,
)

ROOT_ID = np.uint64(1)
OTHER_ROOT = np.uint64(2)


def _root_map(same_root, other_root=()):
    """Build sv_root_map: same_root SVs → ROOT_ID, other_root SVs → OTHER_ROOT."""
    m = {np.uint64(sv): ROOT_ID for sv in same_root}
    m.update({np.uint64(sv): OTHER_ROOT for sv in other_root})
    return m


def _call_get_new_edges(
    edges,
    affinities,
    areas,
    old_new_map,
    distances,
    distance_map,
    new_distance_map,
    sv_root_map,
    new_id_label_map=None,
):
    """Helper to call _get_new_edges with standard ROOT_ID and no cg/kdtrees."""
    return _get_new_edges(
        (edges, affinities, areas),
        old_new_map,
        distances,
        distance_map,
        new_distance_map,
        ROOT_ID,
        sv_root_map,
        None,
        [],
        new_id_label_map,
    )


# ============================================================
# Inf-affinity edge routing
# ============================================================
class TestInfEdgeRouting:
    def test_inf_to_unsplit_partner_closest_only(self):
        """Inf edge to unsplit active partner → only closest fragment gets it."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(50)

        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affs = np.array([np.inf], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([0], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 50, 101, 102])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 50, 101, 102])}
        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}
        # n1 closer to partner than n2
        distances = np.array([[5.0, 2.0, 0.0, 8.0], [6.0, 9.0, 8.0, 0.0]])

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
        )

        inf_edges = result_edges[np.isinf(result_affs)]
        # Only one fragment should connect to partner via inf
        partner_inf = [e for e in inf_edges if partner in e]
        assert len(partner_inf) == 1
        assert n1 in partner_inf[0]  # n1 is closer

    def test_inf_to_split_partner_label_matched(self):
        """Inf edge to split partner → matched by label, not proximity."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(201)  # also split, label 1

        edges = np.array([[10, 201]], dtype=basetypes.NODE_ID)
        affs = np.array([np.inf], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([0], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 101, 102, 201])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 101, 102, 201])}

        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}

        # n2 is CLOSER to partner, but wrong label
        distances = np.array([[5.0, 0.0, 8.0, 9.0], [6.0, 8.0, 0.0, 2.0]])
        label_map = {np.uint64(101): 1, np.uint64(102): 2, np.uint64(201): 1}

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
            label_map,
        )

        inf_edges = result_edges[np.isinf(result_affs)]
        # Should connect n1 (label 1) to partner (label 1), NOT n2
        for e in inf_edges:
            if partner in e:
                assert n1 in e, f"Expected label-matched n1, got {e}"
                assert n2 not in e

    def test_inf_to_split_partner_label_fallback(self):
        """Inf edge to split partner with no matching label → closest fragment."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(201)

        edges = np.array([[10, 201]], dtype=basetypes.NODE_ID)
        affs = np.array([np.inf], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([0], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 101, 102, 201])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 101, 102, 201])}

        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}

        # n2 closer to partner
        distances = np.array([[5.0, 0.0, 8.0, 9.0], [6.0, 8.0, 0.0, 2.0]])
        # Partner has label 3 which matches no fragment
        label_map = {np.uint64(101): 1, np.uint64(102): 2, np.uint64(201): 3}

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
            label_map,
        )

        inf_edges = result_edges[np.isinf(result_affs)]
        partner_edges = [e for e in inf_edges if partner in e]
        assert len(partner_edges) == 1
        # Fallback to closest → n2
        assert n2 in partner_edges[0]


# ============================================================
# Finite-affinity edge routing
# ============================================================
class TestFiniteEdgeRouting:
    def test_finite_to_active_partner_proximity(self):
        """Finite edge to active partner → fragments within threshold."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(50)

        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affs = np.array([0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 50, 101, 102])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 50, 101, 102])}
        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}
        # n1 within threshold (3 < 10), n2 outside (15 > 10)
        distances = np.array([[5.0, 3.0, 0.0, 8.0], [6.0, 15.0, 8.0, 0.0]])

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
        )

        finite_to_partner = [
            e
            for e, a in zip(result_edges, result_affs)
            if partner in e and not np.isinf(a)
        ]
        # Only n1 within threshold
        assert len(finite_to_partner) == 1
        assert n1 in finite_to_partner[0]

    def test_finite_to_active_partner_fallback(self):
        """Finite edge, none within threshold → closest fragment only."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(50)

        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affs = np.array([0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 50, 101, 102])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 50, 101, 102])}
        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}
        # Both outside threshold
        distances = np.array([[5.0, 15.0, 0.0, 8.0], [6.0, 20.0, 8.0, 0.0]])

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
        )

        finite_to_partner = [
            e
            for e, a in zip(result_edges, result_affs)
            if partner in e and not np.isinf(a)
        ]
        # Fallback: closest (n1 at dist 15)
        assert len(finite_to_partner) == 1
        assert n1 in finite_to_partner[0]

    def test_finite_to_inactive_partner_broadcast(self):
        """Finite edge to different-root partner → all fragments get it."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(99)  # different root

        edges = np.array([[10, 99]], dtype=basetypes.NODE_ID)
        affs = np.array([0.5], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([200], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 101, 102], [99])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 99, 101, 102])}

        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}

        distances = np.array([[5.0, 3.0, 0.0, 8.0], [6.0, 4.0, 8.0, 0.0]])

        result_edges, _, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
        )

        partner_edges = [e for e in result_edges if partner in e]
        fragments_connected = {e[0] if e[1] == partner else e[1] for e in partner_edges}
        assert n1 in fragments_connected
        assert n2 in fragments_connected


# ============================================================
# Partner expansion
# ============================================================
class TestPartnerExpansion:
    def test_partner_also_split(self):
        """Partner in old_new_map → expands to its fragments."""
        partners = np.array([np.uint64(50)])
        affs = np.array([0.9])
        areas = np.array([100])
        old_new_map = {np.uint64(50): {np.uint64(501), np.uint64(502)}}

        expanded_partners, expanded_affs, expanded_areas = _expand_partners(
            partners,
            affs,
            areas,
            old_new_map,
        )
        assert len(expanded_partners) == 2
        assert np.uint64(501) in expanded_partners
        assert np.uint64(502) in expanded_partners
        assert all(a == 0.9 for a in expanded_affs)


# ============================================================
# Fragment edges
# ============================================================
class TestFragmentEdges:
    def test_inter_fragment_edges(self):
        """Two fragments → 1 low-affinity inter-fragment edge."""
        old = np.uint64(10)
        n1, n2 = np.uint64(101), np.uint64(102)
        partner = np.uint64(50)

        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affs = np.array([0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2}}
        sv_root_map = _root_map([10, 50, 101, 102])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 50, 101, 102])}
        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1}
        distances = np.array([[5.0, 3.0, 0.0, 8.0], [6.0, 4.0, 8.0, 0.0]])

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
        )

        frag_edges = [
            (e, a) for e, a in zip(result_edges, result_affs) if set(e) == {n1, n2}
        ]
        assert len(frag_edges) == 1
        assert frag_edges[0][1] == pytest.approx(0.001)

    def test_inter_fragment_edges_three_way(self):
        """Three fragments → 3 inter-fragment edges."""
        old = np.uint64(10)
        n1, n2, n3 = np.uint64(101), np.uint64(102), np.uint64(103)
        partner = np.uint64(50)

        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affs = np.array([0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100], dtype=basetypes.EDGE_AREA)

        old_new_map = {old: {n1, n2, n3}}
        sv_root_map = _root_map([10, 50, 101, 102, 103])
        dist_map = {np.uint64(k): i for i, k in enumerate([10, 50, 101, 102, 103])}

        new_dist_map = {np.uint64(101): 0, np.uint64(102): 1, np.uint64(103): 2}

        distances = np.array(
            [
                [5.0, 3.0, 0.0, 8.0, 7.0],
                [6.0, 4.0, 8.0, 0.0, 6.0],
                [7.0, 5.0, 7.0, 6.0, 0.0],
            ]
        )

        result_edges, result_affs, _ = _call_get_new_edges(
            edges,
            affs,
            areas,
            old_new_map,
            distances,
            dist_map,
            new_dist_map,
            sv_root_map,
        )

        frag_pairs = {
            frozenset(e)
            for e, a in zip(result_edges, result_affs)
            if a == pytest.approx(0.001)
        }
        expected = {frozenset([n1, n2]), frozenset([n1, n3]), frozenset([n2, n3])}
        assert frag_pairs == expected


# ============================================================
# Validation
# ============================================================
class TestValidateSplitEdges:
    """All tests use multi-SV old_new_map to match production scenarios."""

    def _make_multi_sv_map(self):
        """Two old SVs split into 2 fragments each."""
        return {
            np.uint64(10): {np.uint64(101), np.uint64(102)},
            np.uint64(20): {np.uint64(201), np.uint64(202)},
        }

    def _make_valid_edges(self, old_new_map, extra_edges=None, extra_affs=None):
        """Build a valid edge set: inter-fragment + cross-SV finite edges."""
        edge_list = []
        aff_list = []
        # Inter-fragment edges for each old SV
        for new_ids in old_new_map.values():
            ids = sorted(new_ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edge_list.append([ids[i], ids[j]])
                    aff_list.append(0.001)
        # Cross-SV edges (fragments from different old SVs connected)
        all_frags = [list(v) for v in old_new_map.values()]
        if len(all_frags) > 1:
            edge_list.append([sorted(all_frags[0])[0], sorted(all_frags[1])[0]])
            aff_list.append(0.5)
        if extra_edges is not None:
            edge_list.extend(extra_edges)
            aff_list.extend(extra_affs)
        return (
            np.array(edge_list, dtype=basetypes.NODE_ID),
            np.array(aff_list, dtype=basetypes.EDGE_AFFINITY),
        )

    def test_valid_multi_sv_edges_pass(self):
        """Well-formed edges with multiple split SVs pass validation."""
        old_new_map = self._make_multi_sv_map()
        partner = np.uint64(50)
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, partner]],
            extra_affs=[np.inf],
        )
        validate_split_edges(edges, affs, old_new_map)

    def test_unsplit_partner_inf_to_fragments_from_different_old_svs(self):
        """Unsplit partner connecting via inf to fragments from different old SVs is valid."""
        old_new_map = self._make_multi_sv_map()
        label_map = {101: 0, 102: 1, 201: 0, 202: 1}
        partner = np.uint64(50)
        # Partner connects to one fragment from each old SV — different old SVs, same label
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, partner], [201, partner]],
            extra_affs=[np.inf, np.inf],
        )
        validate_split_edges(edges, affs, old_new_map, label_map)

    def test_allows_same_label_inf_to_unsplit_partner(self):
        """Multiple fragments with same label connecting to unsplit partner is valid."""
        old_new_map = self._make_multi_sv_map()
        label_map = {101: 0, 102: 0, 201: 0, 202: 1}
        partner = np.uint64(50)
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, partner], [102, partner]],
            extra_affs=[np.inf, np.inf],
        )
        validate_split_edges(edges, affs, old_new_map, label_map)

    def test_catches_cross_label_inf_bridge(self):
        """Fragments with different labels connecting to unsplit partner via inf is invalid."""
        old_new_map = self._make_multi_sv_map()
        label_map = {101: 0, 102: 1, 201: 0, 202: 1}
        partner = np.uint64(50)
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, partner], [102, partner]],
            extra_affs=[np.inf, np.inf],
        )
        with pytest.raises(PostconditionError, match="different labels"):
            validate_split_edges(edges, affs, old_new_map, label_map)

    def test_no_label_map_skips_inf_check(self):
        """Without label map, inf check is skipped (no false positives)."""
        old_new_map = self._make_multi_sv_map()
        partner = np.uint64(50)
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, partner], [102, partner]],
            extra_affs=[np.inf, np.inf],
        )
        validate_split_edges(edges, affs, old_new_map)  # no label_map, should not raise

    def test_catches_self_loop(self):
        """Validation rejects self-loop edges."""
        old_new_map = self._make_multi_sv_map()
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, 101]],
            extra_affs=[0.5],
        )
        with pytest.raises(PostconditionError, match="Self-loop"):
            validate_split_edges(edges, affs, old_new_map)

    def test_catches_missing_fragment_edges(self):
        """Validation rejects missing inter-fragment edges."""
        old_new_map = self._make_multi_sv_map()
        # Only include inter-fragment for second old SV, missing first
        edges = np.array([[201, 202], [101, 201]], dtype=basetypes.NODE_ID)
        affs = np.array([0.001, 0.5], dtype=basetypes.EDGE_AFFINITY)
        with pytest.raises(PostconditionError, match="Missing inter-fragment"):
            validate_split_edges(edges, affs, old_new_map)

    def test_catches_missing_replacement_edges(self):
        """Validation rejects old SV with no edges from any fragment."""
        old_new_map = self._make_multi_sv_map()
        # Only edges for first old SV's fragments
        edges = np.array([[101, 102]], dtype=basetypes.NODE_ID)
        affs = np.array([0.001], dtype=basetypes.EDGE_AFFINITY)
        with pytest.raises(PostconditionError, match="no replacement edges"):
            validate_split_edges(edges, affs, old_new_map)

    def test_empty_edges_pass(self):
        """Empty edge set passes validation."""
        validate_split_edges(
            np.array([], dtype=basetypes.NODE_ID).reshape(0, 2),
            np.array([], dtype=basetypes.EDGE_AFFINITY),
            {},
        )

    def test_inf_to_split_partner_allowed(self):
        """Inf edges to a split partner (in all_new_ids) are allowed from multiple fragments."""
        old_new_map = self._make_multi_sv_map()
        label_map = {101: 0, 102: 1, 201: 0, 202: 1}
        # 201 is a split partner (in all_new_ids), so inf from both 101 and 102 is fine
        edges, affs = self._make_valid_edges(
            old_new_map,
            extra_edges=[[101, 201], [102, 201]],
            extra_affs=[np.inf, np.inf],
        )
        validate_split_edges(edges, affs, old_new_map, label_map)
