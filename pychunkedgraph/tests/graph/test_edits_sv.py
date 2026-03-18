"""Tests for pychunkedgraph.graph.edits_sv"""

import numpy as np
import pytest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from pychunkedgraph.graph.edits_sv import (
    _voxel_crop,
    _parse_results,
    _get_new_edges,
    _match_by_label,
    _match_by_proximity,
)
from pychunkedgraph.graph import basetypes


# ============================================================
# Tests: _voxel_crop
# ============================================================
class TestVoxelCrop:
    def test_no_overlap(self):
        bbs = np.array([10, 20, 30])
        bbe = np.array([20, 30, 40])
        bbs_ = np.array([10, 20, 30])
        bbe_ = np.array([20, 30, 40])
        crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        # No offset and no clipping
        assert crop == np.s_[0:None, 0:None, 0:None]

    def test_with_padding(self):
        bbs = np.array([10, 20, 30])
        bbe = np.array([20, 30, 40])
        bbs_ = np.array([9, 19, 29])
        bbe_ = np.array([21, 31, 41])
        crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        # Start offset = bbs - bbs_ = (1, 1, 1)
        # End offset: bbe_ - bbe = (1,1,1) != 0, so end = -1
        assert crop == np.s_[1:-1, 1:-1, 1:-1]

    def test_partial_padding(self):
        bbs = np.array([10, 20, 30])
        bbe = np.array([20, 30, 40])
        bbs_ = np.array([9, 20, 30])
        bbe_ = np.array([21, 30, 40])
        crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        # Only x has offset
        assert crop == np.s_[1:-1, 0:None, 0:None]


# ============================================================
# Tests: _parse_results
# ============================================================
class TestParseResults:
    def test_basic(self):
        seg = np.array([[[100, 100], [100, 200]]], dtype=basetypes.NODE_ID)
        bbs = np.array([0, 0, 0])
        bbe = np.array([1, 2, 2])
        indices = np.array([[0, 0, 0], [0, 0, 1]])
        old_values = np.array([100, 100], dtype=basetypes.NODE_ID)
        new_values = np.array([300, 301], dtype=basetypes.NODE_ID)
        label_id_map = {1: np.uint64(300), 2: np.uint64(301)}
        results = [(indices, old_values, new_values, label_id_map)]

        updated_seg, old_new_map, slices, new_id_label_map = _parse_results(
            results, seg, bbs, bbe
        )
        assert updated_seg[0, 0, 0] == 300
        assert updated_seg[0, 0, 1] == 301
        assert 300 in old_new_map[100]
        assert 301 in old_new_map[100]
        assert new_id_label_map[np.uint64(300)] == 1
        assert new_id_label_map[np.uint64(301)] == 2

    def test_none_result_skipped(self):
        seg = np.array([[[100]]], dtype=basetypes.NODE_ID)
        bbs = np.array([0, 0, 0])
        bbe = np.array([1, 1, 1])
        results = [None]
        updated_seg, old_new_map, slices, new_id_label_map = _parse_results(
            results, seg, bbs, bbe
        )
        assert updated_seg[0, 0, 0] == 100
        assert len(old_new_map) == 0
        assert len(new_id_label_map) == 0

    def test_multiple_results(self):
        seg = np.array([[[100, 200]]], dtype=basetypes.NODE_ID)
        bbs = np.array([0, 0, 0])
        bbe = np.array([1, 1, 2])
        result1 = (
            np.array([[0, 0, 0]]),
            np.array([100], dtype=basetypes.NODE_ID),
            np.array([300], dtype=basetypes.NODE_ID),
            {1: np.uint64(300)},
        )
        result2 = (
            np.array([[0, 0, 1]]),
            np.array([200], dtype=basetypes.NODE_ID),
            np.array([400], dtype=basetypes.NODE_ID),
            {1: np.uint64(400)},
        )
        results = [result1, result2]

        updated_seg, old_new_map, slices, new_id_label_map = _parse_results(
            results, seg, bbs, bbe
        )
        assert updated_seg[0, 0, 0] == 300
        assert updated_seg[0, 0, 1] == 400
        assert 300 in old_new_map[100]
        assert 400 in old_new_map[200]


# ============================================================
# Tests: _get_new_edges
# ============================================================
class TestGetNewEdges:
    def test_with_active_and_inactive_partners(self):
        """Test with both active partners (in sv_ids) and inactive (not in sv_ids)."""
        old_sv = np.uint64(10)
        new_sv1 = np.uint64(101)
        new_sv2 = np.uint64(102)
        active_partner = np.uint64(50)  # in sv_ids -> active
        inactive_partner = np.uint64(99)  # not in sv_ids -> inactive

        edges = np.array(
            [
                [10, 50],
                [10, 99],
            ],
            dtype=basetypes.NODE_ID,
        )
        affinities = np.array([0.9, 0.5], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100, 200], dtype=basetypes.EDGE_AREA)

        old_new_map = {old_sv: {new_sv1, new_sv2}}
        sv_ids = np.array([10, 50, 101, 102], dtype=basetypes.NODE_ID)

        # distance_map: maps each label to its column index in the distance matrix
        distance_map = {
            np.uint64(10): 0,
            np.uint64(50): 1,
            np.uint64(101): 2,
            np.uint64(102): 3,
        }
        dist_vec = np.vectorize(distance_map.get)
        new_distance_map = {np.uint64(101): 0, np.uint64(102): 1}
        new_dist_vec = np.vectorize(new_distance_map.get)

        # Distances: (new_ids x all_ids)
        distances = np.array(
            [
                [5.0, 3.0, 0.0, 8.0],  # new_sv1
                [6.0, 4.0, 8.0, 0.0],  # new_sv2
            ]
        )

        result_edges, result_affs, result_areas = _get_new_edges(
            (edges, affinities, areas),
            sv_ids,
            old_new_map,
            distances,
            dist_vec,
            new_dist_vec,
        )
        # Should have:
        # - Inactive edges: new_sv1->99, new_sv2->99
        # - Active edges: new_ids -> 50 based on distance
        # - Fragment edges: new_sv1 <-> new_sv2
        assert len(result_edges) >= 3

    def test_edge_between_split_fragments(self):
        """Split fragments should have edges between them with low affinity."""
        old_sv = np.uint64(10)
        new_sv1 = np.uint64(101)
        new_sv2 = np.uint64(102)
        partner = np.uint64(50)

        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affinities = np.array([0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100], dtype=basetypes.EDGE_AREA)

        old_new_map = {old_sv: {new_sv1, new_sv2}}
        sv_ids = np.array([10, 50, 101, 102], dtype=basetypes.NODE_ID)

        distance_map = {
            np.uint64(10): 0,
            np.uint64(50): 1,
            np.uint64(101): 2,
            np.uint64(102): 3,
        }
        dist_vec = np.vectorize(distance_map.get)
        new_distance_map = {np.uint64(101): 0, np.uint64(102): 1}
        new_dist_vec = np.vectorize(new_distance_map.get)
        distances = np.array(
            [
                [5.0, 3.0, 0.0, 8.0],
                [6.0, 4.0, 8.0, 0.0],
            ]
        )

        result_edges, result_affs, result_areas = _get_new_edges(
            (edges, affinities, areas),
            sv_ids,
            old_new_map,
            distances,
            dist_vec,
            new_dist_vec,
        )
        # Check that a fragment-to-fragment edge exists
        fragment_edge_found = False
        for e in result_edges:
            if set(e) == {new_sv1, new_sv2}:
                fragment_edge_found = True
                break
        assert fragment_edge_found

    def test_empty_old_new_map(self):
        """Empty old_new_map should return empty results."""
        edges = np.array([[10, 50]], dtype=basetypes.NODE_ID)
        affinities = np.array([0.9], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([100], dtype=basetypes.EDGE_AREA)

        result_edges, result_affs, result_areas = _get_new_edges(
            (edges, affinities, areas),
            np.array([10], dtype=basetypes.NODE_ID),
            {},
            np.zeros((0, 0)),
            np.vectorize(lambda x: x),
            np.vectorize(lambda x: x),
        )
        assert len(result_edges) == 0

    def test_inf_affinity_uses_label_matching(self):
        """Inf-affinity (cross-chunk) edges should connect only same-label fragments."""
        old_sv = np.uint64(10)
        new_sv1 = np.uint64(101)  # label 1
        new_sv2 = np.uint64(102)  # label 2
        # partner is a cross-chunk fragment also from the split, label 1
        partner = np.uint64(201)

        edges = np.array([[10, 201]], dtype=basetypes.NODE_ID)
        affinities = np.array([np.inf], dtype=basetypes.EDGE_AFFINITY)
        areas = np.array([0], dtype=basetypes.EDGE_AREA)

        old_new_map = {old_sv: {new_sv1, new_sv2}}
        sv_ids = np.array([10, 101, 102, 201], dtype=basetypes.NODE_ID)

        distance_map = {
            np.uint64(10): 0,
            np.uint64(101): 1,
            np.uint64(102): 2,
            np.uint64(201): 3,
        }
        dist_vec = np.vectorize(distance_map.get)
        new_distance_map = {np.uint64(101): 0, np.uint64(102): 1}
        new_dist_vec = np.vectorize(new_distance_map.get)

        # new_sv2 (label 2) is closer to partner 201, but label doesn't match
        distances = np.array(
            [
                [5.0, 0.0, 8.0, 9.0],  # new_sv1 (label 1) — far from partner
                [6.0, 8.0, 0.0, 2.0],  # new_sv2 (label 2) — close to partner
            ]
        )

        new_id_label_map = {
            np.uint64(101): 1,
            np.uint64(102): 2,
            np.uint64(201): 1,  # same label as new_sv1
        }

        result_edges, result_affs, result_areas = _get_new_edges(
            (edges, affinities, areas),
            sv_ids,
            old_new_map,
            distances,
            dist_vec,
            new_dist_vec,
            new_id_label_map,
        )

        # The inf-affinity edge should connect new_sv1 (label 1) to partner 201 (label 1)
        # NOT new_sv2 (label 2) even though it's closer
        inf_edges = result_edges[np.isinf(result_affs)]
        for e in inf_edges:
            assert (
                new_sv2 not in e
            ), f"Inf-affinity edge {e} should not connect label-2 fragment to label-1 partner"
        # Verify new_sv1 <-> 201 inf edge exists
        found = any(set(e) == {new_sv1, partner} for e in inf_edges)
        assert found, "Expected inf-affinity edge between same-label fragments"


# ============================================================
# Tests: _match_by_label / _match_by_proximity
# ============================================================
class TestMatchByLabel:
    def test_matching_label(self):
        new_ids = np.array([101, 102], dtype=basetypes.NODE_ID)
        new_id_label_map = {np.uint64(101): 1, np.uint64(102): 2, np.uint64(201): 1}
        distances_row = np.array([9.0, 2.0])  # 102 is closer

        edges, affs, areas = _match_by_label(
            new_ids, np.uint64(201), np.inf, 0, new_id_label_map, distances_row
        )
        # Should pick 101 (label 1) not 102 (label 2, closer)
        assert all(np.uint64(101) in e for e in edges)
        assert np.uint64(102) not in edges.flatten()

    def test_fallback_to_closest(self):
        new_ids = np.array([101, 102], dtype=basetypes.NODE_ID)
        # partner label 3 doesn't match any new_id
        new_id_label_map = {np.uint64(101): 1, np.uint64(102): 2, np.uint64(201): 3}
        distances_row = np.array([9.0, 2.0])

        edges, affs, areas = _match_by_label(
            new_ids, np.uint64(201), np.inf, 0, new_id_label_map, distances_row
        )
        # Fallback: closest = 102
        assert np.uint64(102) in edges.flatten()


class TestMatchByProximity:
    def test_within_threshold(self):
        new_ids = np.array([101, 102], dtype=basetypes.NODE_ID)
        distances_row = np.array([3.0, 15.0])

        edges, affs, areas = _match_by_proximity(
            new_ids, np.uint64(50), 0.9, 100, distances_row, threshold=10
        )
        # Only 101 is within threshold
        assert len(edges) == 1
        assert np.uint64(101) in edges[0]

    def test_fallback_to_closest(self):
        new_ids = np.array([101, 102], dtype=basetypes.NODE_ID)
        distances_row = np.array([15.0, 20.0])  # both outside threshold

        edges, affs, areas = _match_by_proximity(
            new_ids, np.uint64(50), 0.9, 100, distances_row, threshold=10
        )
        # Fallback: closest = 101
        assert len(edges) == 1
        assert np.uint64(101) in edges[0]
