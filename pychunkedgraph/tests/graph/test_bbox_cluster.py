"""Tests for pychunkedgraph.graph.sv_split.bbox_cluster.tight_bbox"""

from types import SimpleNamespace

import numpy as np
import pytest

from pychunkedgraph.graph.sv_split.bbox_cluster import (
    _cluster_labels,
    _coords_bbox,
    _pick_dense_pair,
    tight_bbox,
)


def _make_cg(resolution=(4, 4, 40), chunk_size=(256, 256, 256), vol_bounds=None):
    if vol_bounds is None:
        vol_bounds = np.array([[0, 0, 0], [100_000, 100_000, 10_000]]).T
    graph_config = SimpleNamespace(CHUNK_SIZE=np.asarray(chunk_size, dtype=int))
    meta = SimpleNamespace(
        resolution=np.asarray(resolution, dtype=float),
        graph_config=graph_config,
        voxel_bounds=np.asarray(vol_bounds, dtype=int),
    )
    return SimpleNamespace(meta=meta)


# ---------------- _cluster_labels ----------------

class TestClusterLabels:
    def test_all_in_one_cluster(self):
        coords = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float)
        labels = _cluster_labels(coords, eps_nm=100.0)
        assert len(np.unique(labels)) == 1

    def test_separated_become_two_clusters(self):
        coords = np.array([[0, 0, 0], [10, 0, 0], [10_000, 0, 0]], dtype=float)
        labels = _cluster_labels(coords, eps_nm=100.0)
        assert len(np.unique(labels)) == 2
        # first two share a label, third is separate
        assert labels[0] == labels[1] and labels[0] != labels[2]

    def test_empty(self):
        labels = _cluster_labels(np.empty((0, 3), dtype=float), eps_nm=100.0)
        assert labels.shape == (0,)

    def test_single_point(self):
        labels = _cluster_labels(np.array([[0, 0, 0]], dtype=float), eps_nm=100.0)
        assert labels.shape == (1,)


# ---------------- _pick_dense_pair ----------------

class TestPickDensePair:
    def test_one_cluster_each_side_picks_everything(self):
        src = np.array([[0, 0, 0], [1, 1, 1]])
        sink = np.array([[2, 2, 2]])
        src_labels = np.zeros(2, dtype=int)
        sink_labels = np.zeros(1, dtype=int)
        s_mask, t_mask = _pick_dense_pair(src_labels, sink_labels, src, sink)
        assert s_mask.all() and t_mask.all()

    def test_picks_smallest_volume_pair(self):
        # Two src clusters: tight cluster near origin, lone outlier far away.
        # One sink cluster near origin.
        src = np.array([[0, 0, 0], [1, 1, 1], [1000, 1000, 1000]])
        sink = np.array([[2, 2, 2]])
        src_labels = np.array([0, 0, 1], dtype=int)
        sink_labels = np.array([0], dtype=int)
        s_mask, t_mask = _pick_dense_pair(src_labels, sink_labels, src, sink)
        # tight cluster (label 0) wins
        assert s_mask.tolist() == [True, True, False]
        assert t_mask.tolist() == [True]

    def test_singleton_outliers_near_each_other_do_not_win(self):
        # Dense cluster has 3 seeds per side, far apart so bbox is large.
        # A singleton src + singleton sink land near each other → tiny bbox.
        # Size-first must pick the dense pair, not the singleton pair.
        src = np.array(
            [[0, 0, 0], [10, 0, 0], [20, 0, 0], [10_000, 10_000, 10_000]]
        )
        sink = np.array(
            [[100, 100, 100], [110, 100, 100], [120, 100, 100], [10_001, 10_001, 10_001]]
        )
        src_labels = np.array([0, 0, 0, 1], dtype=int)
        sink_labels = np.array([0, 0, 0, 1], dtype=int)
        s_mask, t_mask = _pick_dense_pair(src_labels, sink_labels, src, sink)
        assert s_mask.tolist() == [True, True, True, False]
        assert t_mask.tolist() == [True, True, True, False]


# ---------------- tight_bbox ----------------

class TestTightBbox:
    def test_matches_coords_bbox_when_single_cluster(self):
        cg = _make_cg()
        src = np.array([[100, 100, 100], [120, 110, 105]])
        sink = np.array([[200, 200, 200], [210, 195, 202]])
        bbs, bbe = tight_bbox(cg, src, sink)
        bbs_ref, bbe_ref = _coords_bbox(cg, src, sink)
        np.testing.assert_array_equal(bbs, bbs_ref)
        np.testing.assert_array_equal(bbe, bbe_ref)

    def test_outlier_excluded(self):
        cg = _make_cg()
        src = np.array([[100, 100, 100], [120, 110, 105], [50_000, 50_000, 5_000]])
        sink = np.array([[200, 200, 200]])
        bbs, bbe = tight_bbox(cg, src, sink)
        bbs_full, bbe_full = _coords_bbox(cg, src, sink)
        # tight bbox should be strictly inside the full bbox in volume
        vol = float(np.prod(bbe - bbs))
        vol_full = float(np.prod(bbe_full - bbs_full))
        assert vol < vol_full
        # tight bbox should match the no-outlier version
        bbs_no, bbe_no = _coords_bbox(cg, src[:2], sink)
        np.testing.assert_array_equal(bbs, bbs_no)
        np.testing.assert_array_equal(bbe, bbe_no)

    def test_degenerate_single_seed_per_side(self):
        cg = _make_cg()
        src = np.array([[100, 100, 100]])
        sink = np.array([[200, 200, 200]])
        bbs, bbe = tight_bbox(cg, src, sink)
        bbs_ref, bbe_ref = _coords_bbox(cg, src, sink)
        np.testing.assert_array_equal(bbs, bbs_ref)
        np.testing.assert_array_equal(bbe, bbe_ref)

    @pytest.mark.parametrize(
        "vol_offset",
        [(0, 0, 0), (1024, 1024, 64)],
    )
    def test_volume_bounds_clipped(self, vol_offset):
        # bbox grown by a one-chunk margin must clip to voxel_bounds
        cg = _make_cg(
            chunk_size=(256, 256, 256),
            vol_bounds=np.array([list(vol_offset), [vol_offset[0] + 500, vol_offset[1] + 500, vol_offset[2] + 500]]).T,
        )
        src = np.array([[vol_offset[0] + 10, vol_offset[1] + 10, vol_offset[2] + 10]])
        sink = np.array([[vol_offset[0] + 20, vol_offset[1] + 20, vol_offset[2] + 20]])
        bbs, bbe = tight_bbox(cg, src, sink)
        assert (bbs >= np.array(vol_offset)).all()
        assert (bbe <= np.array(vol_offset) + 500).all()

    def test_anisotropic_resolution_keeps_cluster_together(self):
        # Two src seeds spaced 30 voxels in z (= 1200 nm at 40 nm/vx)
        # at xy resolution 4 nm/vx, eps_nm = mean((256, 256, 256) * (4, 4, 40))
        # = mean(1024, 1024, 10240) = 4096 nm
        # 1200 nm < 4096 nm → should be one cluster.
        cg = _make_cg(resolution=(4, 4, 40), chunk_size=(256, 256, 256))
        src = np.array([[100, 100, 100], [100, 100, 130]])
        sink = np.array([[200, 200, 200]])
        bbs, bbe = tight_bbox(cg, src, sink)
        bbs_ref, bbe_ref = _coords_bbox(cg, src, sink)
        np.testing.assert_array_equal(bbs, bbs_ref)
        np.testing.assert_array_equal(bbe, bbe_ref)
