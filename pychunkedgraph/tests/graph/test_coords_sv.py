"""Tests for pychunkedgraph.graph.sv_split._coords."""

import numpy as np
from scipy.spatial import cKDTree

from pychunkedgraph.graph.sv_split._coords import build_coords_by_label


def _min_distance(coords_a, coords_b):
    """Brute min Euclidean distance between two (N,3) coord arrays."""
    return float(cKDTree(coords_a).query(coords_b, k=1)[0].min())


def _random_label_vol(shape=(40, 40, 40), n_labels=5, seed=0):
    """Vol where every voxel is a label in [1..n_labels] (no background gaps)."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, n_labels + 1, size=shape, dtype=np.uint64)


class TestBoundaryInvariant:
    """min-dist(boundary(A), boundary(B)) == min-dist(interior(A), interior(B))."""

    def test_random_label_pairs(self):
        vol = _random_label_vol(seed=1)
        full = build_coords_by_label(vol)
        boundary = build_coords_by_label(vol, boundary_only=True)
        labels = list(full.keys())
        assert len(labels) >= 2
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                d_full = _min_distance(full[a], full[b])
                d_bound = _min_distance(boundary[a], boundary[b])
                assert (
                    d_bound == d_full
                ), f"label pair ({a},{b}): boundary={d_bound} full={d_full}"

    def test_two_solid_blobs(self):
        """Two solid cubes: boundary preserves the closest-face geometry."""
        vol = np.zeros((20, 20, 20), dtype=np.uint64)
        vol[2:8, 2:8, 2:8] = 1
        vol[12:18, 12:18, 12:18] = 2
        full = build_coords_by_label(vol)
        boundary = build_coords_by_label(vol, boundary_only=True)
        d_full = _min_distance(full[1], full[2])
        d_bound = _min_distance(boundary[1], boundary[2])
        assert d_bound == d_full

    def test_touching_labels_adjacent_voxels(self):
        """Two labels sharing a face: boundary-mode reports the adjacent-voxel distance."""
        vol = np.zeros((10, 10, 10), dtype=np.uint64)
        vol[:, :, :5] = 1
        vol[:, :, 5:] = 2
        full = build_coords_by_label(vol)
        boundary = build_coords_by_label(vol, boundary_only=True)
        assert _min_distance(boundary[1], boundary[2]) == _min_distance(
            full[1], full[2]
        )
        assert _min_distance(boundary[1], boundary[2]) == 1.0


class TestBoundarySubset:
    """boundary_only output is a subset of the full output for every label."""

    def test_subset_per_label(self):
        vol = _random_label_vol(seed=2)
        full = build_coords_by_label(vol)
        boundary = build_coords_by_label(vol, boundary_only=True)
        for lab, b_coords in boundary.items():
            full_set = {tuple(r) for r in full[lab]}
            for row in b_coords:
                assert tuple(row) in full_set
