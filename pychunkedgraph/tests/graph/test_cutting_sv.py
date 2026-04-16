"""Tests for pychunkedgraph.graph.cutting_sv"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pychunkedgraph.graph.cutting_sv import (
    _cc_label_26,
    _largest_component_id,
    _to_zyx_sampling,
    _to_internal_zyx_volume,
    _from_internal_zyx_volume,
    _seeds_to_zyx,
    _seeds_from_zyx,
    _extract_mask_boundary,
    _downsample_points,
    snap_seeds_to_segment,
    _compute_edt,
    _upsample_bool,
    _upsample_labels,
    build_kdtrees_by_label,
    pairwise_min_distance_two_sets,
    split_supervoxel_growing,
    connect_both_seeds_via_ridge,
    split_supervoxel_helper,
)


# ============================================================
# Helper: create a simple 3D binary mask with two seed regions
# ============================================================
def _make_dumbbell_mask(shape=(20, 30, 30)):
    """
    Create a dumbbell-shaped mask: two blobs connected by a thin bridge.
    Returns (mask, seeds_a_zyx, seeds_b_zyx) all in ZYX order.
    """
    mask = np.zeros(shape, dtype=bool)
    Z, Y, X = shape
    # blob A: centered at (Z//2, Y//4, X//4)
    cz, cy, cx = Z // 2, Y // 4, X // 4
    r = min(Z, Y, X) // 5
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r**2:
                    mask[z, y, x] = True

    # blob B: centered at (Z//2, 3*Y//4, 3*X//4)
    cz2, cy2, cx2 = Z // 2, 3 * Y // 4, 3 * X // 4
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if (z - cz2) ** 2 + (y - cy2) ** 2 + (x - cx2) ** 2 <= r**2:
                    mask[z, y, x] = True

    # bridge between the two
    mid_y = Y // 2
    mid_x = X // 2
    mask[cz - 1 : cz + 2, cy : cy2 + 1, mid_x - 1 : mid_x + 2] = True

    seeds_a = np.array([[cz, cy, cx]])
    seeds_b = np.array([[cz2, cy2, cx2]])
    return mask, seeds_a, seeds_b


# ============================================================
# Tests: CC label and largest component
# ============================================================
class TestCCLabel26:
    def test_single_component(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        lbl, ncomp = _cc_label_26(mask)
        assert ncomp == 1
        assert lbl[2, 2, 2] > 0
        assert lbl[0, 0, 0] == 0

    def test_two_components(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[7:9, 7:9, 7:9] = True
        lbl, ncomp = _cc_label_26(mask)
        assert ncomp == 2
        assert lbl[2, 2, 2] != lbl[7, 7, 7]

    def test_empty(self):
        mask = np.zeros((3, 3, 3), dtype=bool)
        lbl, ncomp = _cc_label_26(mask)
        assert ncomp == 0

    def test_full(self):
        mask = np.ones((4, 4, 4), dtype=bool)
        lbl, ncomp = _cc_label_26(mask)
        assert ncomp == 1


class TestLargestComponentId:
    def test_single_component(self):
        lbl = np.zeros((5, 5, 5), dtype=np.int32)
        lbl[1:4, 1:4, 1:4] = 1
        assert _largest_component_id(lbl) == 1

    def test_two_components_picks_largest(self):
        lbl = np.zeros((10, 10, 10), dtype=np.int32)
        lbl[0:2, 0:2, 0:2] = 1  # 8 voxels
        lbl[3:8, 3:8, 3:8] = 2  # 125 voxels
        assert _largest_component_id(lbl) == 2

    def test_all_background(self):
        lbl = np.zeros((3, 3, 3), dtype=np.int32)
        assert _largest_component_id(lbl) == 0


# ============================================================
# Tests: Order/utility helpers
# ============================================================
class TestToZyxSampling:
    def test_xyz_order(self):
        result = _to_zyx_sampling((8.0, 8.0, 40.0), "xyz")
        assert result == (40.0, 8.0, 8.0)

    def test_zyx_order(self):
        result = _to_zyx_sampling((40.0, 8.0, 8.0), "zyx")
        assert result == (40.0, 8.0, 8.0)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="vox_order"):
            _to_zyx_sampling((1, 1, 1), "abc")


class TestToInternalZyxVolume:
    def test_zyx_passthrough(self):
        vol = np.zeros((3, 4, 5))
        result, transposed = _to_internal_zyx_volume(vol, "zyx")
        assert result is vol
        assert not transposed

    def test_xyz_transpose(self):
        vol = np.zeros((5, 4, 3))  # X=5, Y=4, Z=3
        result, transposed = _to_internal_zyx_volume(vol, "xyz")
        assert result.shape == (3, 4, 5)
        assert transposed

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="vol_order"):
            _to_internal_zyx_volume(np.zeros((3, 3, 3)), "abc")


class TestFromInternalZyxVolume:
    def test_zyx_passthrough(self):
        vol = np.zeros((3, 4, 5))
        result = _from_internal_zyx_volume(vol, "zyx")
        assert result is vol

    def test_xyz_transpose(self):
        vol = np.zeros((3, 4, 5))  # Z=3, Y=4, X=5
        result = _from_internal_zyx_volume(vol, "xyz")
        assert result.shape == (5, 4, 3)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="vol_order"):
            _from_internal_zyx_volume(np.zeros((3, 3, 3)), "abc")


class TestSeedsToZyx:
    def test_xyz_to_zyx(self):
        seeds = np.array([[10, 20, 30]])  # x, y, z
        result = _seeds_to_zyx(seeds, "xyz")
        np.testing.assert_array_equal(result, [[30, 20, 10]])

    def test_zyx_passthrough(self):
        seeds = np.array([[30, 20, 10]])  # z, y, x
        result = _seeds_to_zyx(seeds, "zyx")
        np.testing.assert_array_equal(result, [[30, 20, 10]])

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="seed_order"):
            _seeds_to_zyx(np.array([[1, 2, 3]]), "abc")


class TestSeedsFromZyx:
    def test_xyz_output(self):
        seeds = np.array([[30, 20, 10]])  # z, y, x
        result = _seeds_from_zyx(seeds, "xyz")
        np.testing.assert_array_equal(result, [[10, 20, 30]])

    def test_zyx_passthrough(self):
        seeds = np.array([[30, 20, 10]])
        result = _seeds_from_zyx(seeds, "zyx")
        np.testing.assert_array_equal(result, [[30, 20, 10]])

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="seed_order"):
            _seeds_from_zyx(np.array([[1, 2, 3]]), "abc")

    def test_roundtrip(self):
        original = np.array([[10, 20, 30], [40, 50, 60]])
        zyx = _seeds_to_zyx(original, "xyz")
        recovered = _seeds_from_zyx(zyx, "xyz")
        np.testing.assert_array_equal(original, recovered)


# ============================================================
# Tests: Snapping (KDTree-based)
# ============================================================
class TestExtractMaskBoundary:
    def test_basic_boundary(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        boundary = _extract_mask_boundary(mask, erosion_iters=1)
        # Interior should not be boundary
        assert boundary[5, 5, 5] == False
        # Edge should be boundary
        assert boundary[2, 2, 2] == True
        # Boundary must be subset of mask
        assert np.all(boundary <= mask)

    def test_zero_erosion_returns_copy(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        result = _extract_mask_boundary(mask, erosion_iters=0)
        np.testing.assert_array_equal(result, mask)

    def test_thin_structure_all_boundary(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, :, :] = True  # single slice - all boundary
        boundary = _extract_mask_boundary(mask, erosion_iters=1)
        # For a single-voxel-thick structure, all voxels are boundary
        assert boundary.sum() > 0


class TestDownsamplePoints:
    def test_stride(self):
        pts = np.arange(30).reshape(10, 3)
        result = _downsample_points(pts, mode="stride", stride=2)
        assert len(result) == 5
        np.testing.assert_array_equal(result[0], pts[0])
        np.testing.assert_array_equal(result[1], pts[2])

    def test_random(self):
        rng = np.random.default_rng(42)
        pts = np.arange(300).reshape(100, 3)
        result = _downsample_points(pts, mode="random", target=10, rng=rng)
        assert len(result) == 10

    def test_random_target_larger_than_n(self):
        pts = np.arange(15).reshape(5, 3)
        result = _downsample_points(pts, mode="random", target=50)
        assert len(result) == 5

    def test_empty_returns_empty(self):
        pts = np.empty((0, 3))
        result = _downsample_points(pts, mode="stride")
        assert len(result) == 0

    def test_invalid_mode_raises(self):
        pts = np.arange(9).reshape(3, 3)
        with pytest.raises(ValueError, match="downsample mode"):
            _downsample_points(pts, mode="invalid")


class TestSnapSeedsToSegment:
    def test_basic_snap(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        seeds = np.array([[0.0, 0.0, 0.0]])  # far outside
        result = snap_seeds_to_segment(
            seeds,
            mask,
            mask_order="zyx",
            use_boundary=False,
            downsample=False,
        )
        # Snapped seed should be on the mask
        x, y, z = result[0]
        assert mask[z, y, x] == True

    def test_seed_inside_mask(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        seeds = np.array([[5.0, 5.0, 5.0]])  # inside the mask
        result = snap_seeds_to_segment(
            seeds,
            mask,
            mask_order="zyx",
            use_boundary=False,
            downsample=False,
        )
        x, y, z = result[0]
        assert mask[z, y, x] == True

    def test_with_boundary_and_downsample(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        seeds = np.array([[0.0, 0.0, 0.0]])
        result = snap_seeds_to_segment(
            seeds,
            mask,
            mask_order="zyx",
            use_boundary=True,
            downsample=True,
            downsample_mode="stride",
            downsample_stride=2,
        )
        x, y, z = result[0]
        assert mask[z, y, x] == True

    def test_xyz_mask_order(self):
        # mask_order='xyz' means shape is (X, Y, Z)
        mask_xyz = np.zeros((10, 12, 8), dtype=bool)
        mask_xyz[3:7, 3:9, 2:6] = True
        seeds = np.array([[5.0, 6.0, 4.0]])  # xyz coords
        result = snap_seeds_to_segment(
            seeds,
            mask_xyz,
            mask_order="xyz",
            use_boundary=False,
            downsample=False,
        )
        x, y, z = result[0]
        assert mask_xyz[x, y, z] == True

    def test_empty_mask_raises(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        seeds = np.array([[2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="no True voxels"):
            snap_seeds_to_segment(
                seeds, mask, mask_order="zyx", use_boundary=False, downsample=False
            )

    def test_non_3d_mask_raises(self):
        mask = np.zeros((5, 5), dtype=bool)
        seeds = np.array([[2.0, 2.0]])
        with pytest.raises(ValueError, match="3D"):
            snap_seeds_to_segment(seeds, mask, mask_order="zyx")

    def test_return_index(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        seeds = np.array([[0.0, 0.0, 0.0]])
        result, idx = snap_seeds_to_segment(
            seeds,
            mask,
            mask_order="zyx",
            use_boundary=False,
            downsample=False,
            return_index=True,
        )
        assert idx.shape[0] == 1

    def test_multiple_seeds(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        seeds = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])
        result = snap_seeds_to_segment(
            seeds,
            mask,
            mask_order="zyx",
            use_boundary=False,
            downsample=False,
        )
        assert result.shape == (2, 3)
        for i in range(2):
            x, y, z = result[i]
            assert mask[z, y, x] == True

    def test_voxel_size_anisotropic(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        seeds = np.array([[0.0, 0.0, 0.0]])
        result = snap_seeds_to_segment(
            seeds,
            mask,
            mask_order="zyx",
            voxel_size=(8.0, 8.0, 40.0),
            use_boundary=False,
            downsample=False,
        )
        x, y, z = result[0]
        assert mask[z, y, x] == True

    def test_invalid_mask_order(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        with pytest.raises(ValueError, match="mask_order"):
            snap_seeds_to_segment(
                np.array([[2, 2, 2]]),
                mask,
                mask_order="bad",
                use_boundary=False,
                downsample=False,
            )


# ============================================================
# Tests: EDT
# ============================================================
class TestComputeEdt:
    def test_basic(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        dist = _compute_edt(mask, (1.0, 1.0, 1.0))
        assert dist.shape == mask.shape
        # Center should have highest distance
        assert dist[5, 5, 5] > dist[3, 3, 3]
        # Outside mask should be zero
        assert dist[0, 0, 0] == 0.0

    def test_anisotropic_sampling(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        dist = _compute_edt(mask, (40.0, 8.0, 8.0))
        assert dist.shape == mask.shape
        assert dist[5, 5, 5] > 0


# ============================================================
# Tests: Upsampling
# ============================================================
class TestUpsample:
    def test_upsample_bool(self):
        mask = np.array([[[True, False], [False, True]]])  # shape (1, 2, 2)
        result = _upsample_bool(mask, (2, 2, 2), (2, 4, 4))
        assert result.shape == (2, 4, 4)
        assert result[0, 0, 0] == True
        assert result[0, 0, 2] == False

    def test_upsample_labels(self):
        lbl = np.array([[[1, 2], [3, 0]]])  # shape (1, 2, 2)
        result = _upsample_labels(lbl, (2, 2, 2), (2, 4, 4))
        assert result.shape == (2, 4, 4)
        assert result[0, 0, 0] == 1
        assert result[0, 0, 2] == 2

    def test_upsample_with_trimming(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        result = _upsample_bool(mask, (3, 3, 3), (5, 5, 5))
        assert result.shape == (5, 5, 5)


# ============================================================
# Tests: build_kdtrees_by_label
# ============================================================
class TestBuildKdtreesByLabel:
    def test_basic(self):
        vol = np.zeros((5, 5, 5), dtype=int)
        vol[1, 1, 1] = 1
        vol[3, 3, 3] = 2
        vol[3, 3, 4] = 2
        trees, counts = build_kdtrees_by_label(vol)
        assert 1 in trees
        assert 2 in trees
        assert 0 not in trees
        assert counts[1] == 1
        assert counts[2] == 2

    def test_empty_volume(self):
        vol = np.zeros((3, 3, 3), dtype=int)
        trees, counts = build_kdtrees_by_label(vol)
        assert len(trees) == 0
        assert len(counts) == 0

    def test_non_zero_background(self):
        vol = np.full((5, 5, 5), 99, dtype=int)
        vol[2, 2, 2] = 1
        trees, counts = build_kdtrees_by_label(vol, background=99)
        assert 1 in trees
        assert 99 not in trees

    def test_min_points_filter(self):
        vol = np.zeros((5, 5, 5), dtype=int)
        vol[1, 1, 1] = 1  # 1 voxel
        vol[2:4, 2:4, 2:4] = 2  # 8 voxels
        trees, counts = build_kdtrees_by_label(vol, min_points=5)
        assert 1 not in trees
        assert 2 in trees

    def test_non_3d_raises(self):
        vol = np.zeros((5, 5), dtype=int)
        with pytest.raises(ValueError, match="3D"):
            build_kdtrees_by_label(vol)

    def test_uint64_labels(self):
        vol = np.zeros((5, 5, 5), dtype=np.uint64)
        vol[1, 1, 1] = np.uint64(2**60)
        trees, counts = build_kdtrees_by_label(vol)
        assert int(2**60) in trees


# ============================================================
# Tests: pairwise_min_distance_two_sets
# ============================================================
class TestPairwiseMinDistanceTwoSets:
    def _make_tree(self, points):
        return cKDTree(np.array(points, dtype=np.float32))

    def test_basic_exact(self):
        tA = self._make_tree([[0, 0, 0]])
        tB = self._make_tree([[3, 4, 0]])
        D = pairwise_min_distance_two_sets([tA], [tB])
        assert D.shape == (1, 1)
        assert D[0, 0] == pytest.approx(5.0)

    def test_multiple_trees(self):
        tA1 = self._make_tree([[0, 0, 0]])
        tA2 = self._make_tree([[10, 10, 10]])
        tB1 = self._make_tree([[1, 0, 0]])
        D = pairwise_min_distance_two_sets([tA1, tA2], [tB1])
        assert D.shape == (2, 1)
        assert D[0, 0] < D[1, 0]

    def test_empty_sets(self):
        D = pairwise_min_distance_two_sets([], [])
        assert D.shape == (0, 0)

    def test_one_empty(self):
        tA = self._make_tree([[0, 0, 0]])
        D = pairwise_min_distance_two_sets([tA], [])
        assert D.shape == (1, 0)

    def test_cutoff_mode(self):
        tA = self._make_tree([[0, 0, 0]])
        tB = self._make_tree([[100, 100, 100]])
        D = pairwise_min_distance_two_sets([tA], [tB], max_distance=5.0)
        assert D[0, 0] == np.inf

    def test_cutoff_mode_within_range(self):
        tA = self._make_tree([[0, 0, 0]])
        tB = self._make_tree([[1, 0, 0]])
        D = pairwise_min_distance_two_sets([tA], [tB], max_distance=5.0)
        assert D[0, 0] == pytest.approx(1.0)

    def test_multi_point_trees(self):
        tA = self._make_tree([[0, 0, 0], [10, 10, 10]])
        tB = self._make_tree([[1, 0, 0], [11, 10, 10]])
        D = pairwise_min_distance_two_sets([tA], [tB])
        assert D.shape == (1, 1)
        assert D[0, 0] == pytest.approx(1.0)

    def test_asymmetric_tree_sizes(self):
        # tA has many points, tB has few
        tA = self._make_tree(np.random.default_rng(0).random((100, 3)) * 10)
        tB = self._make_tree([[5, 5, 5]])
        D = pairwise_min_distance_two_sets([tA], [tB])
        assert D.shape == (1, 1)
        assert D[0, 0] >= 0


# ============================================================
# Tests: split_supervoxel_growing
# ============================================================
class TestSplitSupervoxelGrowing:
    def test_basic_split_xyz(self):
        """Split a dumbbell into two labels."""
        mask, seeds_a_zyx, seeds_b_zyx = _make_dumbbell_mask(shape=(20, 30, 30))
        # Convert to xyz
        mask_xyz = np.transpose(mask, (2, 1, 0))
        seeds_a_xyz = seeds_a_zyx[:, [2, 1, 0]]
        seeds_b_xyz = seeds_b_zyx[:, [2, 1, 0]]

        result = split_supervoxel_growing(
            mask_xyz,
            seeds_a_xyz,
            seeds_b_xyz,
            voxel_size=(1.0, 1.0, 1.0),
            vol_order="xyz",
            vox_order="xyz",
            seed_order="xyz",
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
            enforce_single_cc=True,
            raise_if_multi_cc=False,
        )
        assert result.shape == mask_xyz.shape
        # Should contain labels 1 and 2
        assert np.any(result == 1)
        assert np.any(result == 2)
        # Labels should only be where mask is True
        assert np.all((result > 0) <= mask_xyz)

    def test_basic_split_zyx(self):
        """Split using ZYX order."""
        mask, seeds_a, seeds_b = _make_dumbbell_mask(shape=(20, 30, 30))
        result = split_supervoxel_growing(
            mask,
            seeds_a,
            seeds_b,
            voxel_size=(1.0, 1.0, 1.0),
            vol_order="zyx",
            vox_order="zyx",
            seed_order="zyx",
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
            enforce_single_cc=True,
            raise_if_multi_cc=False,
        )
        assert result.shape == mask.shape
        assert np.any(result == 1)
        assert np.any(result == 2)

    def test_empty_seeds_returns_label1(self):
        """With no seeds on one side, the entire mask gets label 1."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        seeds_a = np.array([[5, 5, 5]])
        seeds_b = np.empty((0, 3), dtype=int)
        result = split_supervoxel_growing(
            mask,
            seeds_a,
            seeds_b,
            vol_order="zyx",
            vox_order="zyx",
            seed_order="zyx",
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
        )
        assert np.all(result[mask] == 1)

    def test_with_downsample_geodesic(self):
        """Test downsampled geodesic computation."""
        mask, seeds_a, seeds_b = _make_dumbbell_mask(shape=(20, 30, 30))
        result = split_supervoxel_growing(
            mask,
            seeds_a,
            seeds_b,
            voxel_size=(1.0, 1.0, 1.0),
            vol_order="zyx",
            vox_order="zyx",
            seed_order="zyx",
            downsample_geodesic=(1, 2, 2),
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
            enforce_single_cc=True,
            raise_if_multi_cc=False,
        )
        assert result.shape == mask.shape
        assert np.any(result == 1)
        assert np.any(result == 2)


# ============================================================
# Tests: connect_both_seeds_via_ridge
# ============================================================
class TestConnectBothSeedsViaRidge:
    def test_basic_connection(self):
        mask, seeds_a_zyx, seeds_b_zyx = _make_dumbbell_mask(shape=(20, 30, 30))
        mask_xyz = np.transpose(mask, (2, 1, 0))
        seeds_a_xyz = seeds_a_zyx[:, [2, 1, 0]]
        seeds_b_xyz = seeds_b_zyx[:, [2, 1, 0]]

        A_aug, B_aug, okA, okB = connect_both_seeds_via_ridge(
            mask_xyz,
            seeds_a_xyz,
            seeds_b_xyz,
            voxel_size=(1.0, 1.0, 1.0),
            vol_order="xyz",
            vox_order="xyz",
            seed_order="xyz",
            downsample=(1, 1, 1),
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
        )
        assert okA
        assert okB
        # Augmented seeds should be at least as many as originals
        assert len(A_aug) >= len(seeds_a_xyz)
        assert len(B_aug) >= len(seeds_b_xyz)

    def test_single_seed_per_team(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        mask_xyz = np.transpose(mask, (2, 1, 0))
        seeds_a = np.array([[4, 4, 4]])
        seeds_b = np.array([[6, 6, 6]])

        A_aug, B_aug, okA, okB = connect_both_seeds_via_ridge(
            mask_xyz,
            seeds_a,
            seeds_b,
            voxel_size=(1.0, 1.0, 1.0),
            vol_order="xyz",
            seed_order="xyz",
            downsample=(1, 1, 1),
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
        )
        assert okA
        assert okB

    def test_empty_seeds(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        mask_xyz = np.transpose(mask, (2, 1, 0))
        seeds_a = np.empty((0, 3), dtype=int)
        seeds_b = np.array([[4, 4, 4]])

        A_aug, B_aug, okA, okB = connect_both_seeds_via_ridge(
            mask_xyz,
            seeds_a,
            seeds_b,
            voxel_size=(1.0, 1.0, 1.0),
            vol_order="xyz",
            seed_order="xyz",
            downsample=(1, 1, 1),
            verbose=False,
            snap_kwargs=dict(use_boundary=False, downsample=False),
        )
        assert not okA


# ============================================================
# Tests: split_supervoxel_helper
# ============================================================
class TestSplitSupervoxelHelper:
    def test_basic_split(self):
        mask, seeds_a_zyx, seeds_b_zyx = _make_dumbbell_mask(shape=(20, 30, 30))
        mask_xyz = np.transpose(mask, (2, 1, 0))
        seeds_a_xyz = seeds_a_zyx[:, [2, 1, 0]]
        seeds_b_xyz = seeds_b_zyx[:, [2, 1, 0]]

        result = split_supervoxel_helper(
            mask_xyz,
            seeds_a_xyz,
            seeds_b_xyz,
            voxel_size=(1.0, 1.0, 1.0),
            verbose=False,
        )
        assert result.shape == mask_xyz.shape
        assert np.any(result == 1)
        assert np.any(result == 2)
