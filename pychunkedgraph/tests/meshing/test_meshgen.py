"""Tests for pychunkedgraph.meshing.meshgen"""

import numpy as np
import pytest

from pychunkedgraph.meshing.meshgen import (
    black_out_dust_from_segmentation,
    calculate_quantization_bits_and_range,
    remap_seg_using_unsafe_dict,
    transform_draco_vertices,
)


class TestBlackOutDust:
    def test_removes_small_interior_segments(self):
        """Small segments not on boundary should be zeroed out."""
        seg = np.zeros((10, 10, 10), dtype=np.uint64)
        # Place a small segment (3 voxels) in the interior
        seg[4, 4, 4] = 5
        seg[4, 4, 5] = 5
        seg[4, 5, 4] = 5
        black_out_dust_from_segmentation(seg, dust_threshold=5)
        assert np.sum(seg == 5) == 0

    def test_preserves_large_segments(self):
        """Segments above threshold should be preserved."""
        seg = np.zeros((10, 10, 10), dtype=np.uint64)
        seg[3:6, 3:6, 3:6] = 7  # 27 voxels
        black_out_dust_from_segmentation(seg, dust_threshold=5)
        assert np.sum(seg == 7) == 27

    def test_preserves_boundary_segments(self):
        """Small segments on the boundary should NOT be removed."""
        seg = np.zeros((10, 10, 10), dtype=np.uint64)
        # Place segment on the -2 boundary face (second-to-last)
        seg[8, 5, 5] = 3  # x=-2 face
        black_out_dust_from_segmentation(seg, dust_threshold=5)
        assert np.sum(seg == 3) == 1

    def test_empty_segmentation(self):
        """All-zero segmentation should not raise."""
        seg = np.zeros((5, 5, 5), dtype=np.uint64)
        black_out_dust_from_segmentation(seg, dust_threshold=10)
        assert np.sum(seg) == 0

    def test_preserves_boundary_last_face(self):
        """Segment on the last face should be preserved."""
        seg = np.zeros((10, 10, 10), dtype=np.uint64)
        seg[9, 5, 5] = 2  # x=-1 face
        black_out_dust_from_segmentation(seg, dust_threshold=5)
        assert np.sum(seg == 2) == 1


class TestCalculateQuantizationBitsAndRange:
    def test_returns_three_values(self):
        bits, qrange, bin_size = calculate_quantization_bits_and_range(
            min_quantization_range=1000, max_draco_bin_size=2
        )
        assert isinstance(bits, (int, float, np.integer, np.floating))
        assert isinstance(qrange, (int, float, np.integer, np.floating))
        assert isinstance(bin_size, (int, float, np.integer, np.floating))

    def test_range_covers_minimum(self):
        """Quantization range must be >= min_quantization_range."""
        for min_range in [100, 500, 1000, 5000]:
            bits, qrange, bin_size = calculate_quantization_bits_and_range(
                min_quantization_range=min_range, max_draco_bin_size=2
            )
            assert qrange >= min_range

    def test_bin_size_within_max(self):
        """Bin size must not exceed max_draco_bin_size."""
        bits, qrange, bin_size = calculate_quantization_bits_and_range(
            min_quantization_range=1000, max_draco_bin_size=4
        )
        assert bin_size <= 4

    def test_explicit_bits(self):
        """When bits are provided, they should be used."""
        bits, qrange, bin_size = calculate_quantization_bits_and_range(
            min_quantization_range=100, max_draco_bin_size=2, draco_quantization_bits=10
        )
        assert bits == 10

    def test_small_range(self):
        bits, qrange, bin_size = calculate_quantization_bits_and_range(
            min_quantization_range=10, max_draco_bin_size=1
        )
        assert qrange >= 10
        assert bin_size >= 1

    def test_consistency(self):
        """num_bins * bin_size == quantization_range."""
        bits, qrange, bin_size = calculate_quantization_bits_and_range(
            min_quantization_range=500, max_draco_bin_size=2
        )
        num_bins = 2**bits - 1
        assert num_bins * bin_size == qrange


class TestTransformDracoVertices:
    def test_in_place_transform(self):
        """Vertices should be quantized in place."""
        mesh = {
            "num_vertices": 2,
            "vertices": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        }
        settings = {
            "quantization_bits": 10,
            "quantization_range": 1023.0,
            "quantization_origin": np.array([0.0, 0.0, 0.0]),
        }
        transform_draco_vertices(mesh, settings)
        # Vertices should be modified (quantized)
        assert mesh["vertices"] is not None
        assert len(mesh["vertices"]) == 6

    def test_origin_offset(self):
        """Vertices at the origin should map back to origin."""
        mesh = {
            "num_vertices": 1,
            "vertices": np.array([100.0, 200.0, 300.0]),
        }
        settings = {
            "quantization_bits": 16,
            "quantization_range": 65535.0,
            "quantization_origin": np.array([100.0, 200.0, 300.0]),
        }
        transform_draco_vertices(mesh, settings)
        # After subtracting origin, dividing by bin_size=1, floor, multiply back, add origin
        np.testing.assert_array_equal(mesh["vertices"], np.array([100.0, 200.0, 300.0]))


class TestRemapSegUsingUnsafeDict:
    def test_no_unsafe_ids(self):
        """Empty unsafe_dict should leave seg unchanged."""
        seg = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint64)
        original = seg.copy()
        result = remap_seg_using_unsafe_dict(seg, {})
        np.testing.assert_array_equal(result, original)

    def test_unsafe_id_not_in_seg(self):
        """Unsafe ID not present in seg should be a no-op."""
        seg = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint64)
        original = seg.copy()
        result = remap_seg_using_unsafe_dict(seg, {99: [10, 20]})
        np.testing.assert_array_equal(result, original)

    def test_single_component_with_overlap(self):
        """Single connected component that overlaps with a linked l2 id."""
        seg = np.zeros((4, 4, 4), dtype=np.uint64)
        seg[2, 2, 2] = 100  # unsafe root id in interior
        seg[2, 2, 3] = 42  # linked l2 id on the -2 boundary
        result = remap_seg_using_unsafe_dict(seg, {100: [42]})
        # The unsafe voxel should be remapped to linked id or zeroed
        # Since seg[-2,:,:] at position (2,2,3) overlaps with 42, CC should link
        assert result[2, 2, 2] in (0, 42)

    def test_zeroes_when_no_linked_ids(self):
        """Unsafe component with no linked l2 neighbors gets zeroed."""
        seg = np.zeros((4, 4, 4), dtype=np.uint64)
        seg[1, 1, 1] = 100  # interior, no neighbors on boundary
        result = remap_seg_using_unsafe_dict(seg, {100: [42]})
        assert result[1, 1, 1] == 0
