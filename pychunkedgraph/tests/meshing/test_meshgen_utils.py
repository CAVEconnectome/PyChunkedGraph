"""Tests for pychunkedgraph.meshing.meshgen_utils"""

import numpy as np
import pytest

from pychunkedgraph.meshing.meshgen_utils import str_to_slice, slice_to_str


class TestStrToSlice:
    def test_basic_conversion(self):
        result = str_to_slice("0-10_0-20_0-30")
        assert result == (slice(0, 10), slice(0, 20), slice(0, 30))

    def test_nonzero_starts(self):
        result = str_to_slice("5-15_10-25_100-200")
        assert result == (slice(5, 15), slice(10, 25), slice(100, 200))

    def test_single_voxel_slices(self):
        result = str_to_slice("0-1_0-1_0-1")
        assert result == (slice(0, 1), slice(0, 1), slice(0, 1))

    def test_large_values(self):
        result = str_to_slice("1024-2048_512-1024_256-512")
        assert result == (slice(1024, 2048), slice(512, 1024), slice(256, 512))


class TestSliceToStr:
    def test_basic_conversion(self):
        slices = (slice(0, 10), slice(0, 20), slice(0, 30))
        assert slice_to_str(slices) == "0-10_0-20_0-30"

    def test_nonzero_starts(self):
        slices = (slice(5, 15), slice(10, 25), slice(100, 200))
        assert slice_to_str(slices) == "5-15_10-25_100-200"

    def test_single_slice(self):
        assert slice_to_str(slice(3, 7)) == "3-7"

    def test_large_values(self):
        slices = (slice(1024, 2048), slice(512, 1024), slice(256, 512))
        assert slice_to_str(slices) == "1024-2048_512-1024_256-512"


class TestRoundTrip:
    def test_str_to_slice_to_str(self):
        original = "0-10_20-30_40-50"
        assert slice_to_str(str_to_slice(original)) == original

    def test_slice_to_str_to_slice(self):
        original = (slice(5, 15), slice(10, 25), slice(100, 200))
        assert str_to_slice(slice_to_str(original)) == original

    @pytest.mark.parametrize(
        "s",
        [
            "0-1_0-1_0-1",
            "128-256_64-128_32-64",
            "0-512_0-512_0-512",
        ],
    )
    def test_roundtrip_parametrized(self, s):
        assert slice_to_str(str_to_slice(s)) == s
