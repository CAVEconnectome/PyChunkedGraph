"""Per-axis clamp on the geodesic downsample factor in
`split_supervoxel_helper`.

The clamp pairs `downsample[::-1]` (zyx) with `binary_seg.shape` (xyz).
A mismatched zip would compare each stride against the wrong extent and
collapse a stride to 1 when the *other* axis is thin. Golden_cube does
not exercise the bug because every extent / `min_grid_per_axis` already
exceeds `max_axis_ds`; a thin-z SV does.
"""

import numpy as np


def _clamp_buggy(downsample_xyz, shape_xyz, max_axis_ds, min_grid_per_axis):
    """Reproduces the pre-fix zip: zyx strides paired with xyz extents."""
    return tuple(
        max(1, min(int(s), max_axis_ds, dim // min_grid_per_axis))
        for s, dim in zip(downsample_xyz[::-1], shape_xyz)
    )


def _clamp_fixed(downsample_xyz, shape_xyz, max_axis_ds, min_grid_per_axis):
    """The current (fixed) clamp: both sides xyz, final tuple reversed."""
    ds_xyz = tuple(
        max(1, min(int(s), max_axis_ds, dim // min_grid_per_axis))
        for s, dim in zip(downsample_xyz, shape_xyz)
    )
    return ds_xyz[::-1]


def test_thin_z_sv_exposes_axis_mismatch():
    voxel_size_xyz = np.array([8, 8, 40])
    downsample_xyz = tuple(int(v) for v in voxel_size_xyz.max() // voxel_size_xyz)
    assert downsample_xyz == (5, 5, 1)
    shape_xyz = (800, 800, 20)
    max_axis_ds = 3
    min_grid_per_axis = 16

    buggy = _clamp_buggy(downsample_xyz, shape_xyz, max_axis_ds, min_grid_per_axis)
    fixed = _clamp_fixed(downsample_xyz, shape_xyz, max_axis_ds, min_grid_per_axis)

    # Pre-fix: x-stride collapses to 1 because it gets paired with z-extent (20),
    # 20 // 16 == 1 dominates `min(5, 3, 1) == 1`. ds_zyx == (1, 3, 1).
    assert buggy == (1, 3, 1)
    # Post-fix: x and y are both wide, z is thin → ds_zyx == (1, 3, 3).
    assert fixed == (1, 3, 3)
    # The bug specifically collapses the x-stride.
    assert fixed[2] == 3 and buggy[2] == 1


def test_golden_cube_shape_bug_is_nonbinding():
    """Reproduces the recent golden_cube run shape: bug exists but the
    `dim // min_grid_per_axis` clamp doesn't bind on any axis, so buggy
    and fixed agree by accident."""
    voxel_size_xyz = np.array([8, 8, 40])
    downsample_xyz = tuple(int(v) for v in voxel_size_xyz.max() // voxel_size_xyz)
    shape_xyz = (987, 658, 144)

    buggy = _clamp_buggy(downsample_xyz, shape_xyz, 3, 16)
    fixed = _clamp_fixed(downsample_xyz, shape_xyz, 3, 16)

    assert buggy == fixed == (1, 3, 3)


def test_isotropic_resolution_is_identity():
    """Isotropic voxel size → downsample (1,1,1) regardless of shape; both
    forms agree trivially."""
    voxel_size_xyz = np.array([8, 8, 8])
    downsample_xyz = tuple(int(v) for v in voxel_size_xyz.max() // voxel_size_xyz)
    assert downsample_xyz == (1, 1, 1)

    for shape_xyz in [(100, 100, 100), (500, 50, 30), (10, 200, 10)]:
        buggy = _clamp_buggy(downsample_xyz, shape_xyz, 3, 16)
        fixed = _clamp_fixed(downsample_xyz, shape_xyz, 3, 16)
        assert buggy == fixed == (1, 1, 1)


def test_extreme_z_anisotropy_thin_xy():
    """Mirror case: x and y are thin, z is wide. Fixed form correctly
    refuses to downsample any axis (xy below min_grid threshold, z stride
    already 1). Buggy form spuriously assigns z-stride 3 because it sends
    the x-stride (10) into z's slot, where 800 // 16 = 50 > 3."""
    voxel_size_xyz = np.array([4, 4, 40])
    downsample_xyz = tuple(int(v) for v in voxel_size_xyz.max() // voxel_size_xyz)
    assert downsample_xyz == (10, 10, 1)
    shape_xyz = (20, 20, 800)

    buggy = _clamp_buggy(downsample_xyz, shape_xyz, 3, 16)
    fixed = _clamp_fixed(downsample_xyz, shape_xyz, 3, 16)

    # Fixed: xy stride 10 capped by 20 // 16 = 1 → 1; z stride 1 → 1.
    # ds_xyz_fixed = (1, 1, 1) → ds_zyx_fixed = (1, 1, 1).
    assert fixed == (1, 1, 1)
    # Buggy: (1, 20) → 1; (10, 20) → min(10, 3, 1) = 1; (10, 800) →
    # min(10, 3, 50) = 3. ds_zyx_buggy = (1, 1, 3) — bogusly downsamples z.
    assert buggy == (1, 1, 3)
    assert buggy != fixed
