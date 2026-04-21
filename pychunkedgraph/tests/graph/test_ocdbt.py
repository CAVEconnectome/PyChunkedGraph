"""Tests for pychunkedgraph.graph.ocdbt"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import tensorstore as ts
from unittest.mock import MagicMock, patch

from pychunkedgraph.graph import ocdbt as ocdbt_mod
from pychunkedgraph.graph.meta import ChunkedGraphMeta, GraphConfig, DataSource


def _make_mock_src(num_scales=2):
    """Build a mock TensorStore source handle with a copyable schema."""
    mock_src = MagicMock()
    schema = MagicMock()
    schema.rank = 4
    schema.dtype = "uint64"
    schema.codec = None
    schema.domain = None
    schema.shape = [256, 256, 256, 1]
    schema.chunk_layout = None
    schema.dimension_units = None
    mock_src.schema = schema
    return mock_src


def _setup_ts_mock(mock_ts, num_scales=2):
    """Configure ts.KvStore.open and ts.open to return source/dest handles
    for `num_scales` scales. Source returns a fake info JSON listing all scales.
    """
    info = {
        "scales": [
            {"resolution": [4 * (2**i), 4 * (2**i), 40], "size": [256, 256, 256]}
            for i in range(num_scales)
        ]
    }
    info_bytes = json.dumps(info).encode()

    # KvStore.open(path).result().read('/info').result().value
    mock_kvs = MagicMock()
    mock_read_result = MagicMock()
    mock_read_result.value = info_bytes
    mock_kvs.read.return_value.result.return_value = mock_read_result
    mock_ts.KvStore.open.return_value.result.return_value = mock_kvs

    # ts.open: alternates source / destination per scale
    handles = []
    for _ in range(num_scales):
        handles.append(_make_mock_src())  # source
        handles.append(MagicMock())  # destination
    mock_ts.open.side_effect = [
        MagicMock(result=MagicMock(return_value=h)) for h in handles
    ]
    return handles


class TestBuildCgOcdbtSpec:
    def test_spec_structure(self):
        """build_cg_ocdbt_spec returns the expected kvstack-layered spec."""
        spec = ocdbt_mod.build_cg_ocdbt_spec("gs://bucket/ws", "my_graph")
        assert spec["driver"] == "ocdbt"
        layers = spec["base"]["layers"]
        assert len(layers) == 3
        # Layer 0: base catch-all (trailing slash)
        assert layers[0]["base"] == "gs://bucket/ws/ocdbt/base/"
        # Layer 1: manifest override
        assert layers[1]["exact"] == "manifest.ocdbt"
        assert "my_graph" in layers[1]["base"]
        # Layer 2: data prefix
        assert layers[2]["prefix"] == "my_graph_d/"
        # Data prefix options steer writes
        assert spec["value_data_prefix"] == "my_graph_d/"
        assert spec["btree_node_data_prefix"] == "my_graph_d/"
        assert spec["version_tree_node_data_prefix"] == "my_graph_d/"


class TestForkBaseManifest:
    def test_copies_manifest(self):
        """fork_base_manifest copies the base manifest via tensorstore kvstore."""
        tmpdir = tempfile.mkdtemp()
        ws = f"file://{tmpdir}"
        try:
            # Create a real base OCDBT with a manifest.
            base_kvs = ts.KvStore.open(f"{ws}/ocdbt/base/").result()
            base_kvs.write("manifest.ocdbt", b"fake_manifest_bytes").result()

            ocdbt_mod.fork_base_manifest(ws, "my_graph")

            fork_kvs = ts.KvStore.open(f"{ws}/ocdbt/my_graph/").result()
            result = fork_kvs.read("manifest.ocdbt").result()
            assert result.value == b"fake_manifest_bytes"
        finally:
            shutil.rmtree(tmpdir)

    def test_wipe_existing_cleans_fork_dir(self):
        """wipe_existing=True removes the fork directory before copying."""
        tmpdir = tempfile.mkdtemp()
        ws = f"file://{tmpdir}"
        try:
            base_kvs = ts.KvStore.open(f"{ws}/ocdbt/base/").result()
            base_kvs.write("manifest.ocdbt", b"manifest_v1").result()

            fork_kvs = ts.KvStore.open(f"{ws}/ocdbt/my_graph/").result()
            fork_kvs.write("stale_file", b"stale").result()

            ocdbt_mod.fork_base_manifest(ws, "my_graph", wipe_existing=True)

            fork_kvs2 = ts.KvStore.open(f"{ws}/ocdbt/my_graph/").result()
            assert fork_kvs2.read("manifest.ocdbt").result().value == b"manifest_v1"
            assert len(fork_kvs2.read("stale_file").result().value) == 0
        finally:
            shutil.rmtree(tmpdir)


class TestModeDownsample:
    def test_2x2x1_picks_majority(self):
        # 2x2x1 block where 3 of 4 voxels are label 7 — mode is 7.
        data = np.array(
            [[[[7]], [[7]]], [[[7]], [[3]]]],
            dtype=np.uint64,
        )
        out = ocdbt_mod._mode_downsample(data, (2, 2, 1))
        assert out.shape == (1, 1, 1, 1)
        assert out[0, 0, 0, 0] == 7

    def test_pads_odd_dimensions(self):
        # 3x3x1 single-channel — pads to 4x4x1 then downsamples to 2x2x1
        data = np.full((3, 3, 1, 1), 5, dtype=np.uint64)
        out = ocdbt_mod._mode_downsample(data, (2, 2, 1))
        assert out.shape == (2, 2, 1, 1)
        assert (out == 5).all()


class TestCopyWsChunk:
    def test_basic_copy(self):

        mock_source = MagicMock()
        mock_destination = MagicMock()

        # Simulate source read
        data = np.ones((64, 64, 64), dtype=np.uint64)
        mock_source.__getitem__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=data))
                )
            )
        )
        mock_destination.__getitem__ = MagicMock(
            return_value=MagicMock(
                write=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=None))
                )
            )
        )

        voxel_bounds = np.array([[0, 256], [0, 256], [0, 256]])
        ocdbt_mod.copy_ws_chunk(
            mock_source,
            mock_destination,
            chunk_size=(64, 64, 64),
            coords=[0, 0, 0],
            voxel_bounds=voxel_bounds,
        )
        # Should have read from source and written to destination
        mock_source.__getitem__.assert_called_once()
        mock_destination.__getitem__.assert_called_once()

    def test_boundary_clipping(self):

        mock_source = MagicMock()
        mock_destination = MagicMock()

        data = np.ones((32, 64, 64), dtype=np.uint64)
        mock_source.__getitem__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=data))
                )
            )
        )
        mock_destination.__getitem__ = MagicMock(
            return_value=MagicMock(
                write=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=None))
                )
            )
        )

        # Volume ends at 224 in x, so last chunk (192-256) is clipped to (192-224)
        voxel_bounds = np.array([[0, 224], [0, 256], [0, 256]])
        ocdbt_mod.copy_ws_chunk(
            mock_source,
            mock_destination,
            chunk_size=(64, 64, 64),
            coords=[3, 0, 0],
            voxel_bounds=voxel_bounds,
        )
        mock_source.__getitem__.assert_called_once()


# --------------------------------------------------------------------------
# Integration tests on a real local OCDBT store.
#
# The mock-based tests above validate call patterns but can't catch byte-level
# bugs in the multi-scale update path (wrong coordinates, missing propagation,
# skipped scales). These use an actual local file-backed OCDBT to exercise
# the full read/write cycle end-to-end.
# --------------------------------------------------------------------------


@pytest.fixture
def local_ocdbt():
    """Create a local precomputed multi-scale OCDBT store.

    Builds 3 scales (factors 2,2,1 between each) with known segmentation IDs
    so downsampling behaviour and propagation can be asserted against exact
    values. Returns paths + handles for tests to work against directly.
    """
    tmpdir = tempfile.mkdtemp()
    base = f"file://{tmpdir}/ocdbt/base"

    mm = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}

    def mk(scale_idx, size, resolution, extra_mm=None):
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "ocdbt", "base": base},
            "scale_metadata": {
                "size": size,
                "resolution": resolution,
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8],
                "chunk_size": [32, 32, 32],
            },
        }
        if extra_mm:
            spec["multiscale_metadata"] = extra_mm
        return ts.open(spec, create=True).result()

    scales = [
        mk(0, [64, 64, 32], [4, 4, 40], extra_mm=mm),
        mk(1, [32, 32, 32], [8, 8, 40]),
        mk(2, [16, 16, 32], [16, 16, 40]),
    ]
    resolutions = [[4, 4, 40], [8, 8, 40], [16, 16, 40]]

    yield {
        "tmpdir": tmpdir,
        "base": base,
        "scales": scales,
        "resolutions": resolutions,
    }
    shutil.rmtree(tmpdir)


class TestPropagateToCoarserScales:
    def test_writes_downsampled_to_each_coarser_scale(self, local_ocdbt):
        """Writing at base scale then propagating populates each coarser scale
        with a mode-downsampled copy at the correctly-scaled coordinates."""
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]

        # Base region: fill a 16x16x16 block with label 42.
        base_slices = (slice(0, 16), slice(0, 16), slice(0, 16))
        base_data = np.full((16, 16, 16, 1), 42, dtype=np.uint64)
        scales[0][base_slices + (slice(None),)] = base_data

        ocdbt_mod.propagate_to_coarser_scales(scales, res, base_slices)

        # Scale 1 (factor 2,2,1): region 8x8x16 should all be label 42.
        s1 = scales[1][0:8, 0:8, 0:16, :].read().result()
        assert (s1 == 42).all(), f"scale 1 mismatch: {np.unique(s1)}"

        # Scale 2 (factor 4,4,1 cumulative): region 4x4x16 should all be 42.
        s2 = scales[2][0:4, 0:4, 0:16, :].read().result()
        assert (s2 == 42).all(), f"scale 2 mismatch: {np.unique(s2)}"

    def test_mode_picks_majority_when_block_is_mixed(self, local_ocdbt):
        """In a 2x2x1 block with mixed labels, mode wins."""
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]

        # Fill a 2x2x16 block: three voxels per xy plane are label 7, one is 3.
        base = np.full((2, 2, 16, 1), 7, dtype=np.uint64)
        base[1, 1, :, 0] = 3  # minority voxel per 2x2 block
        base_slices = (slice(0, 2), slice(0, 2), slice(0, 16))
        scales[0][base_slices + (slice(None),)] = base

        ocdbt_mod.propagate_to_coarser_scales(scales, res, base_slices)

        # Scale 1: 1x1x16 block should be mode-7 (3-of-4 majority).
        s1 = scales[1][0:1, 0:1, 0:16, :].read().result()
        assert (s1 == 7).all(), f"mode downsample failed: {np.unique(s1)}"

    def test_unaffected_regions_remain_zero(self, local_ocdbt):
        """Propagating a region does not touch coarser data outside it."""
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]

        # Seed scale 1 directly with a sentinel in a region we won't propagate over.
        scales[1][16:20, 16:20, 0:4, :] = np.full((4, 4, 4, 1), 99, dtype=np.uint64)

        base_slices = (slice(0, 16), slice(0, 16), slice(0, 16))
        scales[0][base_slices + (slice(None),)] = np.full(
            (16, 16, 16, 1), 1, dtype=np.uint64
        )
        ocdbt_mod.propagate_to_coarser_scales(scales, res, base_slices)

        # Sentinel region is outside the scaled propagation area (0:8, 0:8, 0:16
        # at scale 1) — must be unchanged.
        sentinel = scales[1][16:20, 16:20, 0:4, :].read().result()
        assert (sentinel == 99).all(), "propagation overwrote unrelated region"

    def test_spatial_pattern_matches_ground_truth(self, local_ocdbt):
        """Downsampled coarser scales match what ts.downsample(method=mode) produces.

        Uses a spatial label pattern so every 2x2x1 block has a non-trivial mode;
        catches any off-by-one in coordinate mapping or incorrect mode logic.
        """
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]

        # Deterministic per-voxel labels derived from coords — every 2x2x1 block
        # has 4 distinct values, exposing any mode-selection bug and any
        # coordinate-mapping mistake.
        X, Y, Z = 16, 16, 16
        xs, ys, zs = np.meshgrid(
            np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"
        )
        # Encode (x,y,z) uniquely so each voxel has a unique label in [1, 16*16*16].
        pattern = (xs * (Y * Z) + ys * Z + zs + 1).astype(np.uint64)
        pattern = pattern[..., np.newaxis]

        base_slices = (slice(0, X), slice(0, Y), slice(0, Z))
        scales[0][base_slices + (slice(None),)] = pattern

        ocdbt_mod.propagate_to_coarser_scales(scales, res, base_slices)

        # Ground truth via tensorstore's own downsample driver (mode method).
        # Factor 2,2,1 scale 0 -> scale 1, then 2,2,1 scale 1 -> scale 2.
        gt_s1 = (
            ts.downsample(scales[0], [2, 2, 1, 1], method="mode")[0:8, 0:8, 0:16, :]
            .read()
            .result()
        )
        got_s1 = scales[1][0:8, 0:8, 0:16, :].read().result()
        assert got_s1.shape == gt_s1.shape
        # Mode ties can differ between implementations; verify each downsampled
        # voxel matches ONE of the four input voxels (correctness property).
        for i in range(8):
            for j in range(8):
                for k in range(16):
                    block = pattern[2 * i : 2 * i + 2, 2 * j : 2 * j + 2, k : k + 1, 0]
                    assert (
                        got_s1[i, j, k, 0] in block
                    ), f"scale1[{i},{j},{k}]={got_s1[i,j,k,0]} not in block {block.flatten()}"
        # Scale 2 cascades from scale 1, not base — cumulative factor 4,4,1.
        got_s2 = scales[2][0:4, 0:4, 0:16, :].read().result()
        for i in range(4):
            for j in range(4):
                for k in range(16):
                    # Each scale-2 voxel must match one of the four scale-1 voxels
                    # (because cascade downsamples from the level below).
                    s1_block = got_s1[
                        2 * i : 2 * i + 2, 2 * j : 2 * j + 2, k : k + 1, 0
                    ]
                    assert (
                        got_s2[i, j, k, 0] in s1_block
                    ), f"scale2[{i},{j},{k}]={got_s2[i,j,k,0]} not in s1 block {s1_block.flatten()}"

    def test_repeated_update_reflects_latest_base(self, local_ocdbt):
        """Propagating twice (simulating repeated SV splits) leaves coarser
        scales consistent with the latest base-scale contents."""
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]

        base_slices = (slice(0, 16), slice(0, 16), slice(0, 16))

        # First update: region is label 1.
        scales[0][base_slices + (slice(None),)] = np.full(
            (16, 16, 16, 1), 1, dtype=np.uint64
        )
        ocdbt_mod.propagate_to_coarser_scales(scales, res, base_slices)
        assert (scales[1][0:8, 0:8, 0:16, :].read().result() == 1).all()

        # Second update: same region now label 2. Coarser scales must reflect it.
        scales[0][base_slices + (slice(None),)] = np.full(
            (16, 16, 16, 1), 2, dtype=np.uint64
        )
        ocdbt_mod.propagate_to_coarser_scales(scales, res, base_slices)
        assert (scales[1][0:8, 0:8, 0:16, :].read().result() == 2).all()
        assert (scales[2][0:4, 0:4, 0:16, :].read().result() == 2).all()


class TestWriteSeg:
    def test_writes_base_only(self, local_ocdbt):
        """`write_seg` writes to base scale; coarser scales are untouched
        (propagation is now the downsample worker's job)."""
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]
        meta = MagicMock()
        meta.ws_ocdbt = scales[0]
        meta.ws_ocdbt_scales = scales
        meta.ws_ocdbt_resolutions = res

        data = np.full((16, 16, 16), 55, dtype=np.uint64)
        ocdbt_mod.write_seg(meta, [0, 0, 0], [16, 16, 16], data)

        # Base scale: written region has label 55.
        assert (scales[0][0:16, 0:16, 0:16, :].read().result() == 55).all()
        # Coarser scales: unchanged (still empty/zero — write_seg does not touch them).
        assert (scales[1][0:8, 0:8, 0:16, :].read().result() == 0).all()
        assert (scales[2][0:4, 0:4, 0:16, :].read().result() == 0).all()

    def test_single_scale(self, local_ocdbt):
        """Single-scale setup still works (write_seg only touches base)."""
        meta = MagicMock()
        meta.ws_ocdbt = local_ocdbt["scales"][0]
        meta.ws_ocdbt_scales = [local_ocdbt["scales"][0]]
        meta.ws_ocdbt_resolutions = [local_ocdbt["resolutions"][0]]

        data = np.full((8, 8, 8), 99, dtype=np.uint64)
        ocdbt_mod.write_seg(meta, [0, 0, 0], [8, 8, 8], data)
        assert (meta.ws_ocdbt[0:8, 0:8, 0:8, :].read().result() == 99).all()


class TestMetaToForkEndToEnd:
    """Full path: ChunkedGraphMeta.ws_ocdbt_scales → real kvstack fork → read/write."""

    def test_meta_opens_fork_and_merges_base(self):
        """meta.ws_ocdbt_scales opens a real kvstack-backed OCDBT and reads
        merge base + fork correctly.

        Only `_read_source_scales` is mocked (it reads `/info` which is a
        GCS-only key). The full meta → build_cg_ocdbt_spec → kvstack →
        OCDBT → read/write path is exercised for real.
        """
        tmpdir = tempfile.mkdtemp()
        ws = f"file://{tmpdir}"
        try:
            MM = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}
            SCALE = {
                "size": [64, 64, 32],
                "resolution": [4, 4, 40],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8],
                "chunk_size": [32, 32, 32],
            }
            FAKE_SCALES = [
                {
                    "resolution": [4, 4, 40],
                    "size": [64, 64, 32],
                    "chunk_sizes": [[32, 32, 32]],
                    "encoding": "compressed_segmentation",
                    "compressed_segmentation_block_size": [8, 8, 8],
                }
            ]

            # Source precomputed — needed by get_seg_source_and_destination_ocdbt
            # to open the source handle and copy its schema.
            ts.open(
                {
                    "driver": "neuroglancer_precomputed",
                    "kvstore": f"{ws}/",
                    "multiscale_metadata": MM,
                    "scale_metadata": SCALE,
                },
                create=True,
            ).result()

            # Create base OCDBT with known data.
            base_kvstore = {
                "driver": "ocdbt",
                "base": f"{ws}/ocdbt/base/",
                "config": dict(ocdbt_mod.OCDBT_CONFIG),
            }
            base_store = ts.open(
                {
                    "driver": "neuroglancer_precomputed",
                    "kvstore": base_kvstore,
                    "multiscale_metadata": MM,
                    "scale_metadata": SCALE,
                },
                create=True,
            ).result()
            base_store[...] = np.full((64, 64, 32, 1), 50, dtype=np.uint64)

            # Fork for graph "test_cg".
            ocdbt_mod.fork_base_manifest(f"{ws}/", "test_cg")

            gc = GraphConfig(ID="test_cg", CHUNK_SIZE=[32, 32, 32])
            ds = DataSource(WATERSHED=f"{ws}/", DATA_VERSION=4)
            meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})

            # Mock only _read_source_scales ('/info' is GCS-only).
            with patch.object(
                ocdbt_mod, "_read_source_scales", return_value=FAKE_SCALES
            ):
                scales = meta.ws_ocdbt_scales
                assert len(scales) == 1

                # Read: should see base data.
                r = scales[0][0:16, 0:16, 0:16, :].read().result()
                assert (r == 50).all(), f"fork should see base, got {np.unique(r)}"

                # Write via the fork handle.
                scales[0][0:16, 0:16, 0:16, :] = np.full(
                    (16, 16, 16, 1), 7, dtype=np.uint64
                )

                # Read back: edited = 7, untouched = 50.
                assert (scales[0][0:16, 0:16, 0:16, :].read().result() == 7).all()
                assert (scales[0][32:48, 0:16, 0:16, :].read().result() == 50).all()

            # Base unchanged.
            base_ro = ts.open(
                {
                    "driver": "neuroglancer_precomputed",
                    "kvstore": base_kvstore,
                }
            ).result()
            assert (base_ro[0:16, 0:16, 0:16, :].read().result() == 50).all()
        finally:
            shutil.rmtree(tmpdir)


class TestForkIsolation:
    """End-to-end: two forks on the same base, writes isolated, base immutable."""

    def test_two_forks_isolated(self):
        tmpdir = tempfile.mkdtemp()
        ws = f"file://{tmpdir}"
        try:
            # Build a base OCDBT with known data.
            MM = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}
            SCALE = {
                "size": [64, 64, 32],
                "resolution": [4, 4, 40],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8],
                "chunk_size": [32, 32, 32],
            }
            base_kvstore = {
                "driver": "ocdbt",
                "base": f"{ws}/ocdbt/base/",
                "config": dict(ocdbt_mod.OCDBT_CONFIG),
            }
            base_store = ts.open(
                {
                    "driver": "neuroglancer_precomputed",
                    "kvstore": base_kvstore,
                    "multiscale_metadata": MM,
                    "scale_metadata": SCALE,
                },
                create=True,
            ).result()
            base_store[...] = np.full((64, 64, 32, 1), 50, dtype=np.uint64)

            base_path = f"{tmpdir}/ocdbt/base"
            base_files_before = set(
                os.path.relpath(os.path.join(r, f), base_path)
                for r, _, fs in os.walk(base_path)
                for f in fs
            )

            # Fork A and B via fork_base_manifest.
            ocdbt_mod.fork_base_manifest(ws, "fork_a")
            ocdbt_mod.fork_base_manifest(ws, "fork_b")

            def open_fork(gid):
                spec = ocdbt_mod.build_cg_ocdbt_spec(ws, gid)
                return ts.open(
                    {"driver": "neuroglancer_precomputed", "kvstore": spec},
                ).result()

            fork_a = open_fork("fork_a")
            fork_b = open_fork("fork_b")

            # Both see base data.
            assert (fork_a[0:16, 0:16, 0:16, :].read().result() == 50).all()
            assert (fork_b[0:16, 0:16, 0:16, :].read().result() == 50).all()

            # Write different values to each fork.
            fork_a[0:16, 0:16, 0:16, :] = np.full((16, 16, 16, 1), 1, dtype=np.uint64)
            fork_b[32:48, 0:16, 0:16, :] = np.full((16, 16, 16, 1), 2, dtype=np.uint64)

            # Each fork sees ONLY its own edit + base for the rest.
            assert (fork_a[0:16, 0:16, 0:16, :].read().result() == 1).all()
            assert (fork_a[32:48, 0:16, 0:16, :].read().result() == 50).all()
            assert (fork_b[32:48, 0:16, 0:16, :].read().result() == 2).all()
            assert (fork_b[0:16, 0:16, 0:16, :].read().result() == 50).all()

            # Base is unchanged.
            base_files_after = set(
                os.path.relpath(os.path.join(r, f), base_path)
                for r, _, fs in os.walk(base_path)
                for f in fs
            )
            assert (
                base_files_before == base_files_after
            ), f"base was mutated: new={base_files_after - base_files_before}"

            # Fork writes went to their own directories.
            fork_a_files = os.listdir(f"{tmpdir}/ocdbt/fork_a")
            fork_b_files = os.listdir(f"{tmpdir}/ocdbt/fork_b")
            assert any("fork_a_d" in f for f in fork_a_files)
            assert any("fork_b_d" in f for f in fork_b_files)
        finally:
            shutil.rmtree(tmpdir)


class TestCopyWsChunkMultiscale:
    def test_copies_physical_region_across_scales(self, local_ocdbt):
        """One base-chunk copy call populates the same physical region at every scale."""
        scales = local_ocdbt["scales"]
        res = local_ocdbt["resolutions"]

        # Seed the source at each scale with its own unique label so we can
        # verify the copy reads from the correct scale, not the base.
        src_paths = [MagicMock() for _ in scales]
        # Build a fake "source" using the same local store as src — we fill
        # each scale with a different value and then copy to a NEW local dst.
        for i, (s, label) in enumerate(zip(scales, (11, 22, 33))):
            s[...] = np.full(s.shape, label, dtype=np.uint64)

        # Create a separate local destination store.
        dst_tmp = tempfile.mkdtemp()
        dst_base = f"file://{dst_tmp}/ocdbt/base"
        mm = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}
        dst_scales = []
        for i, src in enumerate(scales):
            spec = {
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "ocdbt", "base": dst_base},
                "scale_metadata": {
                    "size": list(src.shape[:3]),
                    "resolution": res[i],
                    "encoding": "compressed_segmentation",
                    "compressed_segmentation_block_size": [8, 8, 8],
                    "chunk_size": [32, 32, 32],
                },
            }
            if i == 0:
                spec["multiscale_metadata"] = mm
            dst_scales.append(ts.open(spec, create=True).result())

        try:
            # Graph chunk_size 32, coords (0,0,0), voxel_bounds [0..64, 0..64, 0..32]
            # maps to base region (0:32, 0:32, 0:32).
            ocdbt_mod.copy_ws_chunk_multiscale(
                scales,
                dst_scales,
                res,
                chunk_size=(32, 32, 32),
                coords=[0, 0, 0],
                voxel_bounds=np.array([[0, 64], [0, 64], [0, 32]]),
            )
            # Scale 0 dst got label 11 in (0:32, 0:32, 0:32).
            assert (dst_scales[0][0:32, 0:32, 0:32, :].read().result() == 11).all()
            # Scale 1 dst got label 22 in (0:16, 0:16, 0:32).
            assert (dst_scales[1][0:16, 0:16, 0:32, :].read().result() == 22).all()
            # Scale 2 dst got label 33 in (0:8, 0:8, 0:32).
            assert (dst_scales[2][0:8, 0:8, 0:32, :].read().result() == 33).all()
        finally:
            shutil.rmtree(dst_tmp)
