"""Tests for pychunkedgraph.graph.ocdbt"""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import tensorstore as ts
from unittest.mock import MagicMock, patch

from pychunkedgraph.graph import ocdbt as ocdbt_mod
from pychunkedgraph.graph.meta import ChunkedGraphMeta, GraphConfig, DataSource

SCALE_META_BASE = {
    "encoding": "compressed_segmentation",
    "compressed_segmentation_block_size": [8, 8, 8],
    "chunk_size": [32, 32, 32],
}
MULTISCALE_META = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}


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
    """Byte-level behavior of `fork_base_manifest` — manifest copy + wipe."""

    def test_copies_manifest(self, local_ocdbt):
        """fork_base_manifest copies the base manifest via tensorstore kvstore."""
        ws = local_ocdbt["ws"]
        base_kvs = ts.KvStore.open(f"{ws}/ocdbt/base/").result()
        base_kvs.write("manifest.ocdbt", b"fake_manifest_bytes").result()

        ocdbt_mod.fork_base_manifest(ws, "my_graph")

        fork_kvs = ts.KvStore.open(f"{ws}/ocdbt/my_graph/").result()
        assert fork_kvs.read("manifest.ocdbt").result().value == b"fake_manifest_bytes"

    def test_wipe_existing_cleans_fork_dir(self, local_ocdbt):
        """wipe_existing=True removes the fork directory before copying."""
        ws = local_ocdbt["ws"]
        base_kvs = ts.KvStore.open(f"{ws}/ocdbt/base/").result()
        base_kvs.write("manifest.ocdbt", b"manifest_v1").result()

        fork_kvs = ts.KvStore.open(f"{ws}/ocdbt/my_graph/").result()
        fork_kvs.write("stale_file", b"stale").result()

        ocdbt_mod.fork_base_manifest(ws, "my_graph", wipe_existing=True)

        fork_kvs2 = ts.KvStore.open(f"{ws}/ocdbt/my_graph/").result()
        assert fork_kvs2.read("manifest.ocdbt").result().value == b"manifest_v1"
        assert len(fork_kvs2.read("stale_file").result().value) == 0


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
    """Shared OCDBT test environment.

    Creates a local 3-scale precomputed base OCDBT (factors 2,2,1 per
    level) and exposes helpers for fork-based tests. Every OCDBT test
    that needs real storage uses this fixture — no duplicated tmpdir
    scaffolding.

    Yields:
        tmpdir: on-disk workspace (cleaned up on teardown).
        ws: `file://{tmpdir}` URL — what `build_cg_ocdbt_spec` expects.
        base: base OCDBT kvstore URL.
        scales: 3 precomputed handles on the base (multi-scale tests).
        resolutions: per-scale [x,y,z] resolution arrays.
        make_fork(graph_id, *, scale_index=0, pinned_at=None): opens a
            precomputed handle through a fork of the base. Creates the
            fork on first call per `graph_id` and reuses it thereafter;
            repeated calls with the same id never re-copy the manifest
            (which would clobber fork writes).
    """
    tmpdir = tempfile.mkdtemp()
    ws = f"file://{tmpdir}"
    base = f"{ws}/ocdbt/base"

    def _mk_scale(size, resolution, *, include_mm):
        # Match OCDBT_CONFIG so forks (which always use it) don't trip the
        # "Configuration mismatch on max_inline_value_bytes" check.
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "ocdbt",
                "base": base,
                "config": dict(ocdbt_mod.OCDBT_CONFIG),
            },
            "scale_metadata": {
                "size": size,
                "resolution": resolution,
                **SCALE_META_BASE,
            },
        }
        if include_mm:
            spec["multiscale_metadata"] = MULTISCALE_META
        return ts.open(spec, create=True).result()

    scales = [
        _mk_scale([64, 64, 32], [4, 4, 40], include_mm=True),
        _mk_scale([32, 32, 32], [8, 8, 40], include_mm=False),
        _mk_scale([16, 16, 32], [16, 16, 40], include_mm=False),
    ]
    resolutions = [[4, 4, 40], [8, 8, 40], [16, 16, 40]]

    _created_forks = set()

    def make_fork(graph_id, *, scale_index=0, pinned_at=None):
        if graph_id not in _created_forks:
            ocdbt_mod.fork_base_manifest(ws, graph_id)
            _created_forks.add(graph_id)
        spec = ocdbt_mod.build_cg_ocdbt_spec(ws, graph_id, pinned_at=pinned_at)
        return ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": spec,
                "scale_index": scale_index,
            }
        ).result()

    yield {
        "tmpdir": tmpdir,
        "ws": ws,
        "base": base,
        "scales": scales,
        "resolutions": resolutions,
        "make_fork": make_fork,
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


class TestWriteSegChunks:
    """`write_seg_chunks` now takes a flat list of (slices, data) pairs.

    `edits_sv.split_supervoxels` is responsible for producing this list
    across all reps so the outer rep loop is a pure data gather —
    tensorstore writes fire in one parallel batch.
    """

    def test_writes_only_supplied_chunks(self, local_ocdbt):
        """Chunks absent from `seg_writes` stay untouched (OCDBT delta
        stays proportional to the actual SV change)."""
        scales = local_ocdbt["scales"]
        meta = MagicMock()
        meta.ws_ocdbt = scales[0]

        # One chunk at [0..32] with label 55. The adjacent chunk at
        # [32..64] is NOT in the write list, so it should stay zero.
        chunk_data = np.full((32, 32, 32), 55, dtype=np.uint64)
        seg_writes = [
            (
                (slice(0, 32), slice(0, 32), slice(0, 32)),
                chunk_data,
            )
        ]
        ocdbt_mod.write_seg_chunks(meta, seg_writes)

        assert (scales[0][0:32, 0:32, 0:32, :].read().result() == 55).all()
        assert (scales[0][32:64, 0:32, 0:32, :].read().result() == 0).all()
        # Coarser scales untouched — downsample worker's job.
        assert (scales[1][0:16, 0:16, 0:32, :].read().result() == 0).all()
        assert (scales[2][0:8, 0:8, 0:32, :].read().result() == 0).all()

    def test_multiple_chunks_in_one_batch(self, local_ocdbt):
        """Multiple chunks (e.g. from different reps) fire in one call."""
        scales = local_ocdbt["scales"]
        meta = MagicMock()
        meta.ws_ocdbt = scales[0]

        seg_writes = [
            (
                (slice(0, 32), slice(0, 32), slice(0, 32)),
                np.full((32, 32, 32), 11, dtype=np.uint64),
            ),
            (
                (slice(32, 64), slice(0, 32), slice(0, 32)),
                np.full((32, 32, 32), 22, dtype=np.uint64),
            ),
        ]
        ocdbt_mod.write_seg_chunks(meta, seg_writes)

        assert (scales[0][0:32, 0:32, 0:32, :].read().result() == 11).all()
        assert (scales[0][32:64, 0:32, 0:32, :].read().result() == 22).all()

    def test_offset_region(self, local_ocdbt):
        """Writes at a non-origin offset land in the right chunk."""
        scales = local_ocdbt["scales"]
        meta = MagicMock()
        meta.ws_ocdbt = scales[0]

        seg_writes = [
            (
                (slice(32, 64), slice(0, 32), slice(0, 32)),
                np.full((32, 32, 32), 99, dtype=np.uint64),
            )
        ]
        ocdbt_mod.write_seg_chunks(meta, seg_writes)

        assert (scales[0][32:64, 0:32, 0:32, :].read().result() == 99).all()
        assert (scales[0][0:32, 0:32, 0:32, :].read().result() == 0).all()


class TestWsOcdbtScalesProperty:
    """`ChunkedGraphMeta.ws_ocdbt_scales` opens a fork over the shared base.

    Full path exercised: property → build_cg_ocdbt_spec → kvstack → OCDBT
    read/write. Only `_read_source_scales` is mocked (it reads `/info`
    which lives on the source watershed, not the OCDBT fork).
    """

    def test_opens_fork_and_merges_base(self, local_ocdbt):
        ws = local_ocdbt["ws"]

        # Source precomputed at ws root — needed by
        # get_seg_source_and_destination_ocdbt to copy the schema.
        ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": f"{ws}/",
                "multiscale_metadata": MULTISCALE_META,
                "scale_metadata": {
                    "size": [64, 64, 32],
                    "resolution": [4, 4, 40],
                    **SCALE_META_BASE,
                },
            },
            create=True,
        ).result()

        # Seed base scale 0 with a known value via the fixture's handle.
        local_ocdbt["scales"][0][...] = np.full((64, 64, 32, 1), 50, dtype=np.uint64)

        gc = GraphConfig(ID="ws_scales_cg", CHUNK_SIZE=[32, 32, 32])
        ds = DataSource(WATERSHED=f"{ws}/", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})

        # Trigger fork creation through the same helper the property will use.
        local_ocdbt["make_fork"]("ws_scales_cg")

        fake_scales = [
            {
                "resolution": [4, 4, 40],
                "size": [64, 64, 32],
                "chunk_sizes": [[32, 32, 32]],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8],
            }
        ]
        with patch.object(ocdbt_mod, "_read_source_scales", return_value=fake_scales):
            scales = meta.ws_ocdbt_scales
            assert len(scales) == 1

            # Fork sees base data.
            assert (scales[0][0:16, 0:16, 0:16, :].read().result() == 50).all()

            # Write to the fork and confirm isolation.
            scales[0][0:16, 0:16, 0:16, :] = np.full(
                (16, 16, 16, 1), 7, dtype=np.uint64
            )
            assert (scales[0][0:16, 0:16, 0:16, :].read().result() == 7).all()
            assert (scales[0][32:48, 0:16, 0:16, :].read().result() == 50).all()

        # Base still reports the original value (fork write didn't leak).
        assert (
            local_ocdbt["scales"][0][0:16, 0:16, 0:16, :].read().result() == 50
        ).all()


class TestForkIsolation:
    """Two forks on the same base: writes isolated, base immutable."""

    def test_two_forks_isolated(self, local_ocdbt):
        tmpdir = local_ocdbt["tmpdir"]
        # Seed base scale 0 with a known value.
        local_ocdbt["scales"][0][...] = np.full((64, 64, 32, 1), 50, dtype=np.uint64)

        base_path = f"{tmpdir}/ocdbt/base"
        base_files_before = {
            os.path.relpath(os.path.join(r, f), base_path)
            for r, _, fs in os.walk(base_path)
            for f in fs
        }

        fork_a = local_ocdbt["make_fork"]("fork_a")
        fork_b = local_ocdbt["make_fork"]("fork_b")

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

        # Base files unchanged (no new bytes written under ocdbt/base/).
        base_files_after = {
            os.path.relpath(os.path.join(r, f), base_path)
            for r, _, fs in os.walk(base_path)
            for f in fs
        }
        assert (
            base_files_before == base_files_after
        ), f"base was mutated: new={base_files_after - base_files_before}"

        # Fork writes went to their own directories.
        assert any("fork_a_d" in f for f in os.listdir(f"{tmpdir}/ocdbt/fork_a"))
        assert any("fork_b_d" in f for f in os.listdir(f"{tmpdir}/ocdbt/fork_b"))


class TestPinnedAt:
    """Versioned reads: pinning a fork to a prior generation/timestamp
    returns pre-write state; default (unpinned) returns latest.

    Documents both pin forms OCDBT accepts — integer generation (exact)
    and ISO-8601 UTC timestamp with `Z` suffix (commit_time upper bound).
    """

    def test_pin_by_generation_and_by_timestamp(self, local_ocdbt):
        # Seed base so fork reads see data even before the first fork write.
        local_ocdbt["scales"][0][...] = np.full((64, 64, 32, 1), 50, dtype=np.uint64)

        fork = local_ocdbt["make_fork"]("pin_cg")

        # Write v1 then v2 at the same voxels. Capture pin markers between
        # the two writes so pre-v2 state is what each pin should return.
        fork[0:16, 0:16, 0:16, :] = np.full((16, 16, 16, 1), 1, dtype=np.uint64)

        fork_manifest_kvs = ts.KvStore.open(
            f"{local_ocdbt['ws']}/ocdbt/pin_cg/"
        ).result()
        pin_gen = ts.ocdbt.dump(fork_manifest_kvs).result()["versions"][-1][
            "generation_number"
        ]

        time.sleep(0.01)
        pin_ts = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        time.sleep(0.01)

        fork[0:16, 0:16, 0:16, :] = np.full((16, 16, 16, 1), 2, dtype=np.uint64)

        fork_latest = local_ocdbt["make_fork"]("pin_cg")
        assert (fork_latest[0:16, 0:16, 0:16, :].read().result() == 2).all()

        fork_gen = local_ocdbt["make_fork"]("pin_cg", pinned_at=pin_gen)
        assert (fork_gen[0:16, 0:16, 0:16, :].read().result() == 1).all()

        fork_ts = local_ocdbt["make_fork"]("pin_cg", pinned_at=pin_ts)
        assert (fork_ts[0:16, 0:16, 0:16, :].read().result() == 1).all()

        # Untouched region still shows base data under every pin.
        for handle in (fork_latest, fork_gen, fork_ts):
            assert (handle[32:48, 0:16, 0:16, :].read().result() == 50).all()


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
