"""Tests for pychunkedgraph.graph.downsample."""

import shutil
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from pychunkedgraph.graph import downsample as ds
from pychunkedgraph.graph.locks import (
    DownsampleBlockLock,
    _downsample_block_lock_row_key,
)
from pychunkedgraph.graph import exceptions
from pychunkedgraph.tests.helpers import (
    RowKeyLockRegistry,
    make_cg_with_row_key_lock_registry,
)


@pytest.fixture
def local_ocdbt():
    """3-scale file-backed OCDBT store with factor (2,2,1) between scales.

    Matches the fixture in test_ocdbt.py so downsample behaviour can be
    exercised end-to-end against real tensorstore handles.
    """
    tmpdir = tempfile.mkdtemp()
    base = f"file://{tmpdir}/ocdbt/base"
    mm = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}

    def mk(size, resolution, extra_mm=None):
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
        mk([64, 64, 32], [4, 4, 40], extra_mm=mm),
        mk([32, 32, 32], [8, 8, 40]),
        mk([16, 16, 32], [16, 16, 40]),
    ]
    resolutions = [[4, 4, 40], [8, 8, 40], [16, 16, 40]]

    yield {"scales": scales, "resolutions": resolutions}
    shutil.rmtree(tmpdir)


def _make_meta(local_ocdbt_, voxel_bounds=None):
    """Minimal ChunkedGraphMeta stand-in with only the attributes downsample reads."""
    scales = local_ocdbt_["scales"]
    if voxel_bounds is None:
        # Full volume from scale 0.
        dom = scales[0].domain
        voxel_bounds = np.array(
            [
                [dom[0].inclusive_min, dom[0].exclusive_max],
                [dom[1].inclusive_min, dom[1].exclusive_max],
                [dom[2].inclusive_min, dom[2].exclusive_max],
            ],
            dtype=int,
        )
    return SimpleNamespace(
        ws_ocdbt_scales=scales,
        ws_ocdbt_resolutions=local_ocdbt_["resolutions"],
        voxel_bounds=voxel_bounds,
    )


class TestBlockGeometry:
    def test_num_output_mips(self, local_ocdbt):
        meta = _make_meta(local_ocdbt)
        assert ds.num_output_mips(meta) == 2

    def test_uniform_factor(self, local_ocdbt):
        meta = _make_meta(local_ocdbt)
        assert ds.uniform_factor(meta) == (2, 2, 1)

    def test_non_uniform_factor_asserts(self, local_ocdbt):
        meta = _make_meta(local_ocdbt)
        meta.ws_ocdbt_resolutions = [[4, 4, 40], [8, 8, 40], [8, 16, 40]]
        with pytest.raises(AssertionError):
            ds.uniform_factor(meta)

    def test_block_shape_covers_one_coarsest_chunk(self, local_ocdbt):
        # coarsest chunk = 32 mip-2 voxels per axis; factor^2 = (4,4,1).
        # Block = 32 * (4,4,1) = (128, 128, 32) base voxels.
        meta = _make_meta(local_ocdbt)
        assert tuple(ds.block_shape(meta).tolist()) == (128, 128, 32)

    def test_blocks_for_bbox_single(self, local_ocdbt):
        meta = _make_meta(local_ocdbt)
        # Tiny bbox entirely inside block (0,0,0).
        blocks = ds.blocks_for_bbox(meta, [10, 10, 5], [20, 20, 10])
        assert blocks == [(0, 0, 0)]

    def test_blocks_for_bbox_spans_block_boundary(self, local_ocdbt):
        meta = _make_meta(local_ocdbt)
        # Block shape = (128,128,32). Bbox from (120,0,0) to (200,50,10)
        # crosses the x-axis boundary at 128.
        blocks = ds.blocks_for_bbox(meta, [120, 0, 0], [200, 50, 10])
        assert blocks == sorted([(0, 0, 0), (1, 0, 0)])

    def test_block_base_bbox_roundtrip(self, local_ocdbt):
        meta = _make_meta(local_ocdbt)
        lo, hi = ds.block_base_bbox(meta, (0, 0, 0))
        assert tuple(lo.tolist()) == (0, 0, 0)
        assert tuple(hi.tolist()) == (128, 128, 32)

        lo, hi = ds.block_base_bbox(meta, (2, 1, 0))
        assert tuple(lo.tolist()) == (256, 128, 0)
        assert tuple(hi.tolist()) == (384, 256, 32)


class TestProcessBlockInMemory:
    def test_writes_to_every_non_base_scale(self, local_ocdbt):
        """Base region intersected by bbox propagates to mip 1 and mip 2."""
        scales = local_ocdbt["scales"]
        # Seed base with a constant label.
        data = np.full((32, 32, 32), 7, dtype=np.uint64)
        scales[0][0:32, 0:32, 0:32, :].write(data[..., np.newaxis]).result()

        meta = _make_meta(local_ocdbt)
        # Block (0,0,0) has shape (128,128,32); only its (0..32, 0..32, 0..32)
        # subregion has real data — the rest is zeros.
        ds.process_block(
            meta, (0, 0, 0), [(np.array([0, 0, 0]), np.array([32, 32, 32]))]
        )

        mip1 = scales[1][0:16, 0:16, 0:32, :].read().result()
        mip2 = scales[2][0:8, 0:8, 0:32, :].read().result()
        assert (mip1 == 7).all()
        assert (mip2 == 7).all()

    def test_region_outside_bbox_stays_zero(self, local_ocdbt):
        """Mip tiles whose base footprint misses the bbox are not written."""
        scales = local_ocdbt["scales"]
        # Seed base with 3 inside the edit bbox only.
        edit_data = np.full((16, 16, 16), 3, dtype=np.uint64)
        scales[0][0:16, 0:16, 0:16, :].write(edit_data[..., np.newaxis]).result()

        meta = _make_meta(local_ocdbt)
        ds.process_block(
            meta, (0, 0, 0), [(np.array([0, 0, 0]), np.array([16, 16, 16]))]
        )

        # Tile inside edit: written with label 3.
        mip1_inside = scales[1][0:8, 0:8, 0:16, :].read().result()
        assert (mip1_inside == 3).all()
        # Tile outside edit (far corner of block): still zero.
        mip1_outside = scales[1][12:16, 12:16, 16:32, :].read().result()
        assert (mip1_outside == 0).all()


class TestProcessBlockDispatcher:
    def test_selects_in_memory_when_under_budget(self, local_ocdbt, monkeypatch):
        """Typical small affected region → in-memory path."""
        calls = {"in_memory": 0, "per_mip": 0}
        monkeypatch.setattr(
            ds,
            "_process_block_in_memory",
            lambda *a, **kw: calls.__setitem__("in_memory", calls["in_memory"] + 1),
        )
        monkeypatch.setattr(
            ds,
            "_process_block_per_mip",
            lambda *a, **kw: calls.__setitem__("per_mip", calls["per_mip"] + 1),
        )
        meta = _make_meta(local_ocdbt)
        ds.process_block(
            meta, (0, 0, 0), [(np.array([0, 0, 0]), np.array([16, 16, 16]))]
        )
        assert calls == {"in_memory": 1, "per_mip": 0}

    def test_selects_per_mip_when_over_budget(self, local_ocdbt, monkeypatch):
        """When the base read would exceed budget, the per-mip path runs."""
        calls = {"in_memory": 0, "per_mip": 0}
        monkeypatch.setattr(
            ds,
            "_process_block_in_memory",
            lambda *a, **kw: calls.__setitem__("in_memory", calls["in_memory"] + 1),
        )
        monkeypatch.setattr(
            ds,
            "_process_block_per_mip",
            lambda *a, **kw: calls.__setitem__("per_mip", calls["per_mip"] + 1),
        )
        meta = _make_meta(local_ocdbt)
        ds.process_block(
            meta,
            (0, 0, 0),
            [(np.array([0, 0, 0]), np.array([128, 128, 32]))],
            memory_budget_bytes=1,  # force the fallback
        )
        assert calls == {"in_memory": 0, "per_mip": 1}


class TestDownsampleBlockRowKey:
    def test_length(self):
        assert len(_downsample_block_lock_row_key((0, 0, 0))) == 26

    def test_deterministic(self):
        assert _downsample_block_lock_row_key(
            (7, 8, 9)
        ) == _downsample_block_lock_row_key((7, 8, 9))

    def test_distinct_coords_distinct_keys(self):
        a = _downsample_block_lock_row_key((1, 0, 0))
        b = _downsample_block_lock_row_key((0, 1, 0))
        assert a != b

    def test_hash_prefix_scatters(self):
        """Adjacent block coords should not produce adjacent row keys (the whole
        point of the hash prefix)."""
        # Gather hash prefixes for a line of adjacent coords; they should span
        # many distinct first-bytes, not cluster in one byte.
        prefixes = {_downsample_block_lock_row_key((i, 0, 0))[0] for i in range(128)}
        assert len(prefixes) > 32


class TestDownsampleBlockLock:
    def test_acquire_and_release(self):
        registry = RowKeyLockRegistry()
        cg = make_cg_with_row_key_lock_registry(registry)
        with DownsampleBlockLock(cg, [(0, 0, 0), (1, 0, 0)], np.uint64(42)):
            assert len(registry._held) == 2
        assert registry._held == {}

    def test_non_overlapping_concurrent(self):
        """Two locks on disjoint block sets can coexist."""
        registry = RowKeyLockRegistry()
        cg = make_cg_with_row_key_lock_registry(registry)
        l1 = DownsampleBlockLock(cg, [(0, 0, 0)], np.uint64(1))
        l2 = DownsampleBlockLock(cg, [(5, 5, 5)], np.uint64(2))
        l1.__enter__()
        l2.__enter__()
        assert len(registry._held) == 2
        l1.__exit__(None, None, None)
        l2.__exit__(None, None, None)
        assert registry._held == {}

    def test_overlapping_contends(self, monkeypatch):
        """Two overlapping acquisitions serialize: second blocks until first releases."""
        # Short backoff so the waiting thread retries quickly after release.
        monkeypatch.setattr(DownsampleBlockLock, "_ACQUIRE_BACKOFF_BASE_SEC", 0.05)

        registry = RowKeyLockRegistry()
        cg = make_cg_with_row_key_lock_registry(registry)

        l1 = DownsampleBlockLock(cg, [(0, 0, 0)], np.uint64(1))
        l1.__enter__()

        second_entered = threading.Event()
        second_failed = threading.Event()

        def second():
            lock = DownsampleBlockLock(cg, [(0, 0, 0)], np.uint64(2))
            try:
                lock.__enter__()
                second_entered.set()
                lock.__exit__(None, None, None)
            except exceptions.LockingError:
                second_failed.set()

        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.2)
        # l1 is still holding; second should not have entered.
        assert not second_entered.is_set()
        # Now release; second should succeed on its next retry.
        l1.__exit__(None, None, None)
        t.join(timeout=2.0)
        assert second_entered.is_set()
        assert not second_failed.is_set()
        assert registry._held == {}

    def test_partial_acquire_released_on_failure(self, monkeypatch):
        """If any coord in the set fails to lock, prior ones are released."""
        monkeypatch.setattr(DownsampleBlockLock, "_MAX_ACQUIRE_ATTEMPTS", 2)
        monkeypatch.setattr(DownsampleBlockLock, "_ACQUIRE_BACKOFF_BASE_SEC", 0.01)

        registry = RowKeyLockRegistry()
        # Pre-hold (1,0,0) so the second coord always fails.
        registry.lock_by_row_key(
            _downsample_block_lock_row_key((1, 0, 0)), np.uint64(99)
        )

        cg = make_cg_with_row_key_lock_registry(registry)
        lock = DownsampleBlockLock(cg, [(0, 0, 0), (1, 0, 0)], np.uint64(1))
        with pytest.raises(exceptions.LockingError):
            lock.__enter__()
        # Only (1,0,0) should remain held, by the pre-existing holder.
        assert len(registry._held) == 1
        only_key = next(iter(registry._held))
        assert only_key == _downsample_block_lock_row_key((1, 0, 0))
