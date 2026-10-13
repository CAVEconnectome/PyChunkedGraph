# pylint: disable=invalid-name, missing-docstring, redefined-outer-name

from types import SimpleNamespace

import numpy as np
import pytest

from pychunkedgraph.graph import exceptions as cg_exceptions
from pychunkedgraph.graph import limits


CHUNK_SIZE = [256, 256, 512]
# 1000 x 1000 x 1000 level 2 chunks
CHUNK_BOUNDS = np.array([1000, 1000, 1000])
# 8 level 2 chunks per request at most
LIMITS = {"MAX_BYTES": 8 * 1024**2, "BYTES_PER_L2_CHUNK": 1024**2}


@pytest.fixture
def cg():
    """Minimal stand-in exposing only what the bbox estimate reads."""
    meta = SimpleNamespace(
        graph_config=SimpleNamespace(CHUNK_SIZE=CHUNK_SIZE, FANOUT=2),
        layer_chunk_bounds={2: CHUNK_BOUNDS},
        resolution=np.array([8, 8, 40]),
        voxel_bounds=np.array([[0, 0], [0, 0], [0, 0]]),
    )
    return SimpleNamespace(meta=meta)


def _bbox(size):
    """Bounding box of `size` voxels per side, anchored at the origin."""
    return np.array([[0, 0, 0], list(size)])


def test_small_bbox_allowed(cg):
    # exactly 2 x 2 x 2 = 8 chunks, at the limit
    limits.check_subgraph_bounds(cg, _bbox(np.array(CHUNK_SIZE) * 2), LIMITS)


def test_large_bbox_rejected(cg):
    with pytest.raises(cg_exceptions.RequestTooLarge) as exc:
        limits.check_subgraph_bounds(cg, _bbox(np.array(CHUNK_SIZE) * 3), LIMITS)
    assert exc.value.status_code.value == 413
    assert "smaller boxes" in exc.value.message


def test_missing_bbox_rejected(cg):
    """No bounds is a request for the whole dataset."""
    with pytest.raises(cg_exceptions.RequestTooLarge):
        limits.check_subgraph_bounds(cg, None, LIMITS)


def test_bbox_clipped_to_dataset(cg):
    """A box reaching past the dataset only counts the chunks that exist."""
    cg.meta.layer_chunk_bounds = {2: np.array([1, 1, 1])}
    limits.check_subgraph_bounds(cg, _bbox([10**6, 10**6, 10**6]), LIMITS)


def test_no_limits_configured(cg):
    limits.check_subgraph_bounds(cg, None, None)


def test_empty_limits_fall_back_to_defaults(cg):
    """An empty entry is still guarded, with the module defaults."""
    with pytest.raises(cg_exceptions.RequestTooLarge):
        limits.check_subgraph_bounds(cg, None, {})
    # ~86 um^3 per chunk at this chunk size and resolution
    limits.check_subgraph_bounds(cg, _bbox(np.array(CHUNK_SIZE) * 2), {})


def test_default_calibration_tracks_chunk_volume(cg):
    """The default cost per chunk follows the chunk's physical volume."""
    per_chunk = limits.bytes_per_l2_chunk(cg.meta, {})
    chunk_um3 = np.prod(np.array(CHUNK_SIZE) * np.array([8, 8, 40])) / 1e9
    assert per_chunk == pytest.approx(
        chunk_um3 * limits.DEFAULT_BYTES_PER_CUBIC_MICRON
    )

    # a dataset chunked twice as coarsely in z costs twice as much per chunk
    cg.meta.graph_config.CHUNK_SIZE = [256, 256, 1024]
    assert limits.bytes_per_l2_chunk(cg.meta, {}) == pytest.approx(2 * per_chunk)


def test_suggested_box_fits_the_budget(cg):
    chunk_size = np.array(CHUNK_SIZE)
    for max_chunks in [1, 8, 5208, 10**6]:
        box = np.array(limits._suggested_box(cg.meta, max_chunks))
        assert np.all(box % chunk_size == 0)
        assert np.all(box >= chunk_size)
        assert np.prod(box // chunk_size) <= max_chunks

    # roughly cubic in nanometers rather than in chunks
    box = np.array(limits._suggested_box(cg.meta, 5208)) * np.array([8, 8, 40])
    assert box.max() / box.min() < 2


def test_measured_calibration_wins(cg):
    assert limits.bytes_per_l2_chunk(cg.meta, LIMITS) == 1024**2
    assert limits.bytes_per_l2_chunk(cg.meta, {"BYTES_PER_CUBIC_MICRON": 1}) == (
        pytest.approx(np.prod(np.array(CHUNK_SIZE) * np.array([8, 8, 40])) / 1e9)
    )


def test_level2_chunk_count(cg):
    assert limits.level2_chunk_count(cg.meta, _bbox(CHUNK_SIZE)) == 1
    assert limits.level2_chunk_count(cg.meta, _bbox(np.array(CHUNK_SIZE) * 4)) == 64
    # sub-chunk boxes still touch one chunk
    assert limits.level2_chunk_count(cg.meta, _bbox([1, 1, 1])) == 1
    assert limits.level2_chunk_count(cg.meta, None) == int(np.prod(CHUNK_BOUNDS))
