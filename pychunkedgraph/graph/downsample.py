"""Async mip-pyramid downsample worker support.

An SV split writes at base resolution only; coarser mips are produced
afterwards by a pubsub worker that consumes this module's primitives.

Work is organized into `pyramid_block`s. A block is a cubic physical
region sized so that at the coarsest scale in the pyramid it equals
exactly one storage chunk. Because every finer scale's chunk grid is a
power-of-2 refinement of the coarsest, a block aligned at the coarsest
scale is automatically aligned at every finer scale — so two different
blocks never share a storage chunk at any mip. That is what makes a
single lock per block safe.

Within a block we pick one of two code paths:
  1. Fast in-memory path: read the affected base region once, call
     tinybrain with `num_mips=K` (all mips at once), write each mip's
     output. Used when the base read fits a memory budget — the typical
     case because the SV-split bbox is bounded by the /split endpoint
     (source+sink coords + small padding).
  2. Per-mip fallback: read the previous mip, tinybrain one step, write.
     K storage round-trips instead of 1. Kept for pathological inputs
     whose base read would exceed the memory budget.

Uniform downsample factor (e.g. 2x2x2) across all non-base scales is
assumed and asserted.
"""

import numpy as np
import tinybrain

from pychunkedgraph import get_logger

logger = get_logger(__name__)

# Default memory budget for the in-memory path's base read.
# uint64 segmentation is 8 bytes/voxel; 1 GiB ≈ 512^3 voxels. Edits
# produced by the /split endpoint are bounded far below this.
DEFAULT_MEMORY_BUDGET_BYTES = 1 << 30


def num_output_mips(meta) -> int:
    """Count of non-base scales — what the worker actually writes."""
    return len(meta.ws_ocdbt_scales) - 1


def uniform_factor(meta) -> tuple:
    """Per-axis downsample factor between consecutive scales.

    tinybrain takes one factor tuple per call, so the factor must be
    constant across the pyramid. Asserts rather than silently producing
    wrong mips for a dataset with mixed factors.
    """
    resolutions = [np.array(r, dtype=float) for r in meta.ws_ocdbt_resolutions]
    factors = [
        tuple((resolutions[i] / resolutions[i - 1]).astype(int))
        for i in range(1, len(resolutions))
    ]
    assert all(
        f == factors[0] for f in factors
    ), f"non-uniform downsample factors {factors}"
    return factors[0]


def _chunk_size_at_scale(meta, scale_idx: int) -> np.ndarray:
    """Storage chunk size at a given scale (excluding the channel dim)."""
    return np.array(
        meta.ws_ocdbt_scales[scale_idx].chunk_layout.read_chunk.shape[:3], dtype=int
    )


def block_shape(meta) -> np.ndarray:
    """pyramid_block size in base-resolution voxels.

    Chosen so that at the coarsest scale K the block equals exactly one
    storage chunk — which transitively aligns it to every finer scale's
    chunk grid.
    """
    K = num_output_mips(meta)
    coarsest_chunk = _chunk_size_at_scale(meta, K)
    factor = np.array(uniform_factor(meta), dtype=int)
    return coarsest_chunk * factor**K


def blocks_for_bbox(meta, bbs, bbe) -> list:
    """Block coords intersected by a base-resolution bbox.

    Bbox is rounded outward to the block grid — a tiny bbox inside one
    block still yields that one block coord. Returns sorted list of
    `(bx, by, bz)` ints for deadlock-free lock acquisition.
    """
    shape = block_shape(meta)
    lo = np.asarray(bbs, dtype=int) // shape
    hi = -(-np.asarray(bbe, dtype=int) // shape)
    coords = [
        (int(bx), int(by), int(bz))
        for bx in range(lo[0], hi[0])
        for by in range(lo[1], hi[1])
        for bz in range(lo[2], hi[2])
    ]
    return sorted(coords)


def block_base_bbox(meta, block_coord) -> tuple:
    """Inverse of `blocks_for_bbox` for a single coord — base-voxel bbox."""
    shape = block_shape(meta)
    lo = np.asarray(block_coord, dtype=int) * shape
    hi = lo + shape
    return lo, hi


def _seg_bboxes_to_np(seg_bboxes):
    return [
        (np.asarray(bbs, dtype=int), np.asarray(bbe, dtype=int))
        for bbs, bbe in seg_bboxes
    ]


def _affected_region_base(meta, block_coord, seg_bboxes_np):
    """Base-voxel region covering all tiles this block will write, at any mip.

    Starts from the union of (seg bbox ∩ block ∩ volume) then aligns
    outward to the coarsest mip's base-voxel grid (= factor**K per axis).
    That alignment both makes the region tinybrain-valid for num_mips=K
    and guarantees clean chunk-aligned writes at every mip (coarsest
    alignment refines down to every finer scale).

    Returns `(base_lo, base_hi)` or `None` if no overlap.
    """
    K = num_output_mips(meta)
    factor = np.array(uniform_factor(meta), dtype=int)
    align = factor**K

    block_lo, block_hi = block_base_bbox(meta, block_coord)
    vol_lo = meta.voxel_bounds[:, 0]
    vol_hi = meta.voxel_bounds[:, 1]
    clipped_lo = np.maximum(block_lo, vol_lo)
    clipped_hi = np.minimum(block_hi, vol_hi)
    if np.any(clipped_hi <= clipped_lo):
        return None

    union_lo, union_hi = None, None
    for sb, eb in seg_bboxes_np:
        ilo = np.maximum(sb, clipped_lo)
        ihi = np.minimum(eb, clipped_hi)
        if np.any(ihi <= ilo):
            continue
        union_lo = ilo if union_lo is None else np.minimum(union_lo, ilo)
        union_hi = ihi if union_hi is None else np.maximum(union_hi, ihi)
    if union_lo is None:
        return None

    base_lo = (union_lo // align) * align
    base_hi = -(-union_hi // align) * align
    # Keep within the clipped block. Block corners are factor**K-aligned
    # (block_shape is a multiple of factor**K), so this clip preserves
    # alignment.
    base_lo = np.maximum(base_lo, clipped_lo)
    base_hi = np.minimum(base_hi, clipped_hi)
    if np.any(base_hi <= base_lo):
        return None
    return base_lo, base_hi


def _process_block_in_memory(meta, base_region, K, factor):
    """Read base once, tinybrain all mips, write each output.

    Assumes the base region is factor**K-aligned in size (which is what
    `_affected_region_base` returns) so tinybrain with num_mips=K emits
    clean integer voxel counts at every mip.
    """
    base_lo, base_hi = base_region
    base = meta.ws_ocdbt_scales[0]
    arr = (
        base[
            base_lo[0] : base_hi[0],
            base_lo[1] : base_hi[1],
            base_lo[2] : base_hi[2],
            :,
        ]
        .read()
        .result()
    )
    mips = tinybrain.downsample_segmentation(
        arr, factor=tuple(int(f) for f in factor), num_mips=K, sparse=False
    )
    for m, out in enumerate(mips, start=1):
        scale = factor**m
        mip_lo = base_lo // scale
        mip_hi = base_hi // scale
        dst = meta.ws_ocdbt_scales[m]
        dst[
            mip_lo[0] : mip_hi[0],
            mip_lo[1] : mip_hi[1],
            mip_lo[2] : mip_hi[2],
            :,
        ].write(out).result()


def _affected_region_at_mip(
    block_lo_base,
    block_hi_base,
    vol_lo,
    vol_hi,
    seg_bboxes_base,
    mip: int,
    factor: np.ndarray,
    mip_chunk: np.ndarray,
):
    """Write region at this mip in mip-local voxel coords.

    Union of seg bboxes ∩ block ∩ volume, aligned outward to this mip's
    storage-chunk grid. Returns `(mip_lo, mip_hi)` or None.
    """
    scale = factor**mip
    clipped_lo = np.maximum(block_lo_base, vol_lo)
    clipped_hi = np.minimum(block_hi_base, vol_hi)
    if np.any(clipped_hi <= clipped_lo):
        return None

    union_lo, union_hi = None, None
    for sb, eb in seg_bboxes_base:
        ilo = np.maximum(sb, clipped_lo)
        ihi = np.minimum(eb, clipped_hi)
        if np.any(ihi <= ilo):
            continue
        union_lo = ilo if union_lo is None else np.minimum(union_lo, ilo)
        union_hi = ihi if union_hi is None else np.maximum(union_hi, ihi)
    if union_lo is None:
        return None

    mip_lo = union_lo // scale
    mip_hi = -(-union_hi // scale)
    mip_lo = (mip_lo // mip_chunk) * mip_chunk
    mip_hi = -(-mip_hi // mip_chunk) * mip_chunk

    vol_lo_mip = vol_lo // scale
    vol_hi_mip = -(-vol_hi // scale)
    mip_lo = np.maximum(mip_lo, vol_lo_mip)
    mip_hi = np.minimum(mip_hi, vol_hi_mip)
    if np.any(mip_hi <= mip_lo):
        return None
    return mip_lo, mip_hi


def _process_block_per_mip(meta, block_coord, seg_bboxes_np, K, factor):
    """Fallback path: process one mip at a time.

    Used when the full in-memory base read would exceed the memory
    budget. Each mip reads the prior mip from storage, does one
    tinybrain step, writes.

    Safe across mip boundaries only because the caller holds the block
    lock — no other task can write the storage chunks this block owns,
    so reading mip N here always sees what we wrote at mip N in the
    previous iteration.
    """
    vol_lo = meta.voxel_bounds[:, 0]
    vol_hi = meta.voxel_bounds[:, 1]
    block_lo_base, block_hi_base = block_base_bbox(meta, block_coord)

    for mip in range(1, K + 1):
        mip_chunk = _chunk_size_at_scale(meta, mip)
        region = _affected_region_at_mip(
            block_lo_base,
            block_hi_base,
            vol_lo,
            vol_hi,
            seg_bboxes_np,
            mip,
            factor,
            mip_chunk,
        )
        if region is None:
            continue
        mip_lo, mip_hi = region
        src = meta.ws_ocdbt_scales[mip - 1]
        src_lo = mip_lo * factor
        src_hi = mip_hi * factor
        arr = (
            src[
                src_lo[0] : src_hi[0],
                src_lo[1] : src_hi[1],
                src_lo[2] : src_hi[2],
                :,
            ]
            .read()
            .result()
        )
        out = tinybrain.downsample_segmentation(
            arr, factor=tuple(int(f) for f in factor), num_mips=1, sparse=False
        )[0]
        dst = meta.ws_ocdbt_scales[mip]
        dst[
            mip_lo[0] : mip_hi[0],
            mip_lo[1] : mip_hi[1],
            mip_lo[2] : mip_hi[2],
            :,
        ].write(out).result()


def process_block(
    meta,
    block_coord,
    seg_bboxes,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
):
    """Downsample one pyramid_block through every non-base mip.

    Atomic within the block: caller must hold the block lock. Picks the
    in-memory path when the base read fits the memory budget, falls
    back to the per-mip path otherwise. Both paths only touch tiles
    whose footprint intersects `seg_bboxes` — unchanged tiles are
    skipped to keep OCDBT delta growth proportional to the actual edit.

    Args:
        meta: ChunkedGraphMeta with `ws_ocdbt_scales` / `ws_ocdbt_resolutions`.
        block_coord: (bx, by, bz) block grid coord.
        seg_bboxes: iterable of `(bbs, bbe)` base-voxel bbox pairs from
            the SV splits that triggered this job.
    """
    K = num_output_mips(meta)
    factor = np.array(uniform_factor(meta), dtype=int)
    seg_bboxes_np = _seg_bboxes_to_np(seg_bboxes)

    region = _affected_region_base(meta, block_coord, seg_bboxes_np)
    if region is None:
        return
    base_lo, base_hi = region

    bytes_per_voxel = meta.ws_ocdbt_scales[0].dtype.numpy_dtype.itemsize
    base_bytes = int(np.prod(base_hi - base_lo)) * bytes_per_voxel
    if base_bytes <= memory_budget_bytes:
        _process_block_in_memory(meta, region, K, factor)
    else:
        logger.info(
            f"block {block_coord} base read {base_bytes / 1e9:.2f} GB exceeds "
            f"budget {memory_budget_bytes / 1e9:.2f} GB; using per-mip path"
        )
        _process_block_per_mip(meta, block_coord, seg_bboxes_np, K, factor)
