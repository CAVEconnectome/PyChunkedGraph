"""
Manage new supervoxels after a supervoxel split.
"""

import time
from datetime import datetime
from collections import defaultdict
from typing import TYPE_CHECKING, List, Tuple

import fastremap
import numpy as np

from pychunkedgraph import get_logger
from pychunkedgraph.profiler import get_profiler
from pychunkedgraph.graph import (
    attributes,
    cache as cache_utils,
    basetypes,
    serializers,
)
from pychunkedgraph.graph.chunks.utils import chunks_overlapping_bbox
from pychunkedgraph.graph.exceptions import PostconditionError
from .cutting import connect_both_seeds_via_ridge, split_supervoxel_growing
from .edges import update_edges, add_new_edges
from .state import (
    ApplyResult,
    SplitCtx,
    SplitResult,
    SvSplitOutcome,
    SvSplitTask,
)
from pychunkedgraph.graph.utils import get_local_segmentation

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph

logger = get_logger(__name__)


def _coords_bbox(
    cg: "ChunkedGraph",
    src_coords_rep: np.ndarray,
    sink_coords_rep: np.ndarray,
) -> tuple:
    """Base-voxel bbox covering the user's source/sink seeds plus a margin.

    The cut surface lives between the user-placed source and sink
    voxels; voxels of the rep that are far from those seeds never
    contribute to the cut. So the read region is the seeds' envelope,
    not the rep's full chunk envelope — for a physical SV cut into many
    pieces across chunks, this can be orders of magnitude smaller.

    The margin is one CG chunk on each side. It matches the existing
    L2 chunk lock margin and the 1-voxel shell read in
    `split_supervoxel`, and gives `split_supervoxel_helper` headroom
    around the seeds for the cut surface to travel along the SV.

    Pieces of the rep that fall outside the bbox keep their existing
    IDs — they aren't read here and aren't rewritten. Cross-chunk-edge
    routing for boundary-adjacent pieces is handled by the 1-voxel
    shell at read time; cross-chunk edges entirely between unsplit
    pieces don't change because their IDs don't change.
    """
    coords = np.concatenate([src_coords_rep, sink_coords_rep], axis=0)
    margin = np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=int)
    vol_start = cg.meta.voxel_bounds[:, 0]
    vol_end = cg.meta.voxel_bounds[:, 1]
    bbs = np.clip(coords.min(axis=0) - margin, vol_start, vol_end)
    bbe = np.clip(coords.max(axis=0) + margin, vol_start, vol_end)
    return bbs, bbe


def _l2_chunks_for_splits(cg: "ChunkedGraph", per_rep_bboxes: list) -> list[int]:
    """Layer-2 chunk IDs every rep's split will read or write.

    Reads extend 1 voxel past `[bbs, bbe]` so `update_edges` has anchor
    voxels for cross-chunk neighbors; the lock must cover those neighbor
    chunks too, hence the `bbs - 1` / `bbe + 1` expansion. Clipped to
    volume bounds so a bbox on the volume edge doesn't enumerate phantom
    negative-index chunks. Sorted for deterministic lock-acquire order
    (L2ChunkLock relies on sorted input for deadlock avoidance).
    """
    vol_start = cg.meta.voxel_bounds[:, 0]
    vol_end = cg.meta.voxel_bounds[:, 1]
    chunk_size = cg.meta.graph_config.CHUNK_SIZE
    chunk_coords = set()
    for bbs, bbe in per_rep_bboxes:
        read_lo = np.clip(bbs - 1, vol_start, vol_end)
        read_hi = np.clip(bbe + 1, vol_start, vol_end)
        chunk_coords.update(
            chunks_overlapping_bbox(
                read_lo, read_hi, chunk_size, origin=vol_start
            ).keys()
        )
    return sorted(
        int(cg.get_chunk_id(layer=2, x=x, y=y, z=z)) for (x, y, z) in chunk_coords
    )


def _overlapping_reps(
    *,
    sv_remapping: dict,
    source_ids: np.ndarray,
    sink_ids: np.ndarray,
    source_coords: np.ndarray,
    sink_coords: np.ndarray,
):
    """Yield per-rep data for every rep that links source and sink.

    A rep is a cross-chunk-representative SV shared by at least one
    source and one sink in `sv_remapping`. These are the SVs that must
    be split before the multicut can partition source from sink.

    Yields `(sv_id, src_coords_rep, sink_coords_rep, src_mask, sink_mask)`:
        sv_id           — one of the rep's source SV IDs, used as the
                          seed for `split_supervoxel`.
        src_coords_rep  — slice of source_coords whose SV maps to this rep.
        sink_coords_rep — slice of sink_coords whose SV maps to this rep.
        src_mask        — positional boolean mask over source_ids; the
                          caller uses it to splice per-rep results back
                          into the full source arrays.
        sink_mask       — same, for sink_ids.

    Keyword-only signature — positional source/sink args of the same
    shape are easy to swap without noticing.
    """
    sources_remapped = fastremap.remap(
        source_ids, sv_remapping, preserve_missing_labels=True, in_place=False
    )
    sinks_remapped = fastremap.remap(
        sink_ids, sv_remapping, preserve_missing_labels=True, in_place=False
    )
    overlap_mask = np.isin(sources_remapped, sinks_remapped)
    for rep in np.unique(sources_remapped[overlap_mask]):
        src_mask = sources_remapped == rep
        sink_mask = sinks_remapped == rep
        yield (
            source_ids[src_mask][0],
            source_coords[src_mask],
            sink_coords[sink_mask],
            src_mask,
            sink_mask,
        )


def plan_sv_splits(
    cg: "ChunkedGraph",
    *,
    sv_remapping: dict,
    source_ids: np.ndarray,
    sink_ids: np.ndarray,
    source_coords: np.ndarray,
    sink_coords: np.ndarray,
) -> Tuple[List[SvSplitTask], list]:
    """Compute one `SvSplitTask` per rep and the L2 chunk set the splits
    will touch.

    Pure function — no bigtable/OCDBT IO, no locks. Lets the caller
    acquire the L2 chunk locks (both temporal and indefinite) around
    `split_supervoxels` without recomputing the plan inside.

    Returns `(tasks, chunk_ids)` — `tasks` feeds `split_supervoxels`,
    `chunk_ids` is the sorted union of read-expanded L2 chunks the full
    operation touches.
    """
    tasks: List[SvSplitTask] = []
    for (
        sv_id,
        src_coords_rep,
        sink_coords_rep,
        src_mask,
        sink_mask,
    ) in _overlapping_reps(
        sv_remapping=sv_remapping,
        source_ids=source_ids,
        sink_ids=sink_ids,
        source_coords=source_coords,
        sink_coords=sink_coords,
    ):
        bbs, bbe = _coords_bbox(cg, src_coords_rep, sink_coords_rep)
        tasks.append(
            SvSplitTask(
                sv_id=sv_id,
                src_coords=src_coords_rep,
                sink_coords=sink_coords_rep,
                src_mask=src_mask,
                sink_mask=sink_mask,
                bbs=bbs,
                bbe=bbe,
            )
        )
    chunk_ids = _l2_chunks_for_splits(cg, [(t.bbs, t.bbe) for t in tasks])
    return tasks, chunk_ids


def split_supervoxels(
    cg: "ChunkedGraph",
    *,
    tasks: List[SvSplitTask],
    sv_remapping: dict,
    source_ids: np.ndarray,
    sink_ids: np.ndarray,
    operation_id: int,
    timestamp: datetime = None,
    parent_ts: datetime = None,
) -> SplitResult:
    """Pure planner for the SV-split step. Returns a `SplitResult` with
    all the data the caller needs to persist under locks.

    Does **not** write — the caller (`MulticutOperation._apply`) owns
    the L2 chunk lock lifecycle and fires the OCDBT + bigtable writes
    inside `IndefiniteL2ChunkLock`.

    Must be called inside the caller's `L2ChunkLock` for the
    `plan.chunk_ids` set — the seg reads inside `split_supervoxel` need
    to be consistent with concurrent writers.

    `timestamp` is the op's logical write time; threaded down to every
    `mutate_row` in the persist block so all new-SV cells land at the
    same logical time (atomic visibility for `parent_ts`-filtered
    readers, and deterministic replay via `override_ts`).

    Fields on the returned `SplitResult`:
        seg_bboxes: per-task base-resolution `(bbs, bbe)` — downsample
            worker input.
        source_ids_fresh / sink_ids_fresh: input `source_ids`/`sink_ids`
            with positions touched by an overlap task replaced by the
            new SV ID that now lives at that coord. Untouched positions
            stay unchanged. Feeds the retry multicut.
        seg_writes: flat list of `(voxel_slices, data)` pairs across all
            tasks — one tensorstore write per pair, fired in parallel.
        bigtable_rows: flattened rows from `copy_parents_and_add_lineage`
            + `add_new_edges` across all tasks.
    """
    source_ids_fresh = np.asarray(source_ids, dtype=basetypes.NODE_ID).copy()
    sink_ids_fresh = np.asarray(sink_ids, dtype=basetypes.NODE_ID).copy()

    seg_bboxes = []
    seg_writes: List[Tuple[Tuple[slice, slice, slice], np.ndarray]] = []
    bigtable_rows: list = []
    for task in tasks:
        out = split_supervoxel(
            cg,
            task,
            operation_id,
            sv_remapping=sv_remapping,
            time_stamp=timestamp,
            parent_ts=parent_ts,
        )
        seg_bboxes.append(out.seg_bbox)
        source_ids_fresh[task.src_mask] = out.src_new_ids
        sink_ids_fresh[task.sink_mask] = out.sink_new_ids
        seg_writes.extend(out.seg_write_pairs)
        bigtable_rows.extend(out.bigtable_rows)
    return SplitResult(
        seg_bboxes=seg_bboxes,
        source_ids_fresh=source_ids_fresh,
        sink_ids_fresh=sink_ids_fresh,
        seg_writes=seg_writes,
        bigtable_rows=bigtable_rows,
    )


def _update_chunks(cg: "ChunkedGraph", chunks_bbox_map, seg, result_seg, bb_start):
    """Process all chunks in a single pass: assign new SV IDs to split fragments.

    Returns `(results, change_chunks)`:
        results: per-chunk (indices, old_values, new_values, label_id_map)
            tuples; consumed by `_parse_results`.
        change_chunks: `(chunk_coord, chunk_bbox)` for the chunks whose
            voxels received new SV IDs. `write_seg_chunks` uses this to
            rewrite only those chunks (skipping gap chunks that had no
            split activity keeps the OCDBT delta proportional to actual
            label changes).
    """
    results = []
    change_chunks = []
    for chunk_coord, chunk_bbox in chunks_bbox_map.items():
        x, y, z = chunk_coord
        chunk_id = cg.get_chunk_id(layer=1, x=x, y=y, z=z)

        _s, _e = chunk_bbox - bb_start
        og_chunk_seg = seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]
        chunk_seg = result_seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]

        labels = fastremap.unique(chunk_seg[chunk_seg != 0])
        if labels.size < 2:
            continue

        new_ids = cg.id_client.create_node_ids(chunk_id, size=len(labels))
        _indices = []
        _old_values = []
        _new_values = []
        _label_id_map = {}
        for _id, new_id in zip(labels, new_ids):
            _mask = chunk_seg == _id
            voxel_locs = np.where(_mask)
            _og_value = og_chunk_seg[
                voxel_locs[0][0], voxel_locs[1][0], voxel_locs[2][0]
            ]
            _index = np.column_stack(voxel_locs)
            n = len(_index)
            _indices.append(_index)
            _old_values.append(np.full(n, _og_value, dtype=basetypes.NODE_ID))
            _new_values.append(np.full(n, new_id, dtype=basetypes.NODE_ID))
            _label_id_map[int(_id)] = new_id

        _indices = np.concatenate(_indices) + (chunk_bbox[0] - bb_start)
        _old_values = np.concatenate(_old_values)
        _new_values = np.concatenate(_new_values)
        results.append((_indices, _old_values, _new_values, _label_id_map))
        change_chunks.append((chunk_coord, chunk_bbox))
    return results, change_chunks


def _voxel_crop(bbs, bbe, bbs_, bbe_):
    xS, yS, zS = bbs - bbs_
    xE, yE, zE = (None if i == 0 else -1 for i in bbe_ - bbe)
    voxel_overlap_crop = np.s_[xS:xE, yS:yE, zS:zE]
    return voxel_overlap_crop


def _assert_same_chunk(cg: "ChunkedGraph", old_new_map: dict) -> None:
    """Every new SV must live in the same chunk as the SV it split from.

    PCG segment IDs are unique only within a chunk; a split fragment that
    landed in a different chunk than its parent would break the hierarchy.
    """
    olds = np.fromiter(old_new_map.keys(), dtype=basetypes.NODE_ID)
    news = np.fromiter(
        (n for ns in old_new_map.values() for n in ns), dtype=basetypes.NODE_ID
    )
    expected = np.repeat(
        cg.get_chunk_ids_from_node_ids(olds), [len(ns) for ns in old_new_map.values()]
    )
    assert np.array_equal(
        cg.get_chunk_ids_from_node_ids(news), expected
    ), "new supervoxel landed in a different chunk than the SV it split from"


def _parse_results(results, seg, bbs, bbe):
    """Merge per-chunk split results into a single segmentation volume.

    Applies new SV IDs from each chunk's split result to `seg` (in-place)
    and builds the old→new mapping + label→new-id mapping.

    Returns (seg, old_new_map, new_id_label_map).
    """
    old_new_map = defaultdict(set)
    new_id_label_map = {}
    for result in results:
        if result:
            indexer, old_values, new_values, label_id_map = result
            seg[tuple(indexer.T)] = new_values
            for old_sv, new_sv in zip(old_values, new_values):
                old_new_map[old_sv].add(new_sv)
            for label, new_id in label_id_map.items():
                new_id_label_map[new_id] = label

    assert np.all(seg.shape == bbe - bbs), f"{seg.shape} != {bbe - bbs}"
    return seg, old_new_map, new_id_label_map


def _read_seg_and_ids(cg: "ChunkedGraph", bbs, bbe, *, sv_id=None, op_id=None):
    """Read seg over [bbs-1, bbe+1] and return its distinct SV IDs.

    The 1-voxel shell gives update_edges anchor voxels from neighbouring
    SVs. Returns (seg, sv_ids, bbs_, bbe_).
    """
    vol_start = cg.meta.voxel_bounds[:, 0]
    vol_end = cg.meta.voxel_bounds[:, 1]
    bbs_ = np.clip(bbs - 1, vol_start, vol_end)
    bbe_ = np.clip(bbe + 1, vol_start, vol_end)
    _prof = get_profiler()
    t0 = time.time()
    with _prof.profile("seg_read"):
        seg = get_local_segmentation(cg.meta, bbs_, bbe_).squeeze()
    logger.note(f"<{op_id}> {sv_id}: read {seg.shape} ({time.time() - t0:.2f}s)")

    with _prof.profile("seg_unique"):
        # Unique per chunk on the segment-id field only. Segment IDs are
        # injective within a chunk and narrower than uint64, so the per-
        # block sort is cheap; the chunk bits are OR'd back before the
        # final union. The lattice is anchored at voxel_bounds[:, 0] so
        # each block is exactly one chunk. Background 0 carries no chunk
        # and is restored once at the end.
        chunk_map = chunks_overlapping_bbox(
            bbs_, bbe_, cg.meta.graph_config.CHUNK_SIZE, origin=vol_start
        )
        parts = []
        has_zero = False
        for (cx, cy, cz), cbbox in chunk_map.items():
            s, e = cbbox[0] - bbs_, cbbox[1] - bbs_
            sub = seg[s[0] : e[0], s[1] : e[1], s[2] : e[2]]
            chunk_id = np.uint64(cg.get_chunk_id(layer=1, x=cx, y=cy, z=cz))
            limit = np.uint64(cg.get_segment_id_limit(chunk_id))
            narrow = np.min_scalar_type(int(limit))
            u = fastremap.unique((sub & limit).astype(narrow, copy=False))
            if u.size and u[0] == 0:
                has_zero = True
                u = u[1:]
            parts.append(u.astype(np.uint64) | chunk_id)
        sv_ids = np.unique(np.concatenate(parts)) if parts else np.array([], np.uint64)
        if has_zero:
            sv_ids = np.concatenate([[np.uint64(0)], sv_ids])
    return seg, sv_ids, bbs_, bbe_


def _select_cut_supervoxels(sv_id, sv_ids, rep_pieces, *, op_id=None):
    """Narrow the rep to the pieces actually present in the bbox seg.

    Rep pieces whose voxels lie outside the seed-driven bbox don't appear
    in seg and contribute nothing to the cut. Returns (cut_supervoxels,
    supervoxel_ids).
    """
    seg_ids = {int(x) for x in sv_ids if x != 0}
    cut_supervoxels = rep_pieces & seg_ids
    supervoxel_ids = np.array(list(cut_supervoxels), dtype=basetypes.NODE_ID)
    logger.note(
        f"<{op_id}> {sv_id}: whole_sv in_bbox={len(cut_supervoxels)} "
        f"outside_bbox={len(rep_pieces) - len(cut_supervoxels)}"
    )
    logger.verbose(
        f"<{op_id}> {sv_id}: pieces={supervoxel_ids.tolist()}"
    )
    return cut_supervoxels, supervoxel_ids


_SNAP_KWARGS = dict(
    use_boundary=False,
    downsample=False,
    use_bbox=True,
    method="kdtree",
)


def split_supervoxel_helper(ctx: SplitCtx, binary_seg: np.ndarray):
    """Run the geodesic SV cut for one task.

    Wraps ``connect_both_seeds_via_ridge`` + ``split_supervoxel_growing``
    with the SV-split-flow defaults; threads ``ctx.sv_id`` so per-task
    logs inside ``split_supervoxel_growing`` are individually tagged.
    """
    voxel_size = np.array(ctx.cg.meta.resolution)
    downsample = voxel_size.max() // voxel_size  # xyz order
    # Per-axis clamp:
    #   - max_axis_ds bounds the per-axis stride so the cut surface
    #     precision stays within a small multiple of the finest voxel
    #     dimension (3× ≈ 24 nm on pinky).
    #   - min_grid_per_axis ensures ≥ N cells per axis post-downsample so
    #     narrow_band_rel has room to refine and small SVs fall back to
    #     full-res rather than collapsing the geodesic grid.
    # binary_seg.shape is xyz (seg_overlap is xyz from ctx.seg); zip downsample
    # and shape both in xyz, then reverse the final tuple to zyx for the
    # geodesic call's axis convention.
    max_axis_ds = 3
    min_grid_per_axis = 16
    ds_xyz = tuple(
        max(1, min(int(s), max_axis_ds, dim // min_grid_per_axis))
        for s, dim in zip(downsample, binary_seg.shape)
    )
    ds_zyx = ds_xyz[::-1]
    _prof = get_profiler()
    src = ctx.source_coords - ctx.bbs
    sink = ctx.sink_coords - ctx.bbs
    t0 = time.time()
    with _prof.profile("connect_seeds"):
        A_aug, B_aug, okA, okB = connect_both_seeds_via_ridge(
            binary_seg,
            src,
            sink,
            voxel_size=voxel_size,
            downsample=downsample,
            vol_order="xyz",
            vox_order="xyz",
            seed_order="xyz",
            snap_method="kdtree",
            snap_kwargs=dict(_SNAP_KWARGS),
        )
    logger.note(
        f"<{ctx.operation_id}> {ctx.sv_id}: connect_seeds ({time.time() - t0:.2f}s)"
    )
    if not (okA and okB):
        raise RuntimeError(
            "In-mask connection failed for at least one team; skipping split."
        )
    with _prof.profile("split_growing"):
        return split_supervoxel_growing(
            binary_seg,
            A_aug,
            B_aug,
            voxel_size=voxel_size,
            vol_order="xyz",
            vox_order="xyz",
            seed_order="xyz",
            halo=1,
            gamma_neck=1.6,
            narrow_band_rel=0.08,
            nb_dilate=1,
            downsample_geodesic=ds_zyx,
            enforce_single_cc=True,
            raise_if_seed_split=True,
            raise_if_multi_cc=True,
            snap_method="kdtree",
            snap_kwargs=dict(_SNAP_KWARGS),
            sv_id=ctx.sv_id,
            op_id=ctx.operation_id,
        )


def _compute_split(ctx: SplitCtx, supervoxel_ids):
    """Build the binary mask over the overlap crop and run the cut.

    Returns (split_result, voxel_overlap_crop).
    """
    _prof = get_profiler()
    with _prof.profile("binary_seg"):
        # Chunked per-SV OR over the overlap crop. The plain loop would
        # peak at 2× the bool output (binary_seg + one per-iter transient).
        # Slabbing in z caps the transient at the byte budget below; np.isin
        # is worse on memory here because the SV-id value range is too
        # wide for `kind='table'` and `kind='sort'` allocates an int64
        # permutation buffer ≈ 8× the input.
        voxel_overlap_crop = _voxel_crop(ctx.bbs, ctx.bbe, ctx.bbs_, ctx.bbe_)
        seg_overlap = ctx.seg[voxel_overlap_crop]
        binary_seg = np.empty(seg_overlap.shape, dtype=bool)
        yx = int(seg_overlap.shape[1]) * int(seg_overlap.shape[2])
        slab_bytes = 128 * 1024 * 1024
        slab_z = max(1, slab_bytes // max(yx, 1))
        for z0 in range(0, seg_overlap.shape[0], slab_z):
            z1 = min(z0 + slab_z, seg_overlap.shape[0])
            slab = seg_overlap[z0:z1]
            binary_seg[z0:z1] = slab == supervoxel_ids[0]
            for sv in supervoxel_ids[1:]:
                binary_seg[z0:z1] |= slab == sv
    t0 = time.time()
    logger.note(
        f"<{ctx.operation_id}> {ctx.sv_id}: split computation starting shape={binary_seg.shape}"
    )
    split_result = split_supervoxel_helper(ctx, binary_seg)
    logger.note(
        f"<{ctx.operation_id}> {ctx.sv_id}: split computation done "
        f"shape={split_result.shape} ({time.time() - t0:.2f}s)"
    )
    return split_result, voxel_overlap_crop


def _pick_fresh_source_sink_ids(
    old_new_map: dict,
    new_id_label_map: dict,
    n_source: int,
    n_sink: int,
    *,
    sv_id,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick a label-1 and a label-2 fragment from the same old SV (same chunk)
    and broadcast to ``n_source`` / ``n_sink`` length arrays.

    Same-chunk fragments are directly connected by a 0.001 inter-fragment
    bridge in ``add_new_edges``, giving the retry multicut a guaranteed
    one-hop path between sources and sinks irrespective of what extends
    beyond the local-subgraph bbox. Falls back to any label-1 / label-2
    only if no old SV produced both sides (degenerate cut).
    """
    src_frag = sink_frag = None
    for old_sv in sorted(old_new_map):
        new_ids = sorted(old_new_map[old_sv])
        l1 = [n for n in new_ids if new_id_label_map.get(n) == 1]
        l2 = [n for n in new_ids if new_id_label_map.get(n) == 2]
        if l1 and l2:
            src_frag, sink_frag = l1[0], l2[0]
            break
    if src_frag is None:
        src_frag = next(
            (n for n in sorted(new_id_label_map) if new_id_label_map[n] == 1), None
        )
    if sink_frag is None:
        sink_frag = next(
            (n for n in sorted(new_id_label_map) if new_id_label_map[n] == 2), None
        )
    if src_frag is None or sink_frag is None:
        raise PostconditionError(
            f"cut for sv {sv_id} produced no fragments on "
            f"{'source' if src_frag is None else 'sink'} side"
        )
    return (
        np.full(n_source, src_frag, dtype=basetypes.NODE_ID),
        np.full(n_sink, sink_frag, dtype=basetypes.NODE_ID),
    )


def _apply_and_capture(
    ctx: SplitCtx, voxel_overlap_crop, split_result, cut_supervoxels
):
    """Apply fresh IDs to seg's crop and capture the write/lookup outputs.

    Writes fresh SV IDs into seg's overlap crop in place (a view, no
    full-crop copy; _parse_results only writes, never reads crop values),
    then captures the OCDBT write payloads and the src/sink id lookups
    while the crop still holds unmasked neighbour IDs. Everything here
    runs before the root mask, which would otherwise zero the neighbour
    IDs the write must preserve. Returns an `ApplyResult`.
    """
    cg, seg, bbs, bbe = ctx.cg, ctx.seg, ctx.bbs, ctx.bbe
    _prof = get_profiler()
    chunks_bbox_map = chunks_overlapping_bbox(
        bbs, bbe, cg.meta.graph_config.CHUNK_SIZE, origin=cg.meta.voxel_bounds[:, 0]
    )
    t0 = time.time()
    results, change_chunks = _update_chunks(
        cg, chunks_bbox_map, seg[voxel_overlap_crop], split_result, bbs
    )
    logger.note(
        f"<{ctx.operation_id}> {ctx.sv_id}: chunk updates {len(chunks_bbox_map)} chunks, "
        f"{len(change_chunks)} with splits ({time.time() - t0:.2f}s)"
    )

    with _prof.profile("parse_results"):
        new_seg = seg[voxel_overlap_crop]
        new_seg, old_new_map, new_id_label_map = _parse_results(
            results, new_seg, bbs, bbe
        )
    _assert_same_chunk(cg, old_new_map)
    unsplit = cut_supervoxels - set(old_new_map.keys())
    logger.note(
        f"<{ctx.operation_id}> {ctx.sv_id}: split_svs={len(old_new_map)} "
        f"unsplit_kept={len(unsplit)}"
    )
    if unsplit:
        logger.verbose(f"<{ctx.operation_id}> {ctx.sv_id}: unsplit kept IDs: {unsplit}")

    # .copy() per changed chunk detaches each payload from seg before the
    # mask / update_edges mutate it; changed chunks only, so the copies
    # stay proportional to the edit. The caller batches them into one
    # parallel tensorstore write.
    seg_write_pairs: List[Tuple[Tuple[slice, slice, slice], np.ndarray]] = []
    for _, chunk_bbox in change_chunks:
        lo, hi = chunk_bbox[0], chunk_bbox[1]
        local_lo = lo - bbs
        local_hi = hi - bbs
        data = new_seg[
            local_lo[0] : local_hi[0],
            local_lo[1] : local_hi[1],
            local_lo[2] : local_hi[2],
        ].copy()
        voxel_slices = tuple(slice(int(s), int(e)) for s, e in zip(lo, hi))
        seg_write_pairs.append((voxel_slices, data))

    src_new_ids, sink_new_ids = _pick_fresh_source_sink_ids(
        old_new_map,
        new_id_label_map,
        len(ctx.source_coords),
        len(ctx.sink_coords),
        sv_id=ctx.sv_id,
    )
    return ApplyResult(
        old_new_map=old_new_map,
        new_id_label_map=new_id_label_map,
        seg_write_pairs=seg_write_pairs,
        src_new_ids=src_new_ids,
        sink_new_ids=sink_new_ids,
    )


def _route_edges_and_rows(ctx: SplitCtx, old_new_map, new_id_label_map):
    """Resolve the split's root, route edges, build bigtable rows.

    Returns the flat list of bigtable rows.
    """
    cg, seg = ctx.cg, ctx.seg
    _prof = get_profiler()
    with _prof.profile("get_roots"):
        roots = cg.get_roots(ctx.sv_ids, time_stamp=ctx.parent_ts)
        sv_root_map = dict(zip(ctx.sv_ids, roots))
    root = sv_root_map[ctx.sv_id]

    t0 = time.time()
    with _prof.profile("update_edges"):
        edges_tuple = update_edges(
            cg,
            root,
            np.array([ctx.bbs, ctx.bbe]),
            seg,
            old_new_map,
            new_id_label_map,
            parent_ts=ctx.parent_ts,
            sv_id=ctx.sv_id,
            op_id=ctx.operation_id,
        )
    logger.note(
        f"<{ctx.operation_id}> {ctx.sv_id} -> {root} new_edges {edges_tuple[0].shape} "
        f"({time.time() - t0:.2f}s)"
    )

    rows0 = copy_parents_and_add_lineage(
        cg, ctx.operation_id, old_new_map, time_stamp=ctx.time_stamp
    )
    rows1 = add_new_edges(cg, edges_tuple, old_new_map, time_stamp=ctx.time_stamp)
    return rows0 + rows1


def split_supervoxel(
    cg: "ChunkedGraph",
    task: SvSplitTask,
    operation_id: int,
    *,
    sv_remapping: dict,
    time_stamp: datetime = None,
    parent_ts: datetime = None,
) -> SvSplitOutcome:
    """Split one cross-chunk-connected SV into connected components.

    `task.bbs` / `task.bbe` are the base-voxel bbox covering the user's
    source and sink seeds plus a one-chunk margin — `plan_sv_splits`
    pre-computed this via `_coords_bbox`. The bbox is driven by where
    the user wants the cut, not by the rep's full chunk envelope; rep
    pieces outside the bbox aren't read and keep their existing IDs.

    `time_stamp` is the op's logical write time; threaded through to
    `copy_parents_and_add_lineage` + `add_new_edges` so every new-SV
    mutation lands at the same timestamp.
    """
    sv_id = task.sv_id
    bbs = task.bbs
    bbe = task.bbe

    op_id = operation_id
    t_start = time.time()
    logger.note(f"<{op_id}> [sv_split:start] {sv_id} bbox=({bbs}, {bbe})")

    rep = sv_remapping.get(sv_id, sv_id)
    rep_pieces = {int(sv) for sv, r in sv_remapping.items() if r == rep}

    seg, sv_ids, bbs_, bbe_ = _read_seg_and_ids(cg, bbs, bbe, sv_id=sv_id, op_id=op_id)
    ctx = SplitCtx(
        cg=cg,
        seg=seg,
        bbs=bbs,
        bbe=bbe,
        bbs_=bbs_,
        bbe_=bbe_,
        sv_id=sv_id,
        sv_ids=sv_ids,
        source_coords=task.src_coords,
        sink_coords=task.sink_coords,
        operation_id=operation_id,
        time_stamp=time_stamp,
        parent_ts=parent_ts,
    )
    cut_supervoxels, supervoxel_ids = _select_cut_supervoxels(
        sv_id, sv_ids, rep_pieces, op_id=op_id
    )
    split_result, voxel_overlap_crop = _compute_split(ctx, supervoxel_ids)
    applied = _apply_and_capture(ctx, voxel_overlap_crop, split_result, cut_supervoxels)
    rows = _route_edges_and_rows(ctx, applied.old_new_map, applied.new_id_label_map)

    logger.note(
        f"<{op_id}> [sv_split:end] {sv_id} elapsed={time.time() - t_start:.2f}s"
    )
    return SvSplitOutcome(
        seg_bbox=(bbs, bbe),
        src_new_ids=applied.src_new_ids,
        sink_new_ids=applied.sink_new_ids,
        seg_write_pairs=applied.seg_write_pairs,
        bigtable_rows=rows,
    )


def copy_parents_and_add_lineage(
    cg: "ChunkedGraph",
    operation_id: int,
    old_new_map: dict,
    *,
    time_stamp: datetime = None,
) -> list:
    """Copy parent pointers from old SVs onto their new-ID fragments
    and write the lineage (FormerIdentity / NewIdentity) + L2 Child
    list updates.

    `time_stamp` is the op's logical write time — used for every new-SV
    cell this function writes so a `parent_ts`-filtered reader sees the
    op atomically. The Parent-copy and Child-list writes deliberately
    preserve the old cell's timestamp (so pre-op readers still see the
    old hierarchy via the old timestamp).

    Returns a list of mutations to be persisted.
    """
    result = []
    parents = set()
    old_new_map = {k: list(v) for k, v in old_new_map.items()}
    parent_cells_map = cg.client.read_nodes(
        node_ids=list(old_new_map.keys()), properties=attributes.Hierarchy.Parent
    )
    for old_id, new_ids in old_new_map.items():
        for new_id in new_ids:
            val_dict = {
                attributes.Hierarchy.FormerIdentity: np.array(
                    [old_id], dtype=basetypes.NODE_ID
                ),
                attributes.OperationLogs.OperationID: operation_id,
            }
            result.append(
                cg.client.mutate_row(
                    serializers.serialize_uint64(new_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
            for cell in parent_cells_map[old_id]:
                cache_utils.update(cg.cache.parents_cache, [new_id], cell.value)
                parents.add(cell.value)
                result.append(
                    cg.client.mutate_row(
                        serializers.serialize_uint64(new_id),
                        {attributes.Hierarchy.Parent: cell.value},
                        time_stamp=cell.timestamp,
                    )
                )
        val_dict = {
            attributes.Hierarchy.NewIdentity: np.array(new_ids, dtype=basetypes.NODE_ID)
        }
        result.append(
            cg.client.mutate_row(
                serializers.serialize_uint64(old_id),
                val_dict,
                time_stamp=time_stamp,
            )
        )

    children_cells_map = cg.client.read_nodes(
        node_ids=list(parents), properties=attributes.Hierarchy.Child
    )
    for parent, children_cells in children_cells_map.items():
        assert len(children_cells) == 1, children_cells
        for cell in children_cells:
            mask = np.isin(cell.value, list(old_new_map.keys()))
            replace = np.concatenate([old_new_map[x] for x in cell.value[mask]])
            children = np.concatenate([cell.value[~mask], replace])
            cg.cache.children_cache[parent] = children
            result.append(
                cg.client.mutate_row(
                    serializers.serialize_uint64(parent),
                    {attributes.Hierarchy.Child: children},
                    time_stamp=cell.timestamp,
                )
            )
    return result
