"""
Manage new supervoxels after a supervoxel split.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from typing import TYPE_CHECKING, List, Tuple

import fastremap
import numpy as np

from pychunkedgraph import get_logger
from pychunkedgraph.debug.profiler import get_profiler
from pychunkedgraph.graph import (
    attributes,
    cache as cache_utils,
    basetypes,
    serializers,
)
from pychunkedgraph.graph.chunks.utils import chunks_overlapping_bbox
from pychunkedgraph.graph.cutting_sv import split_supervoxel_helper
from pychunkedgraph.graph.edges_sv import update_edges, add_new_edges
from pychunkedgraph.graph.utils import get_local_segmentation

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph

logger = get_logger(__name__)


@dataclass
class SvSplitTask:
    """One SV-split task per cross-chunk rep.

    Produced by `plan_sv_splits` (pure, no IO), consumed by
    `split_supervoxel`. `src_mask`/`sink_mask` are positional masks
    back into the caller's `source_ids`/`sink_ids` arrays so the
    aggregator can splice the per-task fresh IDs in at the right
    positions.
    """

    sv_id: int
    src_coords: np.ndarray
    sink_coords: np.ndarray
    src_mask: np.ndarray
    sink_mask: np.ndarray
    bbs: np.ndarray
    bbe: np.ndarray


@dataclass
class SvSplitOutcome:
    """Output of `split_supervoxel` for one task. Aggregated into
    `SplitResult` by `split_supervoxels`."""

    seg_bbox: Tuple[np.ndarray, np.ndarray]
    src_new_ids: np.ndarray
    sink_new_ids: np.ndarray
    # Per-chunk OCDBT write payloads for this task.
    seg_write_pairs: List[Tuple[Tuple[slice, slice, slice], np.ndarray]]
    bigtable_rows: list


@dataclass
class SplitResult:
    """Pure planner output of `split_supervoxels`.

    The caller (`MulticutOperation._apply`) performs the actual writes
    under the L2 chunk locks:
    - `seg_writes` is fed to `write_seg_chunks` as one flat parallel batch.
    - `bigtable_rows` is written via `cg.client.write` in one batch.
    """

    seg_bboxes: List[Tuple[np.ndarray, np.ndarray]]
    source_ids_fresh: np.ndarray
    sink_ids_fresh: np.ndarray
    # Flat list across all tasks: (voxel_slices, data_block) per OCDBT
    # chunk write. `voxel_slices` is a 3-tuple of `slice` objects; the
    # caller appends the channel slice and writes to `meta.ws_ocdbt`.
    seg_writes: List[Tuple[Tuple[slice, slice, slice], np.ndarray]]
    bigtable_rows: list


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
            chunks_overlapping_bbox(read_lo, read_hi, chunk_size).keys()
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


def split_supervoxel(
    cg: "ChunkedGraph",
    task: SvSplitTask,
    operation_id: int,
    *,
    sv_remapping: dict,
    time_stamp: datetime = None,
    parent_ts: datetime = None,
    verbose: bool = False,
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
    source_coords = task.src_coords
    sink_coords = task.sink_coords
    bbs = task.bbs
    bbe = task.bbe

    vol_start = cg.meta.voxel_bounds[:, 0]
    vol_end = cg.meta.voxel_bounds[:, 1]
    logger.note(f"cg.meta.ws_ocdbt: {cg.meta.ws_ocdbt.shape}; res {cg.meta.resolution}")
    logger.note(f"bbox: {(bbs, bbe)}")

    rep = sv_remapping.get(sv_id, sv_id)
    rep_pieces = {int(sv) for sv, r in sv_remapping.items() if r == rep}

    # one voxel overlap for neighbors — update_edges needs anchor voxels
    # from neighboring SVs to route existing cross-chunk edges onto the
    # new fragments.
    bbs_ = np.clip(bbs - 1, vol_start, vol_end)
    bbe_ = np.clip(bbe + 1, vol_start, vol_end)
    _prof = get_profiler()
    t0 = time.time()
    with _prof.profile("seg_read"):
        seg = get_local_segmentation(cg.meta, bbs_, bbe_).squeeze()
    logger.note(f"segmentation read {seg.shape} ({time.time() - t0:.2f}s)")

    with _prof.profile("seg_unique"):
        # Computed once and reused: seg is not mutated until
        # remap_to_root, so both the cut_supervoxels narrowing below
        # and the get_roots block downstream get the same answer
        # without paying twice for the (seg-size) sort buffer.
        sv_ids = fastremap.unique(seg)

    # Narrow the rep to pieces actually present in the bbox seg. Pieces
    # of the rep whose voxels lie outside the seed-driven bbox don't
    # appear in `seg` and so don't contribute to `binary_seg` anyway —
    # carrying them in `cut_supervoxels` is just log noise plus inflated
    # `unsplit` diff churn.
    seg_ids = {int(x) for x in sv_ids if x != 0}
    cut_supervoxels = rep_pieces & seg_ids
    supervoxel_ids = np.array(list(cut_supervoxels), dtype=basetypes.NODE_ID)
    logger.note(
        f"whole sv {sv_id} -> {supervoxel_ids.tolist()} "
        f"({len(rep_pieces) - len(cut_supervoxels)} rep pieces outside bbox)"
    )

    with _prof.profile("binary_seg"):
        # OR a per-SV boolean comparison instead of np.isin: for a
        # handful of supervoxel_ids, each `seg == sv` is a single C
        # pass with no transient int-array buffers, vs. np.isin's
        # internal sort+search that allocates auxiliaries the size
        # of seg. Empty supervoxel_ids ⇒ all-False, matching np.isin.
        # Compute the boolean only over the overlap crop — that's
        # the only region split_supervoxel_helper consumes, so
        # allocating a full-bbox boolean and then slicing wastes
        # ~seg.size bytes.
        voxel_overlap_crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        seg_overlap = seg[voxel_overlap_crop]
        binary_seg = np.zeros(seg_overlap.shape, dtype=bool)
        for sv in supervoxel_ids:
            binary_seg |= seg_overlap == sv
    t0 = time.time()
    with _prof.profile("geodesic_split"):
        split_result = split_supervoxel_helper(
            binary_seg,
            source_coords - bbs,
            sink_coords - bbs,
            cg.meta.resolution,
            verbose=verbose,
        )
    logger.note(f"split computation {split_result.shape} ({time.time() - t0:.2f}s)")

    chunks_bbox_map = chunks_overlapping_bbox(bbs, bbe, cg.meta.graph_config.CHUNK_SIZE)
    t0 = time.time()
    results, change_chunks = _update_chunks(
        cg, chunks_bbox_map, seg[voxel_overlap_crop], split_result, bbs
    )
    logger.note(
        f"chunk updates {len(chunks_bbox_map)} chunks, "
        f"{len(change_chunks)} with splits ({time.time() - t0:.2f}s)"
    )

    with _prof.profile("parse_results"):
        seg_cropped = seg[voxel_overlap_crop].copy()
        new_seg, old_new_map, new_id_label_map = _parse_results(
            results, seg_cropped, bbs, bbe
        )
    logger.note(
        f"old_new_map: {len(old_new_map)} SVs split, whole_sv: {len(cut_supervoxels)} SVs"
    )
    unsplit = cut_supervoxels - set(old_new_map.keys())
    if unsplit:
        logger.note(f"unsplit SVs (kept IDs): {unsplit}")

    with _prof.profile("get_roots"):
        # sv_ids reused from the seg_unique block above (seg is not
        # mutated between then and here).
        roots = cg.get_roots(sv_ids, time_stamp=parent_ts)
        sv_root_map = dict(zip(sv_ids, roots))
    root = sv_root_map[sv_id]
    logger.note(f"{sv_id} -> {root}")

    # Zero out every label whose root != `root`, in place. The prior
    # implementation materialized a full-size shadow array via
    # fastremap.remap(in_place=False) just to compare against root;
    # mask_except achieves the same filter at C speed without the
    # shadow allocation.
    root_labels = [int(sv) for sv, r in sv_root_map.items() if r == root]
    fastremap.mask_except(seg, root_labels, in_place=True)
    seg[voxel_overlap_crop] = new_seg
    t0 = time.time()
    with _prof.profile("update_edges"):
        edges_tuple = update_edges(
            cg,
            root,
            np.array([bbs, bbe]),
            seg,
            old_new_map,
            new_id_label_map,
            parent_ts=parent_ts,
        )
    logger.note(f"edge update ({time.time() - t0:.2f}s)")

    rows0 = copy_parents_and_add_lineage(
        cg, operation_id, old_new_map, time_stamp=time_stamp
    )
    rows1 = add_new_edges(cg, edges_tuple, old_new_map, time_stamp=time_stamp)
    rows = rows0 + rows1

    # Prepare per-chunk OCDBT write payloads. The caller batches these
    # across all tasks into one parallel tensorstore write — no serial
    # per-task loop.
    seg_write_pairs: List[Tuple[Tuple[slice, slice, slice], np.ndarray]] = []
    for _, chunk_bbox in change_chunks:
        lo, hi = chunk_bbox[0], chunk_bbox[1]
        local_lo = lo - bbs
        local_hi = hi - bbs
        data = new_seg[
            local_lo[0] : local_hi[0],
            local_lo[1] : local_hi[1],
            local_lo[2] : local_hi[2],
        ]
        voxel_slices = tuple(slice(int(s), int(e)) for s, e in zip(lo, hi))
        seg_write_pairs.append((voxel_slices, data))

    # Per-coord fresh IDs: bit-identical to what a post-write seg read
    # would return — new_seg is what the caller is about to write, and
    # the caller holds the L2 chunk lock when it does, so the storage
    # round-trip would see the same bytes.
    local_src = (np.asarray(source_coords, dtype=int) - bbs).astype(int)
    local_sink = (np.asarray(sink_coords, dtype=int) - bbs).astype(int)
    src_new_ids = new_seg[tuple(local_src.T)]
    sink_new_ids = new_seg[tuple(local_sink.T)]
    return SvSplitOutcome(
        seg_bbox=(bbs, bbe),
        src_new_ids=src_new_ids,
        sink_new_ids=sink_new_ids,
        seg_write_pairs=seg_write_pairs,
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
