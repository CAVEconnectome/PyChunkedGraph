"""Per-stage data containers shared across the SV-split modules.

Pulled into a leaf module so that ``edits``, ``profile``, ``inspect``,
``bridge_check``, and any future helper can all type-reference the same
dataclasses without import cycles.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph


@dataclass
class SvSplitTask:
    """One SV-split task per cross-chunk rep.

    Produced by ``plan_sv_splits`` (pure, no IO), consumed by
    ``split_supervoxel``. ``src_mask`` / ``sink_mask`` are positional masks
    back into the caller's ``source_ids`` / ``sink_ids`` arrays so the
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
class SplitCtx:
    """Per-task context shared across the split stage helpers.

    Holds the inputs every stage threads through unchanged. ``seg`` is
    mutated in place across stages (fresh IDs written, then root mask);
    the reference is stable, so storing it here is sound.
    """

    cg: "ChunkedGraph"
    seg: np.ndarray
    bbs: np.ndarray
    bbe: np.ndarray
    bbs_: np.ndarray
    bbe_: np.ndarray
    sv_id: int
    sv_ids: np.ndarray
    source_coords: np.ndarray
    sink_coords: np.ndarray
    operation_id: int
    time_stamp: datetime
    parent_ts: datetime


@dataclass
class ApplyResult:
    """Outputs of ``_apply_and_capture`` consumed by the orchestrator.

    ``full_shape`` / ``fg_shape`` are stamped on by ``split_supervoxel``
    after ``_compute_split`` so the run record exposes the bbox the
    geodesic actually ran on (the foreground crop) vs the bbox derived
    from seed coords. Diff between them measures how much the rep is
    concentrated near the seeds.
    """

    old_new_map: dict
    new_id_label_map: dict
    seg_write_pairs: List[Tuple[Tuple[slice, slice, slice], np.ndarray]]
    src_new_ids: np.ndarray
    sink_new_ids: np.ndarray
    new_edges_tuple: Optional[tuple] = None
    full_shape: Optional[Tuple[int, int, int]] = None
    fg_shape: Optional[Tuple[int, int, int]] = None


@dataclass
class SvSplitOutcome:
    """Output of ``split_supervoxel`` for one task. Aggregated into
    ``SplitResult`` by ``split_supervoxels``."""

    seg_bbox: Tuple[np.ndarray, np.ndarray]
    src_new_ids: np.ndarray
    sink_new_ids: np.ndarray
    # Per-chunk OCDBT write payloads for this task.
    seg_write_pairs: List[Tuple[Tuple[slice, slice, slice], np.ndarray]]
    bigtable_rows: list
    applied: Optional[ApplyResult] = None


@dataclass
class SplitResult:
    """Pure planner output of ``split_supervoxels``.

    The caller (``MulticutOperation._apply``) performs the actual writes
    under the L2 chunk locks:
    - ``seg_writes`` is fed to ``write_seg_chunks`` as one flat parallel batch.
    - ``bigtable_rows`` is written via ``cg.client.write`` in one batch.
    """

    seg_bboxes: List[Tuple[np.ndarray, np.ndarray]]
    source_ids_fresh: np.ndarray
    sink_ids_fresh: np.ndarray
    # Flat list across all tasks: (voxel_slices, data_block) per OCDBT
    # chunk write. ``voxel_slices`` is a 3-tuple of ``slice`` objects; the
    # caller appends the channel slice and writes to ``meta.ws_ocdbt``.
    seg_writes: List[Tuple[Tuple[slice, slice, slice], np.ndarray]]
    bigtable_rows: list
    # Per-task outcomes — carries each task's ApplyResult for post-split
    # graph inspection (defence-in-depth bridge_check).
    task_outcomes: Optional[List[SvSplitOutcome]] = None
