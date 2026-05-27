"""Re-runnable dry-run profile harness for SV splits.

Drives an SV-split operation end-to-end under ``PCG_DRY_RUN=1`` so no
BT or OCDBT state is mutated, captures per-stage timing + memory + IO
metrics into a ``HierarchicalProfiler`` (one ``BlockMetrics`` row per
stage), and snapshots each stage's intermediate result into a
``SplitInputs`` dataclass that's persisted alongside the profiler.

The persisted run lets the user iterate on a single heavy stage in
isolation (e.g. profile just ``split_supervoxels`` after editing it)
without re-running the prior stages.
"""

import hashlib
import json
import pickle
import shutil
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pychunkedgraph.app.segmentation.common import _get_sources_and_sinks
from pychunkedgraph.debug.profiler import (
    HierarchicalProfiler,
    get_profiler,
)
from pychunkedgraph.graph import edits_sv
from pychunkedgraph.graph import utils as _utils_pkg
from pychunkedgraph.graph.dry_run import dry_run_scope
from pychunkedgraph.graph.operation import Cut, MulticutOperation, SvSplitRequired
from pychunkedgraph.graph.utils import generic as _utils_generic
from pychunkedgraph.graph.utils import id_helpers as _utils_id_helpers

_CACHE_ROOT = Path(tempfile.gettempdir()) / "pcg_split_profile"


@dataclass
class SplitInputs:
    """Per-stage inputs/outputs captured during a ``run_split_profile`` run.

    Persisted to disk alongside the profiler so single-stage replays
    can reuse the inputs without re-running prior stages. Every field
    matches the exact value at the corresponding call site in
    ``MulticutOperation._apply``.
    """

    operation_id: Optional[int] = None
    timestamp: Any = None
    source_ids_pre: Optional[np.ndarray] = None
    sink_ids_pre: Optional[np.ndarray] = None
    source_coords: Optional[np.ndarray] = None
    sink_coords: Optional[np.ndarray] = None
    sv_remapping: Optional[dict] = None
    plan_tasks: Any = None
    plan_chunk_ids: Any = None
    sv_result: Any = None
    cut: Any = None


def _payload_canonical(payload: dict) -> str:
    """Canonical JSON encoding used for hashing and collision detection."""
    return json.dumps(payload, sort_keys=True)


def _payload_sha(payload: dict) -> str:
    """First 8 hex chars of sha256 over the canonical-JSON payload."""
    return hashlib.sha256(_payload_canonical(payload).encode()).hexdigest()[:8]


def run_dir(cg, payload: dict) -> Path:
    """Cache directory for ``(cg.graph_id, payload)`` under system tmp."""
    return _CACHE_ROOT / cg.graph_id / _payload_sha(payload)


@contextmanager
def count_io(cg):
    """Count BT row reads + OCDBT bytes read for the wrapped block.

    Wraps ``cg.client.read_nodes`` and ``cg.client.read_log_entries``
    (BT reads), plus ``get_local_segmentation`` at every top-level
    binding site reached from the SV-split flow (OCDBT reads). All
    originals are restored on exit.
    """
    counters: Dict[str, int] = {
        "bt_row_reads": 0,
        "bt_log_reads": 0,
        "ocdbt_reads": 0,
        "ocdbt_bytes": 0,
    }

    orig_read_nodes = cg.client.read_nodes

    def wrap_read_nodes(*a, **k):
        result = orig_read_nodes(*a, **k)
        counters["bt_row_reads"] += len(result) if result is not None else 0
        return result

    orig_read_log = cg.client.read_log_entries

    def wrap_read_log(*a, **k):
        result = orig_read_log(*a, **k)
        counters["bt_log_reads"] += len(result) if result is not None else 0
        return result

    cg.client.read_nodes = wrap_read_nodes
    cg.client.read_log_entries = wrap_read_log

    # Patch every binding of get_local_segmentation reached from the
    # SV-split flow. The source module is _utils_generic; the others
    # imported it by name at module load time, so they hold separate
    # references that need their own swap.
    seg_modules = [_utils_generic, _utils_pkg, edits_sv, _utils_id_helpers]
    orig_seg_fns = {m: m.get_local_segmentation for m in seg_modules}

    def wrap_get_local_seg(meta, bbox_start, bbox_end, mip=0):
        # Always call the source function so we don't double-count if
        # one wrapped binding calls another.
        arr = orig_seg_fns[_utils_generic](meta, bbox_start, bbox_end, mip)
        counters["ocdbt_bytes"] += int(arr.nbytes)
        counters["ocdbt_reads"] += 1
        return arr

    for m in seg_modules:
        m.get_local_segmentation = wrap_get_local_seg

    try:
        yield counters
    finally:
        cg.client.read_nodes = orig_read_nodes
        cg.client.read_log_entries = orig_read_log
        for m, fn in orig_seg_fns.items():
            m.get_local_segmentation = fn


def profile_call(cg, name, fn, *args, **kwargs):
    """Profile a single callable under dry-run with IO counters.

    Standalone replay helper for per-stage profiling (e.g. after
    editing a single function's source). Opens ``dry_run_scope`` +
    ``count_io``, runs ``profiler.profile(name, with_memory=True,
    with_rss=True, counters=counters)`` around ``fn(*args, **kwargs)``,
    returns ``(profiler, result)``. The profiler has exactly one block.
    """
    profiler = HierarchicalProfiler(enabled=True)
    with dry_run_scope(), count_io(cg) as counters:
        with profiler.profile(name, counters=counters):
            result = fn(*args, **kwargs)
    return profiler, result


def build_op(
    cg,
    payload: dict,
    *,
    user_id: str = "dry_run_profile",
    bbox_offset: Tuple[int, int, int] = (240, 240, 24),
) -> MulticutOperation:
    """Decode a /split JSON payload and instantiate ``MulticutOperation``.

    Mirrors ``ChunkedGraph.remove_edges`` direct instantiation pattern.
    The caller drives ``op.execute()``.
    """
    source_ids, sink_ids, source_coords, sink_coords = _get_sources_and_sinks(
        cg, payload
    )
    op = MulticutOperation(
        cg,
        user_id=user_id,
        source_ids=source_ids,
        sink_ids=sink_ids,
        source_coords=source_coords,
        sink_coords=sink_coords,
        bbox_offset=bbox_offset,
    )
    return op


def annotate_chunks(cg, chunk_ids) -> List[str]:
    """Annotate each chunk id with its NGL-navigable center voxel."""
    out: List[str] = []
    for cid in chunk_ids:
        coord = cg.get_chunk_center_voxel(int(cid)).tolist()
        out.append(f"{int(cid):#x} -> voxel {coord}")
    return out


def _save_run(
    cg,
    payload: dict,
    profiler: HierarchicalProfiler,
    inputs: SplitInputs,
) -> Path:
    """Write run artifacts under ``run_dir``; raise on payload-hash collision."""
    target = run_dir(cg, payload)
    payload_path = target / "payload.json"
    incoming = _payload_canonical(payload)
    if payload_path.exists():
        existing = payload_path.read_text()
        if existing != incoming:
            raise RuntimeError(
                f"payload-hash collision at {target}: "
                "existing payload != incoming payload"
            )
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "inputs.pkl", "wb") as f:
        pickle.dump(inputs, f)
    with open(target / "profiler.pkl", "wb") as f:
        pickle.dump(profiler, f)
    buf = StringIO()
    with redirect_stdout(buf):
        profiler.metrics_report()
    (target / "metrics.txt").write_text(buf.getvalue())
    payload_path.write_text(incoming)
    return target


def load_run(cg, payload: dict) -> Tuple[HierarchicalProfiler, SplitInputs]:
    """Restore a prior ``run_split_profile`` result from disk."""
    target = run_dir(cg, payload)
    with open(target / "profiler.pkl", "rb") as f:
        profiler = pickle.load(f)
    with open(target / "inputs.pkl", "rb") as f:
        inputs = pickle.load(f)
    return profiler, inputs


def run_split_profile(
    cg, payload: dict, *, overwrite: bool = False
) -> Tuple[HierarchicalProfiler, SplitInputs]:
    """Drive an SV split under dry-run with per-stage metrics captured.

    Returns ``(profiler, inputs)``. Uses the global profiler so inline
    ``get_profiler().profile()`` blocks inside the SV-split call path
    are captured automatically. ``inputs`` holds each stage's
    intermediate values for standalone replay.

    ``overwrite=True`` wipes any existing cached run for this payload
    before starting.

    Always writes a cache (profiler + inputs + metrics.txt) to
    ``run_dir(cg, payload)`` on completion — even when ``op.execute()``
    raises — and prints the cache path.
    """
    target_dir = run_dir(cg, payload)
    if target_dir.exists():
        if overwrite:
            shutil.rmtree(target_dir)
        else:
            raise FileExistsError(
                f"cached run already exists at {target_dir}; "
                "pass overwrite=True to wipe and re-run, or "
                "load_run(cg, payload) to read it"
            )

    profiler = get_profiler()
    profiler.reset()
    profiler.enabled = True
    inputs = SplitInputs()

    op = build_op(cg, payload)
    inputs.source_ids_pre = op.source_ids.copy()
    inputs.sink_ids_pre = op.sink_ids.copy()
    inputs.source_coords = op.source_coords
    inputs.sink_coords = op.sink_coords

    with dry_run_scope(), count_io(cg) as counters:
        profiler.default_counters = counters

        # Capture-only wrappers for SplitInputs replay — no profile()
        # blocks. The real per-step metrics come from inline profile()
        # blocks inside the called functions.
        orig_run_multicut = MulticutOperation._run_multicut
        orig_plan_sv_splits = edits_sv.plan_sv_splits
        orig_split_supervoxels = edits_sv.split_supervoxels

        mincut_call_count = [0]

        def wrap_run_multicut(self_op, operation_id):
            result = orig_run_multicut(self_op, operation_id)
            mincut_call_count[0] += 1
            if mincut_call_count[0] == 1 and isinstance(result, SvSplitRequired):
                inputs.sv_remapping = result.sv_remapping
            elif isinstance(result, Cut):
                inputs.cut = result
            return result

        def wrap_plan_sv_splits(*a, **k):
            result = orig_plan_sv_splits(*a, **k)
            inputs.plan_tasks, inputs.plan_chunk_ids = result
            return result

        def wrap_split_supervoxels(*a, **k):
            if "operation_id" in k:
                inputs.operation_id = k["operation_id"]
            if "timestamp" in k:
                inputs.timestamp = k["timestamp"]
            result = orig_split_supervoxels(*a, **k)
            inputs.sv_result = result
            return result

        MulticutOperation._run_multicut = wrap_run_multicut
        edits_sv.plan_sv_splits = wrap_plan_sv_splits
        edits_sv.split_supervoxels = wrap_split_supervoxels

        try:
            op.execute()
        except Exception as e:
            print(
                f"[split_profile] op.execute() raised " f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
        finally:
            MulticutOperation._run_multicut = orig_run_multicut
            edits_sv.plan_sv_splits = orig_plan_sv_splits
            edits_sv.split_supervoxels = orig_split_supervoxels
            profiler.default_counters = None

    # Disable so the global profiler is a no-op for callers outside
    # this harness (production code paths included).
    profiler.enabled = False

    try:
        target = _save_run(cg, payload, profiler, inputs)
        print(f"[split_profile] run cached at {target}")
    except Exception as save_err:
        print(f"[split_profile] cache save failed: {save_err}", file=sys.stderr)
    return profiler, inputs
