import os
import time

os.environ["PCG_PROFILER_ENABLED"] = "1"

import numpy as np

import pychunkedgraph.debug.profiler as profiler_mod
from pychunkedgraph.debug.profiler import HierarchicalProfiler
from pychunkedgraph.graph import ChunkedGraph, basetypes
from .utils import extract_structure


def run_current_stitch(graph_id: str, atomic_edges: np.ndarray, do_sanity_check: bool = True) -> dict:
    """
    Run the existing add_edges stitch path on a graph copy.
    Same calling convention as dist/internal/chunkedgraph/operations.py.
    Returns dict with structural result and metadata.
    """

    class SilentProfiler(HierarchicalProfiler):
        def print_report(self, *a, **kw):
            pass

    profiler_mod._profiler = SilentProfiler(enabled=True)

    atomic_edges = np.asarray(atomic_edges, dtype=basetypes.NODE_ID)
    cg = ChunkedGraph(graph_id=graph_id)

    print(f"  [current] stitch ({len(atomic_edges)} edges)...")
    t0 = time.time()
    result = cg.add_edges(
        user_id="test",
        atomic_edges=atomic_edges,
        stitch_mode=True,
        allow_same_segment_merge=True,
        do_sanity_check=do_sanity_check,
    )
    elapsed = time.time() - t0
    new_roots = result.new_root_ids
    new_l2_ids = result.new_lvl2_ids
    print(f"  [current] stitch: {elapsed:.1f}s, {len(new_roots)} roots")

    profiler = profiler_mod._profiler
    perf = {}
    for path, times in profiler.timings.items():
        perf[path] = {
            "total_ms": sum(times) * 1000,
            "calls": profiler.call_counts[path],
            "avg_ms": (sum(times) / profiler.call_counts[path]) * 1000,
        }
    profiler_mod._profiler = HierarchicalProfiler(enabled=False)

    t0 = time.time()
    structure = extract_structure(cg, new_roots)
    print(f"  [current] structure: {time.time() - t0:.1f}s")

    return {
        "structure": structure,
        "new_roots": new_roots.tolist(),
        "new_l2_ids": [int(x) for x in new_l2_ids],
        "operation_id": int(result.operation_id) if result.operation_id else None,
        "elapsed": elapsed,
        "graph_id": graph_id,
        "n_edges": len(atomic_edges),
        "layer_counts": {layer: len(ccs) for layer, ccs in structure["components"].items()},
        "perf": perf,
    }
