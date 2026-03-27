"""
Stitch algorithm orchestrator. All state and logic on LocalChunkedGraph.
This module is a thin wrapper that calls lcg methods in sequence.
"""

import numpy as np

from pychunkedgraph.graph import basetypes

from .local_cg import LocalChunkedGraph
from .stitch_types import StitchResult
from .utils import timed


def stitch(lcg: LocalChunkedGraph, atomic_edges: np.ndarray) -> StitchResult:
    atomic_edges = np.asarray(atomic_edges, dtype=basetypes.NODE_ID)
    perf = {}

    with timed(perf, "phase1_total"):
        lcg.read_upfront(atomic_edges, perf)

    with timed(perf, "phase2_total"):
        new_l2_ids = lcg.merge_l2(perf)

    with timed(perf, "phase2b_total"):
        lcg.discover_siblings(perf)
        lcg.compute_dirty_siblings()
        perf["siblings_dirty"] = len(lcg._cache.dirty_siblings)

    with timed(perf, "phase3_total"):
        roots, layer_perf = lcg.build_hierarchy(perf)
        perf["per_layer"] = layer_perf

    with timed(perf, "phase4_total"):
        rows = lcg.build_rows()
    perf["rpc_log"] = lcg.rpc_log

    return StitchResult(
        new_roots=[int(r) for r in roots],
        new_l2_ids=[int(x) for x in new_l2_ids],
        new_ids_per_layer={int(l): len(ids) for l, ids in lcg.get_all_new_ids().items()},
        rows=rows,
        perf=perf,
    )
