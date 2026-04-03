"""
Stitch algorithm orchestrator. All state and logic on LocalChunkedGraph.
This module is a thin wrapper that calls lcg methods in sequence.
"""

from dataclasses import dataclass, field

import numpy as np

from pychunkedgraph.graph import basetypes

from .local_cg import LocalChunkedGraph
from .utils import timed


@dataclass
class StitchResult:
    new_roots: list
    new_l2_ids: list
    new_ids_per_layer: dict
    rows: dict
    perf: dict = field(default_factory=dict)


def stitch(
    lcg: LocalChunkedGraph, atomic_edges: np.ndarray,
) -> StitchResult:
    atomic_edges = np.asarray(atomic_edges, dtype=basetypes.NODE_ID)
    perf = {}

    with timed(perf, "read_upfront"):
        lcg.read_upfront(atomic_edges, perf)

    with timed(perf, "merge_l2"):
        new_l2_ids = lcg.merge_l2(perf)

    with timed(perf, "discover_siblings"):
        lcg.discover_siblings(perf)

    with timed(perf, "compute_dirty"):
        lcg.compute_dirty_siblings()
    perf["n_dirty_siblings"] = len(lcg._cache.dirty_siblings)

    with timed(perf, "build_hierarchy"):
        roots, layer_perf = lcg.build_hierarchy(perf)
    perf["per_layer"] = layer_perf

    lcg.sanity_check(roots, atomic_edges)

    with timed(perf, "build_rows"):
        rows = lcg.build_rows()
    perf["rpc_log"] = lcg.rpc_log

    return StitchResult(
        new_roots=[int(r) for r in roots],
        new_l2_ids=[int(x) for x in new_l2_ids],
        new_ids_per_layer={int(l): len(ids) for l, ids in lcg.get_all_new_ids().items()},
        rows=rows,
        perf=perf,
    )
