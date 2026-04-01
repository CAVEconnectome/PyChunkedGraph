"""
Stitch algorithm orchestrator. All state and logic on LocalChunkedGraph.
This module is a thin wrapper that calls lcg methods in sequence.
"""

from collections import defaultdict
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
    lcg: LocalChunkedGraph, atomic_edges: np.ndarray, sanity_check: bool = False,
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

    if sanity_check:
        _sanity_check(lcg, roots, atomic_edges)

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


def _sanity_check(
    lcg: LocalChunkedGraph, roots: np.ndarray, atomic_edges: np.ndarray,
) -> None:
    """Verify: no duplicate L2s, all edge SVs under new roots."""
    c = lcg._cache
    all_l2s = []
    root_l2_map = {}
    for root in roots:
        stack = [int(root)]
        root_l2s = []
        while stack:
            nid = stack.pop()
            layer = lcg.get_chunk_layer(np.uint64(nid))
            if layer <= 2:
                root_l2s.append(nid)
                continue
            ch = c.get_children_batch(np.array([nid], dtype=basetypes.NODE_ID))
            for child in ch.get(nid, []):
                stack.append(int(child))
        n_unique = len(set(root_l2s))
        assert n_unique == len(root_l2s), (
            f"Root {root}: {len(root_l2s)} L2 children but only {n_unique} unique"
        )
        all_l2s.extend(root_l2s)
        root_l2_map[int(root)] = set(root_l2s)
    l2_counts = defaultdict(int)
    for l2 in all_l2s:
        l2_counts[l2] += 1
    dupes = {l2: cnt for l2, cnt in l2_counts.items() if cnt > 1}
    if dupes:
        dupe_roots = {}
        for l2 in dupes:
            dupe_roots[l2] = [r for r, l2s in root_l2_map.items() if l2 in l2s]
        assert False, (
            f"Duplicate L2s across roots: {dupes}. "
            f"L2→roots: {dupe_roots}"
        )

    # Verify all edge SVs are under new roots
    edge_svs = np.unique(atomic_edges.ravel())
    sv_parents = c.get_parents(edge_svs)
    edge_l2s = set(int(p) for p in sv_parents if int(p) != 0)
    all_l2_set = set(all_l2s)
    missing = edge_l2s - all_l2_set
    assert not missing, (
        f"{len(missing)} edge L2s not under any new root. "
        f"First 5: {sorted(missing)[:5]}"
    )
