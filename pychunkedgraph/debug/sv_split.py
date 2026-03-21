"""Debug utilities for supervoxel splitting."""

from functools import reduce

import numpy as np
import fastremap

from ..app.segmentation.common import _get_sources_and_sinks as get_sources_and_sinks
from ..graph.chunkedgraph import ChunkedGraph
from ..graph.edges import Edges


def get_subgraph_edges(cg: ChunkedGraph, root_id, bbox):
    """Fetch subgraph edges and return deduplicated (pairs, affinities, areas)."""
    _, edges_tuple = cg.get_subgraph(root_id, bbox=bbox, bbox_is_coordinate=True)
    edges_all = reduce(lambda x, y: x + y, edges_tuple, Edges([], []))
    pairs = edges_all.get_pairs()
    affs = edges_all.affinities
    areas = edges_all.areas

    pairs_sorted = np.sort(pairs, axis=1)
    _, idx = np.unique(pairs_sorted, axis=0, return_index=True)
    idx = idx[pairs_sorted[idx, 0] != pairs_sorted[idx, 1]]
    return pairs[idx], affs[idx], areas[idx]


def compute_bbox(source_coords, sink_coords, padding=240):
    """Compute bounding box from source/sink coordinates with padding."""
    all_coords = np.concatenate([source_coords, sink_coords])
    return np.array(
        [np.min(all_coords, axis=0) - padding, np.max(all_coords, axis=0) + padding],
        dtype=int,
    )


def inspect_sv_edges(cg: ChunkedGraph, svs, bbox):
    """Show edge counts and inf-affinity edges for given SVs within a bbox."""
    root = cg.get_root(svs[0])
    pairs, affs, _ = get_subgraph_edges(cg, root, bbox)

    print(f"root: {root}")
    print(f"total edges: {len(pairs)}")
    print(f"inf-affinity edges: {np.sum(np.isinf(affs))}")
    print()

    for sv in svs:
        m = np.any(pairs == sv, axis=1)
        if m.any():
            inf_m = np.isinf(affs[m])
            print(f"  SV {sv}: {m.sum()} edges, {inf_m.sum()} inf-aff")
        else:
            print(f"  SV {sv}: no edges in subgraph")

    return pairs, affs


def find_inf_bridges(pairs, affs, source_svs, sink_svs):
    """Find partners connected by inf edges to both source and sink sides."""
    source_set = set(np.asarray(source_svs, dtype=np.uint64).tolist())
    sink_set = set(np.asarray(sink_svs, dtype=np.uint64).tolist())
    all_split = source_set | sink_set

    inf_mask = np.isinf(affs)
    inf_pairs = pairs[inf_mask]
    if len(inf_pairs) == 0:
        print("no inf-affinity edges")
        return {}

    partner_sides = {}
    for a, b in inf_pairs:
        a, b = int(a), int(b)
        for sv, other in [(a, b), (b, a)]:
            if other in all_split:
                continue
            if sv in source_set:
                partner_sides.setdefault(other, set()).add("src")
            elif sv in sink_set:
                partner_sides.setdefault(other, set()).add("sink")

    bridges = {k: v for k, v in partner_sides.items() if len(v) > 1}
    if bridges:
        print(f"inf-aff bridge partners (connected to both sides): {len(bridges)}")
        for p, sides in bridges.items():
            print(f"  {p}: {sides}")
    else:
        print("no inf-aff bridge partners")
    return bridges


def check_l2_children(cg: ChunkedGraph, data: dict, svs):
    """Check L2 children for stale/new SVs after a split."""
    original_svs = set(int(node[0]) for k in ["sources", "sinks"] for node in data[k])
    post_svs = set(int(s) for s in svs)
    stale = original_svs - post_svs
    new = post_svs - original_svs

    l2ids = cg.get_parents(np.asarray(list(post_svs), dtype=np.uint64))
    l2_children = cg.get_children(np.unique(l2ids))

    issues = []
    for l2id, children in l2_children.items():
        children_set = set(int(c) for c in children)
        stale_found = stale & children_set
        new_found = new & children_set
        if stale_found:
            print(f"  L2 {l2id}: STALE old SVs: {stale_found}")
            issues.append(("stale", l2id, stale_found))
        if new_found:
            print(f"  L2 {l2id}: new SVs present: {new_found}")

    if not issues:
        print("  no stale SVs found in L2 children")
    return stale, new, issues


def inspect_edited_edges(cg: ChunkedGraph, svs):
    """Show edges from edits for L2 chunks containing the given SVs."""
    l2ids = cg.get_parents(np.asarray(svs, dtype=np.uint64))
    l2chunks = cg.get_chunk_ids_from_node_ids(np.unique(l2ids))
    fedges = cg.get_edges_from_edits(l2chunks)
    for k, v in fedges.items():
        unique_pairs = np.unique(v.get_pairs(), axis=0)
        print(f"  chunk {k}: {unique_pairs.shape[0]} unique edge pairs")
    return fedges


def inspect_split(cg: ChunkedGraph, data: dict):
    """Full diagnostic for a split request: edges, inf bridges, L2 state."""
    sources, sinks, src_coords, snk_coords = get_sources_and_sinks(cg, data)
    all_svs = np.concatenate([sources, sinks])
    bbox = compute_bbox(src_coords, snk_coords)

    print("=== Sources & Sinks ===")
    print(f"  sources: {sources}")
    print(f"  sinks:   {sinks}")
    print()

    print("=== Edited Edges ===")
    inspect_edited_edges(cg, all_svs)
    print()

    print("=== Subgraph Edges ===")
    pairs, affs = inspect_sv_edges(cg, all_svs, bbox)
    print()

    print("=== Inf-aff Bridges ===")
    find_inf_bridges(pairs, affs, sources, sinks)
    print()

    print("=== L2 Children ===")
    check_l2_children(cg, data, all_svs)
