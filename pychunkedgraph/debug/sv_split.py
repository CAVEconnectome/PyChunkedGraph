"""Debug utilities for supervoxel splitting."""

from functools import reduce

import numpy as np
import fastremap

from ..app.app_utils import handle_supervoxel_id_lookup
from ..graph import attributes
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
    node_idents = []
    node_ident_map = {"sources": 0, "sinks": 1}
    coords = []
    node_ids = []
    for k in ["sources", "sinks"]:
        for node in data[k]:
            node_ids.append(node[0])
            coords.append(np.array(node[1:]) / cg.segmentation_resolution)
            node_idents.append(node_ident_map[k])
    node_ids = np.array(node_ids, dtype=np.uint64)
    coords = np.array(coords)
    node_idents = np.array(node_idents)
    sv_ids = handle_supervoxel_id_lookup(cg, coords, node_ids)
    sources = sv_ids[node_idents == 0]
    sinks = sv_ids[node_idents == 1]
    src_coords = coords[node_idents == 0]
    snk_coords = coords[node_idents == 1]
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


def check_unsplit_sv_bridges(cg: ChunkedGraph, sv_remapping: dict, sources, sinks):
    """Check if unsplit SVs on opposite sides still have inf edges bridging them.

    After an SV split, SVs that kept their original IDs (chunk had only 1 label)
    may still have inf edges to SVs on the opposite side in raw chunk data.
    """
    # Find which SVs map to overlapping representatives
    source_set = set(int(s) for s in sources)
    sink_set = set(int(s) for s in sinks)
    all_svs = source_set | sink_set

    # Group by representative
    rep_groups = {}
    for sv, rep in sv_remapping.items():
        rep_groups.setdefault(rep, []).append(sv)

    # Find overlapping reps
    for rep, svs in rep_groups.items():
        src_in = [sv for sv in svs if sv in source_set]
        snk_in = [sv for sv in svs if sv in sink_set]
        if not src_in or not snk_in:
            continue

        print(f"\n=== Overlapping rep {rep} ===")
        print(f"  sources in group: {src_in}")
        print(f"  sinks in group: {snk_in}")

        # Check which SVs have NewIdentity (were split) vs kept original IDs
        all_group_svs = np.array(svs, dtype=np.uint64)
        new_id_cells = cg.client.read_nodes(
            node_ids=all_group_svs,
            properties=attributes.Hierarchy.NewIdentity,
        )
        for sv in svs:
            has_new = bool(new_id_cells.get(sv))
            side = "src" if sv in source_set else "sink" if sv in sink_set else "other"
            print(f"  SV {sv}: side={side}, was_split={has_new}")

        # Check inf edges between unsplit SVs on opposite sides
        unsplit = [sv for sv in svs if not new_id_cells.get(sv)]
        unsplit_src = [sv for sv in unsplit if sv in source_set]
        unsplit_snk = [sv for sv in unsplit if sv in sink_set]
        if unsplit_src and unsplit_snk:
            print(
                f"  WARNING: unsplit SVs on both sides: src={unsplit_src}, sink={unsplit_snk}"
            )
            print(f"  These have inf edges in raw data that were never updated")

        # Show full lineage for the group
        print()
        inspect_split_lineage(cg, svs)


def inspect_split_lineage(cg: ChunkedGraph, whole_sv_ids, old_new_map=None):
    """Inspect NewIdentity/FormerIdentity and old_new_map for a whole SV group.

    Shows which SVs were actually split (got new IDs), which kept their IDs,
    and whether NewIdentity was written correctly.
    """
    sv_arr = np.asarray(whole_sv_ids, dtype=np.uint64)
    print(f"=== Split Lineage for {len(sv_arr)} SVs ===")

    # Read NewIdentity for all SVs in the group
    new_id_cells = cg.client.read_nodes(
        node_ids=sv_arr,
        properties=attributes.Hierarchy.NewIdentity,
    )
    # Read FormerIdentity too
    former_id_cells = cg.client.read_nodes(
        node_ids=sv_arr,
        properties=attributes.Hierarchy.FormerIdentity,
    )

    in_old_new = set()
    if old_new_map:
        in_old_new = set(int(k) for k in old_new_map.keys())
        print(f"\nold_new_map keys: {sorted(in_old_new)}")
        for old, new in old_new_map.items():
            print(f"  {old} -> {new}")

    print(f"\nLineage status:")
    for sv in sv_arr:
        sv_int = int(sv)
        new_id = new_id_cells.get(sv)
        former_id = former_id_cells.get(sv)
        new_vals = [c.value for c in new_id] if new_id else None
        former_vals = [c.value for c in former_id] if former_id else None
        in_map = sv_int in in_old_new
        chunk = cg.get_chunk_coordinates(sv)

        status = []
        if new_vals:
            status.append(f"NewIdentity={new_vals}")
        if former_vals:
            status.append(f"FormerIdentity={former_vals}")
        if in_map:
            status.append("in old_new_map")
        if not status:
            status.append("UNCHANGED (no lineage, not in old_new_map)")

        print(f"  {sv} chunk={chunk}: {', '.join(status)}")

    # Check for SVs that were split (have new fragments) but missing NewIdentity
    if old_new_map:
        missing = [k for k in old_new_map if not new_id_cells.get(np.uint64(k))]
        if missing:
            print(f"\n  WARNING: SVs in old_new_map but missing NewIdentity: {missing}")


def trace_stale_sv(cg: ChunkedGraph, sv_id, bbox=None, root_id=None):
    """Trace why a stale SV still appears in edges after a split.

    Checks: parent, NewIdentity, L2 children membership,
    and where edges referencing it come from.
    """
    sv_id = np.uint64(sv_id)
    print(f"=== Tracing SV {sv_id} ===")

    # Parent
    parents = cg.get_parents([sv_id])
    parent = list(parents.values())[0] if parents else None
    print(f"  parent: {parent}")

    # NewIdentity (set on old SVs after split)
    cells = cg.client.read_nodes(
        node_ids=[sv_id], properties=attributes.Hierarchy.NewIdentity
    )
    if cells.get(sv_id):
        new_ids = [c.value for c in cells[sv_id]]
        print(f"  NewIdentity: {new_ids} (SV was replaced)")
    else:
        print(f"  NewIdentity: not set (SV was NOT replaced)")

    # Is it still in its L2 parent's children?
    if parent is not None:
        children = cg.get_children(parent)
        in_children = sv_id in children
        print(f"  in L2 {parent} children: {in_children}")
        if in_children:
            print(f"    children: {children}")

    # Root
    root = cg.get_root(sv_id)
    print(f"  root: {root}")

    # Check edges in subgraph if bbox provided
    if bbox is not None and root_id is not None:
        print(f"\n  --- Edges referencing {sv_id} in subgraph ---")
        _, edges_tuple = cg.get_subgraph(root_id, bbox, bbox_is_coordinate=True)
        edges_all = reduce(lambda x, y: x + y, edges_tuple, Edges([], []))
        pairs = edges_all.get_pairs()
        affs = edges_all.affinities
        mask = np.any(pairs == sv_id, axis=1)
        print(f"  total edges with this SV: {mask.sum()}")
        if mask.any():
            for p, a in zip(pairs[mask][:10], affs[mask][:10]):
                aff_str = "inf" if np.isinf(a) else f"{a:.4f}"
                print(f"    {p[0]} -- {p[1]}  aff={aff_str}")

        # Check origin: chunk edges vs edit edges
        l2ids = list(
            cg.get_subgraph(
                root_id,
                bbox,
                bbox_is_coordinate=True,
                nodes_only=True,
                return_flattened=True,
            ).values()
        )[0]
        chunk_ids = np.unique(cg.get_chunk_ids_from_node_ids(l2ids))

        from ..io.edges import get_chunk_edges

        chunk_edges_d = cg.read_chunk_edges(chunk_ids)
        chunk_edges_all = reduce(
            lambda x, y: x + y, chunk_edges_d.values(), Edges([], [])
        )
        chunk_pairs = chunk_edges_all.get_pairs()
        chunk_mask = np.any(chunk_pairs == sv_id, axis=1)
        print(f"  from chunk edges (cloud storage): {chunk_mask.sum()}")

        edit_edges_d = cg.get_edges_from_edits(chunk_ids)
        edit_edges_all = reduce(
            lambda x, y: x + y, edit_edges_d.values(), Edges([], [])
        )
        edit_pairs = edit_edges_all.get_pairs()
        edit_mask = np.any(edit_pairs == sv_id, axis=1)
        print(f"  from edit edges (SplitEdges): {edit_mask.sum()}")
        if edit_mask.any():
            edit_affs = edit_edges_all.affinities
            for p, a in zip(edit_pairs[edit_mask][:10], edit_affs[edit_mask][:10]):
                aff_str = "inf" if np.isinf(a) else f"{a:.4f}"
                print(f"    {p[0]} -- {p[1]}  aff={aff_str}")
