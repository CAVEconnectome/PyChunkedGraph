"""
Redesigned stitch algorithm.

Reads AtomicCrossChunkEdge (immutable) and full partner parent chains
upfront in batch, then builds the entire parent hierarchy from cache.

No per-layer BigTable reads. No stale resolution. No sibling lookups.
No _init_old_hierarchy. No _update_neighbor_cx_edges.
Works correctly across any number of accumulated prior stitches.
"""

import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes, serializers, types
from pychunkedgraph.graph import cache as cache_utils
from pychunkedgraph.graph.utils import flatgraph
from pychunkedgraph.graph.utils.generic import filter_failed_node_ids
from pychunkedgraph.graph.edges.utils import get_cross_chunk_edges_layer
from pychunkedgraph.graph.chunks.hierarchy import get_parent_chunk_id

from pychunkedgraph.graph.cache import CacheService

from .utils import extract_structure, batch_get_l2children


def run_proposed_stitch(graph_id: str, atomic_edges: np.ndarray) -> dict:

    atomic_edges = np.asarray(atomic_edges, dtype=basetypes.NODE_ID)
    cg = ChunkedGraph(graph_id=graph_id)
    cg.cache = CacheService(cg)

    print(f"  [proposed] stitch ({len(atomic_edges)} edges)...")
    t0 = time.time()
    result = stitch(cg, atomic_edges, verbose=False)
    elapsed = time.time() - t0
    new_roots = result["new_roots"]
    print(f"  [proposed] stitch: {elapsed:.1f}s, {len(new_roots)} roots")

    t_write = time.time()
    cg.client.write(result["node_entries"])
    cg.client.write(result["parent_entries"])
    result["perf"]["write_entries"] = time.time() - t_write
    print(f"  [proposed] write: {result['perf']['write_entries']:.1f}s")

    t_struct = time.time()
    structure = extract_structure(cg, np.array(new_roots, dtype=basetypes.NODE_ID))
    print(f"  [proposed] structure: {time.time() - t_struct:.1f}s")

    return {
        "structure": structure,
        "new_roots": [int(x) for x in new_roots],
        "new_l2_ids": [int(x) for x in result["new_l2_ids"]],
        "new_ids_per_layer": result.get("new_ids_per_layer", {}),
        "elapsed": elapsed,
        "graph_id": graph_id,
        "n_edges": len(atomic_edges),
        "n_entries_written": len(result["entries"]),
        "layer_counts": {
            layer: len(ccs) for layer, ccs in structure["components"].items()
        },
        "perf": result["perf"],
    }


# ─────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────


def stitch(cg: ChunkedGraph, atomic_edges: np.ndarray, verbose: bool = True) -> dict:
    """
    Stitch algorithm: add cross-boundary edges and build hierarchy.

    1. Batch-read all needed data from BigTable (4 reads)
    2. Merge L2 nodes in-memory
    3. Build parent hierarchy layer-by-layer from cache
    4. Return mutations to write
    """
    perf = {}
    log = print if verbose else lambda *a, **k: None

    # ── Phase 1: batch BigTable reads ────────────────────────────────
    t0 = time.time()
    ctx = _read_upfront(cg, atomic_edges, perf, log)
    perf["phase1_total"] = time.time() - t0
    log(f"    [stitch] phase 1 (reads): {perf['phase1_total']:.1f}s")

    # ── Phase 2: L2 merge ────────────────────────────────────────────
    t0 = time.time()
    new_l2_ids = _merge_l2(cg, ctx, perf)
    perf["phase2_total"] = time.time() - t0
    log(f"    [stitch] phase 2 (L2 merge): {perf['phase2_total']:.1f}s, {len(new_l2_ids)} L2 nodes")

    # ── Phase 2b: discover siblings and warm cache ─────────────────
    t0 = time.time()
    _discover_siblings(cg, ctx, perf, log)
    perf["phase2b_siblings"] = time.time() - t0
    log(f"    [stitch] phase 2b (siblings): {perf['phase2b_siblings']:.1f}s")

    # ── Phase 3: build parent hierarchy ──────────────────────────────
    t0 = time.time()
    new_roots, layer_perf = _build_hierarchy(cg, new_l2_ids, ctx, perf, log)
    perf["phase3_total"] = time.time() - t0
    perf["per_layer"] = layer_perf
    log(f"    [stitch] phase 3 (hierarchy): {perf['phase3_total']:.1f}s, {len(new_roots)} roots")

    # ── Phase 3b: resolve cross edges for writing ──────────────────
    t0 = time.time()
    _resolve_cx_for_write(cg, ctx)
    perf["phase3b_resolve_write"] = time.time() - t0

    # ── Phase 4: build mutations ─────────────────────────────────────
    t0 = time.time()
    node_entries, parent_entries = _build_entries(cg, ctx)
    perf["phase4_total"] = time.time() - t0

    new_ids_per_layer = {
        layer: len(ids) for layer, ids in ctx["new_ids_d"].items() if ids
    }

    return {
        "new_roots": [int(r) for r in new_roots],
        "new_l2_ids": [int(x) for x in new_l2_ids],
        "new_ids_per_layer": new_ids_per_layer,
        "node_entries": node_entries,
        "parent_entries": parent_entries,
        "perf": perf,
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 1: upfront reads
# ─────────────────────────────────────────────────────────────────────


def _get_all_parents_filtered(cg, node_ids, time_stamp=None):
    """
    Like get_all_parents_dict_multiple but applies filter_failed_node_ids
    at every layer to discard orphaned nodes from failed stitch attempts.
    Orphaned parents are remapped to their valid counterparts.
    """
    result = {int(node): {} for node in node_ids}
    nodes = np.array(node_ids, dtype=basetypes.NODE_ID)
    child_parent_map = {}
    layers_map = {}

    while nodes.size > 0:
        parents = cg.get_parents(nodes, time_stamp=time_stamp)
        parent_layers = cg.get_chunk_layers(parents)

        # filter orphaned parents at this level
        remap = {}
        unique_parents = np.unique(parents)
        if len(unique_parents) > 0:
            children_d = cg.get_children(unique_parents)
            max_children_ids = [
                int(np.max(children_d[p])) if len(children_d.get(p, [])) > 0 else 0
                for p in unique_parents
            ]
            segment_ids = np.array([cg.get_segment_id(p) for p in unique_parents])
            valid = set(
                int(x) for x in filter_failed_node_ids(unique_parents, segment_ids, max_children_ids)
            )

            # map orphaned → valid counterpart (same max_children_id)
            mcid_to_valid = {}
            for p, mcid in zip(unique_parents, max_children_ids):
                if int(p) in valid:
                    mcid_to_valid[mcid] = int(p)

            for p, mcid in zip(unique_parents, max_children_ids):
                if int(p) not in valid:
                    remap[int(p)] = mcid_to_valid.get(mcid, int(p))

        for node, parent, layer in zip(nodes, parents, parent_layers):
            resolved_parent = remap.get(int(parent), int(parent))
            layers_map[resolved_parent] = int(layer)
            child_parent_map[int(node)] = resolved_parent

        # continue with valid parents only
        next_nodes = []
        for parent, layer in zip(parents, parent_layers):
            resolved = remap.get(int(parent), int(parent))
            if int(layer) < cg.meta.layer_count:
                next_nodes.append(resolved)
        nodes = np.unique(np.array(next_nodes, dtype=basetypes.NODE_ID)) if next_nodes else np.array([], dtype=basetypes.NODE_ID)

    for node in node_ids:
        current = int(node)
        node_result = {}
        while current in child_parent_map:
            parent = child_parent_map[current]
            parent_layer = layers_map[parent]
            node_result[parent_layer] = parent
            current = parent
        result[int(node)] = node_result
    return result


def _read_upfront(cg, atomic_edges, perf, log=print):
    """
    All BigTable reads happen here. Returns a context dict
    with everything needed for in-memory processing.
    """
    # 1. Classify edges by layer, get L2 parents of all SVs
    t0 = time.time()
    svs = np.unique(atomic_edges)
    sv_parents = cg.get_parents(svs)
    sv_to_l2 = dict(zip(svs.tolist(), sv_parents.tolist()))
    edge_layers = get_cross_chunk_edges_layer(cg.meta, atomic_edges)
    perf["read_sv_parents"] = time.time() - t0
    log(f"    [stitch]   sv parents: {perf['read_sv_parents']:.1f}s, {len(svs)} SVs")

    # Build L2-level edges and per-node cross edge dicts
    l2_edges = []  # within-chunk L2 edges
    l2_cx_edges = defaultdict(lambda: defaultdict(list))  # {l2: {layer: [[sv, sv], ...]}}

    for i, edge in enumerate(atomic_edges):
        layer = edge_layers[i]
        sv0, sv1 = int(edge[0]), int(edge[1])
        p0 = sv_to_l2[sv0]
        p1 = sv_to_l2[sv1]
        if layer == 1:
            l2_edges.append([p0, p1])
        else:
            # store as [sv, sv] — same format as AtomicCrossChunkEdge
            l2_cx_edges[p0][layer].append([sv0, sv1])
            l2_cx_edges[p1][layer].append([sv1, sv0])
            # self-edges so isolated cross-edge nodes form components
            l2_edges.append([p0, p0])
            l2_edges.append([p1, p1])

    l2ids = np.unique(np.array(l2_edges, dtype=basetypes.NODE_ID)) if l2_edges else np.array([], dtype=basetypes.NODE_ID)

    # 2+3. Read children and AtomicCrossChunkEdge in parallel (independent reads)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_children = executor.submit(cg.get_children, l2ids)
        fut_acx = executor.submit(cg.get_atomic_cross_edges, l2ids)
        children_d = fut_children.result()
        atomic_cx = fut_acx.result()
    perf["read_children_and_acx"] = time.time() - t0
    log(f"    [stitch]   children+acx: {perf['read_children_and_acx']:.1f}s")

    # filter out orphaned L2 nodes from prior failed stitch attempts
    if len(l2ids) > 0:
        max_children_ids = [
            int(np.max(children_d[l2id])) if len(children_d.get(l2id, [])) > 0 else 0
            for l2id in l2ids
        ]
        segment_ids = np.array([cg.get_segment_id(l2id) for l2id in l2ids])
        l2ids = filter_failed_node_ids(l2ids, segment_ids, max_children_ids)

    # 4. Collect ALL partner SVs from atomic cross edges + stitch edges
    #    and resolve their full parent chains in one batch
    t0 = time.time()
    partner_svs = set()
    for l2id, layer_d in atomic_cx.items():
        for layer, edges in layer_d.items():
            if len(edges) > 0:
                partner_svs.update(edges[:, 1].tolist())
    for l2id, layer_d in l2_cx_edges.items():
        for layer, edge_list in layer_d.items():
            for e in edge_list:
                partner_svs.add(e[1])

    # remove SVs we already know (our own L2 nodes' children)
    known_svs = set()
    for l2id in l2ids:
        for sv in children_d.get(l2id, []):
            known_svs.add(int(sv))

    unknown_partner_svs = np.array(
        list(partner_svs - known_svs), dtype=basetypes.NODE_ID
    )

    # batch read full parent chains for unknown partner SVs
    # AND start siblings old hierarchy read concurrently (only needs l2ids)
    t_chains = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_partner = executor.submit(
            _get_all_parents_filtered, cg, unknown_partner_svs
        ) if len(unknown_partner_svs) > 0 else None
        fut_old_hier = executor.submit(_get_all_parents_filtered, cg, l2ids)

        partner_chains = fut_partner.result() if fut_partner else {}
        old_hierarchy = fut_old_hier.result()
    perf["read_partner_chains"] = time.time() - t_chains
    log(f"    [stitch]   partner chains + old hierarchy: {perf['read_partner_chains']:.1f}s, {len(unknown_partner_svs)} partner SVs")

    # Build a unified resolver: given any SV, get its identity at any layer
    # Structure: {sv: {layer: identity_at_that_layer}}
    resolver = {}

    # our own SVs: chain starts at their L2 parent
    # (higher layers will be resolved via old_to_new + node_cx during propagation)
    for sv_int, l2_int in sv_to_l2.items():
        resolver[sv_int] = {2: l2_int}

    # partner SVs: full chains from get_all_parents_dict_multiple
    for sv, chain in partner_chains.items():
        resolver[int(sv)] = {int(l): int(p) for l, p in chain.items()}

    return {
        "l2_edges": np.array(l2_edges, dtype=basetypes.NODE_ID) if l2_edges else types.empty_2d,
        "l2ids": l2ids,
        "l2_cx_edges": l2_cx_edges,
        "children_d": children_d,
        "atomic_cx": atomic_cx,
        "resolver": resolver,
        "sv_to_l2": sv_to_l2,
        "old_hierarchy": old_hierarchy,
        "new_ids_d": defaultdict(list),
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 2: L2 merge
# ─────────────────────────────────────────────────────────────────────


def _batch_create_node_ids(cg, size_map, root_chunks=None):
    """Allocate node IDs for multiple chunks in parallel threads."""
    if root_chunks is None:
        root_chunks = set()

    def _alloc(chunk_id):
        return chunk_id, list(
            cg.id_client.create_node_ids(
                np.uint64(chunk_id),
                size=size_map[chunk_id],
                root_chunk=chunk_id in root_chunks,
            )
        )

    result = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_alloc, c): c for c in size_map}
        for fut in as_completed(futures):
            chunk_id, ids = fut.result()
            result[chunk_id] = ids
    return result


def _merge_l2(cg, ctx, _perf):
    """Merge L2 nodes based on within-chunk edge connectivity."""
    l2_edges = ctx["l2_edges"]
    children_d = ctx["children_d"]
    atomic_cx = ctx["atomic_cx"]
    l2_cx_edges = ctx["l2_cx_edges"]

    if l2_edges.size == 0:
        return []

    graph, _, _, graph_ids = flatgraph.build_gt_graph(l2_edges, make_directed=True)
    components = flatgraph.connected_components(graph)

    # allocate new L2 IDs
    chunk_count_map = defaultdict(int)
    for cc_indices in components:
        chunk_count_map[cg.get_chunk_id(graph_ids[cc_indices][0])] += 1

    chunk_ids = list(chunk_count_map.keys())
    random.shuffle(chunk_ids)
    chunk_new_ids_map = _batch_create_node_ids(
        cg, {c: chunk_count_map[c] for c in chunk_ids}
    )

    new_l2_ids = []
    old_to_new = {}
    node_cx = {}       # {node_id: {layer: [node_id, partner_sv]}} — for hierarchy building
    l2_atomic_cx = {}  # {new_l2: {layer: [sv, sv]}} — raw format for BigTable write

    for cc_indices in components:
        old_ids = graph_ids[cc_indices]
        new_id = chunk_new_ids_map[cg.get_chunk_id(old_ids[0])].pop()
        new_l2_ids.append(new_id)

        for old_id in old_ids:
            old_to_new[int(old_id)] = int(new_id)

        # merge SV children
        merged_children = np.concatenate(
            [children_d[l2id] for l2id in old_ids]
        ).astype(basetypes.NODE_ID)
        cg.cache.children_cache[new_id] = merged_children
        cache_utils.update(cg.cache.parents_cache, merged_children, new_id)

        # merge atomic cross edges from all old L2 nodes
        # raw format: [sv, sv] on both sides
        merged = defaultdict(list)
        for old_l2 in old_ids:
            for layer, acx in atomic_cx.get(old_l2, {}).items():
                if len(acx) > 0:
                    merged[layer].append(acx)
            # also include stitch cross edges (also [sv, sv] format)
            for layer, edge_list in l2_cx_edges.get(int(old_l2), {}).items():
                if edge_list:
                    merged[layer].append(
                        np.array(edge_list, dtype=basetypes.NODE_ID)
                    )

        # store raw [sv, sv] for AtomicCrossChunkEdge write
        raw_acx = {}
        for layer, arr_list in merged.items():
            raw_acx[layer] = np.unique(
                np.concatenate(arr_list).astype(basetypes.NODE_ID), axis=0
            )
        l2_atomic_cx[int(new_id)] = raw_acx

        # for hierarchy building: set col 0 to new L2 ID, col 1 stays as partner SV
        node_cx_d = {}
        for layer, edges in raw_acx.items():
            working = edges.copy()
            working[:, 0] = new_id
            node_cx_d[layer] = working
        node_cx[int(new_id)] = node_cx_d

    # add ALL SVs of new L2 nodes to resolver
    # (Phase 1 only added stitch SVs; merged L2s have additional SVs from old L2 children)
    resolver = ctx["resolver"]
    for new_id in new_l2_ids:
        new_id_int = int(new_id)
        for sv in cg.cache.children_cache.get(new_id, []):
            sv_int = int(sv)
            if sv_int not in resolver:
                resolver[sv_int] = {2: new_id_int}

    ctx["old_to_new"] = old_to_new
    ctx["node_cx"] = node_cx
    ctx["l2_atomic_cx"] = l2_atomic_cx
    ctx["new_ids_d"][2] = list(new_l2_ids)
    return new_l2_ids


# ─────────────────────────────────────────────────────────────────────
# Phase 2b: discover siblings
# ─────────────────────────────────────────────────────────────────────


def _discover_siblings(cg, ctx, perf, log=print):
    """
    Find ALL L2 nodes in the affected subtree.

    The old parents of our affected L2 nodes have other L2 descendants
    (siblings). These must participate in hierarchy rebuild because the
    old parents are being replaced.

    We get L2 descendants (not direct children) of old parents, then read
    their AtomicCrossChunkEdge. This is the complete source of truth for
    connectivity at every layer — no CrossChunkEdge read ever needed.
    """
    l2ids = ctx["l2ids"]
    node_cx = ctx["node_cx"]
    resolver = ctx["resolver"]
    atomic_cx = ctx["atomic_cx"]
    children_d = ctx["children_d"]

    # 1. use pre-fetched old hierarchy from phase 1
    old_hierarchy = ctx["old_hierarchy"]

    # collect all old parent IDs (at every layer)
    all_old_parents = set()
    for l2id in l2ids:
        for layer, parent in old_hierarchy.get(l2id, {}).items():
            all_old_parents.add(int(parent))

    # 2. get ALL L2 descendants of old parents (batch, level by level)
    t0 = time.time()
    old_parents_arr = np.array(list(all_old_parents), dtype=basetypes.NODE_ID)
    all_l2_in_subtree = set()
    if len(old_parents_arr) > 0:
        parent_l2_map = batch_get_l2children(cg, old_parents_arr)
        for l2set in parent_l2_map.values():
            all_l2_in_subtree.update(l2set)

    known_l2 = set(int(x) for x in l2ids)
    l2_siblings = np.array(
        list(all_l2_in_subtree - known_l2), dtype=basetypes.NODE_ID
    )
    perf["siblings_get_l2"] = time.time() - t0
    log(f"    [stitch]   L2 siblings: {perf['siblings_get_l2']:.1f}s, {len(l2_siblings)} from {len(all_old_parents)} parents")

    if len(l2_siblings) == 0:
        ctx["siblings_d"] = defaultdict(list)
        return

    # 3. read AtomicCrossChunkEdge + SV children for L2 siblings (parallel)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_acx = executor.submit(cg.get_atomic_cross_edges, l2_siblings)
        fut_children = executor.submit(cg.get_children, l2_siblings)
        sibling_acx = fut_acx.result()
        sibling_children = fut_children.result()

    # filter out orphaned L2 nodes from prior failed stitch attempts
    max_children_ids = [
        int(np.max(sibling_children[l2id])) if len(sibling_children.get(l2id, [])) > 0 else 0
        for l2id in l2_siblings
    ]
    segment_ids = np.array([cg.get_segment_id(l2id) for l2id in l2_siblings])
    l2_siblings = filter_failed_node_ids(l2_siblings, segment_ids, max_children_ids)

    perf["siblings_reads"] = time.time() - t0
    log(f"    [stitch]   siblings reads: {perf['siblings_reads']:.1f}s")

    # 4. resolve partner SVs from siblings' atomic cross edges
    t0 = time.time()
    partner_svs = set()
    for l2id, layer_d in sibling_acx.items():
        for layer, edges in layer_d.items():
            if len(edges) > 0:
                partner_svs.update(edges[:, 0].tolist())
                partner_svs.update(edges[:, 1].tolist())

    known_svs = set(resolver.keys())
    # include ALL SVs from our affected L2 nodes (their new L2 parents
    # don't have parents yet — Phase 3 hasn't run)
    for l2id in l2ids:
        for sv in children_d.get(l2id, []):
            known_svs.add(int(sv))
    for l2id in l2_siblings:
        for sv in sibling_children.get(l2id, []):
            known_svs.add(int(sv))

    unknown_svs = np.array(
        list(partner_svs - known_svs), dtype=basetypes.NODE_ID
    )
    if len(unknown_svs) > 0:
        chains = _get_all_parents_filtered(cg, unknown_svs)
        for sv, chain in chains.items():
            resolver[int(sv)] = {int(l): int(p) for l, p in chain.items()}
    perf["siblings_partner_chains"] = time.time() - t0
    log(f"    [stitch]   siblings partner chains: {perf['siblings_partner_chains']:.1f}s, {len(unknown_svs)} SVs")

    # 5. populate ctx for each L2 sibling
    l2_atomic_cx = ctx.get("l2_atomic_cx", {})
    for l2id in l2_siblings:
        l2id_int = int(l2id)
        children_d[l2id] = sibling_children.get(l2id, np.array([], dtype=basetypes.NODE_ID))
        atomic_cx[l2id] = sibling_acx.get(l2id, {})
        l2_atomic_cx[l2id_int] = sibling_acx.get(l2id, {})

        for sv in sibling_children.get(l2id, []):
            resolver[int(sv)] = {2: l2id_int}

        node_cx_d = {}
        for layer, edges in sibling_acx.get(l2id, {}).items():
            if len(edges) > 0:
                working = edges.copy()
                working[:, 0] = l2id
                node_cx_d[layer] = working
        node_cx[l2id_int] = node_cx_d
    ctx["l2_atomic_cx"] = l2_atomic_cx

    # siblings participate at L2 in hierarchy building
    siblings_d = defaultdict(list)
    siblings_d[2] = [int(x) for x in l2_siblings]
    ctx["siblings_d"] = siblings_d


# ─────────────────────────────────────────────────────────────────────
# Phase 3: build parent hierarchy
# ─────────────────────────────────────────────────────────────────────


def _build_hierarchy(cg, _new_l2_ids, ctx, _perf, log=print):
    """Build parent hierarchy layer by layer from cached data."""
    node_cx = ctx["node_cx"]
    resolver = ctx["resolver"]
    old_to_new = ctx["old_to_new"]
    new_ids_d = ctx["new_ids_d"]
    siblings_d = ctx.get("siblings_d", defaultdict(list))
    # tracks child → parent as we build, so partner resolution works for our own nodes
    child_to_parent = {}
    layer_perf = {}

    for layer in range(2, cg.meta.layer_count):
        new_nodes = new_ids_d[layer]
        sibling_nodes = siblings_d.get(layer, [])
        all_nodes = list(new_nodes) + list(sibling_nodes)
        if not all_nodes:
            continue
        cg.cache.new_ids.update(all_nodes)
        lp = {"n_new_nodes": len(new_nodes), "n_siblings": len(sibling_nodes)}
        log(f"    [stitch] layer {layer}: {len(new_nodes)} new + {len(sibling_nodes)} siblings = {len(all_nodes)} nodes")

        # resolve cross edges at this layer
        t0 = time.time()
        cx_edges = _resolve_cx_at_layer(
            all_nodes, layer, node_cx, resolver, old_to_new, child_to_parent,
            get_layer=lambda nid: cg.get_chunk_layer(np.uint64(nid)),
        )
        lp["resolve_cx"] = time.time() - t0
        lp["n_cross_edges"] = len(cx_edges)

        # connected components
        t0 = time.time()
        nodes_arr = np.array(all_nodes, dtype=basetypes.NODE_ID)
        self_edges = np.vstack([nodes_arr, nodes_arr]).T
        all_edges = np.concatenate([cx_edges, self_edges]).astype(basetypes.NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
        ccs = flatgraph.connected_components(graph)
        lp["cc"] = time.time() - t0
        lp["n_components"] = len(ccs)

        # create parents
        t0 = time.time()
        _create_parents(cg, layer, ccs, graph_ids, new_ids_d, node_cx, child_to_parent)
        lp["create_parents"] = time.time() - t0

        lp["total"] = sum(v for v in lp.values() if isinstance(v, float))
        layer_perf[layer] = lp
        log(f"    [stitch] layer {layer}: {lp['n_components']} CCs, {lp['total']:.1f}s")

    roots = np.array(new_ids_d[cg.meta.layer_count], dtype=basetypes.NODE_ID)
    return roots, layer_perf


def _resolve_cx_at_layer(new_nodes, layer, node_cx, resolver, old_to_new, child_to_parent, get_layer):
    """
    For each node, get its cross edges at `layer` and resolve column 1
    (partner side) to the partner's identity at this layer.

    Column 0 is already the current node ID (set during merge/propagation).
    Column 1 is a partner SV — resolve via:
      1. resolver[sv] gives {layer: identity} from upfront read
      2. For our own SVs, resolver has {2: old_l2}. Apply old_to_new to get new_l2,
         then walk child_to_parent to reach current layer.
      3. For external SVs, resolver has the full chain from get_all_parents_dict_multiple.
    """
    edge_list = []
    for node in new_nodes:
        cx_d = node_cx.get(int(node), {})
        edges = cx_d.get(layer, None)
        if edges is None or len(edges) == 0:
            continue
        edge_list.append(edges)

    if not edge_list:
        return types.empty_2d

    all_edges = np.concatenate(edge_list).astype(basetypes.NODE_ID)

    # resolve column 1 to identity at this layer
    resolved_col1 = np.empty(len(all_edges), dtype=basetypes.NODE_ID)
    for i, partner_sv in enumerate(all_edges[:, 1]):
        resolved_col1[i] = _resolve_sv_to_layer(
            int(partner_sv), layer, resolver, old_to_new, child_to_parent, get_layer
        )

    result = np.column_stack([all_edges[:, 0], resolved_col1])
    result = np.unique(result, axis=0)
    # remove self-edges
    mask = result[:, 0] != result[:, 1]
    return result[mask] if np.any(mask) else types.empty_2d


def _resolve_sv_to_layer(sv, target_layer, resolver, old_to_new, child_to_parent, get_layer):
    """
    Resolve an SV to its identity at target_layer.

    Uses the resolver chain (from upfront read) plus old_to_new (L2 merge)
    and child_to_parent (hierarchy built so far in this stitch).
    get_layer(node_id) returns the chunk layer of a node.
    """
    chain = resolver.get(sv, {})

    # start from the SV itself, walk up the chain
    # chain has {layer: identity} — find highest identity at or below target_layer
    identity = sv
    for l in sorted(chain.keys()):
        if l <= target_layer:
            identity = chain[l]
        else:
            break

    # if identity is an old L2 that was merged in this stitch, remap
    if identity in old_to_new:
        identity = old_to_new[identity]

    # walk child_to_parent, but don't overshoot target_layer
    while identity in child_to_parent:
        next_id = child_to_parent[identity]
        if get_layer(next_id) > target_layer:
            break
        identity = next_id

    return identity


def _create_parents(cg, layer, ccs, graph_ids, new_ids_d, node_cx, child_to_parent):
    """Create parent nodes with skip connections."""
    size_map = defaultdict(int)
    cc_info = {}

    for i, cc_idx in enumerate(ccs):
        cc_ids = graph_ids[cc_idx]
        parent_layer = layer + 1

        if len(cc_ids) == 1:
            # skip connection: find lowest layer above current with cross edges
            parent_layer = cg.meta.layer_count
            cx_d = node_cx.get(int(cc_ids[0]), {})
            for l in range(layer + 1, cg.meta.layer_count):
                if l in cx_d and len(cx_d[l]) > 0:
                    parent_layer = l
                    break

        chunk_id = int(get_parent_chunk_id(cg.meta, cc_ids[0], parent_layer))
        cc_info[i] = (parent_layer, chunk_id)
        size_map[chunk_id] += 1

    # allocate parent IDs
    chunk_ids = list(size_map.keys())
    random.shuffle(chunk_ids)
    root_chunks = {c for c in chunk_ids if cg.get_chunk_layer(np.uint64(c)) == cg.meta.layer_count}
    chunk_new_ids = _batch_create_node_ids(cg, size_map, root_chunks=root_chunks)

    # assign parents and propagate cross edges
    for i, cc_idx in enumerate(ccs):
        cc_ids = graph_ids[cc_idx]
        parent_layer, chunk_id = cc_info[i]
        parent = chunk_new_ids[chunk_id].pop()

        new_ids_d[parent_layer].append(parent)

        # parent inherits cross edges at layers >= parent_layer
        parent_cx = defaultdict(list)
        for child in cc_ids:
            child_cx = node_cx.get(int(child), {})
            for l, edges in child_cx.items():
                if l >= parent_layer and len(edges) > 0:
                    remapped = edges.copy()
                    remapped[:, 0] = parent
                    parent_cx[l].append(remapped)

        merged = {}
        for l, arr_list in parent_cx.items():
            merged[l] = np.unique(
                np.concatenate(arr_list).astype(basetypes.NODE_ID), axis=0
            )
        node_cx[int(parent)] = merged

        # track child → parent so partner resolution works for our own nodes
        for child in cc_ids:
            child_to_parent[int(child)] = int(parent)

        cg.cache.cross_chunk_edges_cache[parent] = merged
        cg.cache.children_cache[parent] = cc_ids
        cache_utils.update(cg.cache.parents_cache, cc_ids, parent)


def _resolve_cx_for_write(cg, ctx):
    """
    Resolve column 1 (partner SVs) in all nodes' cross edges to partner
    identities at the appropriate layer, then store in cg.cache for writing.
    """
    node_cx = ctx["node_cx"]
    resolver = ctx["resolver"]
    old_to_new = ctx.get("old_to_new", {})
    # child_to_parent was built during _build_hierarchy — it's in the closure
    # We need to reconstruct it from cg.cache.parents_cache
    child_to_parent = {}
    for node_id in cg.cache.new_ids:
        children = cg.cache.children_cache.get(node_id)
        if children is not None:
            for child in children:
                child_to_parent[int(child)] = int(node_id)

    get_layer = lambda nid: cg.get_chunk_layer(np.uint64(nid))

    for node_id in cg.cache.new_ids:
        node_layer = int(cg.get_chunk_layer(np.uint64(node_id)))
        raw_cx = node_cx.get(int(node_id), {})
        resolved = {}
        for layer, edges in raw_cx.items():
            if len(edges) == 0:
                continue
            resolved_col1 = np.array([
                _resolve_sv_to_layer(int(sv), node_layer, resolver, old_to_new, child_to_parent, get_layer)
                for sv in edges[:, 1]
            ], dtype=basetypes.NODE_ID)
            resolved_edges = np.column_stack([edges[:, 0], resolved_col1])
            resolved_edges = np.unique(resolved_edges, axis=0)
            # remove self-edges
            mask = resolved_edges[:, 0] != resolved_edges[:, 1]
            if np.any(mask):
                resolved[layer] = resolved_edges[mask]
        cg.cache.cross_chunk_edges_cache[node_id] = resolved


# ─────────────────────────────────────────────────────────────────────
# Phase 4: build mutations
# ─────────────────────────────────────────────────────────────────────


def _build_entries(cg, ctx):
    """Build BigTable mutation entries for all new nodes.
    Returns (node_entries, parent_entries) — write node rows first so
    Parent pointers only ever reference existing rows. This makes
    partial write failures safe to retry.
    """
    node_entries = []
    parent_entries = []
    ts = None
    l2_atomic_cx = ctx.get("l2_atomic_cx", {})

    for node_id in cg.cache.new_ids:
        children = cg.cache.children_cache.get(node_id)
        if children is None:
            continue

        val_dict = {attributes.Hierarchy.Child: children}

        # CrossChunkEdge (resolved, for parent nodes and L2)
        cx = cg.cache.cross_chunk_edges_cache.get(node_id, {})
        for layer, cx_edges in cx.items():
            val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = cx_edges

        # AtomicCrossChunkEdge (raw [sv, sv], only for new L2 nodes)
        acx = l2_atomic_cx.get(int(node_id), {})
        for layer, acx_edges in acx.items():
            val_dict[attributes.Connectivity.AtomicCrossChunkEdge[layer]] = acx_edges

        row_key = serializers.serialize_uint64(node_id)
        node_entries.append(cg.client.mutate_row(row_key, val_dict, time_stamp=ts))

    # Parent pointers — written second so target rows always exist
    for node_id in cg.cache.new_ids:
        children = cg.cache.children_cache.get(node_id)
        if children is None:
            continue
        for child in children:
            val_dict = {attributes.Hierarchy.Parent: node_id}
            row_key = serializers.serialize_uint64(child)
            parent_entries.append(cg.client.mutate_row(row_key, val_dict, time_stamp=ts))

    return node_entries, parent_entries
