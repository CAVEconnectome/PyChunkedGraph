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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes, serializers, types
from pychunkedgraph.graph import cache as cache_utils
from pychunkedgraph.graph.utils import flatgraph

from pychunkedgraph.graph.edges.utils import get_cross_chunk_edges_layer


from .utils import extract_structure, batch_get_l2children
from .reader import (
    CachedReader,
    get_all_parents_filtered,
    filter_orphaned_nodes,
    resolve_partner_sv_parents,
    batch_create_node_ids,
    collect_and_resolve_partner_svs,
    read_l2,
)
from .hierarchy import (
    resolve_cx_at_layer,
    resolve_sv_to_layer,
    create_parents,
    allocate_deferred_roots,
    resolve_cx_for_write,
)


from .stitch_types import StitchResult, StitchContext, RunResult


def run_proposed_stitch(graph_id: str, atomic_edges: np.ndarray) -> dict:

    atomic_edges = np.asarray(atomic_edges, dtype=basetypes.NODE_ID)
    cg = ChunkedGraph(graph_id=graph_id)

    print(f"  [proposed] stitch ({len(atomic_edges)} edges)...")
    t0 = time.time()
    result = stitch(cg, atomic_edges, verbose=False)
    t_stitch = time.time() - t0

    t_write = time.time()
    cg.client.write(result.node_entries)
    cg.client.write(result.parent_entries)
    result.perf["write_entries"] = time.time() - t_write

    elapsed = t_stitch + result.perf["write_entries"]
    print(f"  [proposed] stitch: {t_stitch:.1f}s, write: {result.perf['write_entries']:.1f}s, total: {elapsed:.1f}s, {len(result.new_roots)} roots")

    t_struct = time.time()
    structure = extract_structure(
        cg, np.array(result.new_roots, dtype=basetypes.NODE_ID)
    )
    print(f"  [proposed] structure: {time.time() - t_struct:.1f}s")

    return RunResult(
        structure=structure,
        new_roots=result.new_roots,
        new_l2_ids=result.new_l2_ids,
        new_ids_per_layer=result.new_ids_per_layer,
        elapsed=elapsed,
        graph_id=graph_id,
        n_edges=len(atomic_edges),
        n_entries_written=len(result.node_entries) + len(result.parent_entries),
        layer_counts={layer: len(nodes) for layer, nodes in structure["nodes"].items()},
        perf=result.perf,
    )


# Core algorithm


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
    log(
        f"    [stitch] phase 2 (L2 merge): {perf['phase2_total']:.1f}s, {len(new_l2_ids)} L2 nodes"
    )

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
    log(
        f"    [stitch] phase 3 (hierarchy): {perf['phase3_total']:.1f}s, {len(new_roots)} roots"
    )

    # ── Phase 3b: resolve cross edges for writing ──────────────────
    t0 = time.time()
    resolve_cx_for_write(cg, ctx)
    perf["phase3b_resolve_write"] = time.time() - t0

    # ── Phase 4: build mutations ─────────────────────────────────────
    t0 = time.time()
    node_entries, parent_entries = _build_entries(cg, ctx)
    perf["phase4_total"] = time.time() - t0

    # log RPC summary
    if ctx.reader and ctx.reader.rpc_log:
        log("    [stitch] RPC summary:")
        for method, n_req, n_read, elapsed in ctx.reader.rpc_log:
            log(f"      {method}: {n_read}/{n_req} read, {elapsed:.1f}s")
        perf["rpc_log"] = ctx.reader.rpc_log

    new_ids_per_layer = {layer: len(ids) for layer, ids in ctx.new_ids_d.items() if ids}

    return StitchResult(
        new_roots=[int(r) for r in new_roots],
        new_l2_ids=[int(x) for x in new_l2_ids],
        new_ids_per_layer=new_ids_per_layer,
        node_entries=node_entries,
        parent_entries=parent_entries,
        perf=perf,
    )


def _acx_to_node_cx(acx_dict: dict, node_id: basetypes.NODE_ID) -> dict:
    """Convert AtomicCrossChunkEdge dict to node_cx format: col 0 = node_id."""
    node_cx_d = {}
    for layer, edges in acx_dict.items():
        if len(edges) > 0:
            working = edges.copy()
            working[:, 0] = node_id
            node_cx_d[layer] = working
    return node_cx_d


def _read_upfront(
    cg: ChunkedGraph, atomic_edges: np.ndarray, perf: dict, log=print
) -> "StitchContext":
    """
    All BigTable reads happen here. Returns a context dict
    with everything needed for in-memory processing.
    """
    reader = CachedReader(cg)

    # 1. Classify edges by layer, get L2 parents of all SVs
    t0 = time.time()
    svs = np.unique(atomic_edges)
    sv_parents = reader.get_parents(svs)
    sv_to_l2 = dict(zip(svs, sv_parents))
    edge_layers = get_cross_chunk_edges_layer(cg.meta, atomic_edges)
    perf["read_sv_parents"] = time.time() - t0
    log(f"    [stitch]   sv parents: {perf['read_sv_parents']:.1f}s, {len(svs)} SVs")

    # Build L2-level edges and per-node cross edge dicts
    l2_edges = []  # within-chunk L2 edges
    l2_cx_edges = defaultdict(
        lambda: defaultdict(list)
    )  # {l2: {layer: [[sv, sv], ...]}}

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

    l2ids = (
        np.unique(np.array(l2_edges, dtype=basetypes.NODE_ID))
        if l2_edges
        else np.array([], dtype=basetypes.NODE_ID)
    )

    t0 = time.time()
    children_d, atomic_cx = read_l2(reader, l2ids)
    perf["read_children_and_acx"] = time.time() - t0
    log(f"    [stitch]   children+acx: {perf['read_children_and_acx']:.1f}s")

    l2ids = filter_orphaned_nodes(cg, l2ids, children_d)

    # 4. Collect ALL partner SVs from atomic cross edges + stitch edges
    #    and resolve their full parent chains in one batch
    t0 = time.time()
    partner_svs = set()
    for l2id, layer_d in atomic_cx.items():
        for layer, edges in layer_d.items():
            if len(edges) > 0:
                partner_svs.update(edges[:, 1])
    for l2id, layer_d in l2_cx_edges.items():
        for layer, edge_list in layer_d.items():
            if edge_list:
                partner_svs.update(np.array(edge_list, dtype=basetypes.NODE_ID)[:, 1])

    # remove SVs we already know (our own L2 nodes' children)
    known_svs = set()
    for l2id in l2ids:
        known_svs.update(children_d.get(l2id, []))

    unknown_partner_svs = partner_svs - known_svs

    # resolve partner SVs → L2 parents using sampling when large
    t_chains = time.time()
    partner_sv_to_l2 = resolve_partner_sv_parents(reader, unknown_partner_svs)
    unique_partner_l2s = np.array(
        list(set(partner_sv_to_l2.values())), dtype=basetypes.NODE_ID
    )

    log(
        f"    [stitch]   partner SVs: {len(unknown_partner_svs)} → {len(unique_partner_l2s)} unique L2s"
    )

    # read parent chains from unique partner L2s + our L2s concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_partner = (
            executor.submit(get_all_parents_filtered, reader, unique_partner_l2s)
            if len(unique_partner_l2s) > 0
            else None
        )
        fut_old_hier = executor.submit(get_all_parents_filtered, reader, l2ids)

        partner_l2_chains = fut_partner.result() if fut_partner else {}
        old_hierarchy = fut_old_hier.result()

    # build partner_chains: map each SV to its full chain via its L2 parent
    partner_chains = {}
    for sv_int, l2_int in partner_sv_to_l2.items():
        chain = partner_l2_chains.get(
            l2_int, partner_l2_chains.get(np.uint64(l2_int), {})
        )
        partner_chains[sv_int] = {
            2: l2_int,
            **{int(l): int(p) for l, p in chain.items()},
        }

    perf["read_partner_chains"] = time.time() - t_chains
    log(
        f"    [stitch]   partner chains + old hierarchy: {perf['read_partner_chains']:.1f}s"
    )

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

    return StitchContext(
        l2_edges=(
            np.array(l2_edges, dtype=basetypes.NODE_ID) if l2_edges else types.empty_2d
        ),
        l2ids=l2ids,
        l2_cx_edges=l2_cx_edges,
        children_d=children_d,
        atomic_cx=atomic_cx,
        resolver=resolver,
        sv_to_l2=sv_to_l2,
        old_hierarchy=old_hierarchy,
        reader=reader,
    )


def _merge_l2(cg: ChunkedGraph, ctx: "StitchContext", _perf: dict) -> list:
    """Merge L2 nodes based on within-chunk edge connectivity."""
    l2_edges = ctx.l2_edges
    children_d = ctx.children_d
    atomic_cx = ctx.atomic_cx
    l2_cx_edges = ctx.l2_cx_edges

    if l2_edges.size == 0:
        return []

    t_ = time.time()
    graph, _, _, graph_ids = flatgraph.build_gt_graph(l2_edges, make_directed=True)
    components = flatgraph.connected_components(graph)
    _perf["merge_l2_graph"] = time.time() - t_

    # allocate new L2 IDs
    t_ = time.time()
    chunk_count_map = defaultdict(int)
    for cc_indices in components:
        chunk_count_map[cg.get_chunk_id(graph_ids[cc_indices][0])] += 1

    chunk_ids = list(chunk_count_map.keys())
    random.shuffle(chunk_ids)
    chunk_new_ids_map = batch_create_node_ids(
        cg, {c: chunk_count_map[c] for c in chunk_ids}
    )
    _perf["merge_l2_alloc"] = time.time() - t_

    new_l2_ids = []
    old_to_new = {}
    node_cx = {}  # {node_id: {layer: [node_id, partner_sv]}} — for hierarchy building
    l2_atomic_cx = {}  # {new_l2: {layer: [sv, sv]}} — raw format for BigTable write

    t_ = time.time()
    for cc_indices in components:
        old_ids = graph_ids[cc_indices]
        new_id = chunk_new_ids_map[cg.get_chunk_id(old_ids[0])].pop()
        new_l2_ids.append(new_id)

        for old_id in old_ids:
            old_to_new[int(old_id)] = int(new_id)

        # merge SV children
        merged_children = np.concatenate([children_d[l2id] for l2id in old_ids]).astype(
            basetypes.NODE_ID
        )
        ctx.children_cache[new_id] = merged_children
        cache_utils.update(ctx.parents_cache, merged_children, new_id)

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
                    merged[layer].append(np.array(edge_list, dtype=basetypes.NODE_ID))

        # store raw [sv, sv] for AtomicCrossChunkEdge write
        raw_acx = {}
        for layer, arr_list in merged.items():
            raw_acx[layer] = np.unique(
                np.concatenate(arr_list).astype(basetypes.NODE_ID), axis=0
            )
        l2_atomic_cx[int(new_id)] = raw_acx

        node_cx[int(new_id)] = _acx_to_node_cx(raw_acx, new_id)
    _perf["merge_l2_loop"] = time.time() - t_

    # add ALL SVs of new L2 nodes to resolver
    t_ = time.time()
    resolver = ctx.resolver
    for new_id in new_l2_ids:
        new_id_int = int(new_id)
        for sv in ctx.children_cache.get(new_id, []):
            sv_int = int(sv)
            if sv_int not in resolver:
                resolver[sv_int] = {2: new_id_int}
    _perf["merge_l2_resolver"] = time.time() - t_

    ctx.old_to_new = old_to_new
    ctx.node_cx = node_cx
    ctx.l2_atomic_cx = l2_atomic_cx
    ctx.new_ids_d[2] = list(new_l2_ids)
    return new_l2_ids


def _discover_siblings(
    cg: ChunkedGraph, ctx: StitchContext, perf: dict, log=print
) -> None:
    """
    Find ALL L2 nodes in the affected subtree.

    The old parents of our affected L2 nodes have other L2 descendants
    (siblings). These must participate in hierarchy rebuild because the
    old parents are being replaced.

    We get L2 descendants (not direct children) of old parents, then read
    their AtomicCrossChunkEdge. This is the complete source of truth for
    connectivity at every layer — no CrossChunkEdge read ever needed.
    """
    l2ids = ctx.l2ids
    node_cx = ctx.node_cx
    resolver = ctx.resolver
    atomic_cx = ctx.atomic_cx
    children_d = ctx.children_d

    # 1. use pre-fetched old hierarchy from phase 1
    old_hierarchy = ctx.old_hierarchy

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
        parent_l2_map = batch_get_l2children(ctx.reader, old_parents_arr)
        for l2set in parent_l2_map.values():
            all_l2_in_subtree.update(l2set)

    known_l2 = set(int(x) for x in l2ids)
    l2_siblings = np.array(list(all_l2_in_subtree - known_l2), dtype=basetypes.NODE_ID)
    perf["siblings_get_l2"] = time.time() - t0
    log(
        f"    [stitch]   L2 siblings: {perf['siblings_get_l2']:.1f}s, {len(l2_siblings)} from {len(all_old_parents)} parents"
    )

    if len(l2_siblings) == 0:
        ctx.siblings_d = defaultdict(list)
        return

    t0 = time.time()
    sibling_children, sibling_acx = read_l2(ctx.reader, l2_siblings)

    l2_siblings = filter_orphaned_nodes(cg, l2_siblings, sibling_children)

    perf["siblings_reads"] = time.time() - t0
    log(f"    [stitch]   siblings reads: {perf['siblings_reads']:.1f}s")

    # 4. resolve partner SVs from siblings' atomic cross edges
    t0 = time.time()
    known_svs = set(resolver.keys())
    for l2id in l2ids:
        known_svs.update(children_d.get(l2id, []))
    for l2id in l2_siblings:
        known_svs.update(sibling_children.get(l2id, []))

    unknown_svs = collect_and_resolve_partner_svs(
        ctx.reader, sibling_acx, known_svs, resolver
    )
    perf["siblings_partner_chains"] = time.time() - t0
    log(
        f"    [stitch]   siblings partner chains: {perf['siblings_partner_chains']:.1f}s, {len(unknown_svs)} SVs"
    )

    # 5. populate ctx for each L2 sibling
    l2_atomic_cx = ctx.l2_atomic_cx
    for l2id in l2_siblings:
        l2id_int = int(l2id)
        children_d[l2id] = sibling_children.get(
            l2id, np.array([], dtype=basetypes.NODE_ID)
        )
        atomic_cx[l2id] = sibling_acx.get(l2id, {})

        for sv in sibling_children.get(l2id, []):
            resolver[int(sv)] = {2: l2id_int}

        node_cx[l2id_int] = _acx_to_node_cx(sibling_acx.get(l2id, {}), l2id)
    ctx.l2_atomic_cx = l2_atomic_cx

    # siblings participate at L2 in hierarchy building
    siblings_d = defaultdict(list)
    siblings_d[2] = [int(x) for x in l2_siblings]
    ctx.siblings_d = siblings_d
    ctx.sibling_ids = set(int(x) for x in l2_siblings)


def _build_hierarchy(
    cg: ChunkedGraph, _new_l2_ids: list, ctx: StitchContext, _perf: dict, log=print
) -> tuple:
    """Build parent hierarchy layer by layer from cached data."""
    node_cx = ctx.node_cx
    resolver = ctx.resolver
    old_to_new = ctx.old_to_new
    new_ids_d = ctx.new_ids_d
    siblings_d = ctx.siblings_d
    # tracks child → parent as we build, so partner resolution works for our own nodes
    child_to_parent = {}
    deferred_roots = []  # list of cc_ids arrays for root-layer parents
    layer_perf = {}

    for layer in range(2, cg.meta.layer_count):
        new_nodes = new_ids_d[layer]
        sibling_nodes = siblings_d.get(layer, [])
        all_nodes = list(new_nodes) + list(sibling_nodes)
        if not all_nodes:
            continue
        ctx.new_node_ids.update(new_nodes)
        lp = {"n_new_nodes": len(new_nodes), "n_siblings": len(sibling_nodes)}
        log(
            f"    [stitch] layer {layer}: {len(new_nodes)} new + {len(sibling_nodes)} siblings = {len(all_nodes)} nodes"
        )

        # resolve cross edges at this layer
        t0 = time.time()
        cx_edges = resolve_cx_at_layer(
            all_nodes,
            layer,
            node_cx,
            resolver,
            old_to_new,
            child_to_parent,
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

        # create parents (root-layer parents deferred)
        t0 = time.time()
        create_parents(
            cg,
            ctx,
            layer,
            ccs,
            graph_ids,
            new_ids_d,
            node_cx,
            child_to_parent,
            deferred_roots,
        )
        lp["create_parents"] = time.time() - t0

        lp["total"] = sum(v for v in lp.values() if isinstance(v, float))
        layer_perf[layer] = lp
        log(f"    [stitch] layer {layer}: {lp['n_components']} CCs, {lp['total']:.1f}s")

    # batch-allocate all root IDs at once with collision check
    allocate_deferred_roots(cg, ctx, deferred_roots, new_ids_d)
    roots = np.array(new_ids_d[cg.meta.layer_count], dtype=basetypes.NODE_ID)
    ctx.new_node_ids.update(roots.tolist())
    log(f"    [stitch] allocated {len(roots)} root IDs")
    return roots, layer_perf


def _build_entries(cg: ChunkedGraph, ctx: StitchContext) -> tuple:
    """Build BigTable mutation entries for all new nodes.
    Returns (node_entries, parent_entries) — write node rows first so
    Parent pointers only ever reference existing rows. This makes
    partial write failures safe to retry.
    """
    node_entries = []
    parent_entries = []
    ts = None
    l2_atomic_cx = ctx.l2_atomic_cx
    sibling_ids = ctx.sibling_ids

    # new nodes: write Child + CrossChunkEdge + AtomicCrossChunkEdge (L2 only)
    for node_id in ctx.new_node_ids:
        children = ctx.children_cache.get(node_id)
        if children is None:
            continue
        val_dict = {attributes.Hierarchy.Child: children}
        cx = ctx.cx_cache.get(node_id, {})
        for layer, cx_edges in cx.items():
            val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = cx_edges
        acx = l2_atomic_cx.get(int(node_id), {})
        for layer, acx_edges in acx.items():
            val_dict[attributes.Connectivity.AtomicCrossChunkEdge[layer]] = acx_edges
        row_key = serializers.serialize_uint64(node_id)
        node_entries.append(cg.client.mutate_row(row_key, val_dict, time_stamp=ts))

    # siblings: write only CrossChunkEdge (resolved to new hierarchy IDs)
    for node_id in sibling_ids:
        cx = ctx.cx_cache.get(node_id, {})
        if not cx:
            continue
        val_dict = {}
        for layer, cx_edges in cx.items():
            val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = cx_edges
        row_key = serializers.serialize_uint64(np.uint64(node_id))
        node_entries.append(cg.client.mutate_row(row_key, val_dict, time_stamp=ts))

    # Parent pointers — for new nodes' children only
    # siblings get their Parent updated here because they're children of new L3+ nodes
    for node_id in ctx.new_node_ids:
        children = ctx.children_cache.get(node_id)
        if children is None:
            continue
        for child in children:
            val_dict = {attributes.Hierarchy.Parent: node_id}
            row_key = serializers.serialize_uint64(child)
            parent_entries.append(
                cg.client.mutate_row(row_key, val_dict, time_stamp=ts)
            )

    return node_entries, parent_entries
