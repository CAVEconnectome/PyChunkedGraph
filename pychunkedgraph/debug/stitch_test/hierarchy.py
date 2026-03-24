"""
Hierarchy building utilities for the proposed stitch algorithm.
Cross edge resolution, parent creation, skip connections.
"""

from collections import defaultdict

import numpy as np

from pychunkedgraph.graph import basetypes, types
from pychunkedgraph.graph import cache as cache_utils
from pychunkedgraph.graph.chunks.hierarchy import get_parent_chunk_id

from .reader import batch_create_node_ids
from .stitch_types import StitchContext


def resolve_cx_at_layer(
    new_nodes: list,
    layer: int,
    node_cx: dict,
    resolver: dict,
    old_to_new: dict,
    child_to_parent: dict,
    get_layer,
) -> np.ndarray:
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

    # resolve unique partner SVs once, then map back
    unique_svs = np.unique(all_edges[:, 1])
    sv_to_resolved = {
        int(sv): resolve_sv_to_layer(
            int(sv), layer, resolver, old_to_new, child_to_parent, get_layer
        )
        for sv in unique_svs
    }
    resolved_col1 = np.array(
        [sv_to_resolved[int(sv)] for sv in all_edges[:, 1]], dtype=basetypes.NODE_ID
    )

    result = np.column_stack([all_edges[:, 0], resolved_col1])
    result = np.unique(result, axis=0)
    mask = result[:, 0] != result[:, 1]
    return result[mask] if np.any(mask) else types.empty_2d


def resolve_sv_to_layer(
    sv: int,
    target_layer: int,
    resolver: dict,
    old_to_new: dict,
    child_to_parent: dict,
    get_layer,
) -> int:
    """
    Resolve an SV to its identity at target_layer.

    Uses the resolver chain (from upfront read) plus old_to_new (L2 merge)
    and child_to_parent (hierarchy built so far in this stitch).
    get_layer(node_id) returns the chunk layer of a node.
    """
    chain = resolver.get(sv, {})

    identity = sv
    best_layer = -1
    for l, ident in chain.items():
        if l <= target_layer and l > best_layer:
            best_layer = l
            identity = ident

    if identity in old_to_new:
        identity = old_to_new[identity]

    while identity in child_to_parent:
        next_id = child_to_parent[identity]
        if get_layer(next_id) > target_layer:
            break
        identity = next_id

    return identity


def create_parents(
    cg,
    ctx: StitchContext,
    layer: int,
    ccs: list,
    graph_ids: np.ndarray,
    new_ids_d: dict,
    node_cx: dict,
    child_to_parent: dict,
    deferred_roots: list,
) -> None:
    """Create parent nodes with skip connections.
    Root-layer parents are deferred — their children are recorded in deferred_roots
    for batch allocation after the hierarchy loop.
    """
    size_map = defaultdict(int)
    cc_info = {}
    root_layer = cg.meta.layer_count

    for i, cc_idx in enumerate(ccs):
        cc_ids = graph_ids[cc_idx]
        parent_layer = layer + 1

        if len(cc_ids) == 1:
            parent_layer = root_layer
            cx_d = node_cx.get(int(cc_ids[0]), {})
            for l in range(layer + 1, root_layer):
                if l in cx_d and len(cx_d[l]) > 0:
                    parent_layer = l
                    break

        cc_info[i] = (parent_layer, cc_ids)
        if parent_layer < root_layer:
            chunk_id = int(get_parent_chunk_id(cg.meta, cc_ids[0], parent_layer))
            size_map[chunk_id] += 1
            cc_info[i] = (parent_layer, cc_ids, chunk_id)

    chunk_new_ids = batch_create_node_ids(cg, size_map) if size_map else {}

    for i, cc_idx in enumerate(ccs):
        cc_ids = cc_info[i][1]
        parent_layer = cc_info[i][0]

        if parent_layer == root_layer:
            deferred_roots.append(cc_ids)
            continue

        chunk_id = cc_info[i][2]
        parent = chunk_new_ids[chunk_id].pop()
        new_ids_d[parent_layer].append(parent)

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

        for child in cc_ids:
            child_to_parent[int(child)] = int(parent)

        ctx.cx_cache[parent] = merged
        ctx.children_cache[parent] = cc_ids
        cache_utils.update(ctx.parents_cache, cc_ids, parent)


def allocate_deferred_roots(
    cg, ctx: StitchContext, deferred_roots: list, new_ids_d: dict
) -> None:
    """Batch-allocate root IDs for all deferred root CCs, with collision check."""
    if not deferred_roots:
        return

    root_layer = cg.meta.layer_count
    root_chunk_id = int(get_parent_chunk_id(cg.meta, deferred_roots[0][0], root_layer))
    count = len(deferred_roots)

    root_ids = batch_create_node_ids(
        cg, {root_chunk_id: count}, root_chunks={root_chunk_id}
    )[root_chunk_id]

    for i, cc_ids in enumerate(deferred_roots):
        root_id = root_ids[i]
        new_ids_d[root_layer].append(root_id)
        ctx.children_cache[root_id] = cc_ids
        cache_utils.update(ctx.parents_cache, cc_ids, root_id)


def resolve_cx_for_write(cg, ctx: StitchContext) -> None:
    """
    Resolve column 1 (partner SVs) in all nodes' cross edges to partner
    identities at the appropriate layer, then store in ctx.cx_cache for writing.

    Batched: resolves each unique (sv, target_layer) pair once, then distributes.
    """
    node_cx = ctx.node_cx
    resolver = ctx.resolver
    old_to_new = ctx.old_to_new
    sibling_ids = ctx.sibling_ids
    all_affected = set(ctx.new_node_ids) | {s for s in sibling_ids if s in node_cx}

    child_to_parent = {}
    for node_id in ctx.new_node_ids:
        children = ctx.children_cache.get(node_id)
        if children is not None:
            for child in children:
                child_to_parent[int(child)] = int(node_id)

    get_layer = lambda nid: cg.get_chunk_layer(np.uint64(nid))

    # collect all unique (sv, target_layer) pairs across all nodes
    sv_layer_pairs = set()
    for node_id in all_affected:
        node_layer = int(cg.get_chunk_layer(np.uint64(node_id)))
        raw_cx = node_cx.get(int(node_id), {})
        for layer, edges in raw_cx.items():
            if len(edges) > 0:
                sv_layer_pairs.update((int(sv), node_layer) for sv in edges[:, 1])

    # resolve each unique pair once
    resolved_cache = {}
    for sv, target_layer in sv_layer_pairs:
        resolved_cache[(sv, target_layer)] = resolve_sv_to_layer(
            sv, target_layer, resolver, old_to_new, child_to_parent, get_layer
        )

    # distribute resolved values back to each node
    for node_id in all_affected:
        node_layer = int(cg.get_chunk_layer(np.uint64(node_id)))
        raw_cx = node_cx.get(int(node_id), {})
        resolved = {}
        for layer, edges in raw_cx.items():
            if len(edges) == 0:
                continue
            resolved_col1 = np.array(
                [resolved_cache[(int(sv), node_layer)] for sv in edges[:, 1]],
                dtype=basetypes.NODE_ID,
            )
            resolved_edges = np.unique(
                np.column_stack([edges[:, 0], resolved_col1]), axis=0
            )
            mask = resolved_edges[:, 0] != resolved_edges[:, 1]
            if np.any(mask):
                resolved[layer] = resolved_edges[mask]
        ctx.cx_cache[node_id] = resolved
