"""Resolver operations: SV→identity resolution, CX resolution, partner chain resolution.

Consolidated from tree.py and topology.py. All operations that map SVs or nodes
to their resolved identities at target layers.
"""

from collections import defaultdict

import numpy as np

from pychunkedgraph.graph import basetypes, types


def resolve_svs_to_layer(
    svs: np.ndarray, target_layer: int, cache, get_layer,
) -> dict:
    """Walk SVs upward through the hierarchy to find their ancestor at target_layer.

    Each SV starts at itself. At each step, batch-fetch parents for all current
    positions (deduplicated — many SVs share ancestors). An SV is resolved when
    its current node's parent is above target_layer or missing (root/frontier).
    If no SVs make progress in a step, remaining are returned at current position.

    Returns {sv: ancestor_at_target_layer}.
    """
    result = {}
    active = {int(sv): int(sv) for sv in svs}

    while active:
        current_nodes = np.array(list(set(active.values())), dtype=basetypes.NODE_ID)
        parents = cache.get_parents(current_nodes)
        parent_of = {int(nid): int(p) for nid, p in zip(current_nodes, parents)}

        still_active = {}
        progressed = False
        for sv, current in active.items():
            parent = parent_of.get(current, 0)
            if parent == 0 or get_layer(parent) > target_layer:
                result[sv] = current
                progressed = True
            elif parent != current:
                still_active[sv] = parent
                progressed = True
            else:
                still_active[sv] = current
        if not still_active or not progressed:
            result.update(still_active)
            break
        active = still_active

    return result


def resolve_cx_at_layer(
    nodes: list, layer: int, cache, get_layer,
    resolve_layer: int = None,
) -> np.ndarray:
    if resolve_layer is None:
        resolve_layer = layer
    edge_list = []
    for node in nodes:
        cx_d = cache.unresolved_acx.get(int(node), {})
        edges = cx_d.get(layer)
        if edges is not None and len(edges) > 0:
            edge_list.append(edges)
    if not edge_list:
        return types.empty_2d

    all_edges = np.concatenate(edge_list).astype(basetypes.NODE_ID)
    unique_svs = np.unique(all_edges[:, 1])
    sv_map = resolve_svs_to_layer(unique_svs, resolve_layer, cache, get_layer)
    col1 = np.array([sv_map[int(sv)] for sv in all_edges[:, 1]], dtype=basetypes.NODE_ID)
    result = np.column_stack([all_edges[:, 0], col1])
    result = np.unique(result, axis=0)
    mask = result[:, 0] != result[:, 1]
    return result[mask] if np.any(mask) else types.empty_2d


def store_cx_from_resolved(cache, cx_edges: np.ndarray, layer: int) -> None:
    if len(cx_edges) == 0:
        return
    sort_idx = np.argsort(cx_edges[:, 0])
    sorted_edges = cx_edges[sort_idx]
    unique_ids, starts = np.unique(sorted_edges[:, 0], return_index=True)
    ends = np.append(starts[1:], len(sorted_edges))
    for uid, s, e in zip(unique_ids, starts, ends):
        if int(uid) not in cache.sibling_ids:
            cache.set_cx_layer(int(uid), layer, sorted_edges[s:e])


def resolve_remaining_cx(cache, lcg) -> None:
    get_layer = lambda nid: lcg.get_chunk_layer(np.uint64(nid))
    all_affected = set(cache.new_node_ids)
    groups = defaultdict(list)
    nids_arr = np.array(list(all_affected), dtype=basetypes.NODE_ID)
    if len(nids_arr) == 0:
        return
    cx_batch = cache.get_cx_batch(nids_arr)
    for nid in all_affected:
        nid_layer = get_layer(nid)
        existing = cx_batch.get(int(nid), {})
        raw = cache.unresolved_acx.get(int(nid), {})
        for lyr in raw:
            if lyr > 2 and lyr not in existing and len(raw[lyr]) > 0:
                groups[(lyr, nid_layer)].append(nid)
    for (lyr, nid_layer) in sorted(groups):
        cx = resolve_cx_at_layer(
            groups[(lyr, nid_layer)], lyr, cache, get_layer,
            resolve_layer=nid_layer,
        )
        store_cx_from_resolved(cache, cx, lyr)


def ensure_partners_cached(cache, acx_source: dict) -> None:
    """Batch-ensure all partner SVs from ACX edges are in the cache."""
    partner_svs = set()
    for layer_d in acx_source.values():
        for edges in layer_d.values():
            if len(edges) > 0:
                partner_svs.update(int(sv) for sv in edges[:, 1])
    if partner_svs:
        cache.get_parents(np.array(list(partner_svs), dtype=basetypes.NODE_ID))


def acx_to_cx(acx_dict: dict, node_id) -> dict:
    cx = {}
    for layer, edges in acx_dict.items():
        if len(edges) > 0:
            working = edges.copy()
            working[:, 0] = node_id
            cx[layer] = working
    return cx
