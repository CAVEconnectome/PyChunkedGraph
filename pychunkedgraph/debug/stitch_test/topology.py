"""Topology operations: cross-edge resolve, store, propagate."""

from collections import defaultdict

import numpy as np

from pychunkedgraph.graph import basetypes, types

from . import tree


def acx_to_cx(acx_dict: dict, node_id) -> dict:
    cx = {}
    for layer, edges in acx_dict.items():
        if len(edges) > 0:
            working = edges.copy()
            working[:, 0] = node_id
            cx[layer] = working
    return cx


def resolve_cx_at_layer(
    nodes: list, layer: int, cache, child_to_parent: dict, get_layer,
) -> np.ndarray:
    edge_list = []
    for node in nodes:
        cx_d = cache.raw_cx_edges.get(int(node), {})
        edges = cx_d.get(layer)
        if edges is not None and len(edges) > 0:
            edge_list.append(edges)
    if not edge_list:
        return types.empty_2d

    all_edges = np.concatenate(edge_list).astype(basetypes.NODE_ID)
    unique_svs = np.unique(all_edges[:, 1])
    sv_map = {
        int(sv): tree.resolve_sv_to_layer(int(sv), layer, cache, child_to_parent, get_layer)
        for sv in unique_svs
    }
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
        cache.cx_cache.setdefault(int(uid), {})[layer] = sorted_edges[s:e]


def resolve_remaining_cx(cache, lcg, child_to_parent: dict) -> None:
    get_layer = lambda nid: lcg.get_chunk_layer(np.uint64(nid))
    all_affected = set(cache.new_node_ids) | {
        s for s in cache.sibling_ids if s in cache.raw_cx_edges
    }
    layer_nodes = defaultdict(list)
    for nid in all_affected:
        existing = cache.cx_cache.get(int(nid), {})
        raw = cache.raw_cx_edges.get(int(nid), {})
        for lyr in raw:
            if lyr > 2 and lyr not in existing and len(raw[lyr]) > 0:
                layer_nodes[lyr].append(nid)
    for lyr in sorted(layer_nodes):
        cx = resolve_cx_at_layer(layer_nodes[lyr], lyr, cache, child_to_parent, get_layer)
        store_cx_from_resolved(cache, cx, lyr)


def collect_and_resolve_partners(lcg, acx_source: dict, known_svs: set, resolver: dict) -> np.ndarray:
    partner_svs = set()
    for layer_d in acx_source.values():
        for edges in layer_d.values():
            if len(edges) > 0:
                partner_svs.update(edges[:, 1])

    unknown = np.array(list(partner_svs - known_svs), dtype=basetypes.NODE_ID)
    if len(unknown) > 0:
        chains = tree.get_all_parents_filtered(lcg, unknown)
        for sv, chain in chains.items():
            resolver[int(sv)] = {int(l): int(p) for l, p in chain.items()}
    return unknown
