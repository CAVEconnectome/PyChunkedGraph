"""Tree operations: parent chain walks, resolver, orphan filtering, sibling restore."""

import numpy as np

from pychunkedgraph.graph import basetypes
from pychunkedgraph.graph import cache as cache_utils
from pychunkedgraph.graph.utils.generic import filter_failed_node_ids


def resolve_sv_to_layer(
    sv: int, target_layer: int, cache, child_to_parent: dict, get_layer,
) -> int:
    chain = cache.resolver.get(sv, {})
    identity, best = sv, -1
    for l, ident in chain.items():
        if l <= target_layer and l > best:
            best, identity = l, ident
    if identity in cache.old_to_new:
        identity = cache.old_to_new[identity]
    while identity in child_to_parent:
        nxt = child_to_parent[identity]
        if get_layer(nxt) > target_layer:
            break
        identity = nxt
    return identity


def get_all_parents_filtered(lcg, node_ids: np.ndarray) -> dict:
    result = {int(n): {} for n in node_ids}
    nodes = np.array(node_ids, dtype=basetypes.NODE_ID)
    child_parent = {}
    layer_map = {}

    while nodes.size > 0:
        parents = lcg.get_parents(nodes)
        parent_layers = lcg.get_chunk_layers(parents)

        remap = {}
        unique_parents = np.unique(parents)
        if len(unique_parents) > 0:
            _, ch_d = lcg.bulk_read_parent_child(unique_parents)
            max_ch = [
                int(np.max(ch_d[p])) if len(ch_d.get(p, [])) > 0 else 0
                for p in unique_parents
            ]
            seg_ids = np.array([lcg.get_segment_id(p) for p in unique_parents])
            valid = set(int(x) for x in filter_failed_node_ids(unique_parents, seg_ids, max_ch))
            mcid_valid = {}
            for p, mc in zip(unique_parents, max_ch):
                if int(p) in valid:
                    mcid_valid[mc] = int(p)
            for p, mc in zip(unique_parents, max_ch):
                if int(p) not in valid:
                    remap[int(p)] = mcid_valid.get(mc, int(p))

        for node, parent, layer in zip(nodes, parents, parent_layers):
            resolved = remap.get(int(parent), int(parent))
            layer_map[resolved] = int(layer)
            child_parent[int(node)] = resolved

        nxt = []
        for parent, layer in zip(parents, parent_layers):
            resolved = remap.get(int(parent), int(parent))
            if int(layer) < lcg.meta.layer_count:
                nxt.append(resolved)
        nodes = (
            np.unique(np.array(nxt, dtype=basetypes.NODE_ID))
            if nxt else np.array([], dtype=basetypes.NODE_ID)
        )

    for n in node_ids:
        cur = int(n)
        chain = {}
        while cur in child_parent:
            par = child_parent[cur]
            chain[layer_map[par]] = par
            cur = par
        result[int(n)] = chain
    return result


def filter_orphaned(lcg, node_ids: np.ndarray, children_d: dict) -> np.ndarray:
    if len(node_ids) == 0:
        return node_ids
    max_ch = np.array([
        int(np.max(children_d[n])) if len(children_d.get(n, [])) > 0 else 0
        for n in node_ids
    ])
    seg_ids = np.array([lcg.get_segment_id(n) for n in node_ids])
    return filter_failed_node_ids(node_ids, seg_ids, max_ch)


def resolve_partner_sv_parents(lcg, unknown_svs: set) -> dict:
    THRESHOLD = 25000
    SAMPLE_SIZE = 2500
    STOP = 10000

    if not unknown_svs:
        return {}
    arr = np.array(list(unknown_svs), dtype=basetypes.NODE_ID)

    if len(arr) <= THRESHOLD:
        parents = lcg.get_parents(arr)
        return {int(sv): int(l2) for sv, l2 in zip(arr, parents)}

    rng = np.random.default_rng()
    remaining = set(int(x) for x in arr)
    resolved = {}
    known_l2s = set()

    while len(remaining) > STOP:
        rem = np.array(list(remaining), dtype=basetypes.NODE_ID)
        sample = rng.choice(rem, size=min(SAMPLE_SIZE, len(rem)), replace=False)
        parents = lcg.get_parents(sample)
        for sv, l2 in zip(sample, parents):
            resolved[int(sv)] = int(l2)
        remaining -= set(int(x) for x in sample)

        new_l2s = set(int(x) for x in np.unique(parents)) - known_l2s
        if new_l2s:
            _, ch = lcg.bulk_read_parent_child(np.array(list(new_l2s), dtype=basetypes.NODE_ID))
            for l2_int in new_l2s:
                for sv in ch.get(np.uint64(l2_int), ch.get(l2_int, [])):
                    sv_int = int(sv)
                    if sv_int in remaining:
                        resolved[sv_int] = l2_int
                        remaining.discard(sv_int)
            known_l2s.update(new_l2s)

    if remaining:
        rem = np.array(list(remaining), dtype=basetypes.NODE_ID)
        parents = lcg.get_parents(rem)
        for sv, l2 in zip(rem, parents):
            resolved[int(sv)] = int(l2)

    return resolved


def update_parents_cache(cache, children: np.ndarray, parent) -> None:
    cache_utils.update(cache.parents_cache, children, parent)


def restore_known_siblings(lcg, cache, known: np.ndarray) -> None:
    if len(known) == 0:
        return
    children_d, acx_d = lcg.read_l2(known)
    for sib in known:
        sib_int = int(sib)
        entry = cache.get_sibling(sib_int)
        if entry is None:
            continue
        cache.children_d[sib] = children_d[sib]
        cache.atomic_cx_stitch[sib] = acx_d[sib]
        cache.raw_cx_edges[sib_int] = entry.raw_cx_edges
        for sv_int, resolver_entry in entry.resolver_entries.items():
            cache.resolver[sv_int] = resolver_entry
