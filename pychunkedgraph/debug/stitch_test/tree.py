"""Tree operations: parent chain walks, resolver, orphan filtering, sibling restore."""

import numpy as np

from pychunkedgraph.graph import basetypes
from pychunkedgraph.graph import cache as cache_utils
from pychunkedgraph.graph.utils.generic import filter_failed_node_ids


def resolve_sv_to_layer(
    sv: int, target_layer: int, cache, get_layer,
) -> int:
    """Single-SV resolve. Use resolve_svs_to_layer (resolver.py) for batch."""
    identity = sv
    parents = cache.get_parents(np.array([identity], dtype=basetypes.NODE_ID))
    parent = int(parents[0])
    while parent != 0 and get_layer(parent) <= target_layer:
        identity = parent
        parents = cache.get_parents(np.array([identity], dtype=basetypes.NODE_ID))
        parent = int(parents[0])
    return identity


def get_all_parents_filtered(lcg, node_ids: np.ndarray) -> dict:
    result = {int(n): {} for n in node_ids}
    nodes = np.array(node_ids, dtype=basetypes.NODE_ID)
    child_parent = {}
    layer_map = {}

    while nodes.size > 0:
        parents = lcg._cache.get_parents(nodes)
        parent_layers = lcg.get_chunk_layers(parents)

        # Filter out nodes with no parent (parent=0: root or build frontier)
        has_parent = parents != 0
        nodes_with_parent = nodes[has_parent]
        parents = parents[has_parent]
        parent_layers = parent_layers[has_parent]

        if len(parents) == 0:
            break

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

        for node, parent, layer in zip(nodes_with_parent, parents, parent_layers):
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


def filter_orphaned(lcg, node_ids: np.ndarray) -> np.ndarray:
    if len(node_ids) == 0:
        return node_ids
    ch_d = lcg._cache.get_children_batch(node_ids)
    max_ch = np.array([
        int(np.max(ch_d[int(n)])) if len(ch_d.get(int(n), [])) > 0 else 0
        for n in node_ids
    ])
    seg_ids = np.array([lcg.get_segment_id(n) for n in node_ids])
    return filter_failed_node_ids(node_ids, seg_ids, max_ch)



def update_parents_cache(cache, children: np.ndarray, parent) -> None:
    for ch in children:
        cache.put_parent(int(ch), int(parent))


def restore_known_siblings(lcg, cache, known: np.ndarray) -> None:
    if len(known) == 0:
        return
    for sib in known:
        sib_int = int(sib)
        entry = cache.get_sibling(sib_int)
        if entry is None:
            continue
        cache.unresolved_acx[sib_int] = entry.unresolved_acx
