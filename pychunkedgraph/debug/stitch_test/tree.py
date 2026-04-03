"""Tree operations: parent chain walks, sibling restore."""

import numpy as np

from pychunkedgraph.graph import basetypes


def get_all_parents_filtered(lcg, node_ids: np.ndarray) -> dict:
    """Walk parent chains from node_ids to root. Returns {node: {layer: parent_at_layer}}."""
    result = {int(n): {} for n in node_ids}
    nodes = np.array(node_ids, dtype=basetypes.NODE_ID)
    child_parent = {}
    layer_map = {}

    while nodes.size > 0:
        parents = lcg._cache.get_parents(nodes)
        parent_layers = lcg.get_chunk_layers(parents)

        has_parent = parents != 0
        nodes_with_parent = nodes[has_parent]
        parents = parents[has_parent]
        parent_layers = parent_layers[has_parent]

        if len(parents) == 0:
            break

        unique_parents = np.unique(parents)
        if len(unique_parents) > 0:
            lcg.bulk_read_parent_child(unique_parents)

        for node, parent, layer in zip(nodes_with_parent, parents, parent_layers):
            layer_map[int(parent)] = int(layer)
            child_parent[int(node)] = int(parent)

        nxt = [int(p) for p, layer in zip(parents, parent_layers) if int(layer) < lcg.meta.layer_count]
        nodes = np.unique(np.array(nxt, dtype=basetypes.NODE_ID)) if nxt else np.array([], dtype=basetypes.NODE_ID)

    for n in node_ids:
        cur = int(n)
        chain = {}
        while cur in child_parent:
            par = child_parent[cur]
            chain[layer_map[par]] = par
            cur = par
        result[int(n)] = chain
    return result


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
