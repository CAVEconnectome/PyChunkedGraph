"""
Cache nodes, parents, children and cross edges.
"""
from cachetools import cached
from cachetools import LRUCache

import numpy as np


from .utils.basetypes import NODE_ID

PARENTS = LRUCache(maxsize=1024)
CHILDREN = LRUCache(maxsize=512)
ATOMIC_CX_EDGES = LRUCache(maxsize=512)


def update_cache(cache, keys, vals):
    for k, v in zip(keys, vals):
        cache[k] = v


class CacheService:
    def __init__(self, cg):
        self._cg = cg

        self.parent_vec = np.vectorize(self.parent)
        self.children_vec = np.vectorize(self.children)
        self.atomic_cross_edges_vec = np.vectorize(self.atomic_cross_edges)

    @cached(
        cache=PARENTS, key=lambda self, node_id: node_id,
    )
    def parent(self, node_id):
        return self._cg.get_parent(node_id)

    @cached(
        cache=CHILDREN, key=lambda self, node_id: node_id,
    )
    def children(self, node_id):
        return self._cg.get_children(node_id)

    @cached(
        cache=ATOMIC_CX_EDGES, key=lambda self, node_id: node_id,
    )
    def atomic_cross_edges(self, node_id):
        return self._cg.get_atomic_cross_edges(node_id)

    def parents_multiple(self, node_ids: np.ndarray):
        mask = np.in1d(node_ids, np.fromiter(PARENTS.__data.keys(), dtype=NODE_ID))
        parents = node_ids.copy()
        parents[mask] = self.parent_vec(node_ids[mask])
        parents[~mask] = self._cg.get_parents(node_ids[~mask])
        update_cache(PARENTS, node_ids[~mask], parents[~mask])
        return parents

    def children_multiple(self, node_ids: np.ndarray, *, flatten=False):
        result = {}
        mask = np.in1d(node_ids, np.fromiter(CHILDREN.__data.keys(), dtype=NODE_ID))
        result.update(self.children_vec(node_ids[mask]))
        result.update(self._cg.get_children(node_ids[~mask]))
        update_cache(CHILDREN, node_ids[~mask], [result[k] for k in node_ids[~mask]])
        if flatten:
            return np.concatenate([*result.values()])
        return result

    def atomic_cross_edges_multiple(self, node_ids: np.ndarray):
        result = {}
        mask = np.in1d(
            node_ids, np.fromiter(ATOMIC_CX_EDGES.__data.keys(), dtype=NODE_ID)
        )
        result.update(self.atomic_cross_edges_vec(node_ids[mask]))
        result.update(self._cg.get_atomic_cross_edges(node_ids[~mask]))
        update_cache(
            ATOMIC_CX_EDGES, node_ids[~mask], [result[k] for k in node_ids[~mask]]
        )
        return result
