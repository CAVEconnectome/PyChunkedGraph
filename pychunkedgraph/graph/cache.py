"""
Cache nodes, parents, children and cross edges.
"""
from sys import maxsize
from cachetools import cached
from cachetools import LRUCache

import numpy as np


from .utils.basetypes import NODE_ID


# no limit because we don't want to lose new IDs
PARENTS = LRUCache(maxsize=maxsize)
CHILDREN = LRUCache(maxsize=maxsize)
ATOMIC_CX_EDGES = LRUCache(maxsize=maxsize)


def clear():
    PARENTS.clear()
    CHILDREN.clear()
    ATOMIC_CX_EDGES.clear()


def update(cache, keys, vals):
    try:
        # 1 to 1
        for k, v in zip(keys, vals):
            cache[k] = v
    except TypeError:
        # many to 1
        for k in keys:
            cache[k] = vals


class CacheService:
    def __init__(self, cg):
        self._cg = cg

        self.parent_vec = np.vectorize(self.parent, otypes=[np.uint64])
        self.children_vec = np.vectorize(self.children, otypes=[np.ndarray])
        self.atomic_cross_edges_vec = np.vectorize(
            self.atomic_cross_edges, otypes=[dict]
        )

    @cached(
        cache=PARENTS, key=lambda self, node_id: node_id,
    )
    def parent(self, node_id):
        return self._cg.get_parent(node_id, raw_only=True)

    @cached(
        cache=CHILDREN, key=lambda self, node_id: node_id,
    )
    def children(self, node_id):
        return self._cg.get_children(node_id, raw_only=True)

    @cached(
        cache=ATOMIC_CX_EDGES, key=lambda self, node_id: node_id,
    )
    def atomic_cross_edges(self, node_id):
        edges = self._cg.get_atomic_cross_edges(
            np.array([node_id], dtype=NODE_ID), raw_only=True
        )
        return edges[node_id]

    def parents_multiple(self, node_ids: np.ndarray):
        if not node_ids.size:
            return node_ids
        mask = np.in1d(node_ids, np.fromiter(PARENTS.keys(), dtype=NODE_ID))
        parents = node_ids.copy()
        parents[mask] = self.parent_vec(node_ids[mask])
        parents[~mask] = self._cg.get_parents(node_ids[~mask], raw_only=True)
        update(PARENTS, node_ids[~mask], parents[~mask])
        return parents

    def children_multiple(self, node_ids: np.ndarray, *, flatten=False):
        result = {}
        if not node_ids.size:
            return result
        mask = np.in1d(node_ids, np.fromiter(CHILDREN.keys(), dtype=NODE_ID))
        cached_children_ = self.children_vec(node_ids[mask])
        result.update({id_: c_ for id_, c_ in zip(node_ids[mask], cached_children_)})
        result.update(self._cg.get_children(node_ids[~mask], raw_only=True))
        update(CHILDREN, node_ids[~mask], [result[k] for k in node_ids[~mask]])
        if flatten:
            return np.concatenate([*result.values()])
        return result

    def atomic_cross_edges_multiple(self, node_ids: np.ndarray):
        result = {}
        if not node_ids.size:
            return result
        mask = np.in1d(node_ids, np.fromiter(ATOMIC_CX_EDGES.keys(), dtype=NODE_ID))
        cached_edges_ = self.atomic_cross_edges_vec(node_ids[mask])
        result.update(
            {id_: edges_ for id_, edges_ in zip(node_ids[mask], cached_edges_)}
        )
        result.update(self._cg.get_atomic_cross_edges(node_ids[~mask], raw_only=True))
        update(ATOMIC_CX_EDGES, node_ids[~mask], [result[k] for k in node_ids[~mask]])
        return result
