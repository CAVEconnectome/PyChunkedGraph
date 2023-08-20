# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel
"""
Cache nodes, parents, children and cross edges.
"""
from sys import maxsize
from datetime import datetime

from cachetools import cached
from cachetools import LRUCache

import numpy as np


from .utils.basetypes import NODE_ID


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

        self._parent_vec = np.vectorize(self.parent, otypes=[np.uint64])
        self._children_vec = np.vectorize(self.children, otypes=[np.ndarray])
        self._atomic_cross_edges_vec = np.vectorize(
            self.atomic_cross_edges, otypes=[dict]
        )

        # no limit because we don't want to lose new IDs
        self.parents_cache = LRUCache(maxsize=maxsize)
        self.children_cache = LRUCache(maxsize=maxsize)
        self.atomic_cx_edges_cache = LRUCache(maxsize=maxsize)

    def __len__(self):
        return (
            len(self.parents_cache)
            + len(self.children_cache)
            + len(self.atomic_cx_edges_cache)
        )

    def clear(self):
        self.parents_cache.clear()
        self.children_cache.clear()
        self.atomic_cx_edges_cache.clear()

    def parent(self, node_id: np.uint64, *, time_stamp: datetime = None):
        @cached(cache=self.parents_cache, key=lambda node_id: node_id)
        def parent_decorated(node_id):
            return self._cg.get_parent(node_id, raw_only=True, time_stamp=time_stamp)

        return parent_decorated(node_id)

    def children(self, node_id):
        @cached(cache=self.children_cache, key=lambda node_id: node_id)
        def children_decorated(node_id):
            children = self._cg.get_children(node_id, raw_only=True)
            update(self.parents_cache, children, node_id)
            return children

        return children_decorated(node_id)

    def atomic_cross_edges(self, node_id):
        @cached(cache=self.atomic_cx_edges_cache, key=lambda node_id: node_id)
        def atomic_cross_edges_decorated(node_id):
            edges = self._cg.get_atomic_cross_edges(
                np.array([node_id], dtype=NODE_ID), raw_only=True
            )
            return edges[node_id]

        return atomic_cross_edges_decorated(node_id)

    def parents_multiple(self, node_ids: np.ndarray, *, time_stamp: datetime = None):
        if not node_ids.size:
            return node_ids
        mask = np.in1d(node_ids, np.fromiter(self.parents_cache.keys(), dtype=NODE_ID))
        parents = node_ids.copy()
        parents[mask] = self._parent_vec(node_ids[mask])
        parents[~mask] = self._cg.get_parents(
            node_ids[~mask], raw_only=True, time_stamp=time_stamp
        )
        update(self.parents_cache, node_ids[~mask], parents[~mask])
        return parents

    def children_multiple(self, node_ids: np.ndarray, *, flatten=False):
        result = {}
        if not node_ids.size:
            return result
        mask = np.in1d(node_ids, np.fromiter(self.children_cache.keys(), dtype=NODE_ID))
        cached_children_ = self._children_vec(node_ids[mask])
        result.update({id_: c_ for id_, c_ in zip(node_ids[mask], cached_children_)})
        result.update(self._cg.get_children(node_ids[~mask], raw_only=True))
        update(
            self.children_cache, node_ids[~mask], [result[k] for k in node_ids[~mask]]
        )
        if flatten:
            return np.concatenate([*result.values()])
        return result

    def atomic_cross_edges_multiple(self, node_ids: np.ndarray):
        result = {}
        if not node_ids.size:
            return result
        mask = np.in1d(
            node_ids, np.fromiter(self.atomic_cx_edges_cache.keys(), dtype=NODE_ID)
        )
        cached_edges_ = self._atomic_cross_edges_vec(node_ids[mask])
        result.update(
            {id_: edges_ for id_, edges_ in zip(node_ids[mask], cached_edges_)}
        )
        result.update(self._cg.get_atomic_cross_edges(node_ids[~mask], raw_only=True))
        update(
            self.atomic_cx_edges_cache,
            node_ids[~mask],
            [result[k] for k in node_ids[~mask]],
        )
        return result
