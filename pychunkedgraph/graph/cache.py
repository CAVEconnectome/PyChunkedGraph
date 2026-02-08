# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel
"""
Cache nodes, parents, children and cross edges.
"""
import traceback
from collections import defaultdict as defaultd
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
        self._cross_chunk_edges_vec = np.vectorize(
            self.cross_chunk_edges, otypes=[dict]
        )

        # no limit because we don't want to lose new IDs
        self.parents_cache = LRUCache(maxsize=maxsize)
        self.children_cache = LRUCache(maxsize=maxsize)
        self.cross_chunk_edges_cache = LRUCache(maxsize=maxsize)

        self.new_ids = set()

        # Stats tracking for cache hits/misses
        self.stats = {
            "parents": {"hits": 0, "misses": 0, "calls": 0},
            "children": {"hits": 0, "misses": 0, "calls": 0},
            "cross_chunk_edges": {"hits": 0, "misses": 0, "calls": 0},
        }
        # Track where calls/misses come from
        self.sources = defaultd(lambda: defaultd(lambda: {"calls": 0, "misses": 0}))

    def _get_caller(self, skip_frames=2):
        """Get caller info (filename:line:function)."""
        stack = traceback.extract_stack()
        # Skip frames: _get_caller, the cache method, and go to actual caller
        if len(stack) > skip_frames:
            frame = stack[-(skip_frames + 1)]
            return f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
        return "unknown"

    def _record_call(self, cache_type, misses=0):
        """Record a call and its source."""
        caller = self._get_caller(skip_frames=3)
        self.sources[cache_type][caller]["calls"] += 1
        self.sources[cache_type][caller]["misses"] += misses

    def __len__(self):
        return (
            len(self.parents_cache)
            + len(self.children_cache)
            + len(self.cross_chunk_edges_cache)
        )

    def clear(self):
        self.parents_cache.clear()
        self.children_cache.clear()
        self.cross_chunk_edges_cache.clear()

    def get_stats(self):
        """Return stats with hit rates calculated."""
        result = {}
        for name, s in self.stats.items():
            total = s["hits"] + s["misses"]
            hit_rate = s["hits"] / total if total > 0 else 0
            result[name] = {
                **s,
                "total": total,
                "hit_rate": f"{hit_rate:.1%}",
                "sources": dict(self.sources[name]),
            }
        return result

    def reset_stats(self):
        for s in self.stats.values():
            s["hits"] = 0
            s["misses"] = 0
            s["calls"] = 0
        self.sources.clear()

    def parent(self, node_id: np.uint64, *, time_stamp: datetime = None):
        self.stats["parents"]["calls"] += 1
        is_cached = node_id in self.parents_cache
        miss_count = 0 if is_cached else 1
        if is_cached:
            self.stats["parents"]["hits"] += 1
        else:
            self.stats["parents"]["misses"] += 1
        self._record_call("parents", misses=miss_count)

        @cached(cache=self.parents_cache, key=lambda node_id: node_id)
        def parent_decorated(node_id):
            return self._cg.get_parent(node_id, raw_only=True, time_stamp=time_stamp)

        return parent_decorated(node_id)

    def children(self, node_id):
        self.stats["children"]["calls"] += 1
        is_cached = node_id in self.children_cache
        miss_count = 0 if is_cached else 1
        if is_cached:
            self.stats["children"]["hits"] += 1
        else:
            self.stats["children"]["misses"] += 1
        self._record_call("children", misses=miss_count)

        @cached(cache=self.children_cache, key=lambda node_id: node_id)
        def children_decorated(node_id):
            children = self._cg.get_children(node_id, raw_only=True)
            update(self.parents_cache, children, node_id)
            return children

        return children_decorated(node_id)

    def cross_chunk_edges(self, node_id, *, time_stamp: datetime = None):
        self.stats["cross_chunk_edges"]["calls"] += 1
        is_cached = node_id in self.cross_chunk_edges_cache
        miss_count = 0 if is_cached else 1
        if is_cached:
            self.stats["cross_chunk_edges"]["hits"] += 1
        else:
            self.stats["cross_chunk_edges"]["misses"] += 1
        self._record_call("cross_chunk_edges", misses=miss_count)

        @cached(cache=self.cross_chunk_edges_cache, key=lambda node_id: node_id)
        def cross_edges_decorated(node_id):
            edges = self._cg.get_cross_chunk_edges(
                np.array([node_id], dtype=NODE_ID), raw_only=True, time_stamp=time_stamp
            )
            return edges[node_id]

        return cross_edges_decorated(node_id)

    def parents_multiple(
        self,
        node_ids: np.ndarray,
        *,
        time_stamp: datetime = None,
        fail_to_zero: bool = False,
    ):
        node_ids = np.asarray(node_ids, dtype=NODE_ID)
        if not node_ids.size:
            return node_ids
        self.stats["parents"]["calls"] += 1
        mask = np.isin(node_ids, np.fromiter(self.parents_cache.keys(), dtype=NODE_ID))
        hits = int(np.sum(mask))
        misses = len(node_ids) - hits
        self.stats["parents"]["hits"] += hits
        self.stats["parents"]["misses"] += misses
        self._record_call("parents", misses=misses)
        parents = node_ids.copy()
        parents[mask] = self._parent_vec(node_ids[mask])
        parents[~mask] = self._cg.get_parents(
            node_ids[~mask],
            raw_only=True,
            time_stamp=time_stamp,
            fail_to_zero=fail_to_zero,
        )
        update(self.parents_cache, node_ids[~mask], parents[~mask])
        return parents

    def children_multiple(self, node_ids: np.ndarray, *, flatten=False):
        result = {}
        node_ids = np.asarray(node_ids, dtype=NODE_ID)
        if not node_ids.size:
            return result
        self.stats["children"]["calls"] += 1
        mask = np.isin(node_ids, np.fromiter(self.children_cache.keys(), dtype=NODE_ID))
        hits = int(np.sum(mask))
        misses = len(node_ids) - hits
        self.stats["children"]["hits"] += hits
        self.stats["children"]["misses"] += misses
        self._record_call("children", misses=misses)
        cached_children_ = self._children_vec(node_ids[mask])
        result.update({id_: c_ for id_, c_ in zip(node_ids[mask], cached_children_)})
        result.update(self._cg.get_children(node_ids[~mask], raw_only=True))
        update(
            self.children_cache, node_ids[~mask], [result[k] for k in node_ids[~mask]]
        )
        if flatten:
            return np.concatenate([*result.values()])
        return result

    def cross_chunk_edges_multiple(
        self, node_ids: np.ndarray, *, time_stamp: datetime = None
    ):
        result = {}
        node_ids = np.asarray(node_ids, dtype=NODE_ID)
        if not node_ids.size:
            return result
        self.stats["cross_chunk_edges"]["calls"] += 1
        mask = np.isin(
            node_ids, np.fromiter(self.cross_chunk_edges_cache.keys(), dtype=NODE_ID)
        )
        hits = int(np.sum(mask))
        misses = len(node_ids) - hits
        self.stats["cross_chunk_edges"]["hits"] += hits
        self.stats["cross_chunk_edges"]["misses"] += misses
        self._record_call("cross_chunk_edges", misses=misses)
        cached_edges_ = self._cross_chunk_edges_vec(node_ids[mask])
        result.update(
            {id_: edges_ for id_, edges_ in zip(node_ids[mask], cached_edges_)}
        )
        result.update(
            self._cg.get_cross_chunk_edges(
                node_ids[~mask], raw_only=True, time_stamp=time_stamp
            )
        )
        update(
            self.cross_chunk_edges_cache,
            node_ids[~mask],
            [result[k] for k in node_ids[~mask]],
        )
        return result
