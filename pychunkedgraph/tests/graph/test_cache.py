"""Tests for pychunkedgraph.graph.cache"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from pychunkedgraph.graph.cache import CacheService, update

from ..helpers import create_chunk, to_label
from ...ingest.create.parent_layer import add_parent_chunk


class TestUpdate:
    def test_one_to_one(self):
        cache = {}
        update(cache, [1, 2, 3], [10, 20, 30])
        assert cache == {1: 10, 2: 20, 3: 30}

    def test_many_to_one(self):
        cache = {}
        update(cache, [1, 2, 3], 99)
        assert cache == {1: 99, 2: 99, 3: 99}


class TestCacheService:
    def _build_simple_graph(self, gen_graph):
        """Build a simple 2-chunk graph with 2 SVs per chunk."""
        from math import inf

        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    def test_init(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        assert len(cache) == 0

    def test_len(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        cache.parent(sv)
        assert len(cache) >= 1

    def test_clear(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        cache.parent(sv)
        cache.clear()
        assert len(cache) == 0

    def test_parent_miss_then_hit(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)

        # First call is a miss
        parent1 = cache.parent(sv)
        assert cache.stats["parents"]["misses"] == 1

        # Second call is a hit
        parent2 = cache.parent(sv)
        assert cache.stats["parents"]["hits"] == 1
        assert parent1 == parent2

    def test_children_backfills_parent(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        children = cache.children(root)
        assert len(children) > 0
        # Children should be backfilled as parents
        for child in children:
            assert child in cache.parents_cache

    def test_get_stats(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        cache.parent(sv)
        cache.parent(sv)
        stats = cache.get_stats()
        assert "parents" in stats
        assert stats["parents"]["total"] == 2
        assert "hit_rate" in stats["parents"]

    def test_reset_stats(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        cache.parent(sv)
        cache.reset_stats()
        assert cache.stats["parents"]["hits"] == 0
        assert cache.stats["parents"]["misses"] == 0

    def test_parents_multiple_empty(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        result = cache.parents_multiple(np.array([], dtype=np.uint64))
        assert len(result) == 0

    def test_parents_multiple(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        svs = np.array(
            [
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            dtype=np.uint64,
        )
        result = cache.parents_multiple(svs)
        assert len(result) == 2

    def test_children_multiple(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = cache.children_multiple(np.array([root], dtype=np.uint64))
        assert root in result

    def test_children_multiple_flatten(self, gen_graph):
        graph = self._build_simple_graph(gen_graph)
        cache = CacheService(graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = cache.children_multiple(
            np.array([root], dtype=np.uint64), flatten=True
        )
        assert isinstance(result, np.ndarray)
