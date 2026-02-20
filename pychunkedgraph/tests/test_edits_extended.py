"""Tests for pychunkedgraph.graph.edits - extended coverage"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from pychunkedgraph.graph.edits import flip_ids
from pychunkedgraph.graph.utils import basetypes

from .helpers import create_chunk, to_label
from ..ingest.create.parent_layer import add_parent_chunk


class TestFlipIds:
    def test_basic(self):
        id_map = {
            np.uint64(1): {np.uint64(10), np.uint64(11)},
            np.uint64(2): {np.uint64(20)},
        }
        result = flip_ids(id_map, [np.uint64(1), np.uint64(2)])
        assert np.uint64(10) in result
        assert np.uint64(11) in result
        assert np.uint64(20) in result

    def test_empty(self):
        id_map = {}
        result = flip_ids(id_map, [])
        assert len(result) == 0


class TestInitOldHierarchy:
    def test_basic(self, gen_graph):
        from pychunkedgraph.graph.edits import _init_old_hierarchy

        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        sv = to_label(graph, 1, 0, 0, 0, 0)
        l2_parent = graph.get_parent(sv)
        result = _init_old_hierarchy(graph, np.array([l2_parent], dtype=np.uint64))
        assert l2_parent in result
        assert 2 in result[l2_parent]
