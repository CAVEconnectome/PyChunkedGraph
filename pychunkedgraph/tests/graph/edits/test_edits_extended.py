"""Tests for pychunkedgraph.graph.edits - extended coverage"""

from math import inf

import numpy as np
import pytest

from pychunkedgraph.graph.edits import flip_ids
from pychunkedgraph.graph import basetypes

from ...helpers import SV, build_graph


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

        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
            edges=[("a0", "a1", 0.5)],
        )

        l2_parent = cg.get_parent(sv["a0"])
        result = _init_old_hierarchy(cg, np.array([l2_parent], dtype=np.uint64))
        assert l2_parent in result
        assert 2 in result[l2_parent]
