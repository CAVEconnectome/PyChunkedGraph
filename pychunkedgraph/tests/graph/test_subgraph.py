"""Tests for pychunkedgraph.graph.subgraph"""

from math import inf

import numpy as np

from pychunkedgraph.graph.subgraph import SubgraphProgress, get_subgraph_nodes

from ..helpers import SV, build_graph


class TestSubgraphProgress:
    def test_init(self, gen_graph):
        cg, sv = build_graph(gen_graph, n_layers=4, supervoxels={"a": SV()})
        root = cg.get_root(sv["a"])
        progress = SubgraphProgress(
            cg.meta,
            node_ids=[root],
            return_layers=[2],
            serializable=False,
        )
        assert not progress.done_processing()

    def test_serializable_keys(self, gen_graph):
        cg, sv = build_graph(gen_graph, n_layers=4, supervoxels={"a": SV()})
        root = cg.get_root(sv["a"])
        progress = SubgraphProgress(
            cg.meta,
            node_ids=[root],
            return_layers=[2],
            serializable=True,
        )
        # Keys should be strings when serializable=True
        key = progress.get_dict_key(root)
        assert isinstance(key, str)


class TestGetSubgraphNodes:
    def _build(self, gen_graph):
        return build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "a1": SV(seg=1), "b": SV(x=1)},
            edges=[("a0", "a1", 0.5), ("a0", "b", inf)],
        )

    def test_single_node(self, gen_graph):
        cg, sv = self._build(gen_graph)
        root = cg.get_root(sv["a0"])
        result = get_subgraph_nodes(cg, root)
        assert isinstance(result, dict)
        assert 2 in result

    def test_return_flattened(self, gen_graph):
        cg, sv = self._build(gen_graph)
        root = cg.get_root(sv["a0"])
        result = get_subgraph_nodes(cg, root, return_flattened=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_multiple_nodes(self, gen_graph):
        cg, sv = self._build(gen_graph)
        root = cg.get_root(sv["a0"])
        result = get_subgraph_nodes(cg, [root])
        assert root in result

    def test_serializable(self, gen_graph):
        cg, sv = self._build(gen_graph)
        root = cg.get_root(sv["a0"])
        result = get_subgraph_nodes(cg, root, serializable=True)
        # Keys should be layer ints, values should be arrays
        assert isinstance(result, dict)
