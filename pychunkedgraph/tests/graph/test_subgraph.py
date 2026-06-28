"""Tests for pychunkedgraph.graph.subgraph"""

from math import inf

import numpy as np
import pytest

from pychunkedgraph.graph.subgraph import SubgraphProgress, get_subgraph_nodes

from ..helpers import to_label, build_graph


class TestSubgraphProgress:
    def test_init(self, gen_graph):
        graph, _ = build_graph(gen_graph, chunks=[([(0, 0, 0, 0)], [])])

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        progress = SubgraphProgress(
            graph.meta,
            node_ids=[root],
            return_layers=[2],
            serializable=False,
        )
        assert not progress.done_processing()

    def test_serializable_keys(self, gen_graph):
        graph, _ = build_graph(gen_graph, chunks=[([(0, 0, 0, 0)], [])])

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        progress = SubgraphProgress(
            graph.meta,
            node_ids=[root],
            return_layers=[2],
            serializable=True,
        )
        # Keys should be strings when serializable=True
        key = progress.get_dict_key(root)
        assert isinstance(key, str)


class TestGetSubgraphNodes:
    def _build_graph(self, gen_graph):
        cg, _ = build_graph(
            gen_graph,
            chunks=[
                (
                    [(0, 0, 0, 0), (0, 0, 0, 1)],
                    [((0, 0, 0, 0), (0, 0, 0, 1), 0.5), ((0, 0, 0, 0), (1, 0, 0, 0), inf)],
                ),
                ([(1, 0, 0, 0)], [((1, 0, 0, 0), (0, 0, 0, 0), inf)]),
            ],
        )
        return cg

    def test_single_node(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = get_subgraph_nodes(graph, root)
        assert isinstance(result, dict)
        assert 2 in result

    def test_return_flattened(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = get_subgraph_nodes(graph, root, return_flattened=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_multiple_nodes(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = get_subgraph_nodes(graph, [root])
        assert root in result

    def test_serializable(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = get_subgraph_nodes(graph, root, serializable=True)
        # Keys should be layer ints, values should be arrays
        assert isinstance(result, dict)
