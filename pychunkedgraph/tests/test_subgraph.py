"""Tests for pychunkedgraph.graph.subgraph"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from pychunkedgraph.graph.subgraph import SubgraphProgress, get_subgraph_nodes

from .helpers import create_chunk, to_label
from ..ingest.create.parent_layer import add_parent_chunk


class TestSubgraphProgress:
    def test_init(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        progress = SubgraphProgress(
            graph.meta,
            node_ids=[root],
            return_layers=[2],
            serializable=False,
        )
        assert not progress.done_processing()

    def test_serializable_keys(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

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
