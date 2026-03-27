"""Tests for pychunkedgraph.ingest.create.parent_layer"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from ..helpers import create_chunk, to_label
from ...ingest.create.parent_layer import add_parent_chunk


class TestAddParentChunk:
    def test_single_thread(self, gen_graph):
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

        # Should not raise
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)

        # Verify parent was created
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parent(sv)
        assert parent is not None
        assert graph.get_chunk_layer(parent) == 2

    def test_multi_chunk(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
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

        # Both SVs should share a root
        root0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        root1 = graph.get_root(to_label(graph, 1, 1, 0, 0, 0))
        assert root0 == root1
