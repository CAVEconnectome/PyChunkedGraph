"""Tests for pychunkedgraph.ingest.create.parent_layer"""

from math import inf

import numpy as np
import pytest

from ..helpers import SV, label, create_chunk, fake_timestamp
from ...ingest.create.parent_layer import add_parent_chunk


class TestAddParentChunk:
    def test_single_thread(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = fake_timestamp()

        create_chunk(
            graph,
            vertices=[label(graph, SV()), label(graph, SV(seg=1))],
            edges=[
                (label(graph, SV()), label(graph, SV(seg=1)), 0.5),
            ],
            timestamp=fake_ts,
        )

        # Should not raise
        add_parent_chunk(graph, 3, [0, 0, 0], n_processes=1)

        # Verify parent was created
        sv = label(graph, SV())
        parent = graph.get_parent(sv)
        assert parent is not None
        assert graph.get_chunk_layer(parent) == 2

    def test_multi_chunk(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = fake_timestamp()

        create_chunk(
            graph,
            vertices=[label(graph, SV())],
            edges=[
                (label(graph, SV()), label(graph, SV(x=1)), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[label(graph, SV(x=1))],
            edges=[
                (label(graph, SV(x=1)), label(graph, SV()), inf),
            ],
            timestamp=fake_ts,
        )

        add_parent_chunk(graph, 3, [0, 0, 0], n_processes=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_processes=1)

        # Both SVs should share a root
        root0 = graph.get_root(label(graph, SV()))
        root1 = graph.get_root(label(graph, SV(x=1)))
        assert root0 == root1
