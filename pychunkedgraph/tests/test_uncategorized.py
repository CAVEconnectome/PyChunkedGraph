import collections
import os
import subprocess
import sys
from time import sleep
from datetime import datetime, timedelta
from functools import partial
from math import inf
from signal import SIGTERM
from unittest import mock
from warnings import warn

import numpy as np
import pytest
from google.auth import credentials
from google.cloud import bigtable
from grpc._channel import _Rendezvous

from .helpers import (
    bigtable_emulator,
    create_chunk,
    gen_graph,
    gen_graph_simplequerytest,
    to_label,
    sv_data,
)
from ..graph import types
from ..graph import attributes
from ..graph import exceptions
from ..graph import chunkedgraph
from ..graph.edges import Edges
from ..graph.utils import basetypes
from ..graph.misc import get_delta_roots
from ..graph.cutting import run_multicut
from ..graph.lineage import get_root_id_history
from ..graph.lineage import get_future_root_ids
from ..graph.utils.serializers import serialize_uint64
from ..graph.utils.serializers import deserialize_uint64
from ..ingest.create.parent_layer import add_parent_chunk


class TestGraphNodeConversion:
    @pytest.mark.timeout(30)
    def test_compute_bitmasks(self):
        pass

    @pytest.mark.timeout(30)
    def test_node_conversion(self, gen_graph):
        cg = gen_graph(n_layers=10)

        node_id = cg.get_node_id(np.uint64(4), layer=2, x=3, y=1, z=0)
        assert cg.get_chunk_layer(node_id) == 2
        assert np.all(cg.get_chunk_coordinates(node_id) == np.array([3, 1, 0]))

        chunk_id = cg.get_chunk_id(layer=2, x=3, y=1, z=0)
        assert cg.get_chunk_layer(chunk_id) == 2
        assert np.all(cg.get_chunk_coordinates(chunk_id) == np.array([3, 1, 0]))

        assert cg.get_chunk_id(node_id=node_id) == chunk_id
        assert cg.get_node_id(np.uint64(4), chunk_id=chunk_id) == node_id

    @pytest.mark.timeout(30)
    def test_node_id_adjacency(self, gen_graph):
        cg = gen_graph(n_layers=10)

        assert cg.get_node_id(np.uint64(0), layer=2, x=3, y=1, z=0) + np.uint64(
            1
        ) == cg.get_node_id(np.uint64(1), layer=2, x=3, y=1, z=0)

        assert cg.get_node_id(
            np.uint64(2**53 - 2), layer=10, x=0, y=0, z=0
        ) + np.uint64(1) == cg.get_node_id(
            np.uint64(2**53 - 1), layer=10, x=0, y=0, z=0
        )

    @pytest.mark.timeout(30)
    def test_serialize_node_id(self, gen_graph):
        cg = gen_graph(n_layers=10)

        assert serialize_uint64(
            cg.get_node_id(np.uint64(0), layer=2, x=3, y=1, z=0)
        ) < serialize_uint64(cg.get_node_id(np.uint64(1), layer=2, x=3, y=1, z=0))

        assert serialize_uint64(
            cg.get_node_id(np.uint64(2**53 - 2), layer=10, x=0, y=0, z=0)
        ) < serialize_uint64(
            cg.get_node_id(np.uint64(2**53 - 1), layer=10, x=0, y=0, z=0)
        )

    @pytest.mark.timeout(30)
    def test_deserialize_node_id(self):
        pass

    @pytest.mark.timeout(30)
    def test_serialization_roundtrip(self):
        pass

    @pytest.mark.timeout(30)
    def test_serialize_valid_label_id(self):
        label = np.uint64(0x01FF031234556789)
        assert deserialize_uint64(serialize_uint64(label)) == label


class TestGraphBuild:
    @pytest.mark.timeout(30)
    def test_build_single_node(self, gen_graph):
        """
        Create graph with single RG node 1 in chunk A
        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        cg = gen_graph(n_layers=2)
        # Add Chunk A
        create_chunk(cg, vertices=[to_label(cg, 1, 0, 0, 0, 0)])

        res = cg.client._table.read_rows()
        res.consume_all()

        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 0)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 0, 0, 0, 0))
        assert parent == to_label(cg, 2, 0, 0, 0, 1)

        # Check for the one Level 2 node that should have been created.
        assert serialize_uint64(to_label(cg, 2, 0, 0, 0, 1)) in res.rows
        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([to_label(cg, 2, 0, 0, 0, 1)], dtype=basetypes.NODE_ID)
        )
        attr = attributes.Hierarchy.Child
        row = res.rows[serialize_uint64(to_label(cg, 2, 0, 0, 0, 1))].cells["0"]
        children = attr.deserialize(row[attr.key][0].value)

        for aces in atomic_cross_edge_d.values():
            assert len(aces) == 0

        assert len(children) == 1 and children[0] == to_label(cg, 1, 0, 0, 0, 0)
        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 1 + 1 + 1 + 1 + 1

    @pytest.mark.timeout(30)
    def test_build_single_edge(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        ┌─────┐
        │  A¹ │
        │ 1━2 │
        │     │
        └─────┘
        """

        cg = gen_graph(n_layers=2)

        # Add Chunk A
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5)],
        )

        res = cg.client._table.read_rows()
        res.consume_all()

        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 0)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 0, 0, 0, 0))
        assert parent == to_label(cg, 2, 0, 0, 0, 1)

        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 1)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 0, 0, 0, 1))
        assert parent == to_label(cg, 2, 0, 0, 0, 1)

        # Check for the one Level 2 node that should have been created.
        assert serialize_uint64(to_label(cg, 2, 0, 0, 0, 1)) in res.rows

        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([to_label(cg, 2, 0, 0, 0, 1)], dtype=basetypes.NODE_ID)
        )
        attr = attributes.Hierarchy.Child
        row = res.rows[serialize_uint64(to_label(cg, 2, 0, 0, 0, 1))].cells["0"]
        children = attr.deserialize(row[attr.key][0].value)

        for aces in atomic_cross_edge_d.values():
            assert len(aces) == 0
        assert (
            len(children) == 2
            and to_label(cg, 1, 0, 0, 0, 0) in children
            and to_label(cg, 1, 0, 0, 0, 1) in children
        )

        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 2 + 1 + 1 + 1 + 1

    @pytest.mark.timeout(30)
    def test_build_single_across_edge(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┌─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """

        atomic_chunk_bounds = np.array([2, 1, 1])
        cg = gen_graph(n_layers=3, atomic_chunk_bounds=atomic_chunk_bounds)

        # Chunk A
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf)],
        )

        # Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf)],
        )

        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)
        res = cg.client._table.read_rows()
        res.consume_all()

        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 0)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 0, 0, 0, 0))
        assert parent == to_label(cg, 2, 0, 0, 0, 1)

        assert serialize_uint64(to_label(cg, 1, 1, 0, 0, 0)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
        assert parent == to_label(cg, 2, 1, 0, 0, 1)

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # to_label(cg, 2, 0, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 2, 0, 0, 0, 1)) in res.rows
        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([to_label(cg, 2, 0, 0, 0, 1)], dtype=basetypes.NODE_ID)
        )
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(to_label(cg, 2, 0, 0, 0, 1))
        ]

        attr = attributes.Hierarchy.Child
        row = res.rows[serialize_uint64(to_label(cg, 2, 0, 0, 0, 1))].cells["0"]
        children = attr.deserialize(row[attr.key][0].value)

        test_ace = np.array(
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0)],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert len(children) == 1 and to_label(cg, 1, 0, 0, 0, 0) in children

        # to_label(cg, 2, 1, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 2, 1, 0, 0, 1)) in res.rows
        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([to_label(cg, 2, 1, 0, 0, 1)], dtype=basetypes.NODE_ID)
        )
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(to_label(cg, 2, 1, 0, 0, 1))
        ]

        attr = attributes.Hierarchy.Child
        row = res.rows[serialize_uint64(to_label(cg, 2, 1, 0, 0, 1))].cells["0"]
        children = attr.deserialize(row[attr.key][0].value)

        test_ace = np.array(
            [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0)],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert len(children) == 1 and to_label(cg, 1, 1, 0, 0, 0) in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # to_label(cg, 3, 0, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 3, 0, 0, 0, 1)) in res.rows

        attr = attributes.Hierarchy.Child
        row = res.rows[serialize_uint64(to_label(cg, 3, 0, 0, 0, 1))].cells["0"]
        children = attr.deserialize(row[attr.key][0].value)
        assert (
            len(children) == 2
            and to_label(cg, 2, 0, 0, 0, 1) in children
            and to_label(cg, 2, 1, 0, 0, 1) in children
        )

        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 2 + 2 + 1 + 3 + 1 + 1

    @pytest.mark.timeout(30)
    def test_build_single_edge_and_single_across_edge(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2━1━┿━━3  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
            ],
        )

        # Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf)],
        )

        add_parent_chunk(cg, 3, np.array([0, 0, 0]), n_threads=1)
        res = cg.client._table.read_rows()
        res.consume_all()

        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 0)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 0, 0, 0, 0))
        assert parent == to_label(cg, 2, 0, 0, 0, 1)

        # to_label(cg, 1, 0, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 1)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 0, 0, 0, 1))
        assert parent == to_label(cg, 2, 0, 0, 0, 1)

        # to_label(cg, 1, 1, 0, 0, 0)
        assert serialize_uint64(to_label(cg, 1, 1, 0, 0, 0)) in res.rows
        parent = cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
        assert parent == to_label(cg, 2, 1, 0, 0, 1)

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # to_label(cg, 2, 0, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 2, 0, 0, 0, 1)) in res.rows
        row = res.rows[serialize_uint64(to_label(cg, 2, 0, 0, 0, 1))].cells["0"]
        atomic_cross_edge_d = cg.get_atomic_cross_edges([to_label(cg, 2, 0, 0, 0, 1)])
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(to_label(cg, 2, 0, 0, 0, 1))
        ]
        column = attributes.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array(
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0)],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert (
            len(children) == 2
            and to_label(cg, 1, 0, 0, 0, 0) in children
            and to_label(cg, 1, 0, 0, 0, 1) in children
        )

        # to_label(cg, 2, 1, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 2, 1, 0, 0, 1)) in res.rows
        row = res.rows[serialize_uint64(to_label(cg, 2, 1, 0, 0, 1))].cells["0"]
        atomic_cross_edge_d = cg.get_atomic_cross_edges([to_label(cg, 2, 1, 0, 0, 1)])
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(to_label(cg, 2, 1, 0, 0, 1))
        ]
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array(
            [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0)],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert len(children) == 1 and to_label(cg, 1, 1, 0, 0, 0) in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # to_label(cg, 3, 0, 0, 0, 1)
        assert serialize_uint64(to_label(cg, 3, 0, 0, 0, 1)) in res.rows
        row = res.rows[serialize_uint64(to_label(cg, 3, 0, 0, 0, 1))].cells["0"]
        column = attributes.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        assert (
            len(children) == 2
            and to_label(cg, 2, 0, 0, 0, 1) in children
            and to_label(cg, 2, 1, 0, 0, 1) in children
        )

        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 3 + 2 + 1 + 3 + 1 + 1

    @pytest.mark.timeout(120)
    def test_build_big_graph(self, gen_graph):
        """
        Create graph with RG nodes 1 and 2 in opposite corners of the largest possible dataset
        ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │
        │     │     │     │
        └─────┘     └─────┘
        """

        atomic_chunk_bounds = np.array([8, 8, 8])
        cg = gen_graph(n_layers=5, atomic_chunk_bounds=atomic_chunk_bounds)

        # Preparation: Build Chunk A
        create_chunk(cg, vertices=[to_label(cg, 1, 0, 0, 0, 0)], edges=[])

        # Preparation: Build Chunk Z
        create_chunk(cg, vertices=[to_label(cg, 1, 7, 7, 7, 0)], edges=[])

        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(cg, 3, [3, 3, 3], n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], n_threads=1)
        add_parent_chunk(cg, 5, [0, 0, 0], n_threads=1)

        res = cg.client._table.read_rows()
        res.consume_all()

        assert serialize_uint64(to_label(cg, 1, 0, 0, 0, 0)) in res.rows
        assert serialize_uint64(to_label(cg, 1, 7, 7, 7, 0)) in res.rows
        assert serialize_uint64(to_label(cg, 5, 0, 0, 0, 1)) in res.rows
        assert serialize_uint64(to_label(cg, 5, 0, 0, 0, 2)) in res.rows

    @pytest.mark.timeout(30)
    def test_double_chunk_creation(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘
        """

        atomic_chunk_bounds = np.array([4, 4, 4])
        cg = gen_graph(n_layers=4, atomic_chunk_bounds=atomic_chunk_bounds)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            4,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=2, x=0, y=0, z=0))) == 2
        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=2, x=1, y=0, z=0))) == 1
        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=3, x=0, y=0, z=0))) == 0
        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=4, x=0, y=0, z=0))) == 6

        assert cg.get_chunk_layer(cg.get_root(to_label(cg, 1, 0, 0, 0, 1))) == 4
        assert cg.get_chunk_layer(cg.get_root(to_label(cg, 1, 0, 0, 0, 2))) == 4
        assert cg.get_chunk_layer(cg.get_root(to_label(cg, 1, 1, 0, 0, 1))) == 4

        root_seg_ids = [
            cg.get_segment_id(cg.get_root(to_label(cg, 1, 0, 0, 0, 1))),
            cg.get_segment_id(cg.get_root(to_label(cg, 1, 0, 0, 0, 2))),
            cg.get_segment_id(cg.get_root(to_label(cg, 1, 1, 0, 0, 1))),
        ]

        assert 4 in root_seg_ids
        assert 5 in root_seg_ids
        assert 6 in root_seg_ids


class TestGraphSimpleQueries:
    """
    ┌─────┬─────┬─────┐        L X Y Z S     L X Y Z S     L X Y Z S     L X Y Z S
    │  A¹ │  B¹ │  C¹ │     1: 1 0 0 0 0 ─── 2 0 0 0 1 ───────────────── 4 0 0 0 1
    │  1  │ 3━2━┿━━4  │     2: 1 1 0 0 0 ─┬─ 2 1 0 0 1 ─── 3 0 0 0 1 ─┬─ 4 0 0 0 2
    │     │     │     │     3: 1 1 0 0 1 ─┘                           │
    └─────┴─────┴─────┘     4: 1 2 0 0 0 ─── 2 2 0 0 1 ─── 3 1 0 0 1 ─┘
    """

    @pytest.mark.timeout(30)
    def test_get_parent_and_children(self, gen_graph_simplequerytest):
        cg = gen_graph_simplequerytest

        children10000 = cg.get_children(to_label(cg, 1, 0, 0, 0, 0))
        children11000 = cg.get_children(to_label(cg, 1, 1, 0, 0, 0))
        children11001 = cg.get_children(to_label(cg, 1, 1, 0, 0, 1))
        children12000 = cg.get_children(to_label(cg, 1, 2, 0, 0, 0))

        parent10000 = cg.get_parent(
            to_label(cg, 1, 0, 0, 0, 0),
        )
        parent11000 = cg.get_parent(
            to_label(cg, 1, 1, 0, 0, 0),
        )
        parent11001 = cg.get_parent(
            to_label(cg, 1, 1, 0, 0, 1),
        )
        parent12000 = cg.get_parent(
            to_label(cg, 1, 2, 0, 0, 0),
        )

        children20001 = cg.get_children(to_label(cg, 2, 0, 0, 0, 1))
        children21001 = cg.get_children(to_label(cg, 2, 1, 0, 0, 1))
        children22001 = cg.get_children(to_label(cg, 2, 2, 0, 0, 1))

        parent20001 = cg.get_parent(
            to_label(cg, 2, 0, 0, 0, 1),
        )
        parent21001 = cg.get_parent(
            to_label(cg, 2, 1, 0, 0, 1),
        )
        parent22001 = cg.get_parent(
            to_label(cg, 2, 2, 0, 0, 1),
        )

        children30001 = cg.get_children(to_label(cg, 3, 0, 0, 0, 1))
        # children30002 = cg.get_children(to_label(cg, 3, 0, 0, 0, 2))
        children31001 = cg.get_children(to_label(cg, 3, 1, 0, 0, 1))

        parent30001 = cg.get_parent(
            to_label(cg, 3, 0, 0, 0, 1),
        )
        # parent30002 = cg.get_parent(to_label(cg, 3, 0, 0, 0, 2),  )
        parent31001 = cg.get_parent(
            to_label(cg, 3, 1, 0, 0, 1),
        )

        children40001 = cg.get_children(to_label(cg, 4, 0, 0, 0, 1))
        children40002 = cg.get_children(to_label(cg, 4, 0, 0, 0, 2))

        parent40001 = cg.get_parent(
            to_label(cg, 4, 0, 0, 0, 1),
        )
        parent40002 = cg.get_parent(
            to_label(cg, 4, 0, 0, 0, 2),
        )

        # (non-existing) Children of L1
        assert np.array_equal(children10000, []) is True
        assert np.array_equal(children11000, []) is True
        assert np.array_equal(children11001, []) is True
        assert np.array_equal(children12000, []) is True

        # Parent of L1
        assert parent10000 == to_label(cg, 2, 0, 0, 0, 1)
        assert parent11000 == to_label(cg, 2, 1, 0, 0, 1)
        assert parent11001 == to_label(cg, 2, 1, 0, 0, 1)
        assert parent12000 == to_label(cg, 2, 2, 0, 0, 1)

        # Children of L2
        assert len(children20001) == 1 and to_label(cg, 1, 0, 0, 0, 0) in children20001
        assert (
            len(children21001) == 2
            and to_label(cg, 1, 1, 0, 0, 0) in children21001
            and to_label(cg, 1, 1, 0, 0, 1) in children21001
        )
        assert len(children22001) == 1 and to_label(cg, 1, 2, 0, 0, 0) in children22001

        # Parent of L2
        assert parent20001 == to_label(cg, 4, 0, 0, 0, 1)
        assert parent21001 == to_label(cg, 3, 0, 0, 0, 1)
        assert parent22001 == to_label(cg, 3, 1, 0, 0, 1)

        # Children of L3
        assert len(children30001) == 1 and len(children31001) == 1
        assert to_label(cg, 2, 1, 0, 0, 1) in children30001
        assert to_label(cg, 2, 2, 0, 0, 1) in children31001

        # Parent of L3
        assert parent30001 == parent31001
        assert (
            parent30001 == to_label(cg, 4, 0, 0, 0, 1)
            and parent20001 == to_label(cg, 4, 0, 0, 0, 2)
        ) or (
            parent30001 == to_label(cg, 4, 0, 0, 0, 2)
            and parent20001 == to_label(cg, 4, 0, 0, 0, 1)
        )

        # Children of L4
        assert parent10000 in children40001
        assert parent21001 in children40002 and parent22001 in children40002

        # (non-existing) Parent of L4
        assert parent40001 is None
        assert parent40002 is None

        children2_separate = cg.get_children(
            [
                to_label(cg, 2, 0, 0, 0, 1),
                to_label(cg, 2, 1, 0, 0, 1),
                to_label(cg, 2, 2, 0, 0, 1),
            ]
        )
        assert len(children2_separate) == 3
        assert to_label(cg, 2, 0, 0, 0, 1) in children2_separate and np.all(
            np.isin(children2_separate[to_label(cg, 2, 0, 0, 0, 1)], children20001)
        )
        assert to_label(cg, 2, 1, 0, 0, 1) in children2_separate and np.all(
            np.isin(children2_separate[to_label(cg, 2, 1, 0, 0, 1)], children21001)
        )
        assert to_label(cg, 2, 2, 0, 0, 1) in children2_separate and np.all(
            np.isin(children2_separate[to_label(cg, 2, 2, 0, 0, 1)], children22001)
        )

        children2_combined = cg.get_children(
            [
                to_label(cg, 2, 0, 0, 0, 1),
                to_label(cg, 2, 1, 0, 0, 1),
                to_label(cg, 2, 2, 0, 0, 1),
            ],
            flatten=True,
        )
        assert (
            len(children2_combined) == 4
            and np.all(np.isin(children20001, children2_combined))
            and np.all(np.isin(children21001, children2_combined))
            and np.all(np.isin(children22001, children2_combined))
        )

    @pytest.mark.timeout(30)
    def test_get_root(self, gen_graph_simplequerytest):
        cg = gen_graph_simplequerytest
        root10000 = cg.get_root(
            to_label(cg, 1, 0, 0, 0, 0),
        )
        root11000 = cg.get_root(
            to_label(cg, 1, 1, 0, 0, 0),
        )
        root11001 = cg.get_root(
            to_label(cg, 1, 1, 0, 0, 1),
        )
        root12000 = cg.get_root(
            to_label(cg, 1, 2, 0, 0, 0),
        )

        with pytest.raises(Exception):
            cg.get_root(0)

        assert (
            root10000 == to_label(cg, 4, 0, 0, 0, 1)
            and root11000 == root11001 == root12000 == to_label(cg, 4, 0, 0, 0, 2)
        ) or (
            root10000 == to_label(cg, 4, 0, 0, 0, 2)
            and root11000 == root11001 == root12000 == to_label(cg, 4, 0, 0, 0, 1)
        )

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes(self, gen_graph_simplequerytest):
        cg = gen_graph_simplequerytest
        root1 = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        root2 = cg.get_root(to_label(cg, 1, 1, 0, 0, 0))

        lvl1_nodes_1 = cg.get_subgraph([root1], leaves_only=True)
        lvl1_nodes_2 = cg.get_subgraph([root2], leaves_only=True)
        assert len(lvl1_nodes_1) == 1
        assert len(lvl1_nodes_2) == 3
        assert to_label(cg, 1, 0, 0, 0, 0) in lvl1_nodes_1
        assert to_label(cg, 1, 1, 0, 0, 0) in lvl1_nodes_2
        assert to_label(cg, 1, 1, 0, 0, 1) in lvl1_nodes_2
        assert to_label(cg, 1, 2, 0, 0, 0) in lvl1_nodes_2

        lvl2_parent = cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
        lvl1_nodes = cg.get_subgraph([lvl2_parent], leaves_only=True)
        assert len(lvl1_nodes) == 2
        assert to_label(cg, 1, 1, 0, 0, 0) in lvl1_nodes
        assert to_label(cg, 1, 1, 0, 0, 1) in lvl1_nodes

    @pytest.mark.timeout(30)
    def test_get_subgraph_edges(self, gen_graph_simplequerytest):
        cg = gen_graph_simplequerytest
        root1 = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        root2 = cg.get_root(to_label(cg, 1, 1, 0, 0, 0))

        edges = cg.get_subgraph([root1], edges_only=True)
        assert len(edges) == 0

        edges = cg.get_subgraph([root2], edges_only=True)
        assert [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)] in edges or [
            to_label(cg, 1, 1, 0, 0, 1),
            to_label(cg, 1, 1, 0, 0, 0),
        ] in edges

        assert [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0)] in edges or [
            to_label(cg, 1, 2, 0, 0, 0),
            to_label(cg, 1, 1, 0, 0, 0),
        ] in edges

        lvl2_parent = cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
        edges = cg.get_subgraph([lvl2_parent], edges_only=True)
        assert [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)] in edges or [
            to_label(cg, 1, 1, 0, 0, 1),
            to_label(cg, 1, 1, 0, 0, 0),
        ] in edges

        assert [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0)] in edges or [
            to_label(cg, 1, 2, 0, 0, 0),
            to_label(cg, 1, 1, 0, 0, 0),
        ] in edges

        assert len(edges) == 1

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes_bb(self, gen_graph_simplequerytest):
        cg = gen_graph_simplequerytest
        bb = np.array([[1, 0, 0], [2, 1, 1]], dtype=int)
        bb_coord = bb * cg.meta.graph_config.CHUNK_SIZE
        childs_1 = cg.get_subgraph(
            [cg.get_root(to_label(cg, 1, 1, 0, 0, 1))], bbox=bb, leaves_only=True
        )
        childs_2 = cg.get_subgraph(
            [cg.get_root(to_label(cg, 1, 1, 0, 0, 1))],
            bbox=bb_coord,
            bbox_is_coordinate=True,
            leaves_only=True,
        )
        assert np.all(~(np.sort(childs_1) - np.sort(childs_2)))


class TestGraphMerge:
    @pytest.mark.timeout(30)
    def test_merge_pair_same_chunk(self, gen_graph):
        """
        Add edge between existing RG supervoxels 1 and 2 (same chunk)
        Expected: Same (new) parent for RG 1 and 2 on Layer two
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        atomic_chunk_bounds = np.array([1, 1, 1])
        cg = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Merge
        new_root_ids = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 0)],
            affinities=[0.3],
        ).new_root_ids

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_parent(to_label(cg, 1, 0, 0, 0, 0)) == new_root_id
        assert cg.get_parent(to_label(cg, 1, 0, 0, 0, 1)) == new_root_id
        leaves = np.unique(cg.get_subgraph([new_root_id], leaves_only=True))
        assert len(leaves) == 2
        assert to_label(cg, 1, 0, 0, 0, 0) in leaves
        assert to_label(cg, 1, 0, 0, 0, 1) in leaves

    @pytest.mark.timeout(30)
    def test_merge_pair_neighboring_chunks(self, gen_graph):
        """
        Add edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        # Merge
        new_root_ids = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0)],
            affinities=0.3,
        ).new_root_ids

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root_id
        assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == new_root_id
        leaves = np.unique(cg.get_subgraph([new_root_id], leaves_only=True))
        assert len(leaves) == 2
        assert to_label(cg, 1, 0, 0, 0, 0) in leaves
        assert to_label(cg, 1, 1, 0, 0, 0) in leaves

    @pytest.mark.timeout(120)
    def test_merge_pair_disconnected_chunks(self, gen_graph):
        """
        Add edge between existing RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │     │     │     │      │     │     │     │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        cg = gen_graph(n_layers=5)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk Z
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 7, 7, 7, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            3,
            [3, 3, 3],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            4,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            5,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        # Merge
        result = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 7, 7, 7, 0), to_label(cg, 1, 0, 0, 0, 0)],
            affinities=[0.3],
        )
        new_root_ids, lvl2_node_ids = result.new_root_ids, result.new_lvl2_ids
        print(f"lvl2_node_ids: {lvl2_node_ids}")

        u_layers = np.unique(cg.get_chunk_layers(lvl2_node_ids))
        assert len(u_layers) == 1
        assert u_layers[0] == 2

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root_id
        assert cg.get_root(to_label(cg, 1, 7, 7, 7, 0)) == new_root_id
        leaves = np.unique(cg.get_subgraph(new_root_id, leaves_only=True))
        assert len(leaves) == 2
        assert to_label(cg, 1, 0, 0, 0, 0) in leaves
        assert to_label(cg, 1, 7, 7, 7, 0) in leaves

    @pytest.mark.timeout(30)
    def test_merge_pair_already_connected(self, gen_graph):
        """
        Add edge between already connected RG supervoxels 1 and 2 (same chunk).
        Expected: No change, i.e. same parent (to_label(cg, 2, 0, 0, 0, 1)), affinity (0.5) and timestamp as before
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        cg = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5)],
            timestamp=fake_timestamp,
        )

        res_old = cg.client._table.read_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 0)],
            )
        res_new = cg.client._table.read_rows()
        res_new.consume_all()

        # Check
        if res_old.rows != res_new.rows:
            warn(
                "Rows were modified when merging a pair of already connected supervoxels. "
                "While probably not an error, it is an unnecessary operation."
            )

    @pytest.mark.timeout(30)
    def test_merge_triple_chain_to_full_circle_same_chunk(self, gen_graph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """

        cg = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[
                to_label(cg, 1, 0, 0, 0, 0),
                to_label(cg, 1, 0, 0, 0, 1),
                to_label(cg, 1, 0, 0, 0, 2),
            ],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 2), 0.5),
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2), 0.5),
            ],
            timestamp=fake_timestamp,
        )

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 0)],
                affinities=0.3,
            ).new_root_ids

    @pytest.mark.timeout(30)
    def test_merge_triple_chain_to_full_circle_neighboring_chunks(self, gen_graph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), inf)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0)],
                affinities=1.0,
            ).new_root_ids

    @pytest.mark.timeout(120)
    def test_merge_triple_chain_to_full_circle_disconnected_chunks(self, gen_graph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        cg = gen_graph(n_layers=5)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
                (
                    to_label(cg, 1, 0, 0, 0, 1),
                    to_label(cg, 1, 7, 7, 7, 0),
                    inf,
                ),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 7, 7, 7, 0)],
            edges=[
                (
                    to_label(cg, 1, 7, 7, 7, 0),
                    to_label(cg, 1, 0, 0, 0, 1),
                    inf,
                )
            ],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            3,
            [3, 3, 3],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            4,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            4,
            [1, 1, 1],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            5,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        # Merge
        new_root_ids = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 7, 7, 7, 0), to_label(cg, 1, 0, 0, 0, 0)],
            affinities=1.0,
        ).new_root_ids

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root_id
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 1)) == new_root_id
        assert cg.get_root(to_label(cg, 1, 7, 7, 7, 0)) == new_root_id
        leaves = np.unique(cg.get_subgraph(new_root_id, leaves_only=True))
        assert len(leaves) == 3
        assert to_label(cg, 1, 0, 0, 0, 0) in leaves
        assert to_label(cg, 1, 0, 0, 0, 1) in leaves
        assert to_label(cg, 1, 7, 7, 7, 0) in leaves

    @pytest.mark.timeout(30)
    def test_merge_same_node(self, gen_graph):
        """
        Try to add loop edge between RG supervoxel 1 and itself
        ┌─────┐
        │  A¹ │
        │  1  │  =>  Reject
        │     │
        └─────┘
        """

        cg = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        res_old = cg.client._table.read_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0)],
            )

        res_new = cg.client._table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_merge_pair_abstract_nodes(self, gen_graph):
        """
        Try to add edge between RG supervoxel 1 and abstract node "2"
                    ┌─────┐
                    │  B² │
                    │ "2" │
                    │     │
                    └─────┘
        ┌─────┐              =>  Reject
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        res_old = cg.client._table.read_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 2, 1, 0, 0, 1)],
            )

        res_new = cg.client._table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_diagonal_connections(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2 1━┿━━3  │
        │  /  │     │
        ┌─────┬─────┐
        │  |  │     │
        │  4━━┿━━5  │
        │  C¹ │  D¹ │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 1, 0, 0), inf),
            ],
        )

        # Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf)],
        )

        # Chunk C
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 1, 0, 0)],
            edges=[
                (to_label(cg, 1, 0, 1, 0, 0), to_label(cg, 1, 1, 1, 0, 0), inf),
                (to_label(cg, 1, 0, 1, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf),
            ],
        )

        # Chunk D
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 1, 0, 0)],
            edges=[(to_label(cg, 1, 1, 1, 0, 0), to_label(cg, 1, 0, 1, 0, 0), inf)],
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            n_threads=1,
        )

        rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=3, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())

        assert len(root_ids_t0) == 2

        child_ids = []
        for root_id in root_ids_t0:
            child_ids.extend(cg.get_subgraph(root_id, leaves_only=True))

        new_roots = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            affinities=[0.5],
        ).new_root_ids

        root_ids = []
        for child_id in child_ids:
            root_ids.append(cg.get_root(child_id))

        assert len(np.unique(root_ids)) == 1

        root_id = root_ids[0]
        assert root_id == new_roots[0]

    @pytest.mark.timeout(240)
    def test_cross_edges(self, gen_graph):
        """"""

        cg = gen_graph(n_layers=5)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[
                to_label(cg, 1, 0, 0, 0, 0),
                to_label(cg, 1, 0, 0, 0, 1),
            ],
            edges=[
                (
                    to_label(cg, 1, 0, 0, 0, 1),
                    to_label(cg, 1, 1, 0, 0, 0),
                    inf,
                ),
                (
                    to_label(cg, 1, 0, 0, 0, 0),
                    to_label(cg, 1, 0, 0, 0, 1),
                    inf,
                ),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[
                to_label(cg, 1, 1, 0, 0, 0),
                to_label(cg, 1, 1, 0, 0, 1),
            ],
            edges=[
                (
                    to_label(cg, 1, 1, 0, 0, 0),
                    to_label(cg, 1, 0, 0, 0, 1),
                    inf,
                ),
                (
                    to_label(cg, 1, 1, 0, 0, 0),
                    to_label(cg, 1, 1, 0, 0, 1),
                    inf,
                ),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk C
        create_chunk(
            cg,
            vertices=[
                to_label(cg, 1, 2, 0, 0, 0),
            ],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            3,
            [1, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            4,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        add_parent_chunk(
            cg,
            5,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        new_roots = cg.add_edges(
            "Jane Doe",
            [
                to_label(cg, 1, 1, 0, 0, 0),
                to_label(cg, 1, 2, 0, 0, 0),
            ],
            affinities=0.9,
        ).new_root_ids

        assert len(new_roots) == 1


class TestGraphMergeSplit:
    @pytest.mark.timeout(240)
    def test_multiple_cuts_and_splits(self, gen_graph_simplequerytest):
        """
        ┌─────┬─────┬─────┐        L X Y Z S     L X Y Z S     L X Y Z S     L X Y Z S
        │  A¹ │  B¹ │  C¹ │     1: 1 0 0 0 0 ─── 2 0 0 0 1 ───────────────── 4 0 0 0 1
        │  1  │ 3━2━┿━━4  │     2: 1 1 0 0 0 ─┬─ 2 1 0 0 1 ─── 3 0 0 0 1 ─┬─ 4 0 0 0 2
        │     │     │     │     3: 1 1 0 0 1 ─┘                           │
        └─────┴─────┴─────┘     4: 1 2 0 0 0 ─── 2 2 0 0 1 ─── 3 1 0 0 1 ─┘
        """
        cg = gen_graph_simplequerytest

        rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=4, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())
        child_ids = [types.empty_1d]
        for root_id in root_ids_t0:
            child_ids.append(cg.get_subgraph([root_id], leaves_only=True))
        child_ids = np.concatenate(child_ids)

        for i in range(10):
            print(f"\n\nITERATION {i}/10 - MERGE 1 & 3")
            new_roots = cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
                affinities=0.9,
            ).new_root_ids
            assert len(new_roots) == 1, new_roots
            assert len(cg.get_subgraph([new_roots[0]], leaves_only=True)) == 4

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            print(child_ids)
            print(list(root_ids))
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 1, u_root_ids

            # ------------------------------------------------------------------
            print(f"\n\nITERATION {i}/10 - SPLIT 2 & 3")
            new_roots = cg.remove_edges(
                "John Doe",
                source_ids=to_label(cg, 1, 1, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 1),
                mincut=False,
            ).new_root_ids
            assert len(new_roots) == 2, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            print(child_ids)
            print(list(root_ids))
            u_root_ids = np.unique(root_ids)
            these_child_ids = []
            for root_id in u_root_ids:
                these_child_ids.extend(cg.get_subgraph([root_id], leaves_only=True))

            assert len(these_child_ids) == 4
            assert len(u_root_ids) == 2, u_root_ids

            # ------------------------------------------------------------------
            print(f"\n\nITERATION {i}/10 - SPLIT 1 & 3")
            new_roots = cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 1),
                mincut=False,
            ).new_root_ids
            assert len(new_roots) == 2, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            print(child_ids)
            print(list(root_ids))
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 3, u_root_ids

            # ------------------------------------------------------------------
            print(f"\n\nITERATION {i}/10 - MERGE 2 & 3")
            new_roots = cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
                affinities=0.9,
            ).new_root_ids
            assert len(new_roots) == 1, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            print(child_ids)
            print(list(root_ids))
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 2, u_root_ids

            # for root_id in root_ids:
            #     cross_edge_dict_layers = graph_tests.root_cross_edge_test(
            #         root_id, cg=cg
            #     )  # dict: layer -> cross_edge_dict
            #     n_cross_edges_layer = collections.defaultdict(list)

            #     for child_layer in cross_edge_dict_layers.keys():
            #         for layer in cross_edge_dict_layers[child_layer].keys():
            #             n_cross_edges_layer[layer].append(
            #                 len(cross_edge_dict_layers[child_layer][layer])
            #             )

            #     for layer in n_cross_edges_layer.keys():
            #         assert len(np.unique(n_cross_edges_layer[layer])) == 1


class TestGraphMinCut:
    # TODO: Ideally, those tests should focus only on mincut retrieving the correct edges.
    #       The edge removal part should be tested exhaustively in TestGraphSplit
    @pytest.mark.timeout(30)
    def test_cut_regular_link(self, gen_graph):
        """
        Regular link between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        # Mincut
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            source_coords=[0, 0, 0],
            sink_coords=[
                2 * cg.meta.graph_config.CHUNK_SIZE[0],
                2 * cg.meta.graph_config.CHUNK_SIZE[1],
                cg.meta.graph_config.CHUNK_SIZE[2],
            ],
            mincut=True,
            disallow_isolating_cut=True,
        ).new_root_ids

        # Check New State
        assert len(new_root_ids) == 2
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) != cg.get_root(
            to_label(cg, 1, 1, 0, 0, 0)
        )
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(to_label(cg, 1, 0, 0, 0, 0))], leaves_only=True
            )
        )
        assert len(leaves) == 1 and to_label(cg, 1, 0, 0, 0, 0) in leaves
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(to_label(cg, 1, 1, 0, 0, 0))], leaves_only=True
            )
        )
        assert len(leaves) == 1 and to_label(cg, 1, 1, 0, 0, 0) in leaves

    @pytest.mark.timeout(30)
    def test_cut_no_link(self, gen_graph):
        """
        No connection between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        res_old = cg.client._table.read_rows()
        res_old.consume_all()

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 0),
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        res_new = cg.client._table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_cut_old_link(self, gen_graph):
        """
        Link between 1 and 2 got removed previously (aff = 0.0)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1┅┅╎┅┅2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        cg.remove_edges(
            "John Doe",
            source_ids=to_label(cg, 1, 1, 0, 0, 0),
            sink_ids=to_label(cg, 1, 0, 0, 0, 0),
            mincut=False,
        )

        res_old = cg.client._table.read_rows()
        res_old.consume_all()

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 0),
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        res_new = cg.client._table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_cut_indivisible_link(self, gen_graph):
        """
        Sink: 1, Source: 2
        Link between 1 and 2 is set to `inf` and must not be cut.
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1══╪══2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        original_parents_1 = cg.get_root(
            to_label(cg, 1, 0, 0, 0, 0), get_all_parents=True
        )
        original_parents_2 = cg.get_root(
            to_label(cg, 1, 1, 0, 0, 0), get_all_parents=True
        )

        # Mincut
        with pytest.raises(exceptions.PostconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 0),
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        new_parents_1 = cg.get_root(to_label(cg, 1, 0, 0, 0, 0), get_all_parents=True)
        new_parents_2 = cg.get_root(to_label(cg, 1, 1, 0, 0, 0), get_all_parents=True)

        assert np.all(np.array(original_parents_1) == np.array(new_parents_1))
        assert np.all(np.array(original_parents_2) == np.array(new_parents_2))

    @pytest.mark.timeout(30)
    def test_mincut_disrespects_sources_or_sinks(self, gen_graph):
        """
        When the mincut separates sources or sinks, an error should be thrown.
        Although the mincut is setup to never cut an edge between two sources or
        two sinks, this can happen when an edge along the only path between two
        sources or two sinks is cut.
        """
        cg = gen_graph(n_layers=2)

        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[
                to_label(cg, 1, 0, 0, 0, 0),
                to_label(cg, 1, 0, 0, 0, 1),
                to_label(cg, 1, 0, 0, 0, 2),
                to_label(cg, 1, 0, 0, 0, 3),
            ],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 2), 2),
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2), 3),
                (to_label(cg, 1, 0, 0, 0, 2), to_label(cg, 1, 0, 0, 0, 3), 10),
            ],
            timestamp=fake_timestamp,
        )

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
                sink_ids=[to_label(cg, 1, 0, 0, 0, 3)],
                source_coords=[[0, 0, 0], [10, 0, 0]],
                sink_coords=[[5, 5, 0]],
                mincut=True,
            )


class TestGraphMultiCut:
    @pytest.mark.timeout(30)
    def test_cut_multi_tree(self, gen_graph):
        pass

    @pytest.mark.timeout(30)
    def test_path_augmented_multicut(self, sv_data):
        sv_edges, sv_sources, sv_sinks, sv_affinity, sv_area = sv_data
        edges = Edges(
            sv_edges[:, 0], sv_edges[:, 1], affinities=sv_affinity, areas=sv_area
        )

        cut_edges_aug = run_multicut(edges, sv_sources, sv_sinks, path_augment=True)
        assert cut_edges_aug.shape[0] == 350

        with pytest.raises(exceptions.PreconditionError):
            run_multicut(edges, sv_sources, sv_sinks, path_augment=False)
        pass


class TestGraphHistory:
    """These test inadvertantly also test merge and split operations"""

    @pytest.mark.timeout(120)
    def test_cut_merge_history(self, gen_graph):
        """
        Regular link between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        (1) Split 1 and 2
        (2) Merge 1 and 2
        """
        from ..graph.lineage import lineage_graph

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        first_root = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        assert first_root == cg.get_root(to_label(cg, 1, 1, 0, 0, 0))
        timestamp_before_split = datetime.utcnow()
        split_roots = cg.remove_edges(
            "Jane Doe",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        ).new_root_ids
        assert len(split_roots) == 2
        g = lineage_graph(cg, split_roots[0])
        assert g.size() == 1
        g = lineage_graph(cg, split_roots)
        assert g.size() == 2

        timestamp_after_split = datetime.utcnow()
        merge_roots = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0)],
            affinities=0.4,
        ).new_root_ids
        assert len(merge_roots) == 1
        merge_root = merge_roots[0]
        timestamp_after_merge = datetime.utcnow()

        g = lineage_graph(cg, merge_roots)
        assert g.size() == 4
        assert (
            len(
                get_root_id_history(
                    cg,
                    first_root,
                    time_stamp_past=datetime.min,
                    time_stamp_future=datetime.max,
                )
            )
            == 4
        )
        assert (
            len(
                get_root_id_history(
                    cg,
                    split_roots[0],
                    time_stamp_past=datetime.min,
                    time_stamp_future=datetime.max,
                )
            )
            == 3
        )
        assert (
            len(
                get_root_id_history(
                    cg,
                    split_roots[1],
                    time_stamp_past=datetime.min,
                    time_stamp_future=datetime.max,
                )
            )
            == 3
        )
        assert (
            len(
                get_root_id_history(
                    cg,
                    merge_root,
                    time_stamp_past=datetime.min,
                    time_stamp_future=datetime.max,
                )
            )
            == 4
        )

        new_roots, old_roots = get_delta_roots(
            cg, timestamp_before_split, timestamp_after_split
        )
        assert len(old_roots) == 1
        assert old_roots[0] == first_root
        assert len(new_roots) == 2
        assert np.all(np.isin(new_roots, split_roots))

        new_roots2, old_roots2 = get_delta_roots(
            cg, timestamp_after_split, timestamp_after_merge
        )
        assert len(new_roots2) == 1
        assert new_roots2[0] == merge_root
        assert len(old_roots2) == 2
        assert np.all(np.isin(old_roots2, split_roots))

        new_roots3, old_roots3 = get_delta_roots(
            cg, timestamp_before_split, timestamp_after_merge
        )
        assert len(new_roots3) == 1
        assert new_roots3[0] == merge_root
        assert len(old_roots3) == 1
        assert old_roots3[0] == first_root


class TestGraphLocks:
    @pytest.mark.timeout(30)
    def test_lock_unlock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try unlock (opid = 1)
        (4) Try lock (opid = 2)
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_root(root_id=root_id, operation_id=operation_id_1)

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    @pytest.mark.timeout(30)
    def test_lock_expiration(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """
        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds())

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
            max_tries=10,
            waittime_s=0.5,
        )[0]

    @pytest.mark.timeout(30)
    def test_lock_renew(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.renew_locks(root_ids=[root_id], operation_id=operation_id_1)

    @pytest.mark.timeout(30)
    def test_lock_merge_lock_old_id(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Merge (includes lock opid 1)
        (2) Try lock opid 2 --> should be successful and return new root id
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        new_root_ids = cg.add_edges(
            "Chuck Norris",
            [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            affinities=1.0,
        ).new_root_ids

        assert new_root_ids is not None

        operation_id_2 = cg.id_client.create_operation_id()
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        success, new_root_id = cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
            max_tries=10,
            waittime_s=0.5,
        )

        assert success
        assert new_root_ids[0] == new_root_id

    @pytest.mark.timeout(30)
    def test_indefinite_lock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try indefinite lock (opid = 1), get indefinite lock
        (2) Try normal lock (opid = 2), doesn't get the normal lock
        (3) Try unlock indefinite lock (opid = 1), should unlock indefinite lock
        (4) Try lock (opid = 2), should get the normal lock
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots_indefinitely(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_indefinitely_locked_root(
            root_id=root_id, operation_id=operation_id_1
        )

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    @pytest.mark.timeout(30)
    def test_indefinite_lock_with_normal_lock_expiration(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try normal lock (opid = 1), get normal lock
        (2) Try indefinite lock (opid = 1), get indefinite lock
        (3) Wait until normal lock expires
        (4) Try normal lock (opid = 2), doesn't get the normal lock
        (5) Try unlock indefinite lock (opid = 1), should unlock indefinite lock
        (6) Try lock (opid = 2), should get the normal lock
        """

        # 1. TODO renew lock test when getting indefinite lock
        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.lock_roots_indefinitely(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds())

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_indefinitely_locked_root(
            root_id=root_id, operation_id=operation_id_1
        )

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    # TODO fixme: this scenario can't be tested like this
    # @pytest.mark.timeout(30)
    # def test_normal_lock_expiration(self, gen_graph):
    #     """
    #     No connection between 1, 2 and 3
    #     ┌─────┬─────┐
    #     │  A¹ │  B¹ │
    #     │  1  │  3  │
    #     │  2  │     │
    #     └─────┴─────┘

    #     (1) Try normal lock (opid = 1), get normal lock
    #     (2) Wait until normal lock expires
    #     (3) Try indefinite lock (opid = 1), doesn't get the indefinite lock
    #     """

    #     cg = gen_graph(n_layers=3)

    #     # Preparation: Build Chunk A
    #     fake_timestamp = datetime.utcnow() - timedelta(days=10)
    #     create_chunk(
    #         cg,
    #         vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
    #         edges=[],
    #         timestamp=fake_timestamp,
    #     )

    #     # Preparation: Build Chunk B
    #     create_chunk(
    #         cg,
    #         vertices=[to_label(cg, 1, 1, 0, 0, 1)],
    #         edges=[],
    #         timestamp=fake_timestamp,
    #     )

    #     add_layer(
    #         cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1,
    #     )

    #     operation_id_1 = cg.id_client.create_operation_id()
    #     root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

    #     future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}

    #     assert cg.client.lock_roots(
    #         root_ids=[root_id],
    #         operation_id=operation_id_1,
    #         future_root_ids_d=future_root_ids_d,
    #     )[0]

    #     sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds()+1)

    #     assert not cg.client.lock_roots_indefinitely(
    #         root_ids=[root_id],
    #         operation_id=operation_id_1,
    #         future_root_ids_d=future_root_ids_d,
    #     )[0]


# class MockChunkedGraph:
#     """
#     Dummy class to mock partial functionality of the ChunkedGraph for use in unit tests.
#     Feel free to add more functions as need be. Can pass in alternative member functions into constructor.
#     """

#     def __init__(
#         self, get_chunk_coordinates=None, get_chunk_layer=None, get_chunk_id=None
#     ):
#         if get_chunk_coordinates is not None:
#             self.get_chunk_coordinates = get_chunk_coordinates
#         if get_chunk_layer is not None:
#             self.get_chunk_layer = get_chunk_layer
#         if get_chunk_id is not None:
#             self.get_chunk_id = get_chunk_id

#     def get_chunk_coordinates(self, chunk_id):  # pylint: disable=method-hidden
#         return np.array([0, 0, 0])

#     def get_chunk_layer(self, chunk_id):  # pylint: disable=method-hidden
#         return 2

#     def get_chunk_id(self, *args):  # pylint: disable=method-hidden
#         return 0


# class TestGraphSplit:
#     @pytest.mark.timeout(30)
#     def test_split_pair_same_chunk(self, gen_graph):
#         """
#         Remove edge between existing RG supervoxels 1 and 2 (same chunk)
#         Expected: Different (new) parents for RG 1 and 2 on Layer two
#         ┌─────┐      ┌─────┐
#         │  A¹ │      │  A¹ │
#         │ 1━2 │  =>  │ 1 2 │
#         │     │      │     │
#         └─────┘      └─────┘
#         """

#         cg = gen_graph(n_layers=2)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
#             edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5)],
#             timestamp=fake_timestamp,
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 0, 0, 0, 1),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#             mincut=False,
#         ).new_root_ids

#         # Check New State
#         assert len(new_root_ids) == 2
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) != cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 1)
#         )
#         leaves = np.unique(
#             cg.get_subgraph([cg.get_root(to_label(cg, 1, 0, 0, 0, 0))], leaves_only=True)
#         )
#         assert len(leaves) == 1 and to_label(cg, 1, 0, 0, 0, 0) in leaves
#         leaves = np.unique(
#             cg.get_subgraph([cg.get_root(to_label(cg, 1, 0, 0, 0, 1))], leaves_only=True)
#         )
#         assert len(leaves) == 1 and to_label(cg, 1, 0, 0, 0, 1) in leaves

#         # Check Old State still accessible
#         assert cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp
#         ) == cg.get_root(to_label(cg, 1, 0, 0, 0, 1), time_stamp=fake_timestamp)
#         leaves = np.unique(
#             cg.get_subgraph(
#                 [cg.get_root(to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)],
#                 leaves_only=True,
#             )
#         )
#         assert len(leaves) == 2
#         assert to_label(cg, 1, 0, 0, 0, 0) in leaves
#         assert to_label(cg, 1, 0, 0, 0, 1) in leaves

#         # assert len(cg.get_latest_roots()) == 2
#         # assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     def test_split_nonexisting_edge(self, gen_graph):
#         """
#         Remove edge between existing RG supervoxels 1 and 2 (same chunk)
#         Expected: Different (new) parents for RG 1 and 2 on Layer two
#         ┌─────┐      ┌─────┐
#         │  A¹ │      │  A¹ │
#         │ 1━2 │  =>  │ 1━2 │
#         │   | │      │   | │
#         │   3 │      │   3 │
#         └─────┘      └─────┘
#         """

#         cg = gen_graph(n_layers=2)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
#             edges=[
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 2), to_label(cg, 1, 0, 0, 0, 1), 0.5),
#             ],
#             timestamp=fake_timestamp,
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 0, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 2),
#             mincut=False,
#         ).new_root_ids

#         assert len(new_root_ids) == 1

#     @pytest.mark.timeout(30)
#     def test_split_pair_neighboring_chunks(self, gen_graph):
#         """
#         Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
#         ┌─────┬─────┐      ┌─────┬─────┐
#         │  A¹ │  B¹ │      │  A¹ │  B¹ │
#         │  1━━┿━━2  │  =>  │  1  │  2  │
#         │     │     │      │     │     │
#         └─────┴─────┘      └─────┴─────┘
#         """

#         cg = gen_graph(n_layers=3)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0)],
#             edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 1.0)],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk B
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 1, 0, 0, 0)],
#             edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 1.0)],
#             timestamp=fake_timestamp,
#         )

#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 1, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#             mincut=False,
#         ).new_root_ids

#         # Check New State
#         assert len(new_root_ids) == 2
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) != cg.get_root(
#             to_label(cg, 1, 1, 0, 0, 0)
#         )
#         leaves = np.unique(
#             cg.get_subgraph([cg.get_root(to_label(cg, 1, 0, 0, 0, 0))], leaves_only=True)
#         )
#         assert len(leaves) == 1 and to_label(cg, 1, 0, 0, 0, 0) in leaves
#         leaves = np.unique(
#             cg.get_subgraph([cg.get_root(to_label(cg, 1, 1, 0, 0, 0))], leaves_only=True)
#         )
#         assert len(leaves) == 1 and to_label(cg, 1, 1, 0, 0, 0) in leaves

#         # Check Old State still accessible
#         assert cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp
#         ) == cg.get_root(to_label(cg, 1, 1, 0, 0, 0), time_stamp=fake_timestamp)
#         leaves = np.unique(
#             cg.get_subgraph(
#                 [cg.get_root(to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)],
#                 leaves_only=True,
#             )
#         )
#         assert len(leaves) == 2
#         assert to_label(cg, 1, 0, 0, 0, 0) in leaves
#         assert to_label(cg, 1, 1, 0, 0, 0) in leaves

#         assert len(cg.get_latest_roots()) == 2
#         assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     @pytest.mark.timeout(30)
#     def test_split_verify_cross_chunk_edges(self, gen_graph):
#         """
#         Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
#         ┌─────┬─────┬─────┐      ┌─────┬─────┬─────┐
#         |     │  A¹ │  B¹ │      |     │  A¹ │  B¹ │
#         |     │  1━━┿━━3  │  =>  |     │  1━━┿━━3  │
#         |     │  |  │     │      |     │     │     │
#         |     │  2  │     │      |     │  2  │     │
#         └─────┴─────┴─────┘      └─────┴─────┴─────┘
#         """

#         cg = gen_graph(n_layers=4)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
#             edges=[
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0), inf),
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1), 0.5),
#             ],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk B
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 2, 0, 0, 0)],
#             edges=[(to_label(cg, 1, 2, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf)],
#             timestamp=fake_timestamp,
#         )

#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             4,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )

#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 1, 0, 0, 1)
#         )
#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 2, 0, 0, 0)
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 1, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 1, 0, 0, 1),
#             mincut=False,
#         ).new_root_ids

#         assert len(new_root_ids) == 2

#         svs2 = cg.get_subgraph([new_root_ids[0]], leaves_only=True)
#         svs1 = cg.get_subgraph([new_root_ids[1]], leaves_only=True)
#         len_set = {1, 2}
#         assert len(svs1) in len_set
#         len_set.remove(len(svs1))
#         assert len(svs2) in len_set

#         # Check New State
#         assert len(new_root_ids) == 2
#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) != cg.get_root(
#             to_label(cg, 1, 1, 0, 0, 1)
#         )
#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 2, 0, 0, 0)
#         )

#         cc_dict = cg.get_atomic_cross_edges(
#             cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
#         )
#         assert len(cc_dict[3]) == 1
#         assert cc_dict[3][0][0] == to_label(cg, 1, 1, 0, 0, 0)
#         assert cc_dict[3][0][1] == to_label(cg, 1, 2, 0, 0, 0)

#         assert len(cg.get_latest_roots()) == 2
#         assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     @pytest.mark.timeout(30)
#     def test_split_verify_loop(self, gen_graph):
#         """
#         Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
#         ┌─────┬────────┬─────┐      ┌─────┬────────┬─────┐
#         |     │     A¹ │  B¹ │      |     │     A¹ │  B¹ │
#         |     │  4━━1━━┿━━5  │  =>  |     │  4  1━━┿━━5  │
#         |     │   /    │  |  │      |     │        │  |  │
#         |     │  3  2━━┿━━6  │      |     │  3  2━━┿━━6  │
#         └─────┴────────┴─────┘      └─────┴────────┴─────┘
#         """

#         cg = gen_graph(n_layers=4)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[
#                 to_label(cg, 1, 1, 0, 0, 0),
#                 to_label(cg, 1, 1, 0, 0, 1),
#                 to_label(cg, 1, 1, 0, 0, 2),
#                 to_label(cg, 1, 1, 0, 0, 3),
#             ],
#             edges=[
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0), inf),
#                 (to_label(cg, 1, 1, 0, 0, 1), to_label(cg, 1, 2, 0, 0, 1), inf),
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 2), 0.5),
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 3), 0.5),
#             ],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk B
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 2, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 1)],
#             edges=[
#                 (to_label(cg, 1, 2, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
#                 (to_label(cg, 1, 2, 0, 0, 1), to_label(cg, 1, 1, 0, 0, 1), inf),
#                 (to_label(cg, 1, 2, 0, 0, 1), to_label(cg, 1, 2, 0, 0, 0), 0.5),
#             ],
#             timestamp=fake_timestamp,
#         )

#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             4,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )

#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 1, 0, 0, 1)
#         )
#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 2, 0, 0, 0)
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 1, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 1, 0, 0, 2),
#             mincut=False,
#         ).new_root_ids

#         assert len(new_root_ids) == 2

#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 1, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 1, 0, 0, 3),
#             mincut=False,
#         ).new_root_ids

#         assert len(new_root_ids) == 2

#         cc_dict = cg.get_atomic_cross_edges(
#             cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
#         )
#         assert len(cc_dict[3]) == 1
#         cc_dict = cg.get_atomic_cross_edges(
#             cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
#         )
#         assert len(cc_dict[3]) == 1

#         assert len(cg.get_latest_roots()) == 3
#         assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     @pytest.mark.timeout(30)
#     def test_split_pair_disconnected_chunks(self, gen_graph):
#         """
#         Remove edge between existing RG supervoxels 1 and 2 (disconnected chunks)
#         ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
#         │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
#         │  1━━┿━━━━━┿━━2  │  =>  │  1  │     │  2  │
#         │     │     │     │      │     │     │     │
#         └─────┘     └─────┘      └─────┘     └─────┘
#         """

#         cg = gen_graph(n_layers=9)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0)],
#             edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 7, 7, 7, 0), 1.0,)],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk Z
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 7, 7, 7, 0)],
#             edges=[(to_label(cg, 1, 7, 7, 7, 0), to_label(cg, 1, 0, 0, 0, 0), 1.0,)],
#             timestamp=fake_timestamp,
#         )

#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             4,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             4,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             5,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             5,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             6,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             6,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             7,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             7,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             8,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             8,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )
#         add_layer(
#             cg,
#             9,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )

#         # Split
#         new_roots = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 7, 7, 7, 0),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#             mincut=False,
#         ).new_root_ids

#         # Check New State
#         assert len(new_roots) == 2
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) != cg.get_root(
#             to_label(cg, 1, 7, 7, 7, 0)
#         )
#         leaves = np.unique(
#             cg.get_subgraph([cg.get_root(to_label(cg, 1, 0, 0, 0, 0))], leaves_only=True)
#         )
#         assert len(leaves) == 1 and to_label(cg, 1, 0, 0, 0, 0) in leaves
#         leaves = np.unique(
#             cg.get_subgraph([cg.get_root(to_label(cg, 1, 7, 7, 7, 0))], leaves_only=True)
#         )
#         assert len(leaves) == 1 and to_label(cg, 1, 7, 7, 7, 0) in leaves

#         # Check Old State still accessible
#         assert cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp
#         ) == cg.get_root(to_label(cg, 1, 7, 7, 7, 0), time_stamp=fake_timestamp)
#         leaves = np.unique(
#             cg.get_subgraph(
#                 [cg.get_root(to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)],
#                 leaves_only=True,
#             )
#         )
#         assert len(leaves) == 2
#         assert to_label(cg, 1, 0, 0, 0, 0) in leaves
#         assert to_label(cg, 1, 7, 7, 7, 0) in leaves

#     @pytest.mark.timeout(30)
#     def test_split_pair_already_disconnected(self, gen_graph):
#         """
#         Try to remove edge between already disconnected RG supervoxels 1 and 2 (same chunk).
#         Expected: No change, no error
#         ┌─────┐      ┌─────┐
#         │  A¹ │      │  A¹ │
#         │ 1 2 │  =>  │ 1 2 │
#         │     │      │     │
#         └─────┘      └─────┘
#         """

#         cg = gen_graph(n_layers=2)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
#             edges=[],
#             timestamp=fake_timestamp,
#         )

#         res_old = cg.client._table.read_rows()
#         res_old.consume_all()

#         # Split
#         with pytest.raises(exceptions.PreconditionError):
#             cg.remove_edges(
#                 "Jane Doe",
#                 source_ids=to_label(cg, 1, 0, 0, 0, 1),
#                 sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#                 mincut=False,
#             )

#         res_new = cg.client._table.read_rows()
#         res_new.consume_all()

#         # Check
#         if res_old.rows != res_new.rows:
#             warn(
#                 "Rows were modified when splitting a pair of already disconnected supervoxels. "
#                 "While probably not an error, it is an unnecessary operation."
#             )

#     @pytest.mark.timeout(30)
#     def test_split_full_circle_to_triple_chain_same_chunk(self, gen_graph):
#         """
#         Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (same chunk)
#         ┌─────┐      ┌─────┐
#         │  A¹ │      │  A¹ │
#         │ 1━2 │  =>  │ 1 2 │
#         │ ┗3┛ │      │ ┗3┛ │
#         └─────┘      └─────┘
#         """

#         cg = gen_graph(n_layers=2)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[
#                 to_label(cg, 1, 0, 0, 0, 0),
#                 to_label(cg, 1, 0, 0, 0, 1),
#                 to_label(cg, 1, 0, 0, 0, 2),
#             ],
#             edges=[
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 2), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.3),
#             ],
#             timestamp=fake_timestamp,
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 0, 0, 0, 1),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#             mincut=False,
#         ).new_root_ids

#         # Check New State
#         assert len(new_root_ids) == 1
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root_ids[0]
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 1)) == new_root_ids[0]
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 2)) == new_root_ids[0]
#         leaves = np.unique(cg.get_subgraph([new_root_ids[0]], leaves_only=True))
#         assert len(leaves) == 3
#         assert to_label(cg, 1, 0, 0, 0, 0) in leaves
#         assert to_label(cg, 1, 0, 0, 0, 1) in leaves
#         assert to_label(cg, 1, 0, 0, 0, 2) in leaves

#         # Check Old State still accessible
#         old_root_id = cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp
#         )
#         assert new_root_ids[0] != old_root_id

#         # assert len(cg.get_latest_roots()) == 1
#         # assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     @pytest.mark.timeout(30)
#     def test_split_full_circle_to_triple_chain_neighboring_chunks(self, gen_graph):
#         """
#         Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (neighboring chunks)
#         ┌─────┬─────┐      ┌─────┬─────┐
#         │  A¹ │  B¹ │      │  A¹ │  B¹ │
#         │  1━━┿━━2  │  =>  │  1  │  2  │
#         │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
#         └─────┴─────┘      └─────┴─────┘
#         """

#         cg = gen_graph(n_layers=3)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
#             edges=[
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 1, 0, 0, 0), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.3),
#             ],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk B
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 1, 0, 0, 0)],
#             edges=[
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
#                 (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.3),
#             ],
#             timestamp=fake_timestamp,
#         )

#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 1, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#             mincut=False,
#         ).new_root_ids

#         # Check New State
#         assert len(new_root_ids) == 1
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root_ids[0]
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 1)) == new_root_ids[0]
#         assert cg.get_root(to_label(cg, 1, 1, 0, 0, 0)) == new_root_ids[0]
#         leaves = np.unique(cg.get_subgraph([new_root_ids[0]], leaves_only=True))
#         assert len(leaves) == 3
#         assert to_label(cg, 1, 0, 0, 0, 0) in leaves
#         assert to_label(cg, 1, 0, 0, 0, 1) in leaves
#         assert to_label(cg, 1, 1, 0, 0, 0) in leaves

#         # Check Old State still accessible
#         old_root_id = cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp
#         )
#         assert new_root_ids[0] != old_root_id

#         assert len(cg.get_latest_roots()) == 1
#         assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     @pytest.mark.timeout(30)
#     def test_split_full_circle_to_triple_chain_disconnected_chunks(self, gen_graph):
#         """
#         Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (disconnected chunks)
#         ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
#         │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
#         │  1━━┿━━━━━┿━━2  │  =>  │  1  │     │  2  │
#         │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
#         └─────┘     └─────┘      └─────┘     └─────┘
#         """

#         cg = gen_graph(n_layers=9)

#         loc = 2

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
#             edges=[
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, loc, loc, loc, 0), 0.5,),
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, loc, loc, loc, 0), 0.3,),
#             ],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk Z
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, loc, loc, loc, 0)],
#             edges=[
#                 (to_label(cg, 1, loc, loc, loc, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5,),
#                 (to_label(cg, 1, loc, loc, loc, 0), to_label(cg, 1, 0, 0, 0, 0), 0.3,),
#             ],
#             timestamp=fake_timestamp,
#         )

#         for i_layer in range(3, 10):
#             if loc // 2 ** (i_layer - 3) == 1:
#                 add_layer(
#                     cg,
#                     i_layer,
#                     [0, 0, 0],
#
#                     time_stamp=fake_timestamp,
#                     n_threads=1,
#                 )
#             elif loc // 2 ** (i_layer - 3) == 0:
#                 add_layer(
#                     cg,
#                     i_layer,
#                     [0, 0, 0],
#
#                     time_stamp=fake_timestamp,
#                     n_threads=1,
#                 )
#             else:
#                 add_layer(
#                     cg,
#                     i_layer,
#                     [0, 0, 0],
#
#                     time_stamp=fake_timestamp,
#                     n_threads=1,
#                 )
#                 add_layer(
#                     cg,
#                     i_layer,
#                     [0, 0, 0],
#
#                     time_stamp=fake_timestamp,
#                     n_threads=1,
#                 )

#         assert (
#             cg.get_root(to_label(cg, 1, loc, loc, loc, 0))
#             == cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
#             == cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
#         )

#         # Split
#         new_root_ids = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, loc, loc, loc, 0),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#             mincut=False,
#         ).new_root_ids

#         # Check New State
#         assert len(new_root_ids) == 1
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root_ids[0]
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 1)) == new_root_ids[0]
#         assert cg.get_root(to_label(cg, 1, loc, loc, loc, 0)) == new_root_ids[0]
#         leaves = np.unique(cg.get_subgraph([new_root_ids[0]], leaves_only=True))
#         assert len(leaves) == 3
#         assert to_label(cg, 1, 0, 0, 0, 0) in leaves
#         assert to_label(cg, 1, 0, 0, 0, 1) in leaves
#         assert to_label(cg, 1, loc, loc, loc, 0) in leaves

#         # Check Old State still accessible
#         old_root_id = cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0), time_stamp=fake_timestamp
#         )
#         assert new_root_ids[0] != old_root_id

#         assert len(cg.get_latest_roots()) == 1
#         assert len(cg.get_latest_roots(fake_timestamp)) == 1

#     @pytest.mark.timeout(30)
#     def test_split_same_node(self, gen_graph):
#         """
#         Try to remove (non-existing) edge between RG supervoxel 1 and itself
#         ┌─────┐
#         │  A¹ │
#         │  1  │  =>  Reject
#         │     │
#         └─────┘
#         """

#         cg = gen_graph(n_layers=2)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0)],
#             edges=[],
#             timestamp=fake_timestamp,
#         )

#         res_old = cg.client._table.read_rows()
#         res_old.consume_all()

#         # Split
#         with pytest.raises(exceptions.PreconditionError):
#             cg.remove_edges(
#                 "Jane Doe",
#                 source_ids=to_label(cg, 1, 0, 0, 0, 0),
#                 sink_ids=to_label(cg, 1, 0, 0, 0, 0),
#                 mincut=False,
#             )

#         res_new = cg.client._table.read_rows()
#         res_new.consume_all()

#         assert res_new.rows == res_old.rows

#     @pytest.mark.timeout(30)
#     def test_split_pair_abstract_nodes(self, gen_graph):
#         """
#         Try to remove (non-existing) edge between RG supervoxel 1 and abstract node "2"
#                     ┌─────┐
#                     │  B² │
#                     │ "2" │
#                     │     │
#                     └─────┘
#         ┌─────┐              =>  Reject
#         │  A¹ │
#         │  1  │
#         │     │
#         └─────┘
#         """

#         cg = gen_graph(n_layers=3)

#         # Preparation: Build Chunk A
#         fake_timestamp = datetime.utcnow() - timedelta(days=10)
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0)],
#             edges=[],
#             timestamp=fake_timestamp,
#         )

#         # Preparation: Build Chunk B
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 1, 0, 0, 0)],
#             edges=[],
#             timestamp=fake_timestamp,
#         )

#         add_layer(
#             cg,
#             3,
#             [0, 0, 0],
#
#             time_stamp=fake_timestamp,
#             n_threads=1,
#         )

#         res_old = cg.client._table.read_rows()
#         res_old.consume_all()

#         # Split
#         with pytest.raises(exceptions.PreconditionError):
#             cg.remove_edges(
#                 "Jane Doe",
#                 source_ids=to_label(cg, 1, 0, 0, 0, 0),
#                 sink_ids=to_label(cg, 2, 1, 0, 0, 1),
#                 mincut=False,
#             )

#         res_new = cg.client._table.read_rows()
#         res_new.consume_all()

#         assert res_new.rows == res_old.rows

#     @pytest.mark.timeout(30)
#     def test_diagonal_connections(self, gen_graph):
#         """
#         Create graph with edge between RG supervoxels 1 and 2 (same chunk)
#         and edge between RG supervoxels 1 and 3 (neighboring chunks)
#         ┌─────┬─────┐
#         │  A¹ │  B¹ │
#         │ 2━1━┿━━3  │
#         │  /  │     │
#         ┌─────┬─────┐
#         │  |  │     │
#         │  4━━┿━━5  │
#         │  C¹ │  D¹ │
#         └─────┴─────┘
#         """

#         cg = gen_graph(n_layers=3)

#         # Chunk A
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
#             edges=[
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
#                 (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 1, 0, 0), inf),
#             ],
#         )

#         # Chunk B
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 1, 0, 0, 0)],
#             edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf)],
#         )

#         # Chunk C
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 0, 1, 0, 0)],
#             edges=[
#                 (to_label(cg, 1, 0, 1, 0, 0), to_label(cg, 1, 1, 1, 0, 0), inf),
#                 (to_label(cg, 1, 0, 1, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf),
#             ],
#         )

#         # Chunk D
#         create_chunk(
#             cg,
#             vertices=[to_label(cg, 1, 1, 1, 0, 0)],
#             edges=[(to_label(cg, 1, 1, 1, 0, 0), to_label(cg, 1, 0, 1, 0, 0), inf)],
#         )

#         add_layer(
#             cg, 3, [0, 0, 0],  n_threads=1,
#         )

#         rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=3, x=0, y=0, z=0))
#         root_ids_t0 = list(rr.keys())

#         assert len(root_ids_t0) == 1

#         child_ids = []
#         for root_id in root_ids_t0:
#             child_ids.extend([cg.get_subgraph([root_id])], leaves_only=True)

#         new_roots = cg.remove_edges(
#             "Jane Doe",
#             source_ids=to_label(cg, 1, 0, 0, 0, 0),
#             sink_ids=to_label(cg, 1, 0, 0, 0, 1),
#             mincut=False,
#         ).new_root_ids

#         assert len(new_roots) == 2
#         assert cg.get_root(to_label(cg, 1, 1, 1, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 0, 1, 0, 0)
#         )
#         assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == cg.get_root(
#             to_label(cg, 1, 0, 0, 0, 0)
#         )
