from math import inf

import numpy as np
import pytest

from ..helpers import SV, label, create_chunk, fake_timestamp
from ...graph import attributes, basetypes, serializers
from ...ingest.create.parent_layer import add_parent_chunk


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
        create_chunk(cg, vertices=[label(cg, SV())])

        res = cg.client.read_all_rows()
        res.consume_all()

        assert serializers.serialize_uint64(label(cg, SV())) in res.rows
        parent = cg.get_parent(label(cg, SV()))
        assert parent == label(cg, SV(seg=1), layer=2)

        # Check for the one Level 2 node that should have been created.
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=2)) in res.rows
        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([label(cg, SV(seg=1), layer=2)], dtype=basetypes.NODE_ID)
        )
        attr = attributes.Hierarchy.Child
        row = res.rows[serializers.serialize_uint64(label(cg, SV(seg=1), layer=2))].cells[
            "0"
        ]
        children = attr.deserialize(row[attr.key][0].value)

        for aces in atomic_cross_edge_d.values():
            assert len(aces) == 0

        assert len(children) == 1 and children[0] == label(cg, SV())
        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 1 + 1 + 2 + 1 + 1

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
            vertices=[label(cg, SV()), label(cg, SV(seg=1))],
            edges=[(label(cg, SV()), label(cg, SV(seg=1)), 0.5)],
        )

        res = cg.client.read_all_rows()
        res.consume_all()

        assert serializers.serialize_uint64(label(cg, SV())) in res.rows
        parent = cg.get_parent(label(cg, SV()))
        assert parent == label(cg, SV(seg=1), layer=2)

        assert serializers.serialize_uint64(label(cg, SV(seg=1))) in res.rows
        parent = cg.get_parent(label(cg, SV(seg=1)))
        assert parent == label(cg, SV(seg=1), layer=2)

        # Check for the one Level 2 node that should have been created.
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=2)) in res.rows

        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([label(cg, SV(seg=1), layer=2)], dtype=basetypes.NODE_ID)
        )
        attr = attributes.Hierarchy.Child
        row = res.rows[serializers.serialize_uint64(label(cg, SV(seg=1), layer=2))].cells[
            "0"
        ]
        children = attr.deserialize(row[attr.key][0].value)

        for aces in atomic_cross_edge_d.values():
            assert len(aces) == 0
        assert (
            len(children) == 2
            and label(cg, SV()) in children
            and label(cg, SV(seg=1)) in children
        )

        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 2 + 1 + 2 + 1 + 1

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
            vertices=[label(cg, SV())],
            edges=[(label(cg, SV()), label(cg, SV(x=1)), inf)],
        )

        # Chunk B
        create_chunk(
            cg,
            vertices=[label(cg, SV(x=1))],
            edges=[(label(cg, SV(x=1)), label(cg, SV()), inf)],
        )

        add_parent_chunk(cg, 3, [0, 0, 0], n_processes=1)
        res = cg.client.read_all_rows()
        res.consume_all()

        assert serializers.serialize_uint64(label(cg, SV())) in res.rows
        parent = cg.get_parent(label(cg, SV()))
        assert parent == label(cg, SV(seg=1), layer=2)

        assert serializers.serialize_uint64(label(cg, SV(x=1))) in res.rows
        parent = cg.get_parent(label(cg, SV(x=1)))
        assert parent == label(cg, SV(x=1, seg=1), layer=2)

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # label(cg, SV(seg=1), layer=2)
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=2)) in res.rows
        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([label(cg, SV(seg=1), layer=2)], dtype=basetypes.NODE_ID)
        )
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(label(cg, SV(seg=1), layer=2))
        ]

        attr = attributes.Hierarchy.Child
        row = res.rows[serializers.serialize_uint64(label(cg, SV(seg=1), layer=2))].cells[
            "0"
        ]
        children = attr.deserialize(row[attr.key][0].value)

        test_ace = np.array(
            [label(cg, SV()), label(cg, SV(x=1))],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert len(children) == 1 and label(cg, SV()) in children

        # label(cg, SV(x=1, seg=1), layer=2)
        assert serializers.serialize_uint64(label(cg, SV(x=1, seg=1), layer=2)) in res.rows
        atomic_cross_edge_d = cg.get_atomic_cross_edges(
            np.array([label(cg, SV(x=1, seg=1), layer=2)], dtype=basetypes.NODE_ID)
        )
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(label(cg, SV(x=1, seg=1), layer=2))
        ]

        attr = attributes.Hierarchy.Child
        row = res.rows[serializers.serialize_uint64(label(cg, SV(x=1, seg=1), layer=2))].cells[
            "0"
        ]
        children = attr.deserialize(row[attr.key][0].value)

        test_ace = np.array(
            [label(cg, SV(x=1)), label(cg, SV())],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert len(children) == 1 and label(cg, SV(x=1)) in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # label(cg, SV(seg=1), layer=3)
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=3)) in res.rows

        attr = attributes.Hierarchy.Child
        row = res.rows[serializers.serialize_uint64(label(cg, SV(seg=1), layer=3))].cells[
            "0"
        ]
        children = attr.deserialize(row[attr.key][0].value)
        assert (
            len(children) == 2
            and label(cg, SV(seg=1), layer=2) in children
            and label(cg, SV(x=1, seg=1), layer=2) in children
        )

        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 2 + 2 + 1 + 5 + 1 + 1

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
            vertices=[label(cg, SV()), label(cg, SV(seg=1))],
            edges=[
                (label(cg, SV()), label(cg, SV(seg=1)), 0.5),
                (label(cg, SV()), label(cg, SV(x=1)), inf),
            ],
        )

        # Chunk B
        create_chunk(
            cg,
            vertices=[label(cg, SV(x=1))],
            edges=[(label(cg, SV(x=1)), label(cg, SV()), inf)],
        )

        add_parent_chunk(cg, 3, np.array([0, 0, 0]), n_processes=1)
        res = cg.client.read_all_rows()
        res.consume_all()

        assert serializers.serialize_uint64(label(cg, SV())) in res.rows
        parent = cg.get_parent(label(cg, SV()))
        assert parent == label(cg, SV(seg=1), layer=2)

        # label(cg, SV(seg=1))
        assert serializers.serialize_uint64(label(cg, SV(seg=1))) in res.rows
        parent = cg.get_parent(label(cg, SV(seg=1)))
        assert parent == label(cg, SV(seg=1), layer=2)

        # label(cg, SV(x=1))
        assert serializers.serialize_uint64(label(cg, SV(x=1))) in res.rows
        parent = cg.get_parent(label(cg, SV(x=1)))
        assert parent == label(cg, SV(x=1, seg=1), layer=2)

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # label(cg, SV(seg=1), layer=2)
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=2)) in res.rows
        row = res.rows[serializers.serialize_uint64(label(cg, SV(seg=1), layer=2))].cells[
            "0"
        ]
        atomic_cross_edge_d = cg.get_atomic_cross_edges([label(cg, SV(seg=1), layer=2)])
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(label(cg, SV(seg=1), layer=2))
        ]
        column = attributes.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array(
            [label(cg, SV()), label(cg, SV(x=1))],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert (
            len(children) == 2
            and label(cg, SV()) in children
            and label(cg, SV(seg=1)) in children
        )

        # label(cg, SV(x=1, seg=1), layer=2)
        assert serializers.serialize_uint64(label(cg, SV(x=1, seg=1), layer=2)) in res.rows
        row = res.rows[serializers.serialize_uint64(label(cg, SV(x=1, seg=1), layer=2))].cells[
            "0"
        ]
        atomic_cross_edge_d = cg.get_atomic_cross_edges([label(cg, SV(x=1, seg=1), layer=2)])
        atomic_cross_edge_d = atomic_cross_edge_d[
            np.uint64(label(cg, SV(x=1, seg=1), layer=2))
        ]
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array(
            [label(cg, SV(x=1)), label(cg, SV())],
            dtype=np.uint64,
        )
        assert len(atomic_cross_edge_d[2]) == 1
        assert test_ace in atomic_cross_edge_d[2]
        assert len(children) == 1 and label(cg, SV(x=1)) in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # label(cg, SV(seg=1), layer=3)
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=3)) in res.rows
        row = res.rows[serializers.serialize_uint64(label(cg, SV(seg=1), layer=3))].cells[
            "0"
        ]
        column = attributes.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        assert (
            len(children) == 2
            and label(cg, SV(seg=1), layer=2) in children
            and label(cg, SV(x=1, seg=1), layer=2) in children
        )

        # Make sure there are not any more entries in the table
        # include counters, meta and version rows
        assert len(res.rows) == 3 + 2 + 1 + 5 + 1 + 1

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
        create_chunk(cg, vertices=[label(cg, SV())], edges=[])

        # Preparation: Build Chunk Z
        create_chunk(cg, vertices=[label(cg, SV(x=7, y=7, z=7))], edges=[])

        add_parent_chunk(cg, 3, [0, 0, 0], n_processes=1)
        add_parent_chunk(cg, 3, [3, 3, 3], n_processes=1)
        add_parent_chunk(cg, 4, [0, 0, 0], n_processes=1)
        add_parent_chunk(cg, 5, [0, 0, 0], n_processes=1)

        res = cg.client.read_all_rows()
        res.consume_all()

        assert serializers.serialize_uint64(label(cg, SV())) in res.rows
        assert serializers.serialize_uint64(label(cg, SV(x=7, y=7, z=7))) in res.rows
        assert serializers.serialize_uint64(label(cg, SV(seg=1), layer=5)) in res.rows
        assert serializers.serialize_uint64(label(cg, SV(seg=2), layer=5)) in res.rows

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
        fake_ts = fake_timestamp()
        create_chunk(
            cg,
            vertices=[label(cg, SV(seg=1)), label(cg, SV(seg=2))],
            edges=[],
            timestamp=fake_ts,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[label(cg, SV(x=1, seg=1))],
            edges=[],
            timestamp=fake_ts,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_ts,
            n_processes=1,
        )
        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_ts,
            n_processes=1,
        )
        add_parent_chunk(
            cg,
            4,
            [0, 0, 0],
            time_stamp=fake_ts,
            n_processes=1,
        )

        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=2, x=0, y=0, z=0))) == 2
        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=2, x=1, y=0, z=0))) == 1
        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=3, x=0, y=0, z=0))) == 0
        assert len(cg.range_read_chunk(cg.get_chunk_id(layer=4, x=0, y=0, z=0))) == 6

        assert cg.get_chunk_layer(cg.get_root(label(cg, SV(seg=1)))) == 4
        assert cg.get_chunk_layer(cg.get_root(label(cg, SV(seg=2)))) == 4
        assert cg.get_chunk_layer(cg.get_root(label(cg, SV(x=1, seg=1)))) == 4

        root_seg_ids = [
            cg.get_segment_id(cg.get_root(label(cg, SV(seg=1)))),
            cg.get_segment_id(cg.get_root(label(cg, SV(seg=2)))),
            cg.get_segment_id(cg.get_root(label(cg, SV(x=1, seg=1)))),
        ]

        assert 4 in root_seg_ids
        assert 5 in root_seg_ids
        assert 6 in root_seg_ids
