from datetime import datetime, timedelta, UTC
from math import inf
from warnings import warn

import numpy as np
import pytest

from ..helpers import create_chunk, to_label
from ...graph import ChunkedGraph
from ...graph import serializers
from ...ingest.create.parent_layer import add_parent_chunk


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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
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
        Expected: No change
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        cg = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5)],
            timestamp=fake_timestamp,
        )

        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 0)],
            )
        res_new = cg.client.read_all_rows()
        res_new.consume_all()
        res_new.rows.pop(b"ioperations", None)
        res_new.rows.pop(b"00000000000000000001", None)

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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 7, 7, 7, 0), inf),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 7, 7, 7, 0)],
            edges=[(to_label(cg, 1, 7, 7, 7, 0), to_label(cg, 1, 0, 0, 0, 1), inf)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 3, [3, 3, 3], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [1, 1, 1], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 5, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

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
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0)],
            )

        res_new = cg.client.read_all_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_merge_pair_abstract_nodes(self, gen_graph):
        """
        Try to add edge between RG supervoxel 1 and abstract node "2"
        => Reject
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
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

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 2, 1, 0, 0, 1)],
            )

        res_new = cg.client.read_all_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_diagonal_connections(self, gen_graph):
        """
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

        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)

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
        cg = gen_graph(n_layers=5)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 1, 0, 0, 0), inf),
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), inf),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), inf),
                (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1), inf),
            ],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk C
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 2, 0, 0, 0)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 3, [1, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 5, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        new_roots = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0)],
            affinities=0.9,
        ).new_root_ids

        assert len(new_roots) == 1


class TestGraphMergeSkipConnections:
    """Tests for skip connection behavior during merge operations."""

    @pytest.mark.timeout(120)
    def test_merge_creates_skip_connection(self, gen_graph):
        """
        Merge two isolated nodes in a 5-layer graph. After merge, each
        component that has no sibling at its layer should get a skip-connection
        parent at a higher layer.

        ┌─────┐     ┌─────┐
        │  A¹ │     │  Z¹ │
        │  1  │     │  2  │
        └─────┘     └─────┘
        After merge: 1 and 2 are connected, hierarchy should skip
        intermediate empty layers.
        """
        cg = gen_graph(n_layers=5)

        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 7, 7, 7, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 3, [3, 3, 3], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 5, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Before merge: verify both nodes have root at layer 5
        root1_pre = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        root2_pre = cg.get_root(to_label(cg, 1, 7, 7, 7, 0))
        assert root1_pre != root2_pre
        assert cg.get_chunk_layer(root1_pre) == 5
        assert cg.get_chunk_layer(root2_pre) == 5

        # Merge
        result = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 7, 7, 7, 0)],
            affinities=[0.5],
        )
        new_root_ids = result.new_root_ids
        assert len(new_root_ids) == 1

        # After merge: single root, both supervoxels reachable
        new_root = new_root_ids[0]
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) == new_root
        assert cg.get_root(to_label(cg, 1, 7, 7, 7, 0)) == new_root
        assert cg.get_chunk_layer(new_root) == 5

    @pytest.mark.timeout(120)
    def test_merge_multi_layer_hierarchy_correctness(self, gen_graph):
        """
        After a merge across chunks, verify the full parent chain from
        each supervoxel to root is valid — every node has a parent at
        a higher layer, and the root is reachable.
        """
        cg = gen_graph(n_layers=5)

        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 7, 7, 7, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 3, [3, 3, 3], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 5, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        result = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 7, 7, 7, 0)],
            affinities=[0.5],
        )

        # Verify parent chain for both supervoxels
        for sv in [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 7, 7, 7, 0)]:
            parents = cg.get_root(sv, get_all_parents=True)
            # Each parent should be at a strictly higher layer
            prev_layer = 1
            for p in parents:
                layer = cg.get_chunk_layer(p)
                assert (
                    layer > prev_layer
                ), f"Parent chain not monotonically increasing: {prev_layer} -> {layer}"
                prev_layer = layer
            # Last parent should be the root
            assert parents[-1] == result.new_root_ids[0]

    @pytest.mark.timeout(30)
    def test_merge_no_skip_when_siblings_exist(self, gen_graph):
        """
        When two nodes in neighboring chunks are merged, they should NOT
        create a skip connection — the parent should be at layer+1 since
        they are siblings in the same parent chunk.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  2  │
        └─────┴─────┘
        """
        cg = gen_graph(n_layers=3)

        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Merge
        result = cg.add_edges(
            "Jane Doe",
            [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0)],
            affinities=[0.5],
        )

        new_root = result.new_root_ids[0]
        # Root should be at layer 3 (the top layer), since the two L2 nodes
        # are siblings at layer 3
        assert cg.get_chunk_layer(new_root) == 3
