from math import inf
from warnings import warn

import numpy as np
import pytest

from ..helpers import SV, build_graph, assert_graph_unchanged


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

        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=np.array([1, 1, 1]),
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
        )

        # Merge
        new_root_ids = cg.add_edges(
            "Jane Doe",
            [sv["a1"], sv["a0"]],
            affinities=[0.3],
        ).new_root_ids

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_parent(sv["a0"]) == new_root_id
        assert cg.get_parent(sv["a1"]) == new_root_id
        leaves = np.unique(cg.get_subgraph([new_root_id], leaves_only=True))
        assert len(leaves) == 2
        assert sv["a0"] in leaves
        assert sv["a1"] in leaves

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

        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
        )

        # Merge
        new_root_ids = cg.add_edges(
            "Jane Doe",
            [sv["b"], sv["a0"]],
            affinities=0.3,
        ).new_root_ids

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_root(sv["a0"]) == new_root_id
        assert cg.get_root(sv["b"]) == new_root_id
        leaves = np.unique(cg.get_subgraph([new_root_id], leaves_only=True))
        assert len(leaves) == 2
        assert sv["a0"] in leaves
        assert sv["b"] in leaves

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

        cg, sv = build_graph(
            gen_graph,
            n_layers=5,
            supervoxels={"a0": SV(), "z": SV(x=7, y=7, z=7)},
        )

        # Merge
        result = cg.add_edges(
            "Jane Doe",
            [sv["z"], sv["a0"]],
            affinities=[0.3],
        )
        new_root_ids, lvl2_node_ids = result.new_root_ids, result.new_lvl2_ids

        u_layers = np.unique(cg.get_chunk_layers(lvl2_node_ids))
        assert len(u_layers) == 1
        assert u_layers[0] == 2

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_root(sv["a0"]) == new_root_id
        assert cg.get_root(sv["z"]) == new_root_id
        leaves = np.unique(cg.get_subgraph(new_root_id, leaves_only=True))
        assert len(leaves) == 2
        assert sv["a0"] in leaves
        assert sv["z"] in leaves

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

        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
            edges=[("a0", "a1", 0.5)],
        )

        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [sv["a1"], sv["a0"]],
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

        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={"a0": SV(), "a1": SV(seg=1), "a2": SV(seg=2)},
            edges=[
                ("a0", "a2", 0.5),
                ("a1", "a2", 0.5),
            ],
        )

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [sv["a1"], sv["a0"]],
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

        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "a1": SV(seg=1), "b": SV(x=1)},
            edges=[
                ("a0", "a1", 0.5),
                ("a1", "b", inf),
            ],
        )

        # Merge
        with pytest.raises(Exception):
            cg.add_edges(
                "Jane Doe",
                [sv["b"], sv["a0"]],
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

        cg, sv = build_graph(
            gen_graph,
            n_layers=5,
            supervoxels={"a0": SV(), "a1": SV(seg=1), "z": SV(x=7, y=7, z=7)},
            edges=[
                ("a0", "a1", 0.5),
                ("a1", "z", inf),
            ],
        )

        # Merge
        new_root_ids = cg.add_edges(
            "Jane Doe",
            [sv["z"], sv["a0"]],
            affinities=1.0,
        ).new_root_ids

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cg.get_root(sv["a0"]) == new_root_id
        assert cg.get_root(sv["a1"]) == new_root_id
        assert cg.get_root(sv["z"]) == new_root_id
        leaves = np.unique(cg.get_subgraph(new_root_id, leaves_only=True))
        assert len(leaves) == 3
        assert sv["a0"] in leaves
        assert sv["a1"] in leaves
        assert sv["z"] in leaves

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

        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={"a0": SV()},
        )

        # Merge
        with assert_graph_unchanged(cg):
            with pytest.raises(Exception):
                cg.add_edges(
                    "Jane Doe",
                    [sv["a0"], sv["a0"]],
                )

    @pytest.mark.timeout(30)
    def test_merge_pair_abstract_nodes(self, gen_graph):
        """
        Try to add edge between RG supervoxel 1 and abstract node "2"
        => Reject
        """

        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
        )

        # Merge
        with assert_graph_unchanged(cg):
            with pytest.raises(Exception):
                cg.add_edges(
                    "Jane Doe",
                    [sv["a0"], cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0)],
                )

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

        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={
                "a0": SV(),
                "a1": SV(seg=1),
                "b": SV(x=1),
                "c": SV(y=1),
                "d": SV(x=1, y=1),
            },
            edges=[
                ("a0", "b", inf),
                ("a0", "c", inf),
                ("c", "d", inf),
            ],
        )

        rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=3, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())

        assert len(root_ids_t0) == 2

        child_ids = []
        for root_id in root_ids_t0:
            child_ids.extend(cg.get_subgraph(root_id, leaves_only=True))

        new_roots = cg.add_edges(
            "Jane Doe",
            [sv["a0"], sv["a1"]],
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
        cg, sv = build_graph(
            gen_graph,
            n_layers=5,
            supervoxels={
                "a0": SV(),
                "a1": SV(seg=1),
                "b0": SV(x=1),
                "b1": SV(x=1, seg=1),
                "c": SV(x=2),
            },
            edges=[
                ("a1", "b0", inf),
                ("a0", "a1", inf),
                ("b0", "b1", inf),
            ],
        )

        new_roots = cg.add_edges(
            "Jane Doe",
            [sv["b0"], sv["c"]],
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
        cg, sv = build_graph(
            gen_graph,
            n_layers=5,
            supervoxels={"a0": SV(), "z": SV(x=7, y=7, z=7)},
        )

        # Before merge: verify both nodes have root at layer 5
        root1_pre = cg.get_root(sv["a0"])
        root2_pre = cg.get_root(sv["z"])
        assert root1_pre != root2_pre
        assert cg.get_chunk_layer(root1_pre) == 5
        assert cg.get_chunk_layer(root2_pre) == 5

        # Merge
        result = cg.add_edges(
            "Jane Doe",
            [sv["a0"], sv["z"]],
            affinities=[0.5],
        )
        new_root_ids = result.new_root_ids
        assert len(new_root_ids) == 1

        # After merge: single root, both supervoxels reachable
        new_root = new_root_ids[0]
        assert cg.get_root(sv["a0"]) == new_root
        assert cg.get_root(sv["z"]) == new_root
        assert cg.get_chunk_layer(new_root) == 5

    @pytest.mark.timeout(120)
    def test_merge_multi_layer_hierarchy_correctness(self, gen_graph):
        """
        After a merge across chunks, verify the full parent chain from
        each supervoxel to root is valid — every node has a parent at
        a higher layer, and the root is reachable.
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=5,
            supervoxels={"a0": SV(), "z": SV(x=7, y=7, z=7)},
        )

        result = cg.add_edges(
            "Jane Doe",
            [sv["a0"], sv["z"]],
            affinities=[0.5],
        )

        # Verify parent chain for both supervoxels
        for node in [sv["a0"], sv["z"]]:
            parents = cg.get_root(node, get_all_parents=True)
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
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
        )

        # Merge
        result = cg.add_edges(
            "Jane Doe",
            [sv["a0"], sv["b"]],
            affinities=[0.5],
        )

        new_root = result.new_root_ids[0]
        # Root should be at layer 3 (the top layer), since the two L2 nodes
        # are siblings at layer 3
        assert cg.get_chunk_layer(new_root) == 3
