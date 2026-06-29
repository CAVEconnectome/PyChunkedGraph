from math import inf
from warnings import warn

import numpy as np
import pytest

from ...helpers import SV, build_graph, fake_timestamp, assert_graph_unchanged
from ....graph import exceptions
from ....graph.misc import get_latest_roots


class TestGraphSplit:
    @pytest.mark.timeout(30)
    def test_split_pair_same_chunk(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1 2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
            edges=[("a0", "a1", 0.5)],
            timestamp=fake_ts,
        )

        # Split
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a1"],
            sink_ids=sv["a0"],
            mincut=False,
        ).new_root_ids

        # verify new state
        assert len(new_root_ids) == 2
        assert cg.get_root(sv["a0"]) != cg.get_root(
            sv["a1"]
        )
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(sv["a0"])], leaves_only=True
            )
        )
        assert len(leaves) == 1 and sv["a0"] in leaves
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(sv["a1"])], leaves_only=True
            )
        )
        assert len(leaves) == 1 and sv["a1"] in leaves

        # verify old state
        cg.cache = None
        assert cg.get_root(
            sv["a0"], time_stamp=fake_ts
        ) == cg.get_root(sv["a1"], time_stamp=fake_ts)
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(sv["a0"], time_stamp=fake_ts)],
                leaves_only=True,
            )
        )
        assert len(leaves) == 2
        assert sv["a0"] in leaves
        assert sv["a1"] in leaves

        assert len(get_latest_roots(cg)) == 2
        assert len(get_latest_roots(cg, fake_ts)) == 1

    def test_split_nonexisting_edge(self, gen_graph):
        """
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │   | │      │   | │
        │   3 │      │   3 │
        └─────┘      └─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={"a0": SV(), "a1": SV(seg=1), "a2": SV(seg=2)},
            edges=[
                ("a0", "a1", 0.5),
                ("a2", "a1", 0.5),
            ],
        )
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["a2"],
            mincut=False,
        ).new_root_ids
        assert len(new_root_ids) == 1

    @pytest.mark.timeout(30)
    def test_split_pair_neighboring_chunks(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1━━┿━━2  │  =>  │  1  │  2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", 1.0)],
            timestamp=fake_ts,
        )
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["b"],
            sink_ids=sv["a0"],
            mincut=False,
        ).new_root_ids

        # verify new state
        assert len(new_root_ids) == 2
        assert cg.get_root(sv["a0"]) != cg.get_root(
            sv["b"]
        )
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(sv["a0"])], leaves_only=True
            )
        )
        assert len(leaves) == 1 and sv["a0"] in leaves
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(sv["b"])], leaves_only=True
            )
        )
        assert len(leaves) == 1 and sv["b"] in leaves

        # verify old state
        assert cg.get_root(
            sv["a0"], time_stamp=fake_ts
        ) == cg.get_root(sv["b"], time_stamp=fake_ts)
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(sv["a0"], time_stamp=fake_ts)],
                leaves_only=True,
            )
        )
        assert len(leaves) == 2
        assert sv["a0"] in leaves
        assert sv["b"] in leaves
        assert len(get_latest_roots(cg)) == 2
        assert len(get_latest_roots(cg, fake_ts)) == 1

    @pytest.mark.timeout(30)
    def test_split_verify_cross_chunk_edges(self, gen_graph):
        """
        ┌─────┬─────┬─────┐      ┌─────┬─────┬─────┐
        |     │  A¹ │  B¹ │      |     │  A¹ │  B¹ │
        |     │  1━━┿━━3  │  =>  |     │  1━━┿━━3  │
        |     │  |  │     │      |     │     │     │
        |     │  2  │     │      |     │  2  │     │
        └─────┴─────┴─────┘      └─────┴─────┴─────┘
        """
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"b0": SV(x=1), "b1": SV(x=1, seg=1), "c": SV(x=2)},
            edges=[
                ("b0", "c", inf),
                ("b0", "b1", 0.5),
            ],
            timestamp=fake_ts,
        )

        assert cg.get_root(sv["b0"]) == cg.get_root(
            sv["b1"]
        )
        assert cg.get_root(sv["b0"]) == cg.get_root(
            sv["c"]
        )

        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["b0"],
            sink_ids=sv["b1"],
            mincut=False,
        ).new_root_ids

        assert len(new_root_ids) == 2

        svs2 = cg.get_subgraph([new_root_ids[0]], leaves_only=True)
        svs1 = cg.get_subgraph([new_root_ids[1]], leaves_only=True)
        len_set = {1, 2}
        assert len(svs1) in len_set
        len_set.remove(len(svs1))
        assert len(svs2) in len_set

        # verify new state
        assert len(new_root_ids) == 2
        assert cg.get_root(sv["b0"]) != cg.get_root(
            sv["b1"]
        )
        assert cg.get_root(sv["b0"]) == cg.get_root(
            sv["c"]
        )

        assert len(get_latest_roots(cg)) == 2
        assert len(get_latest_roots(cg, fake_ts)) == 1

    @pytest.mark.timeout(30)
    def test_split_verify_loop(self, gen_graph):
        """
        ┌─────┬────────┬─────┐      ┌─────┬────────┬─────┐
        |     │     A¹ │  B¹ │      |     │     A¹ │  B¹ │
        |     │  4━━1━━┿━━5  │  =>  |     │  4  1━━┿━━5  │
        |     │   /    │  |  │      |     │        │  |  │
        |     │  3  2━━┿━━6  │      |     │  3  2━━┿━━6  │
        └─────┴────────┴─────┘      └─────┴────────┴─────┘
        """
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={
                "b0": SV(x=1),
                "b1": SV(x=1, seg=1),
                "b2": SV(x=1, seg=2),
                "b3": SV(x=1, seg=3),
                "c0": SV(x=2),
                "c1": SV(x=2, seg=1),
            },
            edges=[
                ("b0", "c0", inf),
                ("b1", "c1", inf),
                ("b0", "b2", 0.5),
                ("b0", "b3", 0.5),
                ("c1", "c0", 0.5),
            ],
            timestamp=fake_ts,
        )

        assert cg.get_root(sv["b0"]) == cg.get_root(
            sv["b1"]
        )
        assert cg.get_root(sv["b0"]) == cg.get_root(
            sv["c0"]
        )

        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["b0"],
            sink_ids=sv["b2"],
            mincut=False,
        ).new_root_ids
        assert len(new_root_ids) == 2

        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["b0"],
            sink_ids=sv["b3"],
            mincut=False,
        ).new_root_ids
        assert len(new_root_ids) == 2

        assert len(get_latest_roots(cg)) == 3
        assert len(get_latest_roots(cg, fake_ts)) == 1

    @pytest.mark.timeout(30)
    def test_split_pair_already_disconnected(self, gen_graph):
        """
        Try to remove edge between already disconnected RG supervoxels 1 and 2 (same chunk).
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1 2 │
        │     │      │     │
        └─────┘      └─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
        )
        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=sv["a1"],
                sink_ids=sv["a0"],
                mincut=False,
            )

        res_new = cg.client.read_all_rows()
        res_new.consume_all()

        if res_old.rows != res_new.rows:
            warn(
                "Rows were modified when splitting a pair of already disconnected supervoxels."
                "While probably not an error, it is an unnecessary operation."
            )

    @pytest.mark.timeout(30)
    def test_split_full_circle_to_triple_chain_same_chunk(self, gen_graph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1 2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={
                "a0": SV(),
                "a1": SV(seg=1),
                "a2": SV(seg=2),
            },
            edges=[
                ("a0", "a2", 0.5),
                ("a1", "a2", 0.5),
                ("a0", "a1", 0.3),
            ],
            timestamp=fake_ts,
        )
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a1"],
            sink_ids=sv["a0"],
            mincut=False,
        ).new_root_ids

        # verify new state
        assert len(new_root_ids) == 1
        assert cg.get_root(sv["a0"]) == new_root_ids[0]
        assert cg.get_root(sv["a1"]) == new_root_ids[0]
        assert cg.get_root(sv["a2"]) == new_root_ids[0]
        leaves = np.unique(cg.get_subgraph([new_root_ids[0]], leaves_only=True))
        assert len(leaves) == 3

        # verify old state
        old_root_id = cg.get_root(
            sv["a0"], time_stamp=fake_ts
        )
        assert new_root_ids[0] != old_root_id
        assert len(get_latest_roots(cg)) == 1
        assert len(get_latest_roots(cg, fake_ts)) == 1

    @pytest.mark.timeout(30)
    def test_split_full_circle_to_triple_chain_neighboring_chunks(self, gen_graph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1━━┿━━2  │  =>  │  1  │  2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "a1": SV(seg=1), "b": SV(x=1)},
            edges=[
                ("a0", "a1", 0.5),
                ("a1", "b", 0.5),
                ("a0", "b", 0.3),
            ],
            timestamp=fake_ts,
        )
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["b"],
            sink_ids=sv["a0"],
            mincut=False,
        ).new_root_ids

        # verify new state
        assert len(new_root_ids) == 1
        assert cg.get_root(sv["a0"]) == new_root_ids[0]
        assert cg.get_root(sv["a1"]) == new_root_ids[0]
        assert cg.get_root(sv["b"]) == new_root_ids[0]
        leaves = np.unique(cg.get_subgraph([new_root_ids[0]], leaves_only=True))
        assert len(leaves) == 3

        # verify old state
        old_root_id = cg.get_root(
            sv["a0"], time_stamp=fake_ts
        )
        assert new_root_ids[0] != old_root_id
        assert len(get_latest_roots(cg)) == 1
        assert len(get_latest_roots(cg, fake_ts)) == 1

    @pytest.mark.timeout(30)
    def test_split_same_node(self, gen_graph):
        """
        Try to remove (non-existing) edge between RG supervoxel 1 and itself
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

        with assert_graph_unchanged(cg):
            with pytest.raises(exceptions.PreconditionError):
                cg.remove_edges(
                    "Jane Doe",
                    source_ids=sv["a0"],
                    sink_ids=sv["a0"],
                    mincut=False,
                )

    @pytest.mark.timeout(30)
    def test_split_pair_abstract_nodes(self, gen_graph):
        """
        Try to remove (non-existing) edge between RG supervoxel 1 and abstract node "2"
        => Reject
        """

        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
        )
        with assert_graph_unchanged(cg):
            with pytest.raises((exceptions.PreconditionError, AssertionError)):
                cg.remove_edges(
                    "Jane Doe",
                    source_ids=sv["a0"],
                    sink_ids=cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0),
                    mincut=False,
                )

    @pytest.mark.timeout(30)
    def test_diagonal_connections(self, gen_graph):
        """
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2━1━┿━━3  │
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
                ("a0", "a1", 0.5),
                ("a0", "b", inf),
                ("a0", "c", inf),
                ("c", "d", inf),
            ],
        )

        rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=3, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())
        assert len(root_ids_t0) == 1

        new_roots = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["a1"],
            mincut=False,
        ).new_root_ids

        assert len(new_roots) == 2
        assert cg.get_root(sv["d"]) == cg.get_root(
            sv["c"]
        )
        assert cg.get_root(sv["a0"]) == cg.get_root(
            sv["a0"]
        )


class TestGraphSplitSkipConnections:
    """Tests for skip connection behavior during split operations."""

    @pytest.mark.timeout(120)
    def test_split_multi_layer_hierarchy_correctness(self, gen_graph):
        """
        After a split, verify the full parent chain from each supervoxel
        to its new root is valid.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │  => split => two separate roots
        │     │     │
        └─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", 0.5)],
        )

        result = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )
        assert len(result.new_root_ids) == 2

        # Verify parent chain for both supervoxels
        for sv_id in [sv["a0"], sv["b"]]:
            parents = cg.get_root(sv_id, get_all_parents=True)
            prev_layer = 1
            for p in parents:
                layer = cg.get_chunk_layer(p)
                assert (
                    layer > prev_layer
                ), f"Parent chain not monotonically increasing: {prev_layer} -> {layer}"
                prev_layer = layer
            # Last parent should be one of the new roots
            assert parents[-1] in result.new_root_ids

    @pytest.mark.timeout(120)
    def test_split_creates_isolated_components_with_skip_connections(self, gen_graph):
        """
        After splitting a 3-node chain in a multi-layer graph, the isolated
        node should still have a valid root.

        ┌─────┬─────┬─────┐
        │  A¹ │  B¹ │  C¹ │
        │  1━━┿━━2━━┿━━3  │  => split 1-2 => 1 becomes isolated, 2-3 stay connected
        └─────┴─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "b": SV(x=1), "c": SV(x=2)},
            edges=[
                ("a0", "b", 0.5),
                ("b", "c", inf),
            ],
        )

        # All three should share a root before split
        root_pre = cg.get_root(sv["a0"])
        assert root_pre == cg.get_root(sv["b"])
        assert root_pre == cg.get_root(sv["c"])

        # Split 1 from 2
        result = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )
        assert len(result.new_root_ids) == 2

        # Node 1 should be isolated, nodes 2 and 3 should share a root
        root1 = cg.get_root(sv["a0"])
        root2 = cg.get_root(sv["b"])
        root3 = cg.get_root(sv["c"])
        assert root1 != root2
        assert root2 == root3

        # Both roots should be valid
        assert root1 in result.new_root_ids
        assert root2 in result.new_root_ids
