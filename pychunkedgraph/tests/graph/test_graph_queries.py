from math import inf

import numpy as np
import pytest

from ..helpers import SV, build_graph


class TestGraphSimpleQueries:
    """
    ┌─────┬─────┬─────┐        L X Y Z S     L X Y Z S     L X Y Z S     L X Y Z S
    │  A¹ │  B¹ │  C¹ │     1: 1 0 0 0 0 ─── 2 0 0 0 1 ───────────────── 4 0 0 0 1
    │  1  │ 3━２━┿━━4  │     2: 1 1 0 0 0 ─┬─ 2 1 0 0 1 ─── 3 0 0 0 1 ─┬─ 4 0 0 0 2
    │     │     │     │     3: 1 1 0 0 1 ─┘                           │
    └─────┴─────┴─────┘     4: 1 2 0 0 0 ─── 2 2 0 0 1 ─── 3 1 0 0 1 ─┘
    """

    def _build_graph(self, gen_graph):
        return build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={
                "a0": SV(),
                "b0": SV(x=1),
                "b1": SV(x=1, seg=1),
                "c0": SV(x=2),
            },
            edges=[("b0", "b1", 0.5), ("b0", "c0", inf)],
        )

    @pytest.mark.timeout(30)
    def test_get_parent_and_children(self, gen_graph):
        cg, sv = self._build_graph(gen_graph)

        children10000 = cg.get_children(sv["a0"])
        children11000 = cg.get_children(sv["b0"])
        children11001 = cg.get_children(sv["b1"])
        children12000 = cg.get_children(sv["c0"])

        parent10000 = cg.get_parent(sv["a0"])
        parent11000 = cg.get_parent(sv["b0"])
        parent11001 = cg.get_parent(sv["b1"])
        parent12000 = cg.get_parent(sv["c0"])

        children20001 = cg.get_children(cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0))
        children21001 = cg.get_children(cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0))
        children22001 = cg.get_children(cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0))

        parent20001 = cg.get_parent(cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0))
        parent21001 = cg.get_parent(cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0))
        parent22001 = cg.get_parent(cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0))

        children30001 = cg.get_children(cg.get_node_id(np.uint64(1), layer=3, x=0, y=0, z=0))
        children31001 = cg.get_children(cg.get_node_id(np.uint64(1), layer=3, x=1, y=0, z=0))

        parent30001 = cg.get_parent(cg.get_node_id(np.uint64(1), layer=3, x=0, y=0, z=0))
        parent31001 = cg.get_parent(cg.get_node_id(np.uint64(1), layer=3, x=1, y=0, z=0))

        children40001 = cg.get_children(cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0))
        children40002 = cg.get_children(cg.get_node_id(np.uint64(2), layer=4, x=0, y=0, z=0))

        parent40001 = cg.get_parent(cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0))
        parent40002 = cg.get_parent(cg.get_node_id(np.uint64(2), layer=4, x=0, y=0, z=0))

        # (non-existing) Children of L1
        assert np.array_equal(children10000, []) is True
        assert np.array_equal(children11000, []) is True
        assert np.array_equal(children11001, []) is True
        assert np.array_equal(children12000, []) is True

        # Parent of L1
        assert parent10000 == cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0)
        assert parent11000 == cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0)
        assert parent11001 == cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0)
        assert parent12000 == cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0)

        # Children of L2
        assert len(children20001) == 1 and sv["a0"] in children20001
        assert (
            len(children21001) == 2
            and sv["b0"] in children21001
            and sv["b1"] in children21001
        )
        assert len(children22001) == 1 and sv["c0"] in children22001

        # Parent of L2
        assert parent20001 == cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0)
        assert parent21001 == cg.get_node_id(np.uint64(1), layer=3, x=0, y=0, z=0)
        assert parent22001 == cg.get_node_id(np.uint64(1), layer=3, x=1, y=0, z=0)

        # Children of L3
        assert len(children30001) == 1 and len(children31001) == 1
        assert cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0) in children30001
        assert cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0) in children31001

        # Parent of L3
        assert parent30001 == parent31001
        assert (
            parent30001 == cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0)
            and parent20001 == cg.get_node_id(np.uint64(2), layer=4, x=0, y=0, z=0)
        ) or (
            parent30001 == cg.get_node_id(np.uint64(2), layer=4, x=0, y=0, z=0)
            and parent20001 == cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0)
        )

        # Children of L4
        assert parent10000 in children40001
        assert parent21001 in children40002 and parent22001 in children40002

        # (non-existing) Parent of L4
        assert parent40001 is None
        assert parent40002 is None

        children2_separate = cg.get_children(
            [
                cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0),
                cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0),
                cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0),
            ]
        )
        assert len(children2_separate) == 3
        assert cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0) in children2_separate and np.all(
            np.isin(children2_separate[cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0)], children20001)
        )
        assert cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0) in children2_separate and np.all(
            np.isin(children2_separate[cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0)], children21001)
        )
        assert cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0) in children2_separate and np.all(
            np.isin(children2_separate[cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0)], children22001)
        )

        children2_combined = cg.get_children(
            [
                cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0),
                cg.get_node_id(np.uint64(1), layer=2, x=1, y=0, z=0),
                cg.get_node_id(np.uint64(1), layer=2, x=2, y=0, z=0),
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
    def test_get_root(self, gen_graph):
        cg, sv = self._build_graph(gen_graph)
        root10000 = cg.get_root(sv["a0"])
        root11000 = cg.get_root(sv["b0"])
        root11001 = cg.get_root(sv["b1"])
        root12000 = cg.get_root(sv["c0"])

        with pytest.raises(Exception):
            cg.get_root(0)

        assert (
            root10000 == cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0)
            and root11000 == root11001 == root12000 == cg.get_node_id(np.uint64(2), layer=4, x=0, y=0, z=0)
        ) or (
            root10000 == cg.get_node_id(np.uint64(2), layer=4, x=0, y=0, z=0)
            and root11000 == root11001 == root12000 == cg.get_node_id(np.uint64(1), layer=4, x=0, y=0, z=0)
        )

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes(self, gen_graph):
        cg, sv = self._build_graph(gen_graph)
        root1 = cg.get_root(sv["a0"])
        root2 = cg.get_root(sv["b0"])

        lvl1_nodes_1 = cg.get_subgraph([root1], leaves_only=True)
        lvl1_nodes_2 = cg.get_subgraph([root2], leaves_only=True)
        assert len(lvl1_nodes_1) == 1
        assert len(lvl1_nodes_2) == 3
        assert sv["a0"] in lvl1_nodes_1
        assert sv["b0"] in lvl1_nodes_2
        assert sv["b1"] in lvl1_nodes_2
        assert sv["c0"] in lvl1_nodes_2

        lvl2_parent = cg.get_parent(sv["b0"])
        lvl1_nodes = cg.get_subgraph([lvl2_parent], leaves_only=True)
        assert len(lvl1_nodes) == 2
        assert sv["b0"] in lvl1_nodes
        assert sv["b1"] in lvl1_nodes

    @pytest.mark.timeout(30)
    def test_get_subgraph_edges(self, gen_graph):
        cg, sv = self._build_graph(gen_graph)
        root1 = cg.get_root(sv["a0"])
        root2 = cg.get_root(sv["b0"])

        edges = cg.get_subgraph([root1], edges_only=True)
        assert len(edges) == 0

        edges = cg.get_subgraph([root2], edges_only=True)
        assert [sv["b0"], sv["b1"]] in edges or [
            sv["b1"],
            sv["b0"],
        ] in edges

        assert [sv["b0"], sv["c0"]] in edges or [
            sv["c0"],
            sv["b0"],
        ] in edges

        lvl2_parent = cg.get_parent(sv["b0"])
        edges = cg.get_subgraph([lvl2_parent], edges_only=True)
        assert [sv["b0"], sv["b1"]] in edges or [
            sv["b1"],
            sv["b0"],
        ] in edges

        assert [sv["b0"], sv["c0"]] in edges or [
            sv["c0"],
            sv["b0"],
        ] in edges

        assert len(edges) == 1

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes_bb(self, gen_graph):
        cg, sv = self._build_graph(gen_graph)
        bb = np.array([[1, 0, 0], [2, 1, 1]], dtype=int)
        bb_coord = bb * cg.meta.graph_config.CHUNK_SIZE
        childs_1 = cg.get_subgraph(
            [cg.get_root(sv["b1"])], bbox=bb, leaves_only=True
        )
        childs_2 = cg.get_subgraph(
            [cg.get_root(sv["b1"])],
            bbox=bb_coord,
            bbox_is_coordinate=True,
            leaves_only=True,
        )
        assert np.all(~(np.sort(childs_1) - np.sort(childs_2)))
