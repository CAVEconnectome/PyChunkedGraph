from math import inf

import numpy as np
import pytest

from ..helpers import create_chunk, to_label


class TestGraphSimpleQueries:
    """
    ┌─────┬─────┬─────┐        L X Y Z S     L X Y Z S     L X Y Z S     L X Y Z S
    │  A¹ │  B¹ │  C¹ │     1: 1 0 0 0 0 ─── 2 0 0 0 1 ───────────────── 4 0 0 0 1
    │  1  │ 3━２━┿━━4  │     2: 1 1 0 0 0 ─┬─ 2 1 0 0 1 ─── 3 0 0 0 1 ─┬─ 4 0 0 0 2
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

        parent10000 = cg.get_parent(to_label(cg, 1, 0, 0, 0, 0))
        parent11000 = cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))
        parent11001 = cg.get_parent(to_label(cg, 1, 1, 0, 0, 1))
        parent12000 = cg.get_parent(to_label(cg, 1, 2, 0, 0, 0))

        children20001 = cg.get_children(to_label(cg, 2, 0, 0, 0, 1))
        children21001 = cg.get_children(to_label(cg, 2, 1, 0, 0, 1))
        children22001 = cg.get_children(to_label(cg, 2, 2, 0, 0, 1))

        parent20001 = cg.get_parent(to_label(cg, 2, 0, 0, 0, 1))
        parent21001 = cg.get_parent(to_label(cg, 2, 1, 0, 0, 1))
        parent22001 = cg.get_parent(to_label(cg, 2, 2, 0, 0, 1))

        children30001 = cg.get_children(to_label(cg, 3, 0, 0, 0, 1))
        children31001 = cg.get_children(to_label(cg, 3, 1, 0, 0, 1))

        parent30001 = cg.get_parent(to_label(cg, 3, 0, 0, 0, 1))
        parent31001 = cg.get_parent(to_label(cg, 3, 1, 0, 0, 1))

        children40001 = cg.get_children(to_label(cg, 4, 0, 0, 0, 1))
        children40002 = cg.get_children(to_label(cg, 4, 0, 0, 0, 2))

        parent40001 = cg.get_parent(to_label(cg, 4, 0, 0, 0, 1))
        parent40002 = cg.get_parent(to_label(cg, 4, 0, 0, 0, 2))

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
        root10000 = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        root11000 = cg.get_root(to_label(cg, 1, 1, 0, 0, 0))
        root11001 = cg.get_root(to_label(cg, 1, 1, 0, 0, 1))
        root12000 = cg.get_root(to_label(cg, 1, 2, 0, 0, 0))

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
