from math import inf

import numpy as np
import pytest

from ...helpers import SV, build_graph, assert_graph_unchanged
from ....graph import exceptions


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
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", 0.5)],
        )

        # Mincut
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            source_coords=[0, 0, 0],
            sink_coords=[
                2 * cg.meta.graph_config.CHUNK_SIZE[0],
                2 * cg.meta.graph_config.CHUNK_SIZE[1],
                cg.meta.graph_config.CHUNK_SIZE[2],
            ],
            mincut=True,
            disallow_isolating_cut=True,
        ).new_root_ids

        # verify new state
        assert len(new_root_ids) == 2
        assert cg.get_root(sv["a0"]) != cg.get_root(sv["b"])
        leaves = np.unique(
            cg.get_subgraph([cg.get_root(sv["a0"])], leaves_only=True)
        )
        assert len(leaves) == 1 and sv["a0"] in leaves
        leaves = np.unique(
            cg.get_subgraph([cg.get_root(sv["b"])], leaves_only=True)
        )
        assert len(leaves) == 1 and sv["b"] in leaves

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
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
        )

        # Mincut
        with assert_graph_unchanged(cg):
            with pytest.raises(exceptions.PreconditionError):
                cg.remove_edges(
                    "Jane Doe",
                    source_ids=sv["a0"],
                    sink_ids=sv["b"],
                    source_coords=[0, 0, 0],
                    sink_coords=[
                        2 * cg.meta.graph_config.CHUNK_SIZE[0],
                        2 * cg.meta.graph_config.CHUNK_SIZE[1],
                        cg.meta.graph_config.CHUNK_SIZE[2],
                    ],
                    mincut=True,
                )

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
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", 0.5)],
        )
        cg.remove_edges(
            "John Doe",
            source_ids=sv["b"],
            sink_ids=sv["a0"],
            mincut=False,
        )

        # Mincut
        with assert_graph_unchanged(cg):
            with pytest.raises(exceptions.PreconditionError):
                cg.remove_edges(
                    "Jane Doe",
                    source_ids=sv["a0"],
                    sink_ids=sv["b"],
                    source_coords=[0, 0, 0],
                    sink_coords=[
                        2 * cg.meta.graph_config.CHUNK_SIZE[0],
                        2 * cg.meta.graph_config.CHUNK_SIZE[1],
                        cg.meta.graph_config.CHUNK_SIZE[2],
                    ],
                    mincut=True,
                )

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
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", inf)],
        )

        original_parents_1 = cg.get_root(sv["a0"], get_all_parents=True)
        original_parents_2 = cg.get_root(sv["b"], get_all_parents=True)

        # Mincut
        with pytest.raises(exceptions.PostconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=sv["a0"],
                sink_ids=sv["b"],
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        new_parents_1 = cg.get_root(sv["a0"], get_all_parents=True)
        new_parents_2 = cg.get_root(sv["b"], get_all_parents=True)

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
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            supervoxels={
                "a0": SV(),
                "a1": SV(seg=1),
                "a2": SV(seg=2),
                "a3": SV(seg=3),
            },
            edges=[("a0", "a2", 2), ("a1", "a2", 3), ("a2", "a3", 10)],
        )

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=[sv["a0"], sv["a1"]],
                sink_ids=[sv["a3"]],
                source_coords=[[0, 0, 0], [10, 0, 0]],
                sink_coords=[[5, 5, 0]],
                mincut=True,
            )
