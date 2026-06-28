from math import inf

import numpy as np
import pytest

from ..helpers import SV, build_graph
from ...graph import types


class TestGraphMergeSplit:
    @pytest.mark.timeout(240)
    def test_multiple_cuts_and_splits(self, gen_graph):
        """
        ┌─────┬─────┬─────┐
        │  A¹ │  B¹ │  C¹ │
        │  1  │ 3━2━┿━━4  │
        │     │     │     │
        └─────┴─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={
                "a0": SV(),
                "b0": SV(x=1),
                "b1": SV(x=1, seg=1),
                "c0": SV(x=2),
            },
            edges=[
                ("b0", "b1", 0.5),
                ("b0", "c0", inf),
            ],
        )

        rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=4, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())
        child_ids = [types.empty_1d]
        for root_id in root_ids_t0:
            child_ids.append(cg.get_subgraph([root_id], leaves_only=True))
        child_ids = np.concatenate(child_ids)

        for i in range(10):
            new_roots = cg.add_edges(
                "Jane Doe",
                [sv["a0"], sv["b1"]],
                affinities=0.9,
            ).new_root_ids
            assert len(new_roots) == 1, new_roots
            assert len(cg.get_subgraph([new_roots[0]], leaves_only=True)) == 4

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 1, u_root_ids

            new_roots = cg.remove_edges(
                "John Doe",
                source_ids=sv["b0"],
                sink_ids=sv["b1"],
                mincut=False,
            ).new_root_ids
            assert len(new_roots) == 2, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            these_child_ids = []
            for root_id in u_root_ids:
                these_child_ids.extend(cg.get_subgraph([root_id], leaves_only=True))

            assert len(these_child_ids) == 4
            assert len(u_root_ids) == 2, u_root_ids

            new_roots = cg.remove_edges(
                "Jane Doe",
                source_ids=sv["a0"],
                sink_ids=sv["b1"],
                mincut=False,
            ).new_root_ids
            assert len(new_roots) == 2, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 3, u_root_ids

            new_roots = cg.add_edges(
                "Jane Doe",
                [sv["b0"], sv["b1"]],
                affinities=0.9,
            ).new_root_ids
            assert len(new_roots) == 1, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 2, u_root_ids
