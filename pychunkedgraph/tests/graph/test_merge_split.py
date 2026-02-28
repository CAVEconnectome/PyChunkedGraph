from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from ..helpers import create_chunk, to_label
from ...graph import types


class TestGraphMergeSplit:
    @pytest.mark.timeout(240)
    def test_multiple_cuts_and_splits(self, gen_graph_simplequerytest):
        cg = gen_graph_simplequerytest

        rr = cg.range_read_chunk(chunk_id=cg.get_chunk_id(layer=4, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())
        child_ids = [types.empty_1d]
        for root_id in root_ids_t0:
            child_ids.append(cg.get_subgraph([root_id], leaves_only=True))
        child_ids = np.concatenate(child_ids)

        for i in range(10):
            new_roots = cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
                affinities=0.9,
            ).new_root_ids
            assert len(new_roots) == 1, new_roots
            assert len(cg.get_subgraph([new_roots[0]], leaves_only=True)) == 4

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 1, u_root_ids

            new_roots = cg.remove_edges(
                "John Doe",
                source_ids=to_label(cg, 1, 1, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 1),
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
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 1),
                mincut=False,
            ).new_root_ids
            assert len(new_roots) == 2, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 3, u_root_ids

            new_roots = cg.add_edges(
                "Jane Doe",
                [to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
                affinities=0.9,
            ).new_root_ids
            assert len(new_roots) == 1, new_roots

            root_ids = cg.get_roots(child_ids, assert_roots=True)
            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 2, u_root_ids
