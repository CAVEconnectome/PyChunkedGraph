from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from ..helpers import SV, build_graph, fake_timestamp
from ...graph.lineage import lineage_graph, get_root_id_history
from ...graph.misc import get_delta_roots


class TestGraphHistory:
    """These test inadvertantly also test merge and split operations"""

    @pytest.mark.timeout(120)
    def test_cut_merge_history(self, gen_graph):
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", 0.5)],
            timestamp=fake_ts,
        )

        first_root = cg.get_root(sv["a0"])
        assert first_root == cg.get_root(sv["b"])
        timestamp_before_split = datetime.now(UTC)
        split_roots = cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        ).new_root_ids
        assert len(split_roots) == 2
        g = lineage_graph(cg, split_roots[0])
        assert g.size() == 1
        g = lineage_graph(cg, split_roots)
        assert g.size() == 2

        timestamp_after_split = datetime.now(UTC)
        merge_roots = cg.add_edges(
            "Jane Doe",
            [sv["a0"], sv["b"]],
            affinities=0.4,
        ).new_root_ids
        assert len(merge_roots) == 1
        merge_root = merge_roots[0]
        timestamp_after_merge = datetime.now(UTC)

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
