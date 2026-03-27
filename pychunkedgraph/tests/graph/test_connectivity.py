"""Tests for pychunkedgraph.graph.connectivity.nodes"""

import numpy as np

from pychunkedgraph.graph.types import Agglomeration
from pychunkedgraph.graph.connectivity.nodes import edge_exists


def _make_agg(node_id, supervoxels, out_edges):
    """Helper to create an Agglomeration with the fields needed by edge_exists."""
    return Agglomeration(
        node_id=np.uint64(node_id),
        supervoxels=np.array(supervoxels, dtype=np.uint64),
        in_edges=np.empty((0, 2), dtype=np.uint64),
        out_edges=np.array(out_edges, dtype=np.uint64).reshape(-1, 2),
        cross_edges=np.empty((0, 2), dtype=np.uint64),
    )


class TestEdgeExists:
    def test_edge_exists_true(self):
        """Two agglomerations with edges pointing to each other's supervoxels."""
        # agg1 owns supervoxels [10, 11], agg2 owns supervoxels [20, 21].
        # agg1 has an out_edge from sv 10 -> sv 20 (which belongs to agg2)
        # agg2 has an out_edge from sv 20 -> sv 10 (which belongs to agg1)
        agg1 = _make_agg(
            node_id=1,
            supervoxels=[10, 11],
            out_edges=[[10, 20]],
        )
        agg2 = _make_agg(
            node_id=2,
            supervoxels=[20, 21],
            out_edges=[[20, 10]],
        )
        assert edge_exists([agg1, agg2]) is True

    def test_edge_exists_true_one_direction(self):
        """Edge exists is True even if only one direction has a cross-reference."""
        # agg1 out_edge target (sv 20) belongs to agg2 -> True on the first condition
        agg1 = _make_agg(
            node_id=1,
            supervoxels=[10, 11],
            out_edges=[[10, 20]],
        )
        agg2 = _make_agg(
            node_id=2,
            supervoxels=[20, 21],
            out_edges=[[20, 30]],  # target 30 is not in agg1
        )
        # For this to work, sv 30 must be in the supervoxel_parent_d.
        # Since 30 is not in either agglomeration's supervoxels, a KeyError
        # would occur when checking supervoxel_parent_d[t2].
        # The function iterates zip(targets1, targets2), checking t1 first.
        # If t1 matches, it returns True before checking t2.
        # So agg1.out_edges target=20 (belongs to agg2) triggers True.
        # BUT: zip pairs them, and both t1 and t2 are checked.
        # Actually, the condition uses OR: if t1 belongs to agg2 OR t2 belongs to agg1.
        # However, supervoxel_parent_d[t2] will KeyError if t2=30 is not in the dict.
        # Let's fix: put sv 30 in a third agg, or just make the targets safe.
        # Instead, let's set up so that sv 30 doesn't cause a problem:
        # We need all targets to be in the supervoxel_parent_d.
        # Add sv 30 to agg2's supervoxels.
        agg2_fixed = _make_agg(
            node_id=2,
            supervoxels=[20, 21, 30],
            out_edges=[[20, 30]],  # target 30 belongs to agg2 itself (not agg1)
        )
        assert edge_exists([agg1, agg2_fixed]) is True

    def test_edge_exists_false(self):
        """Two agglomerations with no cross-references between them."""
        # agg1 out_edge targets sv 11 (its own supervoxel),
        # agg2 out_edge targets sv 21 (its own supervoxel).
        # Neither target belongs to the other agglomeration.
        agg1 = _make_agg(
            node_id=1,
            supervoxels=[10, 11],
            out_edges=[[10, 11]],
        )
        agg2 = _make_agg(
            node_id=2,
            supervoxels=[20, 21],
            out_edges=[[20, 21]],
        )
        assert edge_exists([agg1, agg2]) is False

    def test_edge_exists_single_agg(self):
        """Single agglomeration returns False (no combinations to iterate)."""
        agg = _make_agg(
            node_id=1,
            supervoxels=[10, 11],
            out_edges=[[10, 11]],
        )
        assert edge_exists([agg]) is False

    def test_edge_exists_empty_list(self):
        """Empty list of agglomerations returns False."""
        assert edge_exists([]) is False

    def test_edge_exists_three_agglomerations(self):
        """Three agglomerations where only two have a cross-reference."""
        agg1 = _make_agg(
            node_id=1,
            supervoxels=[10, 11],
            out_edges=[[10, 20]],
        )
        agg2 = _make_agg(
            node_id=2,
            supervoxels=[20, 21],
            out_edges=[[20, 10]],
        )
        agg3 = _make_agg(
            node_id=3,
            supervoxels=[30, 31],
            out_edges=[[30, 31]],
        )
        # The combination (agg1, agg2) has cross-references, so True.
        assert edge_exists([agg1, agg2, agg3]) is True
