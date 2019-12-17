from typing import Iterable
from itertools import combinations

from ..types import Agglomeration
from ...utils.general import reverse_dictionary


def edge_exists(agglomerations: Iterable[Agglomeration]):
    """
    Determine if there is an edge (in-active)
    between atleast two of the given nodes (L2 agglomerations).
    """
    supervoxel_parent_d = {}
    for agg in agglomerations:
        supervoxel_parent_d.update(
            zip(agg.supervoxels, [agg.node_id] * len(agg.supervoxels))
        )

    for agg_1, agg_2 in combinations(agglomerations, 2):
        targets1 = agg_1.out_edges[:, 1]
        targets2 = agg_2.out_edges[:, 1]

        for t1, t2 in zip(targets1, targets2):
            if (
                supervoxel_parent_d[t1] == agg_2.node_id
                or supervoxel_parent_d[t2] == agg_1.node_id
            ):
                return True
    return False
