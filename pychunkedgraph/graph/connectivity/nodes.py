from typing import Iterable
from itertools import combinations

from ..types import Agglomeration
from ...utils.general import reverse_dictionary

def edge_exists(agglomerations: Iterable[Agglomeration]):
    """
    Determine if there is an edge (in-active)
    between atleast two of the given agglomerations.
    """
    supervoxel_parent_d = {}
    for agg_1, agg_2 in combinations(agglomerations, 2):
        pass