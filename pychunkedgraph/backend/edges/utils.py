"""
helper functions for edge stuff
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import Edges
from . import EDGE_TYPES

from ..utils.basetypes import NODE_ID
from ..utils.basetypes import EDGE_AFFINITY
from ..utils.basetypes import EDGE_AREA


def concatenate_chunk_edges(chunk_edge_dicts: List) -> Dict:
    """combine edge_dicts of multiple chunks into one edge_dict"""
    edges_dict = {}
    for edge_type in EDGE_TYPES:
        sv_ids1 = [np.array([], dtype=NODE_ID)]
        sv_ids2 = [np.array([], dtype=NODE_ID)]
        affinities = [np.array([], dtype=EDGE_AFFINITY)]
        areas = [np.array([], dtype=EDGE_AREA)]
        for edge_d in chunk_edge_dicts:
            edges = edge_d[edge_type]
            sv_ids1.append(edges.node_ids1)
            sv_ids2.append(edges.node_ids2)
            affinities.append(edges.affinities)
            areas.append(edges.areas)

        sv_ids1 = np.concatenate(sv_ids1)
        sv_ids2 = np.concatenate(sv_ids2)
        affinities = np.concatenate(affinities)
        areas = np.concatenate(areas)
        edges_dict[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
    return edges_dict
