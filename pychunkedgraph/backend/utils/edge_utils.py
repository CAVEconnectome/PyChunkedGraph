"""
helper functions for edge stuff
"""

from typing import Tuple

import numpy as np
from ..definitions.edges import Edges, IN_CHUNK, BT_CHUNK, CX_CHUNK


def concatenate_chunk_edges(chunk_edge_dicts: list) -> dict:
    """combine edge_dicts of all chunks into one edge_dict"""
    edges_dict = {}
    for edge_type in [IN_CHUNK, BT_CHUNK, CX_CHUNK]:
        sv_ids1 = []
        sv_ids2 = []
        affinities = []
        areas = []
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
        edges_dict[edge_type] = Edges(sv_ids1, sv_ids2, affinities, areas)
    return edges_dict


def filter_edges(node_ids: np.ndarray, edges_dict: dict) -> Edges:
    """find edges for the given node_ids from the dict"""
    ids1 = []
    ids2 = []
    affinities = []
    areas = []
    for edge_type in [IN_CHUNK, BT_CHUNK, CX_CHUNK]:
        edges = edges_dict[edge_type]
        filtered = edges.node_ids1 == node_ids
        ids1.append(edges.node_ids1[filtered])
        ids2.append(edges.node_ids2[filtered])
        affinities.append(edges.affinities[filtered])
        areas.append(edges.areas[filtered])
    return Edges(ids1, ids2, affinities, areas)
