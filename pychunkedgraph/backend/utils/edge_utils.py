"""
helper functions for edge stuff
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from ...utils.general import reverse_dictionary
from ..definitions.edges import Edges, IN_CHUNK, BT_CHUNK, CX_CHUNK
from ..connectivity.search import check_reachability
from ..flatgraph_utils import build_gt_graph


def concatenate_chunk_edges(chunk_edge_dicts: list) -> dict:
    """combine edge_dicts of multiple chunks into one edge_dict"""
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
        edges_dict[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
    return edges_dict


def filter_edges(node_ids: np.ndarray, edges: Edges) -> Edges:
    """find edges for the given node_ids"""
    xsorted = np.argsort(edges.node_ids1)
    indices = np.searchsorted(edges.node_ids1[xsorted], node_ids)
    indices = indices[indices < xsorted.size]

    ids1 = edges.node_ids1[indices]
    ids2 = edges.node_ids2[indices]
    affinities = edges.affinities[indices]
    areas = edges.areas[indices]
    return Edges(ids1, ids2, affinities=affinities, areas=areas)


def get_active_edges(edges: Edges, parent_children_d: dict) -> Edges:
    """
    get edges [(v1, v2) ...] where parent(v1) == parent(v2)
    -> assume active if v1 and v2 belong to same connected component
    """
    child_parent_d = reverse_dictionary(parent_children_d)

    sv_ids1 = edges.node_ids1
    sv_ids2 = edges.node_ids2
    affinities = edges.affinities
    areas = edges.areas
    parent_ids1 = np.array([child_parent_d.get(sv_id, sv_id) for sv_id in sv_ids1])
    parent_ids2 = np.array([child_parent_d.get(sv_id, sv_id) for sv_id in sv_ids2])

    sv_ids1 = sv_ids1[parent_ids1 == parent_ids2]
    sv_ids2 = sv_ids2[parent_ids1 == parent_ids2]
    affinities = affinities[parent_ids1 == parent_ids2]
    areas = areas[parent_ids1 == parent_ids2]

    return Edges(sv_ids1, sv_ids2, affinities=affinities, areas=areas)


def filter_fake_edges(added_edges, subgraph_edges) -> List:
    """run bfs to check if a path exists"""
    self_edges = np.array([[node_id, node_id] for node_id in np.unique(added_edges)])
    subgraph_edges = np.concatenate([subgraph_edges, self_edges])

    graph, _, _, original_ids = build_gt_graph(subgraph_edges, is_directed=False)
    reachable = check_reachability(
        graph, added_edges[:, 0], added_edges[:, 1], original_ids
    )
    return added_edges[~reachable]


def map_edges_to_chunks(
    edges: np.ndarray, chunk_ids: np.ndarray, r_indices: np.ndarray
) -> Dict:
    """
    maps a list of edges to corresponding chunks
    returns a dictionary {chuunk_id: [edges that are part of this chunk]}
    """
    chunk_ids_d = defaultdict(list)
    for i, r_index in enumerate(r_indices):
        sv1_index, sv2_index = r_index
        chunk_ids_d[chunk_ids[sv1_index]].append(edges[i])
        if chunk_ids[sv1_index] == chunk_ids[sv2_index]:
            continue
        chunk_ids_d[chunk_ids[sv2_index]].append(edges[i][::-1])
    return {chunk_id: np.array(chunk_ids_d[chunk_id]) for chunk_id in chunk_ids_d}
