"""
helper functions for edge stuff
"""

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable

import numpy as np


from . import Edges
from . import EDGE_TYPES
from ..types import empty_2d
from ..utils import basetypes
from ..chunks import utils as chunk_utils
from ..meta import ChunkedGraphMeta
from ..connectivity.search import check_reachability
from ..utils.flatgraph import build_gt_graph
from ...utils.general import reverse_dictionary


def concatenate_chunk_edges(chunk_edge_dicts: List) -> Dict:
    """combine edge_dicts of multiple chunks into one edge_dict"""
    edges_dict = {}
    for edge_type in EDGE_TYPES:
        sv_ids1 = [np.array([], dtype=basetypes.NODE_ID)]
        sv_ids2 = [np.array([], dtype=basetypes.NODE_ID)]
        affinities = [np.array([], dtype=basetypes.EDGE_AFFINITY)]
        areas = [np.array([], dtype=basetypes.EDGE_AREA)]
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


def concatenate_cross_edge_dicts(cross_edge_dicts: Iterable) -> Dict:
    """Combines multiple cross edge dicts."""
    result_d = {}
    for cross_edge_d in cross_edge_dicts:
        result_d = merge_cross_edge_dicts_single(result_d, cross_edge_d)
    return result_d


def merge_cross_edge_dicts_single(x_edges_d1: Dict, x_edges_d2: Dict) -> Dict:
    """Combines two cross chunk dictionaries of form {layer id : edge list}."""
    layers = np.unique(list(x_edges_d1.keys()) + list(x_edges_d2.keys()))
    result_d = {}
    for layer in range(2, max(layers)):
        edges1 = x_edges_d1.get(layer, empty_2d)
        edges2 = x_edges_d2.get(layer, empty_2d)
        result_d[layer] = np.concatenate([edges1, edges2])
    return result_d


def merge_cross_edge_dicts_multiple(x_edges_d1: Dict, x_edges_d2: Dict) -> Dict:
    """
    Combines two cross chunk dictionaries of form
    {node_id: {layer id : edge list}}.
    """
    node_ids = np.unique(list(x_edges_d1.keys()) + list(x_edges_d2.keys()))
    result_d = {}
    for node_id in node_ids:
        result_d[node_id] = merge_cross_edge_dicts_single(
            x_edges_d1.get(node_id, {}), x_edges_d2.get(node_id, {})
        )
    return result_d


def categorize_edges(
    meta: ChunkedGraphMeta, node_ids: np.ndarray, edges: Edges
) -> Tuple[Edges, Edges]:
    """
    Find edges and categorize them into:
    `in_edges`
        between given node_ids
        (sv1, sv2) - sv1 in node_ids and sv2 in node_ids
    `out_edges`
        originating from given node_ids but within chunk
        (sv1, sv2) - sv1 in node_ids and sv2 not in node_ids
    `cross_edges`
        originating from given node_ids but crossing chunk boundary
    """
    mask1 = np.in1d(edges.node_ids1, node_ids)
    mask2 = np.in1d(edges.node_ids2, node_ids)
    in_mask = mask1 & mask2
    out_mask = mask1 & ~mask2

    ids1 = edges.node_ids1[in_mask]
    ids2 = edges.node_ids2[in_mask]
    affinities = edges.affinities[in_mask]
    areas = edges.areas[in_mask]
    in_edges = Edges(ids1, ids2, affinities=affinities, areas=areas)

    # all_out_edges = out_edges + cross_edges
    ids1 = edges.node_ids1[out_mask]
    ids2 = edges.node_ids2[out_mask]
    affinities = edges.affinities[out_mask]
    areas = edges.areas[out_mask]
    all_out_edges = Edges(ids1, ids2, affinities=affinities, areas=areas)

    edge_layers = get_cross_chunk_edges_layer(meta, all_out_edges.get_pairs())
    cross_edges_m = edge_layers > 1

    ids1 = all_out_edges.node_ids1[~cross_edges_m]
    ids2 = all_out_edges.node_ids2[~cross_edges_m]
    affinities = all_out_edges.affinities[~cross_edges_m]
    areas = all_out_edges.areas[~cross_edges_m]
    out_edges = Edges(ids1, ids2, affinities=affinities, areas=areas)

    cross_edge_layers = edge_layers[cross_edges_m]
    ids1 = all_out_edges.node_ids1[cross_edges_m]
    ids2 = all_out_edges.node_ids2[cross_edges_m]
    affinities = all_out_edges.affinities[cross_edges_m]
    areas = all_out_edges.areas[cross_edges_m]
    cross_edges = Edges(ids1, ids2, affinities=affinities, areas=areas).get_pairs()

    cross_edge_d = {}
    for layer in range(2, meta.layer_count):
        cross_edge_d[layer] = cross_edges[cross_edge_layers == layer]
    return (in_edges, out_edges, cross_edge_d)


def get_active_edges(edges: Edges, parent_children_d: Dict) -> Edges:
    """
    get edges [(v1, v2) ...] where parent(v1) == parent(v2)
    -> assume active if v1 and v2 belong to same connected component
    """
    child_parent_d = reverse_dictionary(parent_children_d)

    sv_ids1 = edges.node_ids1
    sv_ids2 = edges.node_ids2
    parent_ids1 = np.array([child_parent_d.get(sv_id, sv_id) for sv_id in sv_ids1])
    parent_ids2 = np.array([child_parent_d.get(sv_id, sv_id) for sv_id in sv_ids2])

    mask = parent_ids1 == parent_ids2
    sv_ids1 = sv_ids1[mask]
    sv_ids2 = sv_ids2[mask]
    affinities = edges.affinities[mask]
    areas = edges.areas[mask]

    return Edges(sv_ids1, sv_ids2, affinities=affinities, areas=areas)


def filter_fake_edges(added_edges: np.ndarray, subgraph_edges: np.ndarray) -> List:
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
    returns a dictionary {chunk_id: [edges that are part of this chunk]}
    """
    chunk_ids_d = defaultdict(list)
    for i, r_index in enumerate(r_indices):
        sv1_index, sv2_index = r_index
        chunk_ids_d[chunk_ids[sv1_index]].append(edges[i])
        if chunk_ids[sv1_index] == chunk_ids[sv2_index]:
            continue
        chunk_ids_d[chunk_ids[sv2_index]].append(edges[i][::-1])
    return {chunk_id: np.array(chunk_ids_d[chunk_id]) for chunk_id in chunk_ids_d}


def get_linking_edges(
    edges: Edges, parent_children_d: Dict, parent_id1: np.uint64, parent_id2: np.uint64
):
    """
    Find edges that link two level 2 ids
    include (sv1, sv2) if parent(sv1) == parent_id1 and parent(sv2) == parent_id2
    or if parent(sv1) == parent_id2 and parent(sv2) == parent_id1
    """
    child_parent_d = reverse_dictionary(parent_children_d)
    sv_ids1 = edges.node_ids1
    sv_ids2 = edges.node_ids2

    parent_ids1 = np.array([child_parent_d.get(sv_id, sv_id) for sv_id in sv_ids1])
    parent_ids2 = np.array([child_parent_d.get(sv_id, sv_id) for sv_id in sv_ids2])

    mask = (parent_ids1 == parent_id1) & (parent_ids2 == parent_id2)
    mask |= (parent_ids1 == parent_id2) & (parent_ids2 == parent_id1)

    sv_ids1 = sv_ids1[mask]
    sv_ids2 = sv_ids2[mask]
    affinities = edges.affinities[mask]
    areas = edges.areas[mask]
    return Edges(sv_ids1, sv_ids2, affinities=affinities, areas=areas)


def get_cross_chunk_edges_layer(meta: ChunkedGraphMeta, cross_edges: Iterable):
    """ Computes the layer in which a cross chunk edge becomes relevant.
    I.e. if a cross chunk edge links two nodes in layer 4 this function
    returns 3.
    :param cross_edges: n x 2 array
        edges between atomic (level 1) node ids
    :return: array of length n
    """
    if len(cross_edges) == 0:
        return np.array([], dtype=np.int)

    cross_chunk_edge_layers = np.ones(len(cross_edges), dtype=np.int)
    cross_edge_coordinates = []
    for cross_edge in cross_edges:
        cross_edge_coordinates.append(
            [
                chunk_utils.get_chunk_coordinates(meta, cross_edge[0]),
                chunk_utils.get_chunk_coordinates(meta, cross_edge[1]),
            ]
        )

    cross_edge_coordinates = np.array(cross_edge_coordinates, dtype=np.int)
    for _ in range(2, meta.layer_count):
        edge_diff = np.sum(
            np.abs(cross_edge_coordinates[:, 0] - cross_edge_coordinates[:, 1]), axis=1,
        )
        cross_chunk_edge_layers[edge_diff > 0] += 1
        cross_edge_coordinates = cross_edge_coordinates // meta.graph_config.FANOUT
    return cross_chunk_edge_layers

