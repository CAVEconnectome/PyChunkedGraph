import datetime
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Sequence
from collections import defaultdict

from .utils import basetypes
from .utils import flatgraph
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple


def _get_siblings(cg, new_old_ids_d: Dict) -> List:
    """Get parents of `old_node_ids`, their children will include all siblings."""
    return {
        new_node_id: cg.get_children(
            np.unique(cg.get_parents(old_node_ids)), flatten=True
        )
        for new_node_id, old_node_ids in new_old_ids_d.items()
    }


def _create_parents(
    cg,
    new_cross_edges_d: Dict,
    operation_id: basetypes.OPERATION_ID,
    time_stamp: datetime.datetime,
):
    """TODO docs"""
    layer_new_ids_d = defaultdict(list)
    layer_new_ids_d[2] = list(new_cross_edges_d.keys())
    new_root_ids = []
    for layer in range(2, cg.meta.layer_count):
        if len(layer_new_ids_d[layer]) == 0:
            continue
        # silblings_d = _get_siblings(cg, new_old_ids_d)


def _analyze_atomic_edge(cg, atomic_edge) -> Tuple[Iterable, Dict]:
    """
    Determine if the atomic edge is within the chunk.
    If not, consider it as a cross edge between two L2 IDs in different chunks.
    Returns edges and cross edges accordingly.
    """
    edge_layer = cg.get_cross_chunk_edges_layer([atomic_edge])[0]
    parent_edge = cg.get_parents(atomic_edge)

    # edge is within chunk
    if edge_layer == 1:
        return [parent_edge], {}
    # edge crosses atomic chunk boundary
    parent_1 = parent_edge[0]
    parent_2 = parent_edge[1]

    cross_edges_d = {}
    cross_edges_d[parent_1] = {edge_layer: atomic_edge}
    cross_edges_d[parent_2] = {edge_layer: atomic_edge[::-1]}
    edges = [[parent_1, parent_1], [parent_2, parent_2]]
    return (edges, cross_edges_d)


def add_edge_v2(
    cg,
    *,
    edge: np.ndarray,
    operation_id: np.uint64 = None,
    source_coords: Sequence[np.uint64] = None,
    sink_coords: Sequence[np.uint64] = None,
    timestamp: datetime.datetime = None,
):
    """
    Problem: Update parent and children of the new level 2 id
    For each layer >= 2
        get cross edges
        get parents
            get children
        above children + new ID will form a new component
        update parent, former parents and new parents for all affected IDs
    """
    edges, new_cross_edges_d = _analyze_atomic_edge(cg, edge)

    # Read cross chunk edges efficiently
    cross_edges_d = {}
    node_ids = np.unique(edges)

    for node_id in node_ids:
        cross_edges_d[node_id] = cg.get_cross_chunk_edges(node_id)
    
    cross_edges_d = merge_cross_edge_dicts_multiple(cross_edges_d, new_cross_edges_d)
    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)

    rows = []
    # l2_components_d = {}
    new_cross_edges_d = {}
    for cc in ccs:
        l2ids = graph_node_ids[cc]
        new_l2id = cg.get_unique_node_id(cg.get_chunk_id(l2ids[0]))
        # l2_components_d[new_l2id] = l2ids
        new_cross_edges_d[new_l2id] = concatenate_cross_edge_dicts(
            [cross_edges_d[l2id] for l2id in l2ids]
        )

    # changes up the tree
    new_root_ids, new_rows = _create_parents(
        cg,
        new_cross_edges_d,
        operation_id=operation_id,
        time_stamp=timestamp,
    )
    rows.extend(new_rows)
    return new_root_ids, list(l2_components_d.keys()), rows

