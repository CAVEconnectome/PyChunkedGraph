import datetime
import numpy as np
from collections import defaultdict
from typing import Dict
from typing import Tuple
from typing import Iterable
from typing import Sequence

from .utils import basetypes
from .utils import flatgraph
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple


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
    # TODO read cross edges for node IDs in edges
    # add read cross chunk edges method to client
    cross_edges_d = {}
    cross_edges_d = merge_cross_edge_dicts_multiple(cross_edges_d, new_cross_edges_d)
    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)

    rows = []
    l2_components_d = {}
    new_cross_edges_d = {}
    for cc in ccs:
        l2ids = graph_node_ids[cc]
        new_l2id = cg.get_unique_node_id(cg.get_chunk_id(l2ids[0]))
        l2_components_d[new_l2id] = l2ids
        new_cross_edges_d[new_l2id] = concatenate_cross_edge_dicts(
            [cross_edges_d[l2id] for l2id in l2ids]
        )

    # Propagate changes up the tree
    new_root_ids, new_rows = propagate_edits_to_root(
        cg,
        l2_components_d.copy(),
        new_cross_edges_d,
        operation_id=operation_id,
        time_stamp=timestamp,
    )
    rows.extend(new_rows)
    return new_root_ids, list(l2_components_d.keys()), rows

