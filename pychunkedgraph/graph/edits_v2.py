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
from .edges.utils import get_min_layer_cross_edges
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple


class Node:
    def __init__(
        self,
        node_id: basetypes.NODE_ID,
        parent_id: basetypes.NODE_ID = None,
        children: Iterable = None,
    ):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children = children


def _get_all_siblings(cg, new_id_ce_siblings: Iterable) -> List:
    """
    Get parents of `new_id_ce_siblings`
    Children of these parents will include all siblings.
    """
    return cg.get_children(np.unique(cg.get_parents(new_id_ce_siblings)), flatten=True)


def _create_parents(
    cg,
    new_cross_edges_d_d: Dict[np.uint64, Dict],
    operation_id: basetypes.OPERATION_ID,
    time_stamp: datetime.datetime,
):
    """TODO docs"""
    layer_new_ids_d = defaultdict(list)
    all_new_ids = {}  # cache
    layer_new_ids_d[2] = list(new_cross_edges_d_d.keys())
    new_root_ids = []
    for current_layer in range(2, cg.meta.layer_count):
        if len(layer_new_ids_d[current_layer]) == 0:
            continue
        new_ids = layer_new_ids_d[current_layer]
        for new_id in new_ids:
            node = Node(new_id)
            all_new_ids[new_id] = node
            if not new_id in new_cross_edges_d_d:
                new_cross_edges_d_d[new_id] = cg.get_cross_chunk_edges(new_id)
            new_id_ce_d = new_cross_edges_d_d[new_id]
            new_id_ce_layer = list(new_id_ce_d.keys())[0]
            if not new_id_ce_layer == current_layer:
                # create new id at that level
                parent_chunk_id = cg.get_parent_chunk_id_dict(new_id)[new_id_ce_layer]
                new_parent_seg_id = cg.id_client.create_segment_id(parent_chunk_id)
                new_parent_id = parent_chunk_id | new_parent_seg_id
                new_parent_node = Node(new_parent_id)
                new_parent_node.children = [new_id]
                all_new_ids[new_parent_id] = new_parent_node
                layer_new_ids_d[new_id_ce_layer].append(new_parent_id)
            else:
                new_id_ce_siblings = new_id_ce_d[new_id_ce_layer][:, 1]
                new_id_all_siblings = _get_all_siblings(cg, new_id_ce_siblings)


def _analyze_atomic_edge(cg, atomic_edge) -> Tuple[Iterable, Dict]:
    """
    Determine if the atomic edge is within the chunk.
    If not, consider it as a cross edge between two L2 IDs in different chunks.
    Returns edges and cross edges accordingly.
    """
    edge_layer = cg.get_cross_chunk_edges_layer([atomic_edge])[0]
    parent_edge = cg.get_parents(atomic_edge)

    if edge_layer == 1:
        # edge is within chunk
        return [parent_edge], {}
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
    edges, l2_cross_edges_d = _analyze_atomic_edge(cg, edge)
    cross_edges_d = {}
    node_ids = np.unique(edges)

    for node_id in node_ids:
        cross_edges_d[node_id] = cg.get_cross_chunk_edges(node_id)

    cross_edges_d = merge_cross_edge_dicts_multiple(cross_edges_d, l2_cross_edges_d)
    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)

    rows = []
    new_l2ids = []
    l2_cross_edges_d = {}
    for cc in ccs:
        l2ids = graph_node_ids[cc]
        new_l2ids.append(cg.get_unique_node_id(cg.get_chunk_id(l2ids[0])))
        l2_cross_edges_d[new_l2ids[-1]] = concatenate_cross_edge_dicts(
            [cross_edges_d[l2id] for l2id in l2ids]
        )

    #
    new_cross_edges_d_d = {}
    for l2id, cross_edges_d in l2_cross_edges_d.items():
        layer_, edges_ = get_min_layer_cross_edges(cg.meta, cross_edges_d)
        new_cross_edges_d_d[l2id] = {layer_: edges_}

    # changes up the tree
    # TODO pass only relevant layer edges
    new_root_ids, new_rows = _create_parents(
        cg, new_cross_edges_d_d, operation_id=operation_id, time_stamp=timestamp,
    )
    rows.extend(new_rows)
    return new_root_ids, new_l2_ids, rows

