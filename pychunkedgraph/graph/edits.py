import datetime
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Sequence
from collections import defaultdict

from .types import Node
from .utils import basetypes
from .utils import flatgraph
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists
from .edges.utils import get_min_layer_cross_edges
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple


def _get_all_siblings(cg, new_parent_id, new_id_ce_siblings: Iterable) -> List:
    """
    Get parents of `new_id_ce_siblings`
    Children of these parents will include all siblings.
    """
    chunk_ids = cg.get_children_chunk_ids(new_parent_id)
    children = cg.get_children(
        np.unique(cg.get_parents(new_id_ce_siblings)), flatten=True
    )
    children_chunk_ids = cg.get_chunk_ids_from_node_ids(children)
    return children[np.in1d(children_chunk_ids, chunk_ids)]


def _create_parent_node(cg, new_node: Node, parent_layer: int = None) -> Node:
    new_id = new_node.node_id
    parent_chunk_id = cg.get_parent_chunk_id(new_id, parent_layer)
    new_parent_id = cg.id_client.create_node_id(parent_chunk_id)
    new_parent_node = Node(new_parent_id)
    new_node.parent_id = new_parent_id
    return new_parent_node


def _create_parents(
    cg,
    new_cross_edges_d_d: Dict[np.uint64, Dict],
    operation_id: basetypes.OPERATION_ID,
    time_stamp: datetime.datetime,
):
    """
    After new level 2 IDs are built, create parents in higher layers.
    Cross edges are used to determine existing siblings.
    """
    layer_new_ids_d = defaultdict(list)
    # cache for easier access
    new_nodes_d = {}
    # new IDs in each layer
    layer_new_ids_d[2] = list(new_cross_edges_d_d.keys())
    for current_layer in range(2, cg.meta.layer_count):
        if len(layer_new_ids_d[current_layer]) == 0:
            continue
        new_ids = layer_new_ids_d[current_layer]
        for new_id in new_ids:
            if not new_id in new_nodes_d:
                new_nodes_d[new_id] = Node(new_id)
            if not new_id in new_cross_edges_d_d:
                new_cross_edges_d_d[new_id] = cg.get_cross_chunk_edges(
                    new_id, hierarchy=new_nodes_d
                )
            new_id_ce_d = new_cross_edges_d_d[new_id]
            new_node = new_nodes_d[new_id]
            new_id_ce_layer = list(new_id_ce_d.keys())[0]
            if not new_id_ce_layer == current_layer:
                new_parent_node = _create_parent_node(cg, new_node, new_id_ce_layer)
                new_parent_node.children = np.array([new_id], dtype=basetypes.NODE_ID)
                layer_new_ids_d[new_id_ce_layer].append(new_parent_node.node_id)
            else:
                new_parent_node = _create_parent_node(cg, new_node, current_layer + 1)
                new_id_ce_siblings = new_id_ce_d[new_id_ce_layer][:, 1]
                new_id_all_siblings = _get_all_siblings(
                    cg, new_parent_node.node_id, new_id_ce_siblings
                )
                new_parent_node.children = np.concatenate(
                    [[new_id], new_id_all_siblings]
                )
                layer_new_ids_d[current_layer + 1].append(new_parent_node.node_id)
            new_nodes_d[new_parent_node.node_id] = new_parent_node
    return layer_new_ids_d[cg.meta.layer_count]


def _analyze_atomic_edge(
    cg, atomic_edges: Iterable[np.ndarray]
) -> Tuple[Iterable, Dict]:
    """
    Determine if atomic edges are within the chunk.
    If not, they are cross edges between two L2 IDs in adjacent chunks.
    Returns edges and cross edges accordingly.
    """
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    mask = edge_layers == 1

    # initialize with in-chunk edges
    parent_edges = [cg.get_parents(_) for _ in atomic_edges[mask]]

    # cross chunk edges
    atomic_cross_edges_d = {}
    for edge, layer in zip(atomic_edges[~mask], edge_layers[~mask]):
        parent_edge = cg.get_parents(edge)
        parent_1 = parent_edge[0]
        parent_2 = parent_edge[1]
        cross_edges_d[parent_1] = {layer: edge}
        cross_edges_d[parent_2] = {layer: edge[::-1]}
        parent_edges.append([parent_1, parent_1])
        parent_edges.append([parent_2, parent_2])
    return (parent_edges, cross_edges_d)


def add_edge(
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
    edges, l2_atomic_cross_edges_d = _analyze_atomic_edge(cg, edge)
    atomic_cross_edges_d = merge_cross_edge_dicts_multiple(
        cg.get_atomic_cross_edges(np.unique(edges)), l2_atomic_cross_edges_d
    )

    cross_edges_d = {}
    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    # TODO get_cross_chunk_edges multiple node IDs all at once
    # for l2id in graph_node_ids:
    #     cross_edges_d[l2id] = cg.get_cross_chunk_edges(l2id)

    ccs = flatgraph.connected_components(graph)

    new_l2ids = []
    l2_cross_edges_d = {}
    for cc in ccs:
        l2ids = graph_node_ids[cc]
        chunk_id = cg.get_chunk_id(l2ids[0])
        new_l2ids.append(cg.id_client.create_node_id(chunk_id))

        l2_cross_edges_d[new_l2ids[-1]] = concatenate_cross_edge_dicts(
            [cross_edges_d[l2id] for l2id in l2ids]
        )

    new_cross_edges_d_d = {}
    for l2id, cross_edges_d in l2_cross_edges_d.items():
        layer_, edges_ = get_min_layer_cross_edges(cg.meta, cross_edges_d)
        new_cross_edges_d_d[l2id] = {layer_: edges_}

    return _create_parents(
        cg, new_cross_edges_d_d.copy(), operation_id=operation_id, time_stamp=timestamp,
    )


def remove_edge(
    cg,
    operation_id: np.uint64,
    atomic_edges: Sequence[Sequence[np.uint64]],
    time_stamp: datetime.datetime,
):
    pass
