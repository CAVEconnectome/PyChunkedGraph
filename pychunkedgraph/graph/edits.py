import datetime
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Sequence
from collections import defaultdict

from .types import Node
from .types import empty_2d
from .utils import basetypes
from .utils import flatgraph
from .utils.context_managers import TimeIt
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists
from .edges.utils import filter_min_layer_cross_edges
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple
from ..utils.general import in2d


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
        new_ids = np.array(layer_new_ids_d[current_layer], basetypes.NODE_ID)
        new_ids_ = np.fromiter(new_nodes_d.keys(), dtype=basetypes.NODE_ID)
        new_nodes_d.update(
            {id_: Node(id_) for id_ in new_ids[~np.in1d(new_ids, new_ids_)]}
        )
        new_ids_ = np.fromiter(new_cross_edges_d_d.keys(), dtype=basetypes.NODE_ID)
        new_cross_edges_d_d.update(
            cg.get_cross_chunk_edges(
                new_ids[~np.in1d(new_ids, new_ids_)], nodes_cache=new_nodes_d
            )
        )
        for new_id in new_ids:
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


def _analyze_atomic_edges(
    cg, atomic_edges: Iterable[np.ndarray]
) -> Tuple[Iterable, Dict]:
    """
    Determine if atomic edges are within the chunk.
    If not, they are cross edges between two L2 IDs in adjacent chunks.
    Returns edges between L2 IDs and atomic cross edges.
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
        atomic_cross_edges_d[parent_1] = {layer: [edge]}
        atomic_cross_edges_d[parent_2] = {layer: [edge[::-1]]}
        parent_edges.append([parent_1, parent_1])
        parent_edges.append([parent_2, parent_2])
    return (parent_edges, atomic_cross_edges_d)


def add_edge(
    cg,
    *,
    atomic_edges: np.ndarray,
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
    edges, l2_atomic_cross_edges_d = _analyze_atomic_edges(cg, atomic_edges)
    atomic_cross_edges_d = merge_cross_edge_dicts_multiple(
        cg.get_atomic_cross_edges(np.unique(edges)), l2_atomic_cross_edges_d
    )

    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)
    new_l2ids = []
    l2_cross_edges_d = {}
    for cc in ccs:
        l2ids = graph_node_ids[cc]
        new_id = cg.id_client.create_node_id(cg.get_chunk_id(l2ids[0]))
        new_l2ids.append(new_id)
        l2_cross_edges_d[new_id] = concatenate_cross_edge_dicts(
            [atomic_cross_edges_d[l2id] for l2id in l2ids]
        )

        l2_cross_edges_d[new_id] = cg.get_min_layer_cross_edges(
            new_id, [l2_cross_edges_d[new_id]]
        )

    new_cross_edges_d_d = {}
    for l2id, cross_edges_d in l2_cross_edges_d.items():
        layer_, edges_ = filter_min_layer_cross_edges(cg.meta, cross_edges_d)
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
    # This view of the to be removed edges helps us to compute the mask
    # of the retained edges in each chunk
    removed_edges = np.concatenate([atomic_edges, atomic_edges[:, ::-1]], axis=0)

    lvl2_cross_chunk_edge_dict = {}

    # Analyze atomic_edges --> translate them to lvl2 edges and extract cross
    # chunk edges to be removed
    edges, _ = _analyze_atomic_edges(cg, atomic_edges)

    l2_ids = np.unique(edges)
    l2_chunk_ids = cg.get_chunk_ids_from_node_ids(l2_ids)
    l2id_chunk_id_d = dict(zip(l2_ids, l2_chunk_ids))
    l2id_agglomeration_d = cg.get_subgraph(l2_ids, layer_2=True)

    for l2_id, l2_agg in l2id_agglomeration_d.items():
        chunk_edges = l2_agg.in_edges.get_pairs()
        cross_edges = l2_agg.cross_edges.get_pairs()
        chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]
        cross_edges = cross_edges[~in2d(cross_edges, removed_edges)]
        cross_edge_layers = cg.get_cross_chunk_edges_layer(cross_edges)

        isolated_ids = l2_agg.supervoxels[~np.in1d(l2_agg.supervoxels, chunk_edges)]
        isolated_edges = np.vstack([isolated_ids, isolated_ids]).T

        graph, _, _, unique_graph_ids = flatgraph.build_gt_graph(
            np.concatenate([chunk_edges, isolated_edges]), make_directed=True
        )
        ccs = flatgraph.connected_components(graph)
        new_parent_ids = cg.id_client.create_node_ids(l2id_chunk_id_d[l2_id], len(ccs))

        for i_cc, cc in enumerate(ccs):
            new_parent_id = new_parent_ids[i_cc]
            cc_node_ids = unique_graph_ids[cc]

            # Cross edges ---
            cross_edge_m = np.in1d(cross_edges[:, 0], cc_node_ids)
            cc_cross_edges = cross_edges[cross_edge_m]
            cc_cross_edge_layers = cross_edge_layers[cross_edge_m]
            u_cc_cross_edge_layers = np.unique(cc_cross_edge_layers)

            lvl2_cross_chunk_edge_dict[new_parent_id] = {}

            for l in range(2, cg.n_layers):
                empty_edges = column_keys.Connectivity.CrossChunkEdge.deserialize(b"")
                lvl2_cross_chunk_edge_dict[new_parent_id][l] = empty_edges

            val_dict = {}
            for cc_layer in u_cc_cross_edge_layers:
                edge_m = cc_cross_edge_layers == cc_layer
                layer_cross_edges = cc_cross_edges[edge_m]

                if len(layer_cross_edges) > 0:
                    lvl2_cross_chunk_edge_dict[new_parent_id][
                        cc_layer
                    ] = layer_cross_edges

    # Propagate changes up the tree
    if cg.n_layers > 2:
        new_root_ids, new_rows = propagate_edits_to_root(
            cg,
            lvl2_dict.copy(),
            lvl2_cross_chunk_edge_dict,
            operation_id=operation_id,
            time_stamp=time_stamp,
        )
        rows.extend(new_rows)
    else:
        new_root_ids = np.array(list(lvl2_dict.keys()))

    return new_root_ids, list(lvl2_dict.keys()), rows

