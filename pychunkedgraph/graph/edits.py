import datetime
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Sequence
from collections import defaultdict

from . import types
from .utils import basetypes
from .utils import flatgraph
from .utils.context_managers import TimeIt
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists
from .edges.utils import filter_min_layer_cross_edges
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple
from ..utils.general import in2d


"""
Their children might be "too much" due to the split; even within one chunk. How do you deal with that?

a good way to test this is to check all intermediate nodes from the component before the split and then after the split. Basically, get all childrens in all layers of the one component before and the (hopefully) two components afterwards. Check (1) are all intermediate nodes from before in a list after and (2) do all intermediate nodes appear exactly one time after the split (aka is there overlap between the resulting components). (edited) 

for (2) Overlap can be real but then they have to be exactly the same. In that case the removed edges did not split the component in two

"""


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


def _create_parent_node(
    cg, new_node: types.Node, parent_layer: int = None
) -> types.Node:
    new_id = new_node.node_id
    parent_chunk_id = cg.get_parent_chunk_id(new_id, parent_layer)
    new_parent_id = cg.id_client.create_node_id(parent_chunk_id)
    new_parent_node = types.Node(new_parent_id)
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
            {id_: types.Node(id_) for id_ in new_ids[~np.in1d(new_ids, new_ids_)]}
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
    time_stamp: datetime.datetime = None,
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
    new_cross_edges_d_d = {}
    for cc in ccs:
        l2ids = graph_node_ids[cc]
        new_id = cg.id_client.create_node_id(cg.get_chunk_id(l2ids[0]))
        new_cross_edges_d_d[new_id] = cg.get_min_layer_cross_edges(
            new_id, [atomic_cross_edges_d[l2id] for l2id in l2ids]
        )
    return _create_parents(
        cg,
        new_cross_edges_d_d.copy(),
        operation_id=operation_id,
        time_stamp=time_stamp,
    )


def _process_l2_agglomeration(agg: types.Agglomeration, removed_edges: np.ndarray):
    """
    For a given L2 id, remove given edges
    and calculate new connected components.
    """
    chunk_edges = agg.in_edges.get_pairs()
    cross_edges = agg.cross_edges.get_pairs()
    chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]
    cross_edges = cross_edges[~in2d(cross_edges, removed_edges)]

    isolated_ids = agg.supervoxels[~np.in1d(agg.supervoxels, chunk_edges)]
    isolated_edges = np.vstack([isolated_ids, isolated_ids]).T
    graph, _, _, unique_graph_ids = flatgraph.build_gt_graph(
        np.concatenate([chunk_edges, isolated_edges]), make_directed=True
    )
    return flatgraph.connected_components(graph), unique_graph_ids, cross_edges


def _filter_component_cross_edges(
    cc_ids: np.ndarray, cross_edges: np.ndarray, cross_edge_layers: np.ndarray
):
    """
    Filters cross edges for a connected component `cc_ids`
    from `cross_edges` of the complete chunk.
    """
    mask = np.in1d(cross_edges[:, 0], cc_ids)
    cross_edges_ = cross_edges[mask]
    cross_edge_layers_ = cross_edge_layers[mask]

    edges_d = {}
    for layer in np.unique(cross_edge_layers_):
        edge_m = cross_edge_layers_ == layer
        _cross_edges = cross_edges_[edge_m]
        if _cross_edges.size:
            edges_d[layer] = _cross_edges
    return edges_d


def remove_edges(
    cg,
    operation_id: np.uint64,
    atomic_edges: Sequence[Sequence[np.uint64]],
    time_stamp: datetime.datetime,
):
    # This view of the to be removed edges helps us to
    # compute the mask of retained edges in chunk
    removed_edges = np.concatenate([atomic_edges, atomic_edges[:, ::-1]], axis=0)
    edges, _ = _analyze_atomic_edges(cg, atomic_edges)
    l2_ids = np.unique(edges)
    l2_chunk_ids = cg.get_chunk_ids_from_node_ids(l2_ids)
    l2id_chunk_id_d = dict(zip(l2_ids, l2_chunk_ids))
    l2id_agglomeration_d = cg.get_subgraph(l2_ids)

    atomic_cross_edges_d = {}
    for l2_agg in l2id_agglomeration_d.values():
        ccs, unique_graph_ids, cross_edges = _process_l2_agglomeration(
            l2_agg, removed_edges
        )
        cross_edge_layers = cg.get_cross_chunk_edges_layer(cross_edges)
        new_parent_ids = cg.id_client.create_node_ids(
            l2id_chunk_id_d[l2_agg.node_id], len(ccs)
        )
        for i_cc, cc in enumerate(ccs):
            new_parent_id = new_parent_ids[i_cc]
            cc_ids = unique_graph_ids[cc]
            atomic_cross_edges_d[new_parent_id] = _filter_component_cross_edges(
                cc_ids, cross_edges, cross_edge_layers
            )

    new_cross_edges_d_d = {}
    for new_id, cross_edges in atomic_cross_edges_d.items():
        new_cross_edges_d_d[new_id] = cg.get_min_layer_cross_edges(
            new_id, [cross_edges]
        )
    return _create_parents(
        cg,
        new_cross_edges_d_d.copy(),
        operation_id=operation_id,
        time_stamp=time_stamp,
    )

