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
    edges, l2_atomic_cross_edges_d = _analyze_atomic_edge(cg, atomic_edges)
    atomic_cross_edges_d = merge_cross_edge_dicts_multiple(
        cg.get_atomic_cross_edges(np.unique(edges)), l2_atomic_cross_edges_d
    )

    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)

    with TimeIt("get_cross_chunk_edges cross_edges_d"):
        cross_edges_d = {
            id_: cg.get_min_layer_cross_edges(id_, [atomic_cross_edges_d[id_]])
            for id_ in graph_node_ids
        }

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
    atomic_edges_mirrored = np.concatenate(
        [atomic_edges, atomic_edges[:, ::-1]], axis=0
    )
    n_edges = atomic_edges_mirrored.shape[0]  # pylint: disable=E1136
    atomic_edges_mirrored = atomic_edges_mirrored.view(dtype="u8,u8").reshape(n_edges)

    rows = []  # list of rows to be written to BigTable
    lvl2_dict = {}
    lvl2_cross_chunk_edge_dict = {}

    # Analyze atomic_edges --> translate them to lvl2 edges and extract cross
    # chunk edges to be removed
    edges, l2_atomic_cross_edges_d = _analyze_atomic_edge(cg, atomic_edges)
    lvl2_node_ids = np.unique(edges)

    for lvl2_node_id in lvl2_node_ids:
        chunk_id = cg.get_chunk_id(lvl2_node_id)
        chunk_edges, _, _ = cg.get_subgraph_chunk(lvl2_node_id, make_unique=False)

        child_chunk_ids = cg.get_child_chunk_ids(chunk_id)

        assert len(child_chunk_ids) == 1
        child_chunk_id = child_chunk_ids[0]

        children_ids = np.unique(chunk_edges)
        children_chunk_ids = cg.get_chunk_ids_from_node_ids(children_ids)
        children_ids = children_ids[children_chunk_ids == child_chunk_id]

        # These edges still contain the removed edges.
        # For consistency reasons we can only write to BigTable one time.
        # Hence, we have to evict the to be removed "atomic_edges" from the
        # queried edges.
        retained_edges_mask = ~np.in1d(
            chunk_edges.view(dtype="u8,u8").reshape(chunk_edges.shape[0]),
            double_atomic_edges_view,
        )

        chunk_edges = chunk_edges[retained_edges_mask]

        edge_layers = cg.get_cross_chunk_edges_layer(chunk_edges)
        cross_edge_mask = edge_layers != 1

        cross_edges = chunk_edges[cross_edge_mask]
        cross_edge_layers = edge_layers[cross_edge_mask]
        chunk_edges = chunk_edges[~cross_edge_mask]

        isolated_child_ids = children_ids[~np.in1d(children_ids, chunk_edges)]
        isolated_edges = np.vstack([isolated_child_ids, isolated_child_ids]).T

        graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
            np.concatenate([chunk_edges, isolated_edges]), make_directed=True
        )

        ccs = flatgraph_utils.connected_components(graph)

        new_parent_ids = cg.get_unique_node_id_range(chunk_id, len(ccs))

        for i_cc, cc in enumerate(ccs):
            new_parent_id = new_parent_ids[i_cc]
            cc_node_ids = unique_graph_ids[cc]

            lvl2_dict[new_parent_id] = [lvl2_node_id]

            # Write changes to atomic nodes and new lvl2 parent row
            val_dict = {column_keys.Hierarchy.Child: cc_node_ids}
            rows.append(
                cg.mutate_row(
                    serializers.serialize_uint64(new_parent_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )

            for cc_node_id in cc_node_ids:
                val_dict = {column_keys.Hierarchy.Parent: new_parent_id}

                rows.append(
                    cg.mutate_row(
                        serializers.serialize_uint64(cc_node_id),
                        val_dict,
                        time_stamp=time_stamp,
                    )
                )

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
                    val_dict[
                        column_keys.Connectivity.CrossChunkEdge[cc_layer]
                    ] = layer_cross_edges
                    lvl2_cross_chunk_edge_dict[new_parent_id][
                        cc_layer
                    ] = layer_cross_edges

            if len(val_dict) > 0:
                rows.append(
                    cg.mutate_row(
                        serializers.serialize_uint64(new_parent_id),
                        val_dict,
                        time_stamp=time_stamp,
                    )
                )

        if cg.n_layers == 2:
            rows.extend(
                update_root_id_lineage(
                    cg,
                    new_parent_ids,
                    [lvl2_node_id],
                    operation_id=operation_id,
                    time_stamp=time_stamp,
                )
            )

    # Write atomic nodes
    rows.extend(_write_atomic_split_edges(cg, atomic_edges, time_stamp=time_stamp))

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

