import typing

import fastremap
import graph_tool
import numpy as np

from pychunkedgraph.graph.utils import flatgraph

from ..subgraph import get_subgraph_nodes


def get_first_shared_parent(
    cg, first_node_id: np.uint64, second_node_id: np.uint64, time_stamp=None
):
    """
    Get the common parent of first_node_id and second_node_id with the lowest layer.
    Returns None if the two nodes belong to different root ids.
    :param first_node_id: np.uint64
    :param second_node_id: np.uint64
    :return: np.uint64 or None
    """
    first_node_parent_ids = set()
    second_node_parent_ids = set()
    cur_first_node_parent = first_node_id
    cur_second_node_parent = second_node_id
    while cur_first_node_parent is not None or cur_second_node_parent is not None:
        if cur_first_node_parent is not None:
            first_node_parent_ids.add(cur_first_node_parent)
        if cur_second_node_parent is not None:
            second_node_parent_ids.add(cur_second_node_parent)
        if cur_first_node_parent in second_node_parent_ids:
            return cur_first_node_parent
        if cur_second_node_parent in first_node_parent_ids:
            return cur_second_node_parent
        if cur_first_node_parent is not None:
            cur_first_node_parent = cg.get_parent(
                cur_first_node_parent, time_stamp=time_stamp
            )
        if cur_second_node_parent is not None:
            cur_second_node_parent = cg.get_parent(
                cur_second_node_parent, time_stamp=time_stamp
            )
    return None


def get_children_at_layer(
    cg,
    agglomeration_id: np.uint64,
    layer: int,
    allow_lower_layers: bool = False,
):
    """
    Get the children of agglomeration_id that have layer = layer.
    :param agglomeration_id: np.uint64
    :param layer: int
    :return: [np.uint64]
    """
    nodes_to_query = [agglomeration_id]
    children_at_layer = []
    while True:
        children = cg.get_children(nodes_to_query, flatten=True)
        children_layers = cg.get_chunk_layers(children)
        if allow_lower_layers:
            stop_layer_mask = children_layers <= layer
        else:
            stop_layer_mask = children_layers == layer
        continue_layer_mask = children_layers > layer
        found_children_at_layer = children[stop_layer_mask]
        children_at_layer.append(found_children_at_layer)
        nodes_to_query = children[continue_layer_mask]
        if not np.any(nodes_to_query):
            break
    return np.concatenate(children_at_layer)


def get_lvl2_edge_list(
    cg,
    node_id: np.uint64,
    bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
):
    """get an edge list of lvl2 ids for a particular node

    :param cg: ChunkedGraph object
    :param node_id: np.uint64 that you want the edge list for
    :param bbox: Optional[Sequence[Sequence[int]]] a bounding box to limit the search
    """

    if bbox is None:
        # maybe temporary, this was the old implementation
        lvl2_ids = get_children_at_layer(cg, node_id, 2)
    else:
        lvl2_ids = get_subgraph_nodes(
            cg,
            node_id,
            bbox=bbox,
            bbox_is_coordinate=True,
            return_layers=[2],
            return_flattened=True,
        )

    edges = _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True)
    return edges


def _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=False):
    # protect in case there are no lvl2 ids
    if len(lvl2_ids) == 0:
        return np.empty((0, 2), dtype=np.uint64)

    cce_dict = cg.get_atomic_cross_edges(lvl2_ids)

    # Gather all of the supervoxel ids into two lists, we will map them to
    # their parent lvl2 ids
    edge_array = []
    for l2_id in cce_dict:
        for level in cce_dict[l2_id]:
            edge_array.append(cce_dict[l2_id][level])
    edge_array = np.concatenate(edge_array)
    known_supervoxels_list = []
    known_l2_list = []
    unknown_supervoxel_list = []
    for lvl2_id in cce_dict:
        for level in cce_dict[lvl2_id]:
            known_supervoxels_for_lv2_id = cce_dict[lvl2_id][level][:, 0]
            unknown_supervoxels_for_lv2_id = cce_dict[lvl2_id][level][:, 1]
            known_supervoxels_list.append(known_supervoxels_for_lv2_id)
            known_l2_list.append(np.full(known_supervoxels_for_lv2_id.shape, lvl2_id))
            unknown_supervoxel_list.append(unknown_supervoxels_for_lv2_id)

    # Create two arrays to map supervoxels for which we know their parents
    known_supervoxel_array, unique_indices = np.unique(
        np.concatenate(known_supervoxels_list), return_index=True
    )
    known_l2_array = (np.concatenate(known_l2_list))[unique_indices]
    unknown_supervoxel_array = np.unique(np.concatenate(unknown_supervoxel_list))

    # Call get_parents on any supervoxels for which we don't know their parents
    supervoxels_to_query_parent = np.setdiff1d(
        unknown_supervoxel_array, known_supervoxel_array
    )
    if len(supervoxels_to_query_parent) > 0:
        missing_l2_ids = cg.get_parents(supervoxels_to_query_parent)
        known_supervoxel_array = np.concatenate(
            (known_supervoxel_array, supervoxels_to_query_parent)
        )
        known_l2_array = np.concatenate((known_l2_array, missing_l2_ids))

    # Map the cross-chunk edges from supervoxels to lvl2 ids
    edge_view = edge_array.view()
    edge_view.shape = -1
    fastremap.remap_from_array_kv(edge_view, known_supervoxel_array, known_l2_array)

    edge_array = np.unique(np.sort(edge_array, axis=1), axis=0)

    if induced:
        # make this an induced subgraph
        # keep only the edges that are between the lvl2 ids asked for
        edge_array = edge_array[
            np.isin(edge_array[:, 0], lvl2_ids) & np.isin(edge_array[:, 1], lvl2_ids)
        ]

    return edge_array


def find_l2_shortest_path(
    cg, source_l2_id: np.uint64, target_l2_id: np.uint64, time_stamp=None
):
    """
    Find a path of level 2 ids that connect two level 2 node ids through cross chunk edges.
    Return a list of level 2 ids representing this path.
    Return None if the two level 2 ids do not belong to the same object.
    :param cg: ChunkedGraph object
    :param source_l2_id: np.uint64
    :param target_l2_id: np.uint64
    :return: [np.uint64] or None
    """
    # Get the cross-chunk edges that we need to build the graph
    shared_parent_id = get_first_shared_parent(
        cg, source_l2_id, target_l2_id, time_stamp
    )
    if shared_parent_id is None:
        return None

    edge_array = get_lvl2_edge_list(cg, shared_parent_id)
    # Create a graph-tool graph of the mapped cross-chunk-edges
    weighted_graph, _, _, graph_indexed_l2_ids = flatgraph.build_gt_graph(
        edge_array, is_directed=False
    )

    # Find the shortest path from the source_l2_id to the target_l2_id
    source_graph_id = np.where(graph_indexed_l2_ids == source_l2_id)[0][0]
    target_graph_id = np.where(graph_indexed_l2_ids == target_l2_id)[0][0]
    source_vertex = weighted_graph.vertex(source_graph_id)
    target_vertex = weighted_graph.vertex(target_graph_id)
    vertex_list, _ = graph_tool.topology.shortest_path(
        weighted_graph, source=source_vertex, target=target_vertex
    )

    # Remap the graph-tool ids to lvl2 ids and return the path
    vertex_indices = [weighted_graph.vertex_index[vertex] for vertex in vertex_list]
    l2_traversal_path = graph_indexed_l2_ids[vertex_indices]
    return l2_traversal_path


def compute_rough_coordinate_path(cg, l2_ids):
    """
    Given a list of l2_ids, return a list of rough coordinates representing
    the path the l2_ids form.
    :param cg: ChunkedGraph object
    :param l2_ids: Sequence[np.uint64]
    :return: [np.ndarray]
    """
    coordinate_path = []
    for l2_id in l2_ids:
        chunk_center = cg.get_chunk_coordinates(l2_id) + np.array([0.5, 0.5, 0.5])
        coordinate = chunk_center * np.array(
            cg.meta.graph_config.CHUNK_SIZE
        ) + np.array(cg.meta.cv.mip_voxel_offset(0))
        coordinate = coordinate * np.array(cg.meta.cv.mip_resolution(0))
        coordinate = coordinate.astype(np.float32)
        coordinate_path.append(coordinate)
    return coordinate_path
