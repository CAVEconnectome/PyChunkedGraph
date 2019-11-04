import fastremap
import graph_tool
import numpy as np
from pychunkedgraph.backend import flatgraph_utils
from pychunkedgraph.meshing import meshgen, meshgen_utils
from cloudvolume import Storage


def find_l2_shortest_path(cg, source_l2_id: np.uint64, target_l2_id: np.uint64):
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
    shared_parent_id = cg.get_first_shared_parent(source_l2_id, target_l2_id)
    if shared_parent_id is None:
        return None
    lvl2_ids = cg.get_children_at_layer(shared_parent_id, 2)
    cce_dict = cg.read_cross_chunk_edges_for_nodes(
        lvl2_ids, start_layer=2, end_layer=cg.get_chunk_layer(shared_parent_id)
    )

    # Gather all of the supervoxel ids into two lists, we will map them to
    # their parent lvl2 ids
    edge_array = np.concatenate(list(cce_dict.values()))
    known_supervoxels_list = []
    known_l2_list = []
    unknown_supervoxel_list = []
    for lvl2_id in cce_dict:
        known_supervoxels_for_lv2_id = cce_dict[lvl2_id][:, 0]
        unknown_supervoxels_for_lv2_id = cce_dict[lvl2_id][:, 1]
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

    # Create a graph-tool graph of the mapped cross-chunk-edges
    weighted_graph, _, _, graph_indexed_l2_ids = flatgraph_utils.build_gt_graph(
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


def compute_centroid_by_range(vertices):
    bbox_min = np.amin(vertices, axis=0)
    bbox_max = np.amax(vertices, axis=0)
    return bbox_min + ((bbox_max - bbox_min) / 2)


def compute_centroid_with_chunk_boundary(cg, vertices, l2_id, last_l2_id):
    """
    Given a level 2 id, the vertices of its mesh, and the level 2 id preceding it in
    a path, return the center point of the mesh on the chunk boundary separating the two
    ids, and the center point of the entire mesh.

    :param cg: ChunkedGraph object
    :param vertices: [[np.float]]
    :param l2_id: np.uint64
    :param last_l2_id: np.uint64 or None
    :return: [np.float]
    """
    centroid_by_range = compute_centroid_by_range(vertices)
    if last_l2_id is None:
        return [centroid_by_range]
    l2_id_cc = cg.get_chunk_coordinates(l2_id)
    last_l2_id_cc = cg.get_chunk_coordinates(last_l2_id)

    # Given the coordinates of the two level 2 ids, find the chunk boundary
    axis_change = 2
    look_for_max = True
    if l2_id_cc[0] != last_l2_id_cc[0]:
        axis_change = 0
    elif l2_id_cc[1] != last_l2_id_cc[1]:
        axis_change = 1
    if np.sum(l2_id_cc - last_l2_id_cc) > 0:
        look_for_max = False
    if look_for_max:
        value_to_filter = np.amax(vertices[:, axis_change])
    else:
        value_to_filter = np.amin(vertices[:, axis_change])
    chunk_boundary_vertices = vertices[
        np.where(vertices[:, axis_change] == value_to_filter)
    ]

    # Get the center point of the mesh on the chunk boundary
    bbox_min = np.amin(chunk_boundary_vertices, axis=0)
    bbox_max = np.amax(chunk_boundary_vertices, axis=0)
    return [bbox_min + ((bbox_max - bbox_min) / 2), centroid_by_range]


def compute_mesh_centroids_of_l2_ids(cg, l2_ids, flatten=False):
    """
    Given a list of l2_ids, return a tuple containing a dict that maps l2_ids to their
    mesh's centroid (a global coordinate), and a list of the l2_ids for which the mesh does not exist.

    :param cg: ChunkedGraph object
    :param l2_ids: Sequence[np.uint64]
    :return: Union[Dict[np.uint64, np.ndarray], [np.uint64], [np.uint64]]
    """
    fragments_to_fetch = [
        f"{l2_id}:0:{meshgen_utils.get_chunk_bbox_str(cg, cg.get_chunk_id(l2_id))}"
        for l2_id in l2_ids
    ]
    if flatten:
        centroids_with_chunk_boundary_points = []
    else:
        centroids_with_chunk_boundary_points = {}
    last_l2_id = None
    failed_l2_ids = []
    with Storage(cg.cv_mesh_path) as storage:
        files_contents = storage.get_files(fragments_to_fetch)
        fragment_map = {}
        for i in range(len(files_contents)):
            fragment_map[files_contents[i]["filename"]] = files_contents[i]
        for i in range(len(fragments_to_fetch)):
            fragment_to_fetch = fragments_to_fetch[i]
            l2_id = l2_ids[i]
            try:
                fragment = fragment_map[fragment_to_fetch]
                if fragment["content"] is not None and fragment["error"] is None:
                    mesh = meshgen.decode_draco_mesh_buffer(fragment["content"])
                    if flatten:
                        centroids_with_chunk_boundary_points.extend(
                            compute_centroid_with_chunk_boundary(
                                cg, mesh["vertices"], l2_id, last_l2_id
                            )
                        )
                    else:
                        centroids_with_chunk_boundary_points[
                            l2_id
                        ] = compute_centroid_with_chunk_boundary(
                            cg, mesh["vertices"], l2_id, last_l2_id
                        )
            except:
                failed_l2_ids.append(l2_id)
            last_l2_id = l2_id
    return centroids_with_chunk_boundary_points, failed_l2_ids
