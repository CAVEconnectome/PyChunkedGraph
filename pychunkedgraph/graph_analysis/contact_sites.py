import collections
from enum import auto, Enum
import itertools

from contact_points import find_contact_points
import numpy as np

from pychunkedgraph.backend import flatgraph_utils
from pychunkedgraph.backend.utils import column_keys

def _get_edges_for_contact_site_graph(
    cg, root_id, bounding_box, bb_is_coordinate, end_time
):
    """
    Helper function for get_contact_sites that queries the ChunkedGraph for edges and returns
    a list of edges that are involved in the root_id's contact sites.
    """
    # Get information about the root id
    # All supervoxels
    sv_ids = cg.get_subgraph_nodes(
        root_id, bounding_box=bounding_box, bb_is_coordinate=bb_is_coordinate
    )
    # All edges that are _not_ connected / on
    edges, _, areas = cg.get_subgraph_edges(
        root_id,
        bounding_box=bounding_box,
        bb_is_coordinate=bb_is_coordinate,
        connected_edges=False,
    )
    # Build area lookup dictionary
    edge_mask = ~np.in1d(edges, sv_ids).reshape(-1, 2)
    area_mask = np.where(edge_mask)[0]
    masked_areas = areas[area_mask]
    contact_sites_svs = edges[edge_mask]
    contact_sites_svs_area_dict = collections.defaultdict(int)

    for area, sv_id in zip(masked_areas, contact_sites_svs):
        contact_sites_svs_area_dict[sv_id] += area

    # Extract svs from contacting root ids
    unique_contact_sites_svs = np.unique(contact_sites_svs)

    # Load edges of these contact sites' supervoxels
    edges_contact_sites_svs_rows = cg.read_node_id_rows(
        node_ids=unique_contact_sites_svs,
        columns=[column_keys.Connectivity.Partner, column_keys.Connectivity.Connected],
        end_time=end_time,
        end_time_inclusive=True,
    )

    contact_sites_edges = []
    for row_information in edges_contact_sites_svs_rows.items():
        sv_edges, _, _ = cg._retrieve_connectivity(row_information)
        contact_sites_edges.extend(sv_edges)
    if len(contact_sites_edges) == 0:
        return None, None, False
    contact_sites_edges_array = np.array(contact_sites_edges)
    contact_sites_edge_mask = np.isin(
        contact_sites_edges_array[:, 1], unique_contact_sites_svs
    )
    # Fake edges to ensure lone contact site supervoxels show up as a connected component
    self_edges = np.stack((unique_contact_sites_svs, unique_contact_sites_svs), axis=-1)
    contact_sites_graph_edges = np.concatenate(
        (contact_sites_edges_array[contact_sites_edge_mask], self_edges), axis=0
    )
    return contact_sites_graph_edges, contact_sites_svs_area_dict, True

def get_contact_sites(
    cg,
    root_id,
    bounding_box=None,
    bb_is_coordinate=True,
    compute_partner=True,
    end_time=None,
    voxel_location=False,
    areas_only=False,
    as_list=False
):
    """
    Given a root id, return a dictionary containing all the contact sites with other roots in the dataset.

    If compute_partner=True, the keys of the dictionary are other root ids and the values are lists of all the contact sites
    these two roots make. Each value in the list is a tuple containing three entries. If voxel_location=False, 
    then the first two entries are chunk coordinates that bound part of the contact site; the third entry 
    is the area of the contact site. If voxel_location=True, the first two entries are the positions of those two chunks
    in global coordinates instead. If areas_only=True, then the value is just the area and no location is returned.
    
    If compute_partner=False, the keys of the dictionary are unsigned integers counting up from 0. The values are 
    the lists containing one tuple of the kind specified in the above paragraph.
    """
    contact_sites_graph_edges, contact_sites_svs_area_dict, any_contact_sites = _get_edges_for_contact_site_graph(
        cg, root_id, bounding_box, bb_is_coordinate, end_time
    )

    if not any_contact_sites:
        return collections.defaultdict(list)

    contact_sites_svs_area_dict_vec = np.vectorize(contact_sites_svs_area_dict.get)

    graph, _, _, unique_sv_ids = flatgraph_utils.build_gt_graph(
        contact_sites_graph_edges, make_directed=True
    )

    connected_components = flatgraph_utils.connected_components(graph)

    contact_site_dict = collections.defaultdict(list)
    # First create intermediary map of supervoxel to contact sites, so we
    # can call cg.get_roots() on all supervoxels at once.
    intermediary_sv_dict = {}
    for cc in connected_components:
        cc_sv_ids = unique_sv_ids[cc]
        contact_sites_areas = contact_sites_svs_area_dict_vec(cc_sv_ids)

        representative_sv = cc_sv_ids[0]
        # Tuple of location and area of contact site
        chunk_coordinates = cg.get_chunk_coordinates(representative_sv)
        if areas_only:
            data_pair = np.sum(contact_sites_areas)
        elif voxel_location:
            voxel_lower_bound = (
                cg.vx_vol_bounds[:, 0] + cg.chunk_size * chunk_coordinates
            )
            voxel_upper_bound = cg.vx_vol_bounds[:, 0] + cg.chunk_size * (
                chunk_coordinates + 1
            )
            data_pair = (
                voxel_lower_bound * cg.segmentation_resolution,
                voxel_upper_bound * cg.segmentation_resolution,
                np.sum(contact_sites_areas),
            )
        else:
            data_pair = (
                chunk_coordinates,
                chunk_coordinates + 1,
                np.sum(contact_sites_areas),
            )

        if compute_partner:
            # Cast np.uint64 to int for dict key because int is hashable
            intermediary_sv_dict[int(representative_sv)] = data_pair
        else:
            contact_site_dict[len(contact_site_dict)].append(data_pair)
    if compute_partner:
        sv_list = np.array(list(intermediary_sv_dict.keys()), dtype=np.uint64)
        partner_roots = cg.get_roots(sv_list)
        for i in range(len(partner_roots)):
            contact_site_dict[int(partner_roots[i])].append(
                intermediary_sv_dict.get(int(sv_list[i]))
            )

    if as_list:
        contact_site_list = []
        for partner_id in contact_site_dict:
            if compute_partner:
                contact_site_list.append({
                    'segment_id': np.uint64(partner_id),
                    'contact_site_areas': contact_site_dict[partner_id]
                })
            else:
                contact_site_list.append({
                    'segment_id': partner_id,
                    'contact_site_areas': contact_site_dict[partner_id]
                })
        return contact_site_list

    return contact_site_dict


def _retrieve_connectivity_optimized(cg, dict_item):
    """
    An altered version of cg._retrieve_connectivity that is optimized
    for the purpose of retrieving contact sites.

    WARNING: This method of determining connectivity is soon to be
    deprecated. - 10/15/19
    """
    node_id, row = dict_item

    tmp = set()
    for x in itertools.chain.from_iterable(
        generation.value for generation in row[column_keys.Connectivity.Connected][::-1]
    ):
        tmp.remove(x) if x in tmp else tmp.add(x)

    connected_indices = np.fromiter(tmp, np.uint64)
    if column_keys.Connectivity.Partner in row:
        edges = np.fromiter(
            itertools.chain.from_iterable(
                (node_id, partner_id)
                for generation in row[column_keys.Connectivity.Partner][::-1]
                for partner_id in generation.value
            ),
            dtype=basetypes.NODE_ID,
        ).reshape((-1, 2))
        edges_in = cg._connected_or_not(edges, connected_indices, True)
        edges_out = cg._connected_or_not(edges, connected_indices, False)
    else:
        edges_in = np.empty((0, 2), basetypes.NODE_ID)
        edges_out = np.empty((0, 2), basetypes.NODE_ID)

    if column_keys.Connectivity.Area in row:
        areas = np.fromiter(
            itertools.chain.from_iterable(
                generation.value
                for generation in row[column_keys.Connectivity.Area][::-1]
            ),
            dtype=basetypes.EDGE_AREA,
        )
        areas_out = cg._connected_or_not(areas, connected_indices, False)
    else:
        areas_out = np.empty(0, basetypes.EDGE_AREA)

    return edges_in, edges_out, areas_out


def find_approximate_middle_point(points, resolution):
    """
    Given a list of points and a resolution, choose a point from
    the list that is approximately in the center.
    """
    bbox_min = np.amin(points, axis=0)
    bbox_max = np.amax(points, axis=0)
    distances_from_bbox_max = np.sum(((bbox_max - points) * resolution), axis=1)
    distances_from_bbox_min = np.sum(((points - bbox_min) * resolution), axis=1)
    # For each point, greater distance of distance to bbox_max or bbox_min
    greater_distance = np.amax(
        np.vstack((distances_from_bbox_max, distances_from_bbox_min)), axis=0
    )
    return points[np.argmin(greater_distance)]


def _find_points_nm_coordinate(cg, chunk_coordinates, coordinate_within_chunk):
    """
    Given a lvl1/lvl2 chunk coordinate and a position in the chunk, return a global
    coordinate representing that voxel's position in the dataset.
    """
    points_voxel_coordinate = (
        cg.get_chunk_voxel_location(chunk_coordinates) + coordinate_within_chunk
    )
    return points_voxel_coordinate * cg.segmentation_resolution


def _get_approximate_middle_contact_point_same_chunk(cg, sv1, sv2, ws_seg):
    """
    Given two supervoxels in the same chunk, return a contact point amongst
    their contact points that is approximately in the center of the contact. 
    If there are no contact points, return a point that is 
    approximately in the center of sv1.
    """
    chunk_coordinate = cg.get_chunk_coordinates(sv1)
    contact_points = find_contact_points(ws_seg, sv1, sv2)
    if len(contact_points) == 0:
        sv1_locations = np.where(ws_seg == sv1)
        sv1_points = np.vstack((sv1_locations[0], sv1_locations[1], sv1_locations[2]))
        middle_point = find_approximate_middle_point(
            sv1_points, cg.segmentation_resolution
        )
    else:
        middle_point = find_approximate_middle_point(
            contact_points[:, 0, :], cg.segmentation_resolution
        )
    return _find_points_nm_coordinate(cg, chunk_coordinate, middle_point)


def _get_approximate_middle_contact_point_across_chunks(cg, sv1, sv2, sv1_chunk_seg):
    """
    Given two supervoxels in neighboring chunks, return a contact point amongst
    their contact points that is approximately in the center of the contact. 
    If there are no contact points, return a point that is 
    approximately in the center of sv1.
    """
    chunk_coordinate = cg.get_chunk_coordinates(sv1)
    other_chunk_coordinate = cg.get_chunk_coordinates(sv2)
    sv2_chunk_seg = cg.download_chunk_segmentation(other_chunk_coordinate)
    crossing_index = np.where(chunk_coordinate - other_chunk_coordinate)[0][0]
    positive_crossing = (chunk_coordinate - other_chunk_coordinate)[crossing_index] < 0
    sv1_chunk_plane_index = -1 if positive_crossing else 0
    sv2_chunk_plane_index = 0 if positive_crossing else -1
    if crossing_index == 0:
        sv1_chunk_plane = sv1_chunk_seg[sv1_chunk_plane_index, :, :]
        sv2_chunk_plane = sv2_chunk_seg[sv2_chunk_plane_index, :, :]
    elif crossing_index == 1:
        sv1_chunk_plane = sv1_chunk_seg[:, sv1_chunk_plane_index, :]
        sv2_chunk_plane = sv2_chunk_seg[:, sv2_chunk_plane_index, :]
    else:
        sv1_chunk_plane = sv1_chunk_seg[:, :, sv1_chunk_plane_index]
        sv2_chunk_plane = sv2_chunk_seg[:, :, sv2_chunk_plane_index]
    chunk_border = np.stack((sv1_chunk_plane, sv2_chunk_plane), axis=crossing_index)
    contact_points = find_contact_points(chunk_border, sv1, sv2)
    if len(contact_points) == 0:
        sv1_locations = np.where(chunk_border == sv1)
        sv1_points = np.vstack((sv1_locations[0], sv1_locations[1], sv1_locations[2]))
        middle_point = find_approximate_middle_point(
            sv1_points, cg.segmentation_resolution
        )
    else:
        middle_point = find_approximate_middle_point(
            contact_points[:, 0, :], cg.segmentation_resolution
        )
    return _find_points_nm_coordinate(cg, chunk_coordinate, middle_point)


def _get_approximate_middle_sv_point(cg, sv, ws_seg):
    """
    Get a point in a given supervoxel that is approximately
    in its center.
    """
    chunk_coordinate = cg.get_chunk_coordinates(sv)
    sv_locations = np.where(ws_seg == sv)
    sv_points = np.vstack((sv_locations[0], sv_locations[1], sv_locations[2]))
    middle_point = find_approximate_middle_point(sv_points, cg.segmentation_resolution)
    return _find_points_nm_coordinate(cg, chunk_coordinate, middle_point)


class EdgeType(Enum):
    SameChunk = auto()
    AdjacentChunks = auto()
    NonAdjacentChunks = auto()


def _choose_contact_site_edge(cg, first_node_unconnected_edges, second_node_sv_ids):
    """
    Choose an edge to represent a contact site from a list of edges. For performance reasons,
    try to choose an edge that is not a cross chunk edge (to avoid downloading multiple chunks).
    """
    edge_mask = np.where(
        np.isin(first_node_unconnected_edges[:, 1], second_node_sv_ids)
    )[0]
    filtered_unconnected_edges = first_node_unconnected_edges[edge_mask]
    smallest_distance = None
    best_edge = None
    for edge in filtered_unconnected_edges:
        chunk_coordinates_first_node = cg.get_chunk_coordinates(edge[0])
        chunk_coordinates_second_node = cg.get_chunk_coordinates(edge[1])
        if np.array_equal(chunk_coordinates_first_node, chunk_coordinates_second_node):
            return (edge, EdgeType.SameChunk)
        distance = abs(
            np.sum(chunk_coordinates_first_node) - np.sum(chunk_coordinates_second_node)
        )
        if best_edge is None or distance < smallest_distance:
            smallest_distance = distance
            best_edge = edge
    if smallest_distance == 1:
        return (edge, EdgeType.AdjacentChunks)
    return (edge, EdgeType.NonAdjacentChunks)


def _get_contact_site_edges(
    cg,
    first_node_unconnected_edges,
    first_node_unconnected_areas,
    second_node_connected_edges,
    second_node_sv_ids,
):
    """
    Given two sets of supervoxels, find all contact sites between the two sets, and for each
    contact site return an edge that represents the site.
    """
    # Retrieve edges that connect first node with second node
    unconnected_edge_mask = np.where(
        np.isin(first_node_unconnected_edges[:, 1], second_node_sv_ids)
    )[0]
    filtered_unconnected_edges = first_node_unconnected_edges[unconnected_edge_mask]
    filtered_areas = first_node_unconnected_areas[unconnected_edge_mask]
    contact_sites_svs_area_dict = collections.defaultdict(int)
    for i in range(filtered_unconnected_edges.shape[0]):
        sv_id = filtered_unconnected_edges[i, 1]
        area = filtered_areas[i]
        contact_sites_svs_area_dict[sv_id] += area
    contact_sites_svs_area_dict_vec = np.vectorize(contact_sites_svs_area_dict.get)
    unique_contact_sites_svs = np.unique(filtered_unconnected_edges[:, 1])
    # Retrieve edges that connect second node contact sites with other second node contact sites
    connected_edge_test = np.isin(second_node_connected_edges, unique_contact_sites_svs)
    connected_edges_mask = np.where(np.all(connected_edge_test, axis=1))[0]
    filtered_connected_edges = second_node_connected_edges[connected_edges_mask]
    # Make fake edges from contact site svs to themselves to make sure they appear in the created graph
    self_edges = np.stack((unique_contact_sites_svs, unique_contact_sites_svs), axis=-1)
    contact_sites_graph_edges = np.concatenate(
        (filtered_connected_edges, self_edges), axis=0
    )

    graph, _, _, unique_sv_ids = flatgraph_utils.build_gt_graph(
        contact_sites_graph_edges, make_directed=True
    )
    connected_components = flatgraph_utils.connected_components(graph)

    contact_site_edges = []
    for cc in connected_components:
        cc_sv_ids = unique_sv_ids[cc]
        contact_sites_areas = contact_sites_svs_area_dict_vec(cc_sv_ids)
        contact_site_edge, contact_site_edge_type = _choose_contact_site_edge(
            cg, filtered_unconnected_edges, cc_sv_ids
        )
        contact_site_edge_info = (
            contact_site_edge,
            contact_site_edge_type,
            np.sum(contact_sites_areas),
        )
        contact_site_edges.append(contact_site_edge_info)
    return contact_site_edges


def _get_contact_site_edges_to_inspect(
    cg, sv_ids_set1, sv_ids_set2, end_time, optimize_unsafe
):
    """
    Given two sets of supervoxels, read their connectivity data from the graph,
    find all contact sites between the two sets, and for each
    contact site return an edge that represents the site.
    """
    # Combine the supervoxel into one set to optimize the call to read_node_id_rows
    all_sv_ids = np.concatenate((sv_ids_set1, sv_ids_set2))
    is_in_set1 = np.concatenate((np.ones(len(sv_ids_set1)), np.zeros(len(sv_ids_set2))))
    is_in_set1_dict = dict(zip(all_sv_ids, is_in_set1))

    all_sv_rows = cg.read_node_id_rows(
        node_ids=all_sv_ids,
        columns=[
            column_keys.Connectivity.Area,
            column_keys.Connectivity.Partner,
            column_keys.Connectivity.Connected,
        ],
        end_time=end_time,
        end_time_inclusive=True,
    )

    # Retrieve the connectivity data
    edges_in1 = []
    edges_out1 = []
    areas_out1 = []
    edges_in2 = []
    edges_out2 = []
    areas_out2 = []
    for row_information in all_sv_rows.items():
        sv_id, _ = row_information
        if optimize_unsafe:
            # WARNING: We are deprecating the old way of retrieving connectivity so _retrieve_connectivity_optimized
            # will soon have to be updated - 10/15/2019
            sv_edges_in, sv_edges_out, sv_areas_out = _retrieve_connectivity_optimized(
                cg, row_information
            )
        else:
            sv_edges_in, _, _ = cg._retrieve_connectivity(
                row_information, connected_edges=True
            )
            sv_edges_out, _, sv_areas_out = cg._retrieve_connectivity(
                row_information, connected_edges=False
            )
        if is_in_set1_dict[sv_id]:
            edges_in1.append(sv_edges_in)
            edges_out1.append(sv_edges_out)
            areas_out1.append(sv_areas_out)
        else:
            edges_in2.append(sv_edges_in)
            edges_out2.append(sv_edges_out)
            areas_out2.append(sv_areas_out)

    def _concat_nonempty(x):
        if len(x) > 0:
            return np.concatenate(x)
        return x

    edges_in1 = _concat_nonempty(edges_in1)
    edges_out1 = _concat_nonempty(edges_out1)
    areas_out1 = _concat_nonempty(areas_out1)
    edges_in2 = _concat_nonempty(edges_in2)
    edges_out2 = _concat_nonempty(edges_out2)
    areas_out2 = _concat_nonempty(areas_out2)


    # Check the number of contact sites from both sides. The number may be different; the higher
    # number is always more accurate.
    contact_sites_edges = _get_contact_site_edges(
        cg, edges_out1, areas_out1, edges_in2, sv_ids_set2
    )
    contact_sites_edges_other_direction = _get_contact_site_edges(
        cg, edges_out2, areas_out2, edges_in1, sv_ids_set1
    )

    if len(contact_sites_edges_other_direction) > len(contact_sites_edges):
        edges_to_inspect = contact_sites_edges_other_direction
    else:
        edges_to_inspect = contact_sites_edges

    return edges_to_inspect


def _get_sv_contact_site_candidates(cg, first_node_id, second_node_id):
    """
    Given two node ids, return their two sets of children supervoxels
    that could possibly have a contact point with a supervoxel of the
    other node id.
    """
    # Get lvl2 ids of both roots
    first_node_l2_ids = cg.get_children_at_layer(first_node_id, 2)
    second_node_l2_ids = cg.get_children_at_layer(second_node_id, 2)

    if len(first_node_l2_ids) > len(second_node_l2_ids):
        smaller_set_l2_ids = second_node_l2_ids
        bigger_set_l2_ids = first_node_l2_ids
    else:
        smaller_set_l2_ids = first_node_l2_ids
        bigger_set_l2_ids = second_node_l2_ids

    # Make a dict holding all the chunk coordinates of all the lvl2 ids of the "smaller" root, and the coordinates of the neighboring chunks
    chunk_coordinate_dict = collections.defaultdict(set)
    for i in range(len(smaller_set_l2_ids)):
        l2_id = smaller_set_l2_ids[i]
        chunk_coordinates = cg.get_chunk_coordinates(l2_id)
        chunk_coordinate_dict[tuple(chunk_coordinates)].add(i)
        for j in range(3):
            adjacent_chunk_coordinates = np.copy(chunk_coordinates)
            adjacent_chunk_coordinates[j] += 1
            chunk_coordinate_dict[tuple(adjacent_chunk_coordinates)].add(i)
            adjacent_chunk_coordinates[j] -= 2
            chunk_coordinate_dict[tuple(adjacent_chunk_coordinates)].add(i)

    candidate_smaller_set_l2_ids = set()
    candidate_bigger_set_l2_ids = []

    # For the "bigger" root id, filter out all lvl2 ids that do not appear in the chunk_coordinate_dict
    for l2_id in bigger_set_l2_ids:
        chunk_coordinates = cg.get_chunk_coordinates(l2_id)
        new_candidate_smaller_set_l2_ids = chunk_coordinate_dict.get(
            tuple(chunk_coordinates)
        )
        if new_candidate_smaller_set_l2_ids is not None:
            for new_candidate in new_candidate_smaller_set_l2_ids:
                candidate_smaller_set_l2_ids.add(smaller_set_l2_ids[new_candidate])
            candidate_bigger_set_l2_ids.append(l2_id)

    # Get the supervoxel of the filtered lvl2 ids to retrieve their edges
    sv_ids_set1 = cg.get_children(list(candidate_smaller_set_l2_ids), flatten=True)
    sv_ids_set2 = cg.get_children(candidate_bigger_set_l2_ids, flatten=True)

    return (sv_ids_set1, sv_ids_set2)


def _get_exact_contact_sites(cg, edges_to_inspect):
    """
    Given a list of edges between two supervoxels, for each edge get a global
    coordinate in the dataset where the two supervoxels contact.
    """
    chunk_coordinate_list = []
    for edge_to_inspect in edges_to_inspect:
        edge = edge_to_inspect[0]
        chunk_coordinate_list.append(tuple(cg.get_chunk_coordinates(edge[0])))
    chunk_coordinate_array = np.array(
        chunk_coordinate_list, dtype=[("x", "i8"), ("y", "i8"), ("z", "i8")]
    )
    sorted_chunk_coordinates_indices = np.argsort(
        chunk_coordinate_array, order=("x", "y", "z")
    )

    memoized_chunk = None
    last_chunk_coordinate = None
    contact_sites = []
    for index in sorted_chunk_coordinates_indices:
        edge, edgeType, area = edges_to_inspect[index]
        cur_chunk_coordinate = chunk_coordinate_array[index]
        if (
            last_chunk_coordinate is None
            or last_chunk_coordinate != cur_chunk_coordinate
        ):
            last_chunk_coordinate = cur_chunk_coordinate
            memoized_chunk = cg.download_chunk_segmentation(
                np.array(
                    (
                        cur_chunk_coordinate[0],
                        cur_chunk_coordinate[1],
                        cur_chunk_coordinate[2],
                    )
                )
            )
        if edgeType == EdgeType.SameChunk:
            contact_point = _get_approximate_middle_contact_point_same_chunk(
                cg, edge[0], edge[1], memoized_chunk
            )
        elif edgeType == EdgeType.AdjacentChunks:
            contact_point = _get_approximate_middle_contact_point_across_chunks(
                cg, edge[0], edge[1], memoized_chunk
            )
        else:
            contact_point = _get_approximate_middle_sv_point(
                cg, edge[0], memoized_chunk
            )
        contact_sites.append((contact_point, area))
    return contact_sites


def get_contact_sites_pairwise(
    cg,
    first_node_id,
    second_node_id,
    end_time=None,
    exact_location=True,
    optimize_unsafe=False,
):
    """
    Given two node ids, find the locations and areas of their contact sites in the dataset.

    If exact_location=True, this function returns a list of tuples of two elements,
    where the first element in the tuple is a global coordinate in the dataset, and
    the second is an area.

    If exact_location=False, this function returns a list of tuples of three elements.
    The first and second elements are the global coordinates that the contact site appears somewhere
    between, and the third is an area.
    """
    sv_ids_set1, sv_ids_set2 = _get_sv_contact_site_candidates(
        cg, first_node_id, second_node_id
    )
    if len(sv_ids_set1) == 0 or len(sv_ids_set2) == 0:
        return []
    edges_to_inspect = _get_contact_site_edges_to_inspect(
        cg, sv_ids_set1, sv_ids_set2, end_time, optimize_unsafe
    )
    if exact_location:
        return _get_exact_contact_sites(cg, edges_to_inspect)
    else:
        contact_sites = []
        for edge_to_inspect in edges_to_inspect:
            edge, _, area = edge_to_inspect
            contact_sites.append(
                (
                    cg.get_chunk_voxel_location(cg.get_chunk_coordinates(edge[0])) * cg.segmentation_resolution,
                    cg.get_chunk_voxel_location(cg.get_chunk_coordinates(edge[0]) + 1) * cg.segmentation_resolution,
                    area,
                )
            )
    return contact_sites
