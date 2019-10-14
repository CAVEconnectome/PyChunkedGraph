import numpy as np
import datetime
import collections
import itertools

from contact_points import find_contact_points
from enum import auto, Enum
from pychunkedgraph.backend import chunkedgraph, flatgraph_utils
from pychunkedgraph.backend.utils import column_keys, basetypes
from multiwrapper import multiprocessing_utils as mu

from typing import Optional, Sequence


def _read_delta_root_rows_thread(args) -> Sequence[list]:
    start_seg_id, end_seg_id, serialized_cg_info, time_stamp_start, time_stamp_end  = args

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    start_id = cg.get_node_id(segment_id=start_seg_id,
                              chunk_id=cg.root_chunk_id)
    end_id = cg.get_node_id(segment_id=end_seg_id,
                            chunk_id=cg.root_chunk_id)

    # apply column filters to avoid Lock columns
    rows = cg.read_node_id_rows(
        start_id=start_id,
        start_time=time_stamp_start,
        end_id=end_id,
        end_id_inclusive=False,
        columns=[column_keys.Hierarchy.FormerParent, column_keys.Hierarchy.NewParent],
        end_time=time_stamp_end,
        end_time_inclusive=True)

    # new roots are those that have no NewParent in this time window
    new_root_ids = [k for (k, v) in rows.items()
                    if column_keys.Hierarchy.NewParent not in v]

    # expired roots are the IDs of FormerParent's 
    # whose timestamp is before the start_time 
    expired_root_ids = []
    for k, v in rows.items():
        if column_keys.Hierarchy.FormerParent in v:
            fp = v[column_keys.Hierarchy.FormerParent]
            for cell_entry in fp:
                expired_root_ids.extend(cell_entry.value)

    return new_root_ids, expired_root_ids


def _read_root_rows_thread(args) -> list:
    start_seg_id, end_seg_id, serialized_cg_info, time_stamp = args

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    start_id = cg.get_node_id(segment_id=start_seg_id,
                              chunk_id=cg.root_chunk_id)
    end_id = cg.get_node_id(segment_id=end_seg_id,
                            chunk_id=cg.root_chunk_id)

    rows = cg.read_node_id_rows(
        start_id=start_id,
        end_id=end_id,
        end_id_inclusive=False,
        end_time=time_stamp,
        end_time_inclusive=True)

    root_ids = [k for (k, v) in rows.items()
                if column_keys.Hierarchy.NewParent not in v]

    return root_ids


def get_latest_roots(cg,
                     time_stamp: Optional[datetime.datetime] = None,
                     n_threads: int = 1) -> Sequence[np.uint64]:

    # Create filters: time and id range
    max_seg_id = cg.get_max_seg_id(cg.root_chunk_id) + 1

    if n_threads == 1:
        n_blocks = 1
    else:
        n_blocks = int(np.min([n_threads * 3 + 1, max_seg_id]))

    seg_id_blocks = np.linspace(1, max_seg_id, n_blocks + 1, dtype=np.uint64)

    cg_serialized_info = cg.get_serialized_info()

    if n_threads > 1:
        del cg_serialized_info["credentials"]

    multi_args = []
    for i_id_block in range(0, len(seg_id_blocks) - 1):
        multi_args.append([seg_id_blocks[i_id_block],
                           seg_id_blocks[i_id_block + 1],
                           cg_serialized_info, time_stamp])

    if n_threads == 1:
        results = mu.multiprocess_func(_read_root_rows_thread,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_read_root_rows_thread,
                                          multi_args, n_threads=n_threads)

    root_ids = []
    for result in results:
        root_ids.extend(result)

    return np.array(root_ids, dtype=np.uint64)


def get_delta_roots(cg,
                    time_stamp_start: datetime.datetime,
                    time_stamp_end: Optional[datetime.datetime] = None,
                    min_seg_id: int = 1,
                    n_threads: int = 1) -> Sequence[np.uint64]:

    # Create filters: time and id range
    max_seg_id = cg.get_max_seg_id(cg.root_chunk_id) + 1

    n_blocks = int(np.min([n_threads + 1, max_seg_id-min_seg_id+1]))
    seg_id_blocks = np.linspace(min_seg_id, max_seg_id, n_blocks,
                                dtype=np.uint64)

    cg_serialized_info = cg.get_serialized_info()

    if n_threads > 1:
        del cg_serialized_info["credentials"]

    multi_args = []
    for i_id_block in range(0, len(seg_id_blocks) - 1):
        multi_args.append([seg_id_blocks[i_id_block],
                           seg_id_blocks[i_id_block + 1],
                           cg_serialized_info, time_stamp_start, time_stamp_end])

    # Run parallelizing
    if n_threads == 1:
        results = mu.multiprocess_func(_read_delta_root_rows_thread,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_read_delta_root_rows_thread,
                                          multi_args, n_threads=n_threads)

    # aggregate all the results together
    new_root_ids = []
    expired_root_id_candidates = []
    for r1, r2 in results:
        new_root_ids.extend(r1)
        expired_root_id_candidates.extend(r2)
    expired_root_id_candidates = np.array(expired_root_id_candidates, dtype=np.uint64)
    # filter for uniqueness
    expired_root_id_candidates = np.unique(expired_root_id_candidates)

    # filter out the expired root id's whose creation (measured by the timestamp
    # of their Child links) is after the time_stamp_start
    rows = cg.read_node_id_rows(node_ids=expired_root_id_candidates,
                                columns=[column_keys.Hierarchy.Child],
                                end_time=time_stamp_start)  
    expired_root_ids = np.array([k for (k, v) in rows.items()], dtype=np.uint64)

    return np.array(new_root_ids, dtype=np.uint64), expired_root_ids

def get_contact_sites(cg, root_id, bounding_box=None, bb_is_coordinate=True, compute_partner=True, end_time=None, voxel_location=False):
    # Get information about the root id
    # All supervoxels
    sv_ids = cg.get_subgraph_nodes(root_id,
                                   bounding_box=bounding_box,
                                   bb_is_coordinate=bb_is_coordinate)
    
    # All edges that are _not_ connected / on
    edges, _, areas = cg.get_subgraph_edges(root_id,
                                               bounding_box=bounding_box,
                                               bb_is_coordinate=bb_is_coordinate,
                                               connected_edges=False)

    # Build area lookup dictionary
    edge_mask = ~np.in1d(edges, sv_ids).reshape(-1, 2)
    area_mask = np.where(edge_mask)[0]
    masked_areas = areas[area_mask]
    contact_sites_svs = edges[edge_mask]
    contact_sites_svs_area_dict = collections.defaultdict(int)

    for area, sv_id in zip(masked_areas, contact_sites_svs):
        contact_sites_svs_area_dict[sv_id] += area

    contact_sites_svs_area_dict_vec = np.vectorize(contact_sites_svs_area_dict.get)

    # Extract svs from contacting root ids
    unique_contact_sites_svs = np.unique(contact_sites_svs)

    # Load edges of these contact sites' supervoxels
    edges_contact_sites_svs_rows = cg.read_node_id_rows(node_ids=unique_contact_sites_svs,
                                             columns=[column_keys.Connectivity.Partner,
                                                      column_keys.Connectivity.Connected], end_time=end_time, end_time_inclusive=True)
    
    contact_sites_edges = []
    for row_information in edges_contact_sites_svs_rows.items():
        sv_edges, _, _ = cg._retrieve_connectivity(row_information)
        contact_sites_edges.extend(sv_edges)

    if len(contact_sites_edges) == 0:
        return collections.defaultdict(list)

    contact_sites_edges_array = np.array(contact_sites_edges)
    contact_sites_edge_mask = np.isin(contact_sites_edges_array[:,1], unique_contact_sites_svs)
    # Fake edges to ensure lone contact site supervoxels show up as a connected component
    self_edges = np.stack((unique_contact_sites_svs, unique_contact_sites_svs), axis=-1)
    contact_sites_graph_edges = np.concatenate((contact_sites_edges_array[contact_sites_edge_mask], self_edges), axis=0)

    graph, _, _, unique_sv_ids = flatgraph_utils.build_gt_graph(
        contact_sites_graph_edges, make_directed=True)

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
        if voxel_location:
            voxel_lower_bound = cg.vx_vol_bounds[:,0] + cg.chunk_size * chunk_coordinates
            voxel_upper_bound = cg.vx_vol_bounds[:,0] + cg.chunk_size * (chunk_coordinates + 1)
            data_pair = (voxel_lower_bound * cg.segmentation_resolution, voxel_upper_bound * cg.segmentation_resolution, np.sum(contact_sites_areas))
        else:
            data_pair = (chunk_coordinates, chunk_coordinates + 1, np.sum(contact_sites_areas))

        if compute_partner:
            # Cast np.uint64 to int for dict key because int is hashable
            intermediary_sv_dict[int(representative_sv)] = data_pair
        else:
            contact_site_dict[len(contact_site_dict)].append(data_pair)

    if compute_partner:
        sv_list = np.array(list(intermediary_sv_dict.keys()), dtype=np.uint64)
        partner_roots = cg.get_roots(sv_list)
        for i in range(len(partner_roots)):
            contact_site_dict[int(partner_roots[i])].append(intermediary_sv_dict.get(int(sv_list[i])))

    return contact_site_dict

def _retrieve_connectivity_optimized(cg, dict_item):
    node_id, row = dict_item

    tmp = set()
    for x in itertools.chain.from_iterable(generation.value for generation in row[column_keys.Connectivity.Connected][::-1]):
        tmp.remove(x) if x in tmp else tmp.add(x)

    connected_indices = np.fromiter(tmp, np.uint64)
    if column_keys.Connectivity.Partner in row:
        edges = np.fromiter(itertools.chain.from_iterable(
            (node_id, partner_id)
            for generation in row[column_keys.Connectivity.Partner][::-1]
            for partner_id in generation.value),
            dtype=basetypes.NODE_ID).reshape((-1, 2))
        edges_in = cg._connected_or_not(edges, connected_indices, True)
        edges_out = cg._connected_or_not(edges, connected_indices, False)
    else:
        edges_in = np.empty((0, 2), basetypes.NODE_ID)
        edges_out = np.empty((0, 2), basetypes.NODE_ID)

    if column_keys.Connectivity.Area in row:
        areas = np.fromiter(itertools.chain.from_iterable(
            generation.value for generation in row[column_keys.Connectivity.Area][::-1]),
            dtype=basetypes.EDGE_AREA)
        areas_out = cg._connected_or_not(areas, connected_indices, False)
    else:
        areas_out = np.empty(0, basetypes.EDGE_AREA)

    return edges_in, edges_out, areas_out

# def _get_approximate_centroid(cg, sv_ids):
#     chunk_coordinate_vote_dict = collections.defaultdict(list)
#     most_popular_chunk_coordinate = None
#     most_popular_chunk_coordinate_vote_count = 0
#     for sv_id in sv_ids:
#         chunk_coordinates = cg.get_chunk_coordinates(sv_id)
#         chunk_coordinate_vote_dict[tuple(chunk_coordinates)].append(sv_id)
#         if len(chunk_coordinate_vote_dict[tuple(chunk_coordinates)]) > most_popular_chunk_coordinate_vote_count:
#             most_popular_chunk_coordinate_vote_count = len(chunk_coordinate_vote_dict[tuple(chunk_coordinates)])
#             most_popular_chunk_coordinate = chunk_coordinates
#     chunk_start = np.array((cg.vx_vol_bounds[:,0] + cg.chunk_size * most_popular_chunk_coordinate), dtype=np.int)
#     chunk_end = np.array((cg.vx_vol_bounds[:,0] + cg.chunk_size * (most_popular_chunk_coordinate + 1)), dtype=np.int)
#     ws_seg = cg.cv[
#         chunk_start[0] : chunk_end[0],
#         chunk_start[1] : chunk_end[1],
#         chunk_start[2] : chunk_end[2],
#     ].squeeze()
#     import ipdb
#     ipdb.set_trace()

def find_approximate_middle_point(points, resolution):
    bbox_min = np.amin(points, axis=0)
    bbox_max = np.amax(points, axis=0)
    distances_from_bbox_max = np.sum(((bbox_max - points) * resolution), axis=1)
    distances_from_bbox_min = np.sum(((points - bbox_min) * resolution), axis=1)
    # For each point, greater distance of distance to bbox_max or bbox_min
    greater_distance = np.amax(np.vstack((distances_from_bbox_max, distances_from_bbox_min)), axis=0)
    return points[np.argmin(greater_distance)]

def _find_points_nm_coordinate(cg, chunk_coordinates, coordinate_within_chunk):
    points_voxel_coordinate = cg.get_chunk_voxel_location(chunk_coordinates) + coordinate_within_chunk
    return points_voxel_coordinate * cg.segmentation_resolution

@profile
def _get_approximate_middle_contact_point_same_chunk(cg, sv1, sv2):
    chunk_coordinate = cg.get_chunk_coordinates(sv1)
    ws_seg = cg.download_chunk_segmentation(chunk_coordinate)
    contact_points = find_contact_points(ws_seg, sv1, sv2)
    if len(contact_points) == 0:
        sv1_locations = np.where(ws_seg == sv1)
        sv1_points = np.vstack((sv1_locations[0], sv1_locations[1], sv1_locations[2]))
        middle_point = find_approximate_middle_point(sv1_points, cg.segmentation_resolution)
    else:
        middle_point = find_approximate_middle_point(contact_points[:,0,:], cg.segmentation_resolution)
    return _find_points_nm_coordinate(cg, chunk_coordinate, middle_point)

def _get_approximate_middle_contact_point_across_chunks(cg, sv1, sv2):
    chunk_coordinate = cg.get_chunk_coordinates(sv1)
    other_chunk_coordinate = cg.get_chunk_coordinates(sv2)
    sv1_chunk_seg = cg.download_chunk_segmentation(chunk_coordinate)
    sv2_chunk_seg = cg.download_chunk_segmentation(other_chunk_coordinate)
    crossing_index = np.where(chunk_coordinate - other_chunk_coordinate)[0][0]
    positive_crossing = (chunk_coordinate - other_chunk_coordinate)[crossing_index] < 0
    sv1_chunk_plane_index = -1 if positive_crossing else 0
    sv2_chunk_plane_index = 0 if positive_crossing else -1
    if crossing_index == 0:
        sv1_chunk_plane = sv1_chunk_seg[sv1_chunk_plane_index,:,:]
        sv2_chunk_plane = sv2_chunk_seg[sv2_chunk_plane_index,:,:]
    elif crossing_index == 1:
        sv1_chunk_plane = sv1_chunk_seg[:,sv1_chunk_plane_index,:]
        sv2_chunk_plane = sv2_chunk_seg[:,sv2_chunk_plane_index,:]
    else:
        sv1_chunk_plane = sv1_chunk_seg[:,:,sv1_chunk_plane_index]
        sv2_chunk_plane = sv2_chunk_seg[:,:,sv2_chunk_plane_index]
    chunk_border = np.stack((sv1_chunk_plane, sv2_chunk_plane), axis=crossing_index)
    contact_points = find_contact_points(chunk_border, sv1, sv2)
    if len(contact_points) == 0:
        sv1_locations = np.where(chunk_border == sv1)
        sv1_points = np.vstack((sv1_locations[0], sv1_locations[1], sv1_locations[2]))
        middle_point = find_approximate_middle_point(sv1_points, cg.segmentation_resolution)
    else:
        middle_point = find_approximate_middle_point(contact_points[:,0,:], cg.segmentation_resolution)
    return _find_points_nm_coordinate(cg, chunk_coordinate, middle_point)    

def _get_approximate_middle_sv_point(cg, sv):
    chunk_coordinate = cg.get_chunk_coordinates(sv)
    ws_seg = cg.download_chunk_segmentation(chunk_coordinate)
    sv_locations = np.where(ws_seg == sv)
    sv_points = np.vstack((sv_locations[0], sv_locations[1], sv_locations[2]))
    middle_point = find_approximate_middle_point(sv_points, cg.segmentation_resolution)
    return _find_points_nm_coordinate(cg, chunk_coordinate, middle_point)

@profile
def _get_approximate_contact_point(cg, first_root_unconnected_edges, second_root_sv_ids):
    class EdgeType(Enum):
        SameChunk = auto()
        AdjacentChunks = auto()
        NonAdjancentChunks = auto()
    def _choose_contact_site_edge():
        edge_mask = np.where(np.isin(first_root_unconnected_edges[:,1], second_root_sv_ids))[0]
        filtered_unconnected_edges = first_root_unconnected_edges[edge_mask]
        smallest_distance = None
        best_edge = None
        for edge in filtered_unconnected_edges:
            chunk_coordinates_first_root = cg.get_chunk_coordinates(edge[0])
            chunk_coordinates_second_root = cg.get_chunk_coordinates(edge[1])
            if np.array_equal(chunk_coordinates_first_root, chunk_coordinates_second_root):
                return (edge, EdgeType.SameChunk)
            distance = abs(np.sum(chunk_coordinates_first_root) - np.sum(chunk_coordinates_second_root))
            if best_edge is None or distance < smallest_distance:
                smallest_distance = distance
                best_edge = edge
        if smallest_distance == 1:
            return (edge, EdgeType.AdjacentChunks)
        return (edge, EdgeType.NonAdjancentChunks)
    edge, edgeType = _choose_contact_site_edge()
    if edgeType == EdgeType.SameChunk:
        return _get_approximate_middle_contact_point_same_chunk(cg, edge[0], edge[1])
    if edgeType == EdgeType.AdjacentChunks:
        return _get_approximate_middle_contact_point_across_chunks(cg, edge[0], edge[1])
    return _get_approximate_middle_sv_point(cg, edge[0])
    
@profile
def _compute_contact_sites_from_edges(cg, first_root_unconnected_edges, first_root_unconnected_areas, second_root_connected_edges, second_root_sv_ids):
    # Retrieve edges that connect first root with second root
    unconnected_edge_mask = np.where(np.isin(first_root_unconnected_edges[:,1], second_root_sv_ids))[0]
    filtered_unconnected_edges = first_root_unconnected_edges[unconnected_edge_mask]
    filtered_areas = first_root_unconnected_areas[unconnected_edge_mask]
    contact_sites_svs_area_dict = collections.defaultdict(int)
    for i in range(filtered_unconnected_edges.shape[0]):
        sv_id = filtered_unconnected_edges[i,1]
        area = filtered_areas[i]
        contact_sites_svs_area_dict[sv_id] += area
    contact_sites_svs_area_dict_vec = np.vectorize(contact_sites_svs_area_dict.get)
    unique_contact_sites_svs = np.unique(filtered_unconnected_edges[:,1])
    # Retrieve edges that connect second root contact sites with other second root contact sites
    connected_edge_test = np.isin(second_root_connected_edges, unique_contact_sites_svs)
    connected_edges_mask = np.where(np.all(connected_edge_test, axis=1))[0]
    filtered_connected_edges = second_root_connected_edges[connected_edges_mask]
    # Make fake edges from contact site svs to themselves to make sure they appear in the created graph
    self_edges = np.stack((unique_contact_sites_svs, unique_contact_sites_svs), axis=-1)
    contact_sites_graph_edges = np.concatenate((filtered_connected_edges, self_edges), axis=0)

    graph, _, _, unique_sv_ids = flatgraph_utils.build_gt_graph(
        contact_sites_graph_edges, make_directed=True)

    connected_components = flatgraph_utils.connected_components(graph)

    contact_site_list = []
    for cc in connected_components:
        cc_sv_ids = unique_sv_ids[cc]
        contact_sites_areas = contact_sites_svs_area_dict_vec(cc_sv_ids)
        contact_site_point = _get_approximate_contact_point(cg, filtered_unconnected_edges, cc_sv_ids)
        data_pair = (contact_site_point, np.sum(contact_sites_areas))
        contact_site_list.append(data_pair)
    return contact_site_list

@profile
def get_contact_sites_pairwise(cg, first_root_id, second_root_id, bounding_box=None, bb_is_coordinate=True, end_time=None, voxel_location=False, optimize_unsafe=False):
    # Get lvl2 ids of both roots
    first_root_l2_ids = cg.get_children_at_layer(first_root_id, 2)
    second_root_l2_ids = cg.get_children_at_layer(second_root_id, 2)

    if len(first_root_l2_ids) > len(second_root_l2_ids):
        smaller_set_l2_ids = second_root_l2_ids
        bigger_set_l2_ids = first_root_l2_ids
    else:
        smaller_set_l2_ids = first_root_l2_ids
        bigger_set_l2_ids = second_root_l2_ids

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
        new_candidate_smaller_set_l2_ids = chunk_coordinate_dict.get(tuple(chunk_coordinates))
        if new_candidate_smaller_set_l2_ids is not None:
            for new_candidate in new_candidate_smaller_set_l2_ids:
                candidate_smaller_set_l2_ids.add(smaller_set_l2_ids[new_candidate])
            candidate_bigger_set_l2_ids.append(l2_id)
    
    # Get the supervoxel of the filtered lvl2 ids to retrieve their edges
    sv_ids_set1 = cg.get_children(list(candidate_smaller_set_l2_ids), flatten=True)
    sv_ids_set2 = cg.get_children(candidate_bigger_set_l2_ids, flatten=True)

    # Combine the supervoxel into one set to optimize the call to read_node_id_rows
    all_sv_ids = np.concatenate((sv_ids_set1, sv_ids_set2))
    is_in_set1 = np.concatenate((np.ones(len(sv_ids_set1)), np.zeros(len(sv_ids_set2))))
    is_in_set1_dict = dict(zip(all_sv_ids, is_in_set1))
    
    all_sv_rows = cg.read_node_id_rows(node_ids=all_sv_ids,
                                          columns=[column_keys.Connectivity.Area,
                                                   column_keys.Connectivity.Partner,
                                                   column_keys.Connectivity.Connected],
                                          end_time=end_time,
                                          end_time_inclusive=True)

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
            # will soon have to be updated
            sv_edges_in, sv_edges_out, sv_areas_out = _retrieve_connectivity_optimized(cg, row_information)
        else:
            sv_edges_in, _, _ = cg._retrieve_connectivity(row_information, connected_edges=True)
            sv_edges_out, _, sv_areas_out = cg._retrieve_connectivity(row_information, connected_edges=False)
        if is_in_set1_dict[sv_id]:
            edges_in1.append(sv_edges_in)
            edges_out1.append(sv_edges_out)
            areas_out1.append(sv_areas_out)
        else:
            edges_in2.append(sv_edges_in)
            edges_out2.append(sv_edges_out)
            areas_out2.append(sv_areas_out)
    edges_in1 = np.concatenate(edges_in1)
    edges_out1 = np.concatenate(edges_out1)
    areas_out1 = np.concatenate(areas_out1)
    edges_in2 = np.concatenate(edges_in2)
    edges_out2 = np.concatenate(edges_out2)
    areas_out2 = np.concatenate(areas_out2)

    contact_sites = _compute_contact_sites_from_edges(cg, edges_out1, areas_out1, edges_in2, sv_ids_set2)
    contact_sites_other_direction = _compute_contact_sites_from_edges(cg, edges_out2, areas_out2, edges_in1, sv_ids_set1)

    if len(contact_sites_other_direction) > len(contact_sites):
        return contact_sites_other_direction
    
    return contact_sites
