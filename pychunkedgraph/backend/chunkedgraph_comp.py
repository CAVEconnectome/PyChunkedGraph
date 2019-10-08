import numpy as np
import datetime
import collections
import itertools

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
    contact_sites_svs = edges[~np.in1d(edges, sv_ids).reshape(-1, 2)]
    contact_sites_svs_area_dict = collections.defaultdict(int)

    for area, sv_id in zip(areas, contact_sites_svs):
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
            data_pair = (cg.vx_vol_bounds[:,0] + cg.chunk_size * chunk_coordinates, np.sum(contact_sites_areas))
        else:
            data_pair = (chunk_coordinates, np.sum(contact_sites_areas))

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

def get_contact_sites_pairwise(cg, first_root_id, second_root_id, bounding_box=None, bb_is_coordinate=True, end_time=None, voxel_location=False):
    # Get information about the root id
    # All supervoxels
    first_root_l2_ids = cg.get_children_at_layer(first_root_id, 2)
    second_root_l2_ids = cg.get_children_at_layer(second_root_id, 2)

    if len(first_root_l2_ids) > len(second_root_l2_ids):
        smaller_set_l2_ids = second_root_l2_ids
        bigger_set_l2_ids = first_root_l2_ids
    else:
        smaller_set_l2_ids = first_root_l2_ids
        bigger_set_l2_ids = second_root_l2_ids

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

    for l2_id in bigger_set_l2_ids:
        chunk_coordinates = cg.get_chunk_coordinates(l2_id)
        new_candidate_smaller_set_l2_ids = chunk_coordinate_dict.get(tuple(chunk_coordinates))
        if new_candidate_smaller_set_l2_ids is not None:
            for new_candidate in new_candidate_smaller_set_l2_ids:
                candidate_smaller_set_l2_ids.add(smaller_set_l2_ids[new_candidate])
            candidate_bigger_set_l2_ids.append(l2_id)
    
    sv_ids_set1 = cg.get_children(list(candidate_smaller_set_l2_ids), flatten=True)
    sv_ids_set2 = cg.get_children(candidate_bigger_set_l2_ids, flatten=True)

    all_sv_ids = np.concatenate((sv_ids_set1, sv_ids_set2))
    is_in_set1 = np.concatenate((np.ones(len(sv_ids_set1)), np.zeros(len(sv_ids_set2))))
    is_in_set1_dict = dict(zip(all_sv_ids, is_in_set1))
    
    sv_rows_conc = cg.read_node_id_rows(node_ids=all_sv_ids,
                                          columns=[column_keys.Connectivity.Area,
                                                   column_keys.Connectivity.Partner,
                                                   column_keys.Connectivity.Connected],
                                          end_time=end_time,
                                          end_time_inclusive=True)

    def _retrieve_connectivity_optimized(dict_item):
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


    edges_in1 = []
    edges_out1 = []
    areas_out1 = []
    edges_in2 = []
    edges_out2 = []
    areas_out2 = []
    for row_information in sv_rows_conc.items():
        sv_id, _ = row_information
        # sv_edges_in, _, _ = cg._retrieve_connectivity(row_information, connected_edges=True)
        # sv_edges_out, _, sv_areas_out = cg._retrieve_connectivity(row_information, connected_edges=False)
        sv_edges_in, sv_edges_out, sv_areas_out = _retrieve_connectivity_optimized(row_information)
        if is_in_set1_dict[sv_id] == 1:
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

