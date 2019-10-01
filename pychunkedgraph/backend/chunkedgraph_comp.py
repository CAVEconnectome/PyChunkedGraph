import numpy as np
import datetime
import collections

from pychunkedgraph.backend import chunkedgraph, flatgraph_utils
from pychunkedgraph.backend.utils import column_keys

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


def get_contact_sites(cg, root_id, bounding_box=None, bb_is_coordinate=True, compute_partner=True):
    # Get information about the root id
    # All supervoxels
    sv_ids = cg.get_subgraph_nodes(root_id,
                                   bounding_box=bounding_box,
                                   bb_is_coordinate=bb_is_coordinate)
    # All edges that are _not_ connected / on
    edges, affs, areas = cg.get_subgraph_edges(root_id,
                                               bounding_box=bounding_box,
                                               bb_is_coordinate=bb_is_coordinate,
                                               connected_edges=False)

    # Build area lookup dictionary
    cs_svs = edges[~np.in1d(edges, sv_ids).reshape(-1, 2)]
    area_dict = collections.defaultdict(int)

    for area, sv_id in zip(areas, cs_svs):
        area_dict[sv_id] += area

    area_dict_vec = np.vectorize(area_dict.get)

    # Extract svs from contacting root ids
    u_cs_svs = np.unique(cs_svs)

    # Load edges of these cs_svs
    edges_cs_svs_rows = cg.read_node_id_rows(node_ids=u_cs_svs,
                                             columns=[column_keys.Connectivity.Partner,
                                                      column_keys.Connectivity.Connected])

    temp_root_id = np.uint64(648518346344081009)
    temp_sv_ids = cg.get_subgraph_nodes(temp_root_id,
                                   bounding_box=bounding_box,
                                   bb_is_coordinate=bb_is_coordinate)
    temp_intersect = np.intersect1d(temp_sv_ids, u_cs_svs)
    
    pre_cs_edges = []
    intersect_edges = []
    for ri in edges_cs_svs_rows.items():
        r = cg._retrieve_connectivity(ri)
        pre_cs_edges.extend(r[0])
        ttt = r[0]
        temp_add = ttt[np.in1d(ttt, temp_sv_ids).reshape(-1, 2)]
        if len(temp_add) > 0:
            intersect_edges.append(temp_add)

    graph, _, _, unique_ids = flatgraph_utils.build_gt_graph(
        pre_cs_edges, make_directed=True)

    # connected components in this graph will be combined in one component
    ccs = flatgraph_utils.connected_components(graph)


    cs_dict = collections.defaultdict(list)
    temp_sv_dict = collections.defaultdict(list)

    for cc in ccs:
        cc_sv_ids_all = unique_ids[cc]

        cc_sv_ids = cc_sv_ids_all[np.in1d(cc_sv_ids_all, u_cs_svs)]
        cs_areas = area_dict_vec(cc_sv_ids)

        if compute_partner:
            temp_sv_dict[int(cc_sv_ids[0])] = np.sum(cs_areas)
            partner_root_id = len(temp_sv_dict)
            cc_sv_ids_roots = cg.get_roots(cc_sv_ids)
            uni_sv_roots = np.unique(cc_sv_ids_roots)
            temp_intersect_test = np.intersect1d(temp_sv_ids, cc_sv_ids)
            if len(temp_intersect_test) > 0:
                import ipdb
                ipdb.set_trace()
            # print('intersect test', len(temp_intersect_test))
            # if uni_sv_roots.shape[0] > 1:
            # import ipdb
            # ipdb.set_trace()
        else:
            partner_root_id = len(cs_dict)
            cs_dict[partner_root_id].append(np.sum(cs_areas))

        # if partner_root_id % 100 == 0:
        print(partner_root_id, np.sum(cs_areas))

    if compute_partner:
        sv_list = np.array(list(temp_sv_dict.keys()), dtype=np.uint64)
        partner_roots = cg.get_roots(sv_list)
        for i in range(len(partner_roots)):
            cs_dict[int(partner_roots[i])].append(temp_sv_dict.get(int(sv_list[i])))

    import ipdb
    ipdb.set_trace()

    return cs_dict