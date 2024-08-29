# pylint: disable=invalid-name, missing-docstring, c-extension-no-member, import-outside-toplevel

import datetime
import collections
from typing import Dict
from typing import Optional
from typing import Sequence

import fastremap
import numpy as np
from multiwrapper import multiprocessing_utils as mu

from . import ChunkedGraph
from . import attributes
from .edges import Edges
from .utils import flatgraph
from .types import Agglomeration


def _read_delta_root_rows(
    cg,
    start_id,
    end_id,
    time_stamp_start,
    time_stamp_end,
) -> Sequence[list]:
    # apply column filters to avoid Lock columns
    rows = cg.client.read_nodes(
        start_id=start_id,
        start_time=time_stamp_start,
        end_id=end_id,
        end_id_inclusive=False,
        properties=[attributes.Hierarchy.FormerParent, attributes.Hierarchy.NewParent],
        end_time=time_stamp_end,
        end_time_inclusive=True,
    )

    # new roots are those that have no NewParent in this time window
    new_root_ids = [
        k for (k, v) in rows.items() if attributes.Hierarchy.NewParent not in v
    ]

    # expired roots are the IDs of FormerParent's
    # whose timestamp is before the start_time
    expired_root_ids = []
    for v in rows.values():
        if attributes.Hierarchy.FormerParent in v:
            fp = v[attributes.Hierarchy.FormerParent]
            for cell_entry in fp:
                expired_root_ids.extend(cell_entry.value)
    return new_root_ids, expired_root_ids


def _read_root_rows_thread(args) -> list:
    start_seg_id, end_seg_id, serialized_cg_info, time_stamp = args
    cg = ChunkedGraph(**serialized_cg_info)
    start_id = cg.get_node_id(segment_id=start_seg_id, chunk_id=cg.root_chunk_id)
    end_id = cg.get_node_id(segment_id=end_seg_id, chunk_id=cg.root_chunk_id)
    rows = cg.client.read_nodes(
        start_id=start_id,
        end_id=end_id,
        end_id_inclusive=False,
        end_time=time_stamp,
        end_time_inclusive=True,
    )
    root_ids = [k for (k, v) in rows.items() if attributes.Hierarchy.NewParent not in v]
    return root_ids


def get_proofread_root_ids(
    cg: ChunkedGraph,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None,
):
    log_entries = cg.client.read_log_entries(
        start_time=start_time,
        end_time=end_time,
        properties=[attributes.OperationLogs.RootID],
        end_time_inclusive=True,
    )
    root_chunks = [e[attributes.OperationLogs.RootID] for e in log_entries.values()]
    if len(root_chunks) == 0:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
    new_roots = np.concatenate(root_chunks)

    root_rows = cg.client.read_nodes(
        node_ids=new_roots, properties=[attributes.Hierarchy.FormerParent]
    )
    old_root_chunks = [np.empty(0, dtype=np.uint64)]
    for e in root_rows.values():
        old_root_chunks.append(e[attributes.Hierarchy.FormerParent][0].value)
    old_roots = np.concatenate(old_root_chunks)
    return old_roots, new_roots


def get_latest_roots(
    cg, time_stamp: Optional[datetime.datetime] = None, n_threads: int = 1
) -> Sequence[np.uint64]:
    # Create filters: time and id range
    max_seg_id = cg.get_max_seg_id(cg.root_chunk_id) + 1
    n_blocks = 1 if n_threads == 1 else int(np.min([n_threads * 3 + 1, max_seg_id]))
    seg_id_blocks = np.linspace(1, max_seg_id, n_blocks + 1, dtype=np.uint64)
    cg_serialized_info = cg.get_serialized_info()
    if n_threads > 1:
        del cg_serialized_info["credentials"]

    multi_args = []
    for i_id_block in range(0, len(seg_id_blocks) - 1):
        multi_args.append(
            [
                seg_id_blocks[i_id_block],
                seg_id_blocks[i_id_block + 1],
                cg_serialized_info,
                time_stamp,
            ]
        )

    if n_threads == 1:
        results = mu.multiprocess_func(
            _read_root_rows_thread,
            multi_args,
            n_threads=n_threads,
            verbose=False,
            debug=n_threads == 1,
        )
    else:
        results = mu.multisubprocess_func(
            _read_root_rows_thread, multi_args, n_threads=n_threads
        )
    root_ids = []
    for result in results:
        root_ids.extend(result)
    return np.array(root_ids, dtype=np.uint64)


def get_delta_roots(
    cg: ChunkedGraph,
    time_stamp_start: datetime.datetime,
    time_stamp_end: Optional[datetime.datetime] = None,
) -> Sequence[np.uint64]:
    # Create filters: time and id range
    start_id = np.uint64(cg.get_chunk_id(layer=cg.meta.layer_count) + 1)
    end_id = cg.id_client.get_max_node_id(
        cg.get_chunk_id(layer=cg.meta.layer_count), root_chunk=True
    ) + np.uint64(1)
    new_root_ids, expired_root_id_candidates = _read_delta_root_rows(
        cg, start_id, end_id, time_stamp_start, time_stamp_end
    )

    # aggregate all the results together
    new_root_ids = np.array(new_root_ids, dtype=np.uint64)
    expired_root_id_candidates = np.array(expired_root_id_candidates, dtype=np.uint64)
    # filter for uniqueness
    expired_root_id_candidates = np.unique(expired_root_id_candidates)

    # filter out the expired root id's whose creation (measured by the timestamp
    # of their Child links) is after the time_stamp_start
    rows = cg.client.read_nodes(
        node_ids=expired_root_id_candidates,
        properties=[attributes.Hierarchy.Child],
        end_time=time_stamp_start,
    )
    expired_root_ids = np.array([k for (k, v) in rows.items()], dtype=np.uint64)
    return np.array(new_root_ids, dtype=np.uint64), expired_root_ids


def get_contact_sites(
    cg: ChunkedGraph,
    root_id,
    bounding_box=None,
    bbox_is_coordinate=True,
    compute_partner=True,
    time_stamp=None,
):
    # Get information about the root id
    # All supervoxels
    sv_ids = cg.get_subgraph(
        root_id,
        bbox=bounding_box,
        bbox_is_coordinate=bbox_is_coordinate,
        nodes_only=True,
        return_flattened=True,
    )
    # All edges that are _not_ connected / on
    edges, _, areas = cg.get_subgraph_edges(
        root_id,
        bbox=bounding_box,
        bbox_is_coordinate=bbox_is_coordinate,
        connected_edges=False,
    )

    # Build area lookup dictionary
    cs_svs = edges[~np.in1d(edges, sv_ids).reshape(-1, 2)]
    area_dict = collections.defaultdict(int)

    for area, sv_id in zip(areas, cs_svs):
        area_dict[sv_id] += area

    area_dict_vec = np.vectorize(area_dict.get)
    # Extract svs from contacting root ids
    u_cs_svs = np.unique(cs_svs)
    # Load edges of these cs_svs
    edges_cs_svs_rows = cg.client.read_nodes(
        node_ids=u_cs_svs,
        # columns=[attributes.Connectivity.Partner, attributes.Connectivity.Connected],
    )
    pre_cs_edges = []
    for ri in edges_cs_svs_rows.items():
        r = cg._retrieve_connectivity(ri)
        pre_cs_edges.extend(r[0])
    graph, _, _, unique_ids = flatgraph.build_gt_graph(pre_cs_edges, make_directed=True)
    # connected components in this graph will be combined in one component
    ccs = flatgraph.connected_components(graph)
    cs_dict = collections.defaultdict(list)
    for cc in ccs:
        cc_sv_ids = unique_ids[cc]
        cc_sv_ids = cc_sv_ids[np.in1d(cc_sv_ids, u_cs_svs)]
        cs_areas = area_dict_vec(cc_sv_ids)
        partner_root_id = (
            int(cg.get_root(cc_sv_ids[0], time_stamp=time_stamp))
            if compute_partner
            else len(cs_dict)
        )
        cs_dict[partner_root_id].append(np.sum(cs_areas))
    return cs_dict


def get_agglomerations(
    l2id_children_d: Dict,
    in_edges: Edges,
    ot_edges: Edges,
    cx_edges: Edges,
    sv_parent_d: Dict,
) -> Dict[np.uint64, Agglomeration]:
    l2id_agglomeration_d = {}
    _in = fastremap.remap(in_edges.node_ids1, sv_parent_d, preserve_missing_labels=True)
    _ot = fastremap.remap(ot_edges.node_ids1, sv_parent_d, preserve_missing_labels=True)
    _cx = fastremap.remap(cx_edges.node_ids1, sv_parent_d, preserve_missing_labels=True)
    for l2id in l2id_children_d:
        l2id_agglomeration_d[l2id] = Agglomeration(
            l2id,
            l2id_children_d[l2id],
            in_edges[_in == l2id],
            ot_edges[_ot == l2id],
            cx_edges[_cx == l2id],
        )
    return l2id_agglomeration_d


def get_activated_edges(
    cg: ChunkedGraph, operation_id: int, delta: Optional[int] = 100
) -> np.ndarray:
    """
    Returns edges that were made active by a merge operation.
    """
    from datetime import timedelta
    from .edits import merge_preprocess
    from .operation import GraphEditOperation
    from .operation import MergeOperation
    from .utils.generic import get_bounding_box as get_bbox

    log, time_stamp = cg.client.read_log_entry(operation_id)
    assert (
        GraphEditOperation.get_log_record_type(log) == MergeOperation
    ), "Must be a merge operation."

    time_stamp -= timedelta(milliseconds=delta)
    operation = GraphEditOperation.from_log_record(cg, log)
    bbox = get_bbox(
        operation.source_coords, operation.sink_coords, operation.bbox_offset
    )

    root_ids = set(
        cg.get_roots(
            operation.added_edges.ravel(), assert_roots=True, time_stamp=time_stamp
        )
    )
    assert len(root_ids) > 1, "More than one segment is required for merge."
    edges = operation.cg.get_subgraph(
        root_ids,
        bbox=bbox,
        bbox_is_coordinate=True,
        edges_only=True,
    )
    return merge_preprocess(
        cg,
        subgraph_edges=edges,
        supervoxels=operation.added_edges.ravel(),
        parent_ts=time_stamp,
    )
