import time
import datetime
import os
import collections
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, NamedTuple

import pytz
import numpy as np

from pychunkedgraph.backend import cutting, chunkedgraph_comp, flatgraph_utils
from pychunkedgraph.backend.utils import serializers, column_keys, row_keys, basetypes
from pychunkedgraph.backend.chunkedgraph_utils import compute_indices_pandas, \
    compute_bitmasks, get_google_compatible_time_stamp, \
    get_time_range_filter, get_time_range_and_column_filter, get_max_time, \
    combine_cross_chunk_edge_dicts, get_min_time, partial_row_data_to_column_dict


UTC = pytz.UTC

def add_atomic_edges_in_chunks_v2(cg, edge_id_dict: dict,
                                edge_aff_dict: dict, edge_area_dict: dict,
                                isolated_node_ids: Sequence[np.uint64],
                                verbose: bool = True,
                                time_stamp: Optional[datetime.datetime] = None):
    """ Creates atomic nodes in first abstraction layer for a SINGLE chunk
        and all abstract nodes in the second for the same chunk

    Alle edges (edge_ids) need to be from one chunk and no nodes should
    exist for this chunk prior to calling this function. All cross edges
    (cross_edge_ids) have to point out the chunk (first entry is the id
    within the chunk)

    :param edge_id_dict: dict
    :param edge_aff_dict: dict
    :param edge_area_dict: dict
    :param isolated_node_ids: list of uint64s
        ids of nodes that have no edge in the chunked graph
    :param verbose: bool
    :param time_stamp: datetime
    """
    if time_stamp is None:
        time_stamp = datetime.datetime.utcnow()

    if time_stamp.tzinfo is None:
        time_stamp = UTC.localize(time_stamp)

    # Comply to resolution of BigTables TimeRange
    time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                    round_up=False)

    edge_aff_keys = [
        'in_connected','in_disconnected','between_connected','between_disconnected']
    edge_id_keys = edge_aff_keys[:].insert(2, 'cross')

    # Check if keys exist and include an empty array if not
    n_edge_ids = 0
    empty_edges_array = np.array([], dtype=np.uint64).reshape(0, 2)

    for key in edge_id_keys:
        edge_id_dict[key] = np.concatenate(
            edge_id_dict.get(key, empty_edges_array.copy(),
                                empty_edges_array.copy()))
        n_edge_ids += len(edge_id_dict[key])

    for key in edge_aff_keys:
        edge_aff_dict[key] = np.concatenate(
            edge_aff_dict.get(key, empty_edges_array.copy(),
            empty_edges_array.copy()))

    # Get connected component within the chunk
    chunk_node_ids = np.concatenate([
            isolated_node_ids.astype(np.uint64),
            np.unique(edge_id_dict["in_connected"]),
            np.unique(edge_id_dict["in_disconnected"]),
            np.unique(edge_id_dict["cross"][:, 0]),
            np.unique(edge_id_dict["between_connected"][:, 0]),
            np.unique(edge_id_dict["between_disconnected"][:, 0])])
    
    if not len(chunk_node_ids): return 0

    chunk_node_ids = np.unique(chunk_node_ids)
    node_chunk_ids = np.array(
        [cg.get_chunk_id(c) for c in chunk_node_ids], dtype=np.uint64)

    u_node_chunk_ids, c_node_chunk_ids = np.unique(
        node_chunk_ids, return_counts=True)
    if len(u_node_chunk_ids) > 1:
        raise Exception("%d: %d chunk ids found in node id list. "
                        "Some edges might be in the wrong order. "
                        "Number of occurences:" %
                        (u_node_chunk_ids[0], len(u_node_chunk_ids)), c_node_chunk_ids)


    # add self edge to all node_ids to make sure they're
    # part of connected components because the graph is processed component wise
    # if not, the node_ids won't be stored
    edge_ids = np.concatenate([
        edge_id_dict["in_connected"].copy(),
        np.vstack([chunk_node_ids, chunk_node_ids]).T])

    graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
        edge_ids, make_directed=True)

    time_start = time.time()
    ccs = flatgraph_utils.connected_components(graph)

    if verbose:
        cg.logger.debug("CC in chunk: %.3fs" % (time.time() - time_start))

    # Add rows for nodes that are in this chunk
    # a connected component at a time
    node_c = 0  # Just a counter for the log / speed measurement

    n_ccs = len(ccs)

    # Make parent id creation easier
    parent_chunk_id = cg.get_chunk_id(
        layer=2, *cg.get_chunk_coordinates(u_node_chunk_ids[0]))

    parent_ids = cg.get_unique_node_id_range(parent_chunk_id, step=n_ccs)
    
    time_start = time.time()
    time_dict = collections.defaultdict(list)

    time_start_1 = time.time()
    sparse_indices = {}
    remapping = {}
    for k in edge_id_dict.keys():
        # Circumvent datatype issues

        u_ids, inv_ids = np.unique(edge_id_dict[k], return_inverse=True)
        mapped_ids = np.arange(len(u_ids), dtype=np.int32)
        remapped_arr = mapped_ids[inv_ids].reshape(edge_id_dict[k].shape)

        sparse_indices[k] = compute_indices_pandas(remapped_arr)
        remapping[k] = dict(zip(u_ids, mapped_ids))

    time_dict["sparse_indices"].append(time.time() - time_start_1)

    rows = []

    for i_cc, cc in enumerate(ccs):
        parent_id = parent_ids[i_cc]
        parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

        # Add rows for nodes that are in this chunk
        node_ids = unique_graph_ids[cc]
        for node_id in node_ids:
            time_start_2 = time.time()

            val_dict = {column_keys.Hierarchy.Parent: parent_id}
            rows.append(cg.mutate_row(serializers.serialize_uint64(node_id),
                                        val_dict, time_stamp=time_stamp))
            node_c += 1
            time_dict["creating_lv1_row"].append(time.time() - time_start_2)

        time_start_1 = time.time()
        # Create parent node
        rows.append(cg.mutate_row(serializers.serialize_uint64(parent_id),
                                    {column_keys.Hierarchy.Child: node_ids},
                                    time_stamp=time_stamp))

        time_dict["creating_lv2_row"].append(time.time() - time_start_1)
        time_start_1 = time.time()

        cce_layers = cg.get_cross_chunk_edges_layer(parent_cross_edges)
        u_cce_layers = np.unique(cce_layers)

        val_dict = {}
        for cc_layer in u_cce_layers:
            layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

            if len(layer_cross_edges) > 0:
                val_dict[column_keys.Connectivity.CrossChunkEdge[cc_layer]] = \
                    layer_cross_edges

        if len(val_dict) > 0:
            rows.append(cg.mutate_row(serializers.serialize_uint64(parent_id),
                                        val_dict, time_stamp=time_stamp))
        node_c += 1

        time_dict["adding_cross_edges"].append(time.time() - time_start_1)

        if len(rows) > 100000:
            time_start_1 = time.time()
            cg.bulk_write(rows)
            time_dict["writing"].append(time.time() - time_start_1)

    if len(rows) > 0:
        time_start_1 = time.time()
        cg.bulk_write(rows)
        time_dict["writing"].append(time.time() - time_start_1)

    if verbose:
        cg.logger.debug("Time creating rows: %.3fs for %d ccs with %d nodes" %
                            (time.time() - time_start, len(ccs), node_c))

        for k in time_dict.keys():
            cg.logger.debug("%s -- %.3fms for %d instances -- avg = %.3fms" %
                                (k, np.sum(time_dict[k])*1000, len(time_dict[k]),
                                np.mean(time_dict[k])*1000))