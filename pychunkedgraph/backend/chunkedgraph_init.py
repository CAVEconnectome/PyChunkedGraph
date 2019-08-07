"""
Module for stuff related to creating the initial chunkedgraph
"""

import time
import datetime
import collections
from typing import Optional, Sequence

import pytz
import numpy as np

from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.column_family import MaxVersionsGCRule


from .utils import basetypes
from ..edges.definitions import Edges, IN_CHUNK, TYPES as EDGE_TYPES
from .chunkedgraph_utils import compute_indices_pandas, get_google_compatible_time_stamp
from .flatgraph_utils import build_gt_graph, connected_components
from .utils import serializers, column_keys


def add_atomic_edges(
    cg_instance,
    chunk_coord,
    chunk_edges: dict,
    isolated: Sequence[np.uint64],
    time_stamp: Optional[datetime.datetime] = None,
):
    """
    Creates atomic nodes in first abstraction layer for a SINGLE chunk
    and all abstract nodes in the second for the same chunk.
    All the edges (edge_ids) need to be from one chunk and no nodes should
    exist for this chunk prior to calling this function. All cross edges
    (cross_edge_ids) have to point out the chunk (first entry is the id
    within the chunk)

    :param chunk_edges: dict
    :param isolated: list of isolated node ids
    :param time_stamp: datetime
    """

    chunk_node_ids, chunk_edge_ids = _get_chunk_nodes_and_edges(chunk_edges, isolated)
    if not chunk_node_ids:
        return 0

    chunk_ids = cg_instance.get_chunk_ids_from_node_ids(chunk_node_ids)
    assert len(np.unique(chunk_ids)) == 1

    graph, _, _, unique_ids = build_gt_graph(chunk_edge_ids, make_directed=True)
    ccs = connected_components(graph)

    node_c = 0  # Just a counter for the log / speed measurement
    parent_chunk_id = cg_instance.get_chunk_id(layer=2, *chunk_coord)
    parent_ids = cg_instance.get_unique_node_id_range(parent_chunk_id, step=len(ccs))

    sparse_indices = {}
    remapping = {}
    for k in edge_id_dict.keys():
        u_ids, inv_ids = np.unique(edge_id_dict[k], return_inverse=True)
        mapped_ids = np.arange(len(u_ids), dtype=np.int32)
        remapped_arr = mapped_ids[inv_ids].reshape(edge_id_dict[k].shape)
        sparse_indices[k] = compute_indices_pandas(remapped_arr)
        remapping[k] = dict(zip(u_ids, mapped_ids))

    time_stamp = _get_valid_timestamp(time_stamp)
    rows = []
    for i_cc, component in enumerate(ccs):
        node_ids = unique_ids[component]
        parent_id = parent_ids[i_cc]
        parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

        for node_id in node_ids:

            # out chunk + connected
            if node_id in remapping["between_connected"]:
                row_ids, column_ids = sparse_indices["between_connected"][
                    remapping["between_connected"][node_id]
                ]

                row_ids = row_ids[column_ids == 0]
                parent_cross_edges = np.concatenate(
                    [parent_cross_edges, edge_id_dict["between_connected"][row_ids]]
                )

            # cross
            if node_id in remapping["cross"]:
                row_ids, column_ids = sparse_indices["cross"][
                    remapping["cross"][node_id]
                ]

                row_ids = row_ids[column_ids == 0]

                parent_cross_edges = np.concatenate(
                    [parent_cross_edges, edge_id_dict["cross"][row_ids]]
                )

            val_dict = {column_keys.Hierarchy.Parent: parent_id}

            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(node_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
            node_c += 1

        # Create parent node
        rows.append(
            cg_instance.mutate_row(
                serializers.serialize_uint64(parent_id),
                {column_keys.Hierarchy.Child: node_ids},
                time_stamp=time_stamp,
            )
        )

        cce_layers = cg_instance.get_cross_chunk_edges_layer(parent_cross_edges)
        u_cce_layers = np.unique(cce_layers)

        val_dict = {}
        for cc_layer in u_cce_layers:
            layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

            if layer_cross_edges:
                val_dict[
                    column_keys.Connectivity.CrossChunkEdge[cc_layer]
                ] = layer_cross_edges

        if val_dict:
            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(parent_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
        node_c += 1

        if len(rows) > 100000:
            cg_instance.bulk_write(rows)

    if rows:
        cg_instance.bulk_write(rows)


def _get_chunk_nodes_and_edges(chunk_edges: dict, isolated_ids: Sequence[np.uint64]):
    """get all nodes and edges in the chunk"""
    isolated_nodes_self_edges = np.vstack([isolated_ids, isolated_ids]).T
    node_ids = [isolated_ids]
    edge_ids = [isolated_nodes_self_edges]
    for edge_type in EDGE_TYPES:
        edges = chunk_edges[edge_type]
        node_ids.append(edges.node_ids1)
        if edge_type == IN_CHUNK:
            node_ids.append(edges.node_ids2)
            edge_ids.append(np.vstack([edges.node_ids1, edges.node_ids2]).T)

    chunk_node_ids = np.unique(np.concatenate(node_ids))
    chunk_edge_ids = np.concatenate(edge_ids)

    return (chunk_node_ids, chunk_edge_ids)


def _get_valid_timestamp(timestamp):
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)

    # Comply to resolution of BigTables TimeRange
    return get_google_compatible_time_stamp(timestamp, round_up=False)
