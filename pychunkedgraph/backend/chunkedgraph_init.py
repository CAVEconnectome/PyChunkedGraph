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


from .chunkedgraph import ChunkedGraph
from .utils import basetypes
from ..edges.definitions import Edges, IN_CHUNK, BT_CHUNK, CX_CHUNK, TYPES as EDGE_TYPES
from .chunkedgraph_utils import compute_indices_pandas, get_google_compatible_time_stamp
from .flatgraph_utils import build_gt_graph, connected_components
from .utils import serializers, column_keys


def add_atomic_edges(
    cg_instance: ChunkedGraph,
    chunk_coord: np.ndarray,
    chunk_edges_d: dict,
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

    :param cg_instance:
    :param chunk_coord: [x,y,z]
    :param chunk_edges_d: dict of {"edge_type": Edges}
    :param isolated: list of isolated node ids
    :param time_stamp: datetime
    """

    chunk_node_ids, chunk_edge_ids = _get_chunk_nodes_and_edges(chunk_edges_d, isolated)
    if not chunk_node_ids:
        return 0

    chunk_ids = cg_instance.get_chunk_ids_from_node_ids(chunk_node_ids)
    assert len(np.unique(chunk_ids)) == 1

    graph, _, _, unique_ids = build_gt_graph(chunk_edge_ids, make_directed=True)
    ccs = connected_components(graph)

    parent_chunk_id = cg_instance.get_chunk_id(layer=2, *chunk_coord)
    parent_ids = cg_instance.get_unique_node_id_range(parent_chunk_id, step=len(ccs))

    sparse_indices, remapping = _get_remapping(chunk_edges_d)
    time_stamp = _get_valid_timestamp(time_stamp)
    rows = []
    for i_cc, component in enumerate(ccs):
        node_ids = unique_ids[component]
        parent_id = parent_ids[i_cc]
        parent_cross_edges = []

        for node_id in node_ids:
            _edges = _get_parent_cross_edges(
                node_id, chunk_edges_d, sparse_indices, remapping
            )
            parent_cross_edges.append(_edges)
            val_dict = {column_keys.Hierarchy.Parent: parent_id}

            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(node_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )

        parent_cross_edges = np.concatenate(parent_cross_edges)
        cce_layers = cg_instance.get_cross_chunk_edges_layer(parent_cross_edges)
        u_cce_layers = np.unique(cce_layers)

        val_dict = {column_keys.Hierarchy.Child: node_ids}
        for cc_layer in u_cce_layers:
            layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

            if layer_cross_edges:
                val_dict[
                    column_keys.Connectivity.CrossChunkEdge[cc_layer]
                ] = layer_cross_edges

        rows.append(
            cg_instance.mutate_row(
                serializers.serialize_uint64(parent_id), val_dict, time_stamp=time_stamp
            )
        )

        if len(rows) > 100000:
            cg_instance.bulk_write(rows)
            rows = []

    cg_instance.bulk_write(rows)


def _get_chunk_nodes_and_edges(chunk_edges_d: dict, isolated_ids: Sequence[np.uint64]):
    """
    returns IN_CHUNK edges and nodes_ids
    """
    isolated_nodes_self_edges = np.vstack([isolated_ids, isolated_ids]).T
    node_ids = [isolated_ids]
    edge_ids = [isolated_nodes_self_edges]
    for edge_type in EDGE_TYPES:
        edges = chunk_edges_d[edge_type]
        node_ids.append(edges.node_ids1)
        if edge_type == IN_CHUNK:
            node_ids.append(edges.node_ids2)
            edge_ids.append(edges.get_pairs())

    chunk_node_ids = np.unique(np.concatenate(node_ids))
    chunk_edge_ids = np.concatenate(edge_ids)

    return (chunk_node_ids, chunk_edge_ids)


def _get_remapping(chunk_edges_d: dict):
    """
    TODO add logic
    """
    sparse_indices = {}
    remapping = {}
    for edge_type in EDGE_TYPES:
        edges = chunk_edges_d[edge_type].get_pairs()
        u_ids, inv_ids = np.unique(edges, return_inverse=True)
        mapped_ids = np.arange(len(u_ids), dtype=np.int32)
        remapped_arr = mapped_ids[inv_ids].reshape(edges.shape)
        sparse_indices[edge_type] = compute_indices_pandas(remapped_arr)
        remapping[edge_type] = dict(zip(u_ids, mapped_ids))
    return sparse_indices, remapping


def _get_valid_timestamp(timestamp):
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)

    # Comply to resolution of BigTables TimeRange
    return get_google_compatible_time_stamp(timestamp, round_up=False)


def _get_parent_cross_edges(node_id, chunk_edges_d, sparse_indices, remapping):
    """
    TODO add docs
    """
    parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)
    if node_id in remapping[BT_CHUNK]:
        row_ids, column_ids = sparse_indices[BT_CHUNK][remapping[BT_CHUNK][node_id]]
        row_ids = row_ids[column_ids == 0]
        participating_edges = chunk_edges_d[BT_CHUNK][row_ids]
        parent_cross_edges = np.concatenate(
            [parent_cross_edges, participating_edges.get_pairs()]
        )
    if node_id in remapping[CX_CHUNK]:
        row_ids, column_ids = sparse_indices[CX_CHUNK][remapping[CX_CHUNK][node_id]]
        row_ids = row_ids[column_ids == 0]
        participating_edges = chunk_edges_d[CX_CHUNK][row_ids]
        parent_cross_edges = np.concatenate(
            [parent_cross_edges, participating_edges.get_pairs()]
        )
    return parent_cross_edges
