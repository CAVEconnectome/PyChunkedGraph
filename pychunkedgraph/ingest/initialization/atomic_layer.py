"""
Functions for creating atomic nodes and their level 2 abstract parents
"""

import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import pytz
import numpy as np

from ...backend.chunkedgraph import ChunkedGraph
from ...backend.utils import basetypes
from ...backend.utils import serializers
from ...backend.utils import column_keys
from ...backend.edges import Edges
from ...backend.edges import EDGE_TYPES
from ...backend.chunkedgraph_utils import compute_indices_pandas
from ...backend.chunkedgraph_utils import get_valid_timestamp
from ...backend.flatgraph_utils import build_gt_graph
from ...backend.flatgraph_utils import connected_components


def add_atomic_edges(
    cg_instance: ChunkedGraph,
    chunk_coord: np.ndarray,
    chunk_edges_d: Dict[str, Edges],
    isolated: Sequence[int],
    time_stamp: Optional[datetime.datetime] = None,
):
    chunk_node_ids, chunk_edge_ids = _get_chunk_nodes_and_edges(chunk_edges_d, isolated)
    if not chunk_node_ids.size:
        return 0

    chunk_ids = cg_instance.get_chunk_ids_from_node_ids(chunk_node_ids)
    assert len(np.unique(chunk_ids)) == 1

    graph, _, _, unique_ids = build_gt_graph(chunk_edge_ids, make_directed=True)
    ccs = connected_components(graph)

    parent_chunk_id = cg_instance.get_chunk_id(
        layer=2, x=chunk_coord[0], y=chunk_coord[1], z=chunk_coord[2]
    )
    parent_ids = cg_instance.get_unique_node_id_range(parent_chunk_id, step=len(ccs))

    sparse_indices, remapping = _get_remapping(chunk_edges_d)
    time_stamp = get_valid_timestamp(time_stamp)
    rows = []
    for i_cc, component in enumerate(ccs):
        _rows = _process_component(
            cg_instance,
            chunk_edges_d,
            parent_ids[i_cc],
            unique_ids[component],
            sparse_indices,
            remapping,
            time_stamp,
        )
        rows.extend(_rows)

        if len(rows) > 100000:
            cg_instance.bulk_write(rows)
            rows = []
    cg_instance.bulk_write(rows)


def _get_chunk_nodes_and_edges(chunk_edges_d: dict, isolated_ids: Sequence[int]):
    """
    in-chunk edges and nodes_ids
    """
    isolated_nodes_self_edges = np.vstack([isolated_ids, isolated_ids]).T
    node_ids = [isolated_ids]
    edge_ids = [isolated_nodes_self_edges]
    for edge_type in EDGE_TYPES:
        edges = chunk_edges_d[edge_type]
        node_ids.append(edges.node_ids1)
        if edge_type == EDGE_TYPES.in_chunk:
            node_ids.append(edges.node_ids2)
            edge_ids.append(edges.get_pairs())

    chunk_node_ids = np.unique(np.concatenate(node_ids))
    return (chunk_node_ids, np.concatenate(edge_ids))


def _get_remapping(chunk_edges_d: dict):
    """
    TODO add logic explanation
    """
    sparse_indices = {}
    remapping = {}
    for edge_type in [EDGE_TYPES.between_chunk, EDGE_TYPES.cross_chunk]:
        edges = chunk_edges_d[edge_type].get_pairs()
        u_ids, inv_ids = np.unique(edges, return_inverse=True)
        mapped_ids = np.arange(len(u_ids), dtype=np.int32)
        remapped_arr = mapped_ids[inv_ids].reshape(edges.shape)
        sparse_indices[edge_type] = compute_indices_pandas(remapped_arr)
        remapping[edge_type] = dict(zip(u_ids, mapped_ids))
    return sparse_indices, remapping


def _process_component(
    cg_instance,
    chunk_edges_d,
    parent_id,
    node_ids,
    sparse_indices,
    remapping,
    time_stamp,
):
    rows = []
    chunk_out_edges = []  # out = between + cross
    # print(node_ids)
    for node_id in node_ids:
        _edges = _get_outgoing_edges(node_id, chunk_edges_d, sparse_indices, remapping)
        chunk_out_edges.append(_edges)
        val_dict = {column_keys.Hierarchy.Parent: parent_id}
        r_key = serializers.serialize_uint64(node_id)
        rows.append(cg_instance.mutate_row(r_key, val_dict, time_stamp=time_stamp))

    chunk_out_edges = np.concatenate(chunk_out_edges)
    cce_layers = cg_instance.get_cross_chunk_edges_layer(chunk_out_edges)
    u_cce_layers = np.unique(cce_layers)

    val_dict = {column_keys.Hierarchy.Child: node_ids}
    for cc_layer in u_cce_layers:
        layer_out_edges = chunk_out_edges[cce_layers == cc_layer]
        if layer_out_edges.size:
            col = column_keys.Connectivity.CrossChunkEdge[cc_layer]
            val_dict[col] = layer_out_edges

    r_key = serializers.serialize_uint64(parent_id)
    rows.append(cg_instance.mutate_row(r_key, val_dict, time_stamp=time_stamp))
    return rows


def _get_outgoing_edges(node_id, chunk_edges_d, sparse_indices, remapping):
    """
    edges of node_id pointing outside the chunk (between and cross)
    """
    chunk_out_edges = np.array([], dtype=basetypes.NODE_ID).reshape(0, 2)
    for edge_type in remapping:
        if node_id in remapping[edge_type]:
            edges_obj = chunk_edges_d[edge_type]
            edges = edges_obj.get_pairs()

            row_ids, column_ids = sparse_indices[edge_type][
                remapping[edge_type][node_id]
            ]
            row_ids = row_ids[column_ids == 0]
            # edges that this node is part of
            chunk_out_edges = np.concatenate([chunk_out_edges, edges[row_ids]])
    return chunk_out_edges
