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
from .utils.edges import Edges
from ..backend.utils.edges import TYPES as EDGE_TYPES, Edges
from .chunkedgraph_utils import compute_indices_pandas, get_google_compatible_time_stamp
from .flatgraph_utils import build_gt_graph, connected_components
from .utils import serializers, column_keys


def add_atomic_edges_in_chunks(
    cg_instance,
    chunk_coord,
    chunk_edges: dict,
    isolated: Sequence[np.uint64],
    verbose: bool = True,
    time_stamp: Optional[datetime.datetime] = None,
):
    """
    Creates atomic nodes in first abstraction layer for a SINGLE chunk
    and all abstract nodes in the second for the same chunk.
    All the edges (edge_ids) need to be from one chunk and no nodes should
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

    time_start = time.time()
    chunk_node_ids, chunk_edge_ids = _get_chunk_nodes_and_edges(chunk_edges, isolated)
    if not chunk_node_ids:
        return 0

    node_chunk_ids = cg_instance.get_chunk_ids_from_node_ids(chunk_node_ids)
    u_node_chunk_ids = np.unique(node_chunk_ids)
    assert len(u_node_chunk_ids) == 1

    graph, _, _, unique_ids = build_gt_graph(chunk_edge_ids, make_directed=True)
    ccs = connected_components(graph)
    if verbose:
        cg_instance.logger.debug(f"CC in chunk: {(time.time() - time_start):.3f}s")

    node_c = 0  # Just a counter for the log / speed measurement
    time_dict = collections.defaultdict(list)
    time_start = time.time()
    sparse_indices = {}
    remapping = {}
    for k in edge_id_dict.keys():
        # Circumvent datatype issues
        u_ids, inv_ids = np.unique(edge_id_dict[k], return_inverse=True)
        mapped_ids = np.arange(len(u_ids), dtype=np.int32)
        remapped_arr = mapped_ids[inv_ids].reshape(edge_id_dict[k].shape)

        sparse_indices[k] = compute_indices_pandas(remapped_arr)
        remapping[k] = dict(zip(u_ids, mapped_ids))

    time_dict["sparse_indices"].append(time.time() - time_start)

    parent_chunk_id = cg_instance.get_chunk_id(layer=2, *chunk_coord)
    parent_ids = cg_instance.get_unique_node_id_range(parent_chunk_id, step=len(ccs))

    time_stamp = _get_valid_timestamp(time_stamp)
    rows = []
    for i_cc, cc in enumerate(ccs):
        node_ids = unique_ids[cc]

        u_chunk_ids = np.unique([cg_instance.get_chunk_id(n) for n in node_ids])

        if len(u_chunk_ids) > 1:
            cg_instance.logger.error(f"Found multiple chunk ids: {u_chunk_ids}")
            raise Exception()

        # Create parent id
        parent_id = parent_ids[i_cc]

        parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

        # Add rows for nodes that are in this chunk
        for i_node_id, node_id in enumerate(node_ids):
            # Extract edges relevant to this node

            # in chunk + connected
            time_start_2 = time.time()
            if node_id in remapping["in_connected"]:
                row_ids, column_ids = sparse_indices["in_connected"][
                    remapping["in_connected"][node_id]
                ]

                inv_column_ids = (column_ids + 1) % 2

                connected_ids = edge_id_dict["in_connected"][row_ids, inv_column_ids]
                connected_affs = edge_aff_dict["in_connected"][row_ids]
                connected_areas = edge_area_dict["in_connected"][row_ids]
                time_dict["in_connected"].append(time.time() - time_start_2)
                time_start_2 = time.time()
            else:
                connected_ids = np.array([], dtype=np.uint64)
                connected_affs = np.array([], dtype=np.float32)
                connected_areas = np.array([], dtype=np.uint64)

            # in chunk + disconnected
            if node_id in remapping["in_disconnected"]:
                row_ids, column_ids = sparse_indices["in_disconnected"][
                    remapping["in_disconnected"][node_id]
                ]
                inv_column_ids = (column_ids + 1) % 2

                disconnected_ids = edge_id_dict["in_disconnected"][
                    row_ids, inv_column_ids
                ]
                disconnected_affs = edge_aff_dict["in_disconnected"][row_ids]
                disconnected_areas = edge_area_dict["in_disconnected"][row_ids]
                time_dict["in_disconnected"].append(time.time() - time_start_2)
                time_start_2 = time.time()
            else:
                disconnected_ids = np.array([], dtype=np.uint64)
                disconnected_affs = np.array([], dtype=np.float32)
                disconnected_areas = np.array([], dtype=np.uint64)

            # out chunk + connected
            if node_id in remapping["between_connected"]:
                row_ids, column_ids = sparse_indices["between_connected"][
                    remapping["between_connected"][node_id]
                ]

                row_ids = row_ids[column_ids == 0]
                column_ids = column_ids[column_ids == 0]
                inv_column_ids = (column_ids + 1) % 2
                time_dict["out_connected_mask"].append(time.time() - time_start_2)
                time_start_2 = time.time()

                connected_ids = np.concatenate(
                    [
                        connected_ids,
                        edge_id_dict["between_connected"][row_ids, inv_column_ids],
                    ]
                )
                connected_affs = np.concatenate(
                    [connected_affs, edge_aff_dict["between_connected"][row_ids]]
                )
                connected_areas = np.concatenate(
                    [connected_areas, edge_area_dict["between_connected"][row_ids]]
                )

                parent_cross_edges = np.concatenate(
                    [parent_cross_edges, edge_id_dict["between_connected"][row_ids]]
                )

                time_dict["out_connected"].append(time.time() - time_start_2)
                time_start_2 = time.time()

            # out chunk + disconnected
            if node_id in remapping["between_disconnected"]:
                row_ids, column_ids = sparse_indices["between_disconnected"][
                    remapping["between_disconnected"][node_id]
                ]

                row_ids = row_ids[column_ids == 0]
                column_ids = column_ids[column_ids == 0]
                inv_column_ids = (column_ids + 1) % 2
                time_dict["out_disconnected_mask"].append(time.time() - time_start_2)
                time_start_2 = time.time()

                disconnected_ids = np.concatenate(
                    [
                        disconnected_ids,
                        edge_id_dict["between_disconnected"][row_ids, inv_column_ids],
                    ]
                )
                disconnected_affs = np.concatenate(
                    [disconnected_affs, edge_aff_dict["between_disconnected"][row_ids]]
                )
                disconnected_areas = np.concatenate(
                    [
                        disconnected_areas,
                        edge_area_dict["between_disconnected"][row_ids],
                    ]
                )

                time_dict["out_disconnected"].append(time.time() - time_start_2)
                time_start_2 = time.time()

            # cross
            if node_id in remapping["cross"]:
                row_ids, column_ids = sparse_indices["cross"][
                    remapping["cross"][node_id]
                ]

                row_ids = row_ids[column_ids == 0]
                column_ids = column_ids[column_ids == 0]
                inv_column_ids = (column_ids + 1) % 2
                time_dict["cross_mask"].append(time.time() - time_start_2)
                time_start_2 = time.time()

                connected_ids = np.concatenate(
                    [connected_ids, edge_id_dict["cross"][row_ids, inv_column_ids]]
                )
                connected_affs = np.concatenate(
                    [connected_affs, np.full((len(row_ids)), np.inf, dtype=np.float32)]
                )
                connected_areas = np.concatenate(
                    [connected_areas, np.ones((len(row_ids)), dtype=np.uint64)]
                )

                parent_cross_edges = np.concatenate(
                    [parent_cross_edges, edge_id_dict["cross"][row_ids]]
                )
                time_dict["cross"].append(time.time() - time_start_2)
                time_start_2 = time.time()

            # Create node
            partners = np.concatenate([connected_ids, disconnected_ids])
            affinities = np.concatenate([connected_affs, disconnected_affs])
            areas = np.concatenate([connected_areas, disconnected_areas])
            connected = np.arange(len(connected_ids), dtype=np.int)

            val_dict = {
                column_keys.Connectivity.Partner: partners,
                column_keys.Connectivity.Affinity: affinities,
                column_keys.Connectivity.Area: areas,
                column_keys.Connectivity.Connected: connected,
                column_keys.Hierarchy.Parent: parent_id,
            }

            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(node_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
            node_c += 1
            time_dict["creating_lv1_row"].append(time.time() - time_start_2)

        time_start = time.time()
        # Create parent node
        rows.append(
            cg_instance.mutate_row(
                serializers.serialize_uint64(parent_id),
                {column_keys.Hierarchy.Child: node_ids},
                time_stamp=time_stamp,
            )
        )

        time_dict["creating_lv2_row"].append(time.time() - time_start)
        time_start = time.time()

        cce_layers = cg_instance.get_cross_chunk_edges_layer(parent_cross_edges)
        u_cce_layers = np.unique(cce_layers)

        val_dict = {}
        for cc_layer in u_cce_layers:
            layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

            if layer_cross_edges:
                val_dict[
                    column_keys.Connectivity.CrossChunkEdge[cc_layer]
                ] = layer_cross_edges

        if val_dict > 0:
            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(parent_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
        node_c += 1

        time_dict["adding_cross_edges"].append(time.time() - time_start)

        if len(rows) > 100000:
            time_start = time.time()
            cg_instance.bulk_write(rows)
            time_dict["writing"].append(time.time() - time_start)

    if rows:
        time_start = time.time()
        cg_instance.bulk_write(rows)
        time_dict["writing"].append(time.time() - time_start)


def _get_chunk_nodes_and_edges(chunk_edges: dict, isolated_ids: Sequence[np.uint64]):
    """get all nodes and edges in the chunk"""
    isolated_nodes_self_edges = np.vstack([isolated_ids, isolated_ids]).T
    node_ids = [isolated_ids]
    edge_ids = [isolated_nodes_self_edges]
    for edge_type in EDGE_TYPES:
        edges = chunk_edges[edge_type]
        node_ids.append(edges.node_ids1)
        node_ids.append(edges.node_ids2)
        edge_ids.append(np.vstack([edges.node_ids1, edges.node_ids2]).T)

    chunk_node_ids = np.unique(np.concatenate(node_ids))
    chunk_edge_ids = np.unique(np.concatenate(edge_ids))

    return (chunk_node_ids, chunk_edge_ids)


def _get_valid_timestamp(timestamp):
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)

    # Comply to resolution of BigTables TimeRange
    return get_google_compatible_time_stamp(timestamp, round_up=False)
