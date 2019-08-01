import time
import datetime
import collections

import numpy as np
import pytz

from pychunkedgraph.backend.utils import serializers, column_keys, row_keys, basetypes
from pychunkedgraph.backend import flatgraph_utils
from pychunkedgraph.backend.chunkedgraph_utils import (
    compute_indices_pandas,
    get_google_compatible_time_stamp,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    NamedTuple,
)

UTC = pytz.UTC


def add_atomic_edges_in_chunks(
    self,
    edge_id_dict: dict,
    edge_aff_dict: dict,
    edge_area_dict: dict,
    isolated_node_ids: Sequence[np.uint64],
    verbose: bool = True,
    time_stamp: Optional[datetime.datetime] = None,
):
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
    time_stamp = get_google_compatible_time_stamp(time_stamp, round_up=False)

    edge_id_keys = [
        "in_connected",
        "in_disconnected",
        "cross",
        "between_connected",
        "between_disconnected",
    ]
    edge_aff_keys = [
        "in_connected",
        "in_disconnected",
        "between_connected",
        "between_disconnected",
    ]

    # Check if keys exist and include an empty array if not
    n_edge_ids = 0
    chunk_id = None
    for edge_id_key in edge_id_keys:
        if not edge_id_key in edge_id_dict:
            empty_edges = np.array([], dtype=np.uint64).reshape(0, 2)
            edge_id_dict[edge_id_key] = empty_edges
        else:
            n_edge_ids += len(edge_id_dict[edge_id_key])

            if len(edge_id_dict[edge_id_key]) > 0:
                node_id = edge_id_dict[edge_id_key][0, 0]
                chunk_id = self.get_chunk_id(node_id)

    for edge_aff_key in edge_aff_keys:
        if not edge_aff_key in edge_aff_dict:
            edge_aff_dict[edge_aff_key] = np.array([], dtype=np.float32)

    time_start = time.time()

    # Catch trivial case
    if n_edge_ids == 0 and len(isolated_node_ids) == 0:
        return 0

    # Make parent id creation easier
    if chunk_id is None:
        chunk_id = self.get_chunk_id(isolated_node_ids[0])

    chunk_id_c = self.get_chunk_coordinates(chunk_id)
    parent_chunk_id = self.get_chunk_id(
        layer=2, x=chunk_id_c[0], y=chunk_id_c[1], z=chunk_id_c[2]
    )

    # Get connected component within the chunk
    chunk_node_ids = np.concatenate(
        [
            isolated_node_ids.astype(np.uint64),
            np.unique(edge_id_dict["in_connected"]),
            np.unique(edge_id_dict["in_disconnected"]),
            np.unique(edge_id_dict["cross"][:, 0]),
            np.unique(edge_id_dict["between_connected"][:, 0]),
            np.unique(edge_id_dict["between_disconnected"][:, 0]),
        ]
    )

    chunk_node_ids = np.unique(chunk_node_ids)

    node_chunk_ids = np.array(
        [self.get_chunk_id(c) for c in chunk_node_ids], dtype=np.uint64
    )

    u_node_chunk_ids, c_node_chunk_ids = np.unique(node_chunk_ids, return_counts=True)
    if len(u_node_chunk_ids) > 1:
        raise Exception(
            "%d: %d chunk ids found in node id list. "
            "Some edges might be in the wrong order. "
            "Number of occurences:" % (chunk_id, len(u_node_chunk_ids)),
            c_node_chunk_ids,
        )

    add_edge_ids = np.vstack([chunk_node_ids, chunk_node_ids]).T
    edge_ids = np.concatenate([edge_id_dict["in_connected"].copy(), add_edge_ids])

    graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
        edge_ids, make_directed=True
    )

    ccs = flatgraph_utils.connected_components(graph)

    if verbose:
        self.logger.debug("CC in chunk: %.3fs" % (time.time() - time_start))

    # Add rows for nodes that are in this chunk
    # a connected component at a time
    node_c = 0  # Just a counter for the log / speed measurement

    n_ccs = len(ccs)

    parent_ids = self.get_unique_node_id_range(parent_chunk_id, step=n_ccs)
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
        node_ids = unique_graph_ids[cc]

        u_chunk_ids = np.unique([self.get_chunk_id(n) for n in node_ids])

        if len(u_chunk_ids) > 1:
            self.logger.error(f"Found multiple chunk ids: {u_chunk_ids}")
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
                self.mutate_row(
                    serializers.serialize_uint64(node_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
            node_c += 1
            time_dict["creating_lv1_row"].append(time.time() - time_start_2)

        time_start_1 = time.time()
        # Create parent node
        rows.append(
            self.mutate_row(
                serializers.serialize_uint64(parent_id),
                {column_keys.Hierarchy.Child: node_ids},
                time_stamp=time_stamp,
            )
        )

        time_dict["creating_lv2_row"].append(time.time() - time_start_1)
        time_start_1 = time.time()

        cce_layers = self.get_cross_chunk_edges_layer(parent_cross_edges)
        u_cce_layers = np.unique(cce_layers)

        val_dict = {}
        for cc_layer in u_cce_layers:
            layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

            if len(layer_cross_edges) > 0:
                val_dict[
                    column_keys.Connectivity.CrossChunkEdge[cc_layer]
                ] = layer_cross_edges

        if len(val_dict) > 0:
            rows.append(
                self.mutate_row(
                    serializers.serialize_uint64(parent_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )
        node_c += 1

        time_dict["adding_cross_edges"].append(time.time() - time_start_1)

        if len(rows) > 100000:
            time_start_1 = time.time()
            self.bulk_write(rows)
            time_dict["writing"].append(time.time() - time_start_1)

    if len(rows) > 0:
        time_start_1 = time.time()
        self.bulk_write(rows)
        time_dict["writing"].append(time.time() - time_start_1)

    if verbose:
        self.logger.debug(
            "Time creating rows: %.3fs for %d ccs with %d nodes"
            % (time.time() - time_start, len(ccs), node_c)
        )

        for k in time_dict.keys():
            self.logger.debug(
                "%s -- %.3fms for %d instances -- avg = %.3fms"
                % (
                    k,
                    np.sum(time_dict[k]) * 1000,
                    len(time_dict[k]),
                    np.mean(time_dict[k]) * 1000,
                )
            )

