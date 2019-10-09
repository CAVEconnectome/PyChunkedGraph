"""
Functions for creating parents in level 3 and above
"""

import datetime
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from ...backend import flatgraph_utils
from ...backend.chunkedgraph import ChunkedGraph
from ...backend.chunkedgraph_utils import get_valid_timestamp
from ...backend.utils import serializers, column_keys


def add_layer(
    cg_instance,
    layer_id: int,
    parent_coords: Sequence[int],
    children_coords: Sequence[Sequence[int]],
    *,
    time_stamp: Optional[datetime.datetime] = None,
    n_threads: int = 8,
) -> None:
    x, y, z = parent_coords
    parent_chunk_id = cg_instance.get_chunk_id(layer=layer_id, x=x, y=y, z=z)
    children_ids, cross_edge_dict = _read_children_chunks(
        n_threads, cg_instance, layer_id, children_coords
    )

    # cross_edge_dict, children_ids = _process_chunks(cg_instance, layer_id, children_coords)
    edge_ids = _resolve_cross_chunk_edges(layer_id, children_ids, cross_edge_dict)

    # Extract connected components
    isolated_node_mask = ~np.in1d(children_ids, np.unique(edge_ids))
    add_node_ids = children_ids[isolated_node_mask].squeeze()
    add_edge_ids = np.vstack([add_node_ids, add_node_ids]).T
    edge_ids.extend(add_edge_ids)

    graph, _, _, graph_ids = flatgraph_utils.build_gt_graph(
        edge_ids, make_directed=True
    )

    ccs = flatgraph_utils.connected_components(graph)
    _write_out_connected_components(
        cg_instance,
        layer_id,
        parent_chunk_id,
        ccs,
        cross_edge_dict,
        graph_ids,
        time_stamp,
    )
    return f"{layer_id}_{'_'.join(map(str, (x, y, z)))}"


def _read_children_chunks(n_threads, cg_instance, layer_id, children_coords):
    cg_info = cg_instance.get_serialized_info()
    multi_args = []
    for child_coord in children_coords:
        multi_args.append((cg_info, layer_id, child_coord))
    chunk_info = mu.multithread_func(
        _process_chunk, multi_args, n_threads=min(n_threads, len(multi_args))
    )

    children_ids = []
    cross_edge_dict = {}
    for ids, cross_edge_d in chunk_info:
        children_ids.append(ids)
        cross_edge_dict = {**cross_edge_dict, **cross_edge_d}
    return np.concatenate(children_ids), cross_edge_dict


def _process_chunk(cg_info, layer_id, chunk_coord):
    cg_instance = ChunkedGraph(**cg_info)
    cross_edge_dict = defaultdict(dict)
    row_ids, cross_edge_columns_d = _read_chunk(cg_instance, layer_id, chunk_coord)
    for row_id in row_ids:
        if row_id in cross_edge_columns_d:
            cell_family = cross_edge_columns_d[row_id]
            for l in range(layer_id - 1, cg_instance.n_layers):
                cross_edges_key = column_keys.Connectivity.CrossChunkEdge[l]
                if cross_edges_key in cell_family:
                    cross_edge_dict[row_id][l] = cell_family[cross_edges_key][0].value
    return row_ids, cross_edge_dict


def _read_chunk(cg_instance, layer_id, chunk_coord):
    x, y, z = chunk_coord
    columns = [column_keys.Hierarchy.Child] + [
        column_keys.Connectivity.CrossChunkEdge[l]
        for l in range(layer_id - 1, cg_instance.n_layers)
    ]
    range_read = cg_instance.range_read_chunk(layer_id - 1, x, y, z, columns=columns)
    # Deserialize row keys and store child with highest id for
    # comparison
    row_ids = np.fromiter(range_read.keys(), dtype=np.uint64)
    segment_ids = np.array([cg_instance.get_segment_id(r_id) for r_id in row_ids])
    cross_edge_columns_d = {}
    max_child_ids = []
    for row_id, row_data in range_read.items():
        cross_edge_columns = {
            k: v
            for (k, v) in row_data.items()
            if k.family_id == cg_instance.cross_edge_family_id
        }
        if cross_edge_columns:
            cross_edge_columns_d[row_id] = cross_edge_columns
        node_child_ids = row_data[column_keys.Hierarchy.Child][0].value
        max_child_ids.append(np.max(node_child_ids))

    sorting = np.argsort(segment_ids)[::-1]
    row_ids = row_ids[sorting]
    max_child_ids = np.array(max_child_ids, dtype=np.uint64)[sorting]

    counter = defaultdict(int)
    max_child_ids_occ_so_far = np.zeros(len(max_child_ids), dtype=np.int)
    for i_row in range(len(max_child_ids)):
        max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
        counter[max_child_ids[i_row]] += 1
    row_ids = row_ids[max_child_ids_occ_so_far == 0]
    return row_ids, cross_edge_columns_d


def _resolve_cross_chunk_edges(layer_id, node_ids, cross_edge_dict) -> None:
    cross_edge_dict = defaultdict(dict, cross_edge_dict)
    atomic_partner_id_dict = {}
    atomic_child_id_dict_pairs = []
    for node_id in node_ids:
        if int(layer_id - 1) in cross_edge_dict[node_id]:
            atomic_cross_edges = cross_edge_dict[node_id][layer_id - 1]
            if len(atomic_cross_edges) > 0:
                atomic_partner_id_dict[node_id] = atomic_cross_edges[:, 1]
                new_pairs = zip(
                    atomic_cross_edges[:, 0], [node_id] * len(atomic_cross_edges)
                )
                atomic_child_id_dict_pairs.extend(new_pairs)

    d = dict(atomic_child_id_dict_pairs)
    atomic_child_id_dict = defaultdict(np.uint64, d)

    edge_ids = []
    for child_key in atomic_partner_id_dict:
        this_atomic_partner_ids = atomic_partner_id_dict[child_key]
        partners = {
            atomic_child_id_dict[atomic_cross_id]
            for atomic_cross_id in this_atomic_partner_ids
            if atomic_child_id_dict[atomic_cross_id] != 0
        }
        if len(partners) > 0:
            partners = np.array(list(partners), dtype=np.uint64)[:, None]

            this_ids = np.array([child_key] * len(partners), dtype=np.uint64)[:, None]
            these_edges = np.concatenate([this_ids, partners], axis=1)

            edge_ids.extend(these_edges)
    return edge_ids


def _write_out_connected_components(
    cg_instance, layer_id, parent_chunk_id, ccs, cross_edge_dict, graph_ids, time_stamp
) -> None:
    time_stamp = get_valid_timestamp(time_stamp)
    parent_layer_ids = range(layer_id, cg_instance.n_layers + 1)
    cc_connections = {l: [] for l in parent_layer_ids}
    for i_cc, cc in enumerate(ccs):
        node_ids = graph_ids[cc]
        parent_cross_edges = defaultdict(list)

        # Collect row info for nodes that are in this chunk
        for node_id in node_ids:
            if node_id in cross_edge_dict:
                # Extract edges relevant to this node
                for l in range(layer_id, cg_instance.n_layers):
                    if (
                        l in cross_edge_dict[node_id]
                        and len(cross_edge_dict[node_id][l]) > 0
                    ):
                        parent_cross_edges[l].append(cross_edge_dict[node_id][l])

        if cg_instance.use_skip_connections and len(node_ids) == 1:
            for l in parent_layer_ids:
                if l == cg_instance.n_layers or len(parent_cross_edges[l]) > 0:
                    cc_connections[l].append([node_ids, parent_cross_edges])
                    break
        else:
            cc_connections[layer_id].append([node_ids, parent_cross_edges])

    # Write out cc info
    rows = []
    parent_chunk_id_dict = cg_instance.get_parent_chunk_id_dict(parent_chunk_id)
    # Iterate through layers
    for parent_layer_id in parent_layer_ids:
        if len(cc_connections[parent_layer_id]) == 0:
            continue

        parent_chunk_id = parent_chunk_id_dict[parent_layer_id]
        reserved_parent_ids = cg_instance.get_unique_node_id_range(
            parent_chunk_id, step=len(cc_connections[parent_layer_id])
        )

        for i_cc, cc_info in enumerate(cc_connections[parent_layer_id]):
            node_ids, parent_cross_edges = cc_info

            parent_id = reserved_parent_ids[i_cc]
            val_dict = {column_keys.Hierarchy.Parent: parent_id}

            for node_id in node_ids:
                rows.append(
                    cg_instance.mutate_row(
                        serializers.serialize_uint64(node_id),
                        val_dict,
                        time_stamp=time_stamp,
                    )
                )

            val_dict = {column_keys.Hierarchy.Child: node_ids}
            for l in range(parent_layer_id, cg_instance.n_layers):
                if l in parent_cross_edges and len(parent_cross_edges[l]) > 0:
                    val_dict[
                        column_keys.Connectivity.CrossChunkEdge[l]
                    ] = np.concatenate(parent_cross_edges[l])

            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(parent_id),
                    val_dict,
                    time_stamp=time_stamp,
                )
            )

            if len(rows) > 100000:
                cg_instance.bulk_write(rows)
                rows = []

    if len(rows) > 0:
        cg_instance.bulk_write(rows)
