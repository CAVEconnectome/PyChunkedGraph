"""
Functions for creating parents in level 3 and above
"""

import time
import datetime
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from .helpers import get_touching_atomic_chunks
from ...utils.general import chunked
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
    n_threads: int = 32,
) -> None:
    x, y, z = parent_coords
    parent_chunk_id = cg_instance.get_chunk_id(layer=layer_id, x=x, y=y, z=z)

    start = time.time()
    children_ids = _read_children_chunks(
        n_threads, cg_instance, layer_id, children_coords
    )
    print(f"_read_children_chunks: {time.time()-start}")

    # cross_edge_dict, children_ids = _process_chunks(cg_instance, layer_id, children_coords)
    start = time.time()
    edge_ids = _resolve_cross_chunk_edges(
        n_threads, layer_id, children_ids, cross_edge_dict
    )
    print(f"_resolve_cross_chunk_edges: {time.time()-start}")
    print(len(children_ids), len(edge_ids))

    # Extract connected components
    isolated_node_mask = ~np.in1d(children_ids, np.unique(edge_ids))
    add_node_ids = children_ids[isolated_node_mask].squeeze()
    add_edge_ids = np.vstack([add_node_ids, add_node_ids]).T
    edge_ids = np.concatenate([edge_ids, add_edge_ids])

    graph, _, _, graph_ids = flatgraph_utils.build_gt_graph(
        edge_ids, make_directed=True
    )

    ccs = flatgraph_utils.connected_components(graph)
    start = time.time()
    _write_out_connected_components(
        cg_instance,
        layer_id,
        parent_chunk_id,
        ccs,
        cross_edge_dict,
        graph_ids,
        time_stamp,
    )
    print(f"_write_out_connected_components: {time.time()-start}")
    return f"{layer_id}_{'_'.join(map(str, (x, y, z)))}"


def _read_children_chunks(n_threads, cg_instance, layer_id, children_coords):
    cg_info = cg_instance.get_serialized_info()
    del cg_info["credentials"]
    
    multi_args = []
    for child_coord in children_coords:
        multi_args.append((cg_info, layer_id-1, child_coord))
    children_ids = mu.multithread_func(
        _read_chunk_thread, multi_args, n_threads=len(multi_args)
    )
    return np.concatenate(children_ids)


def _get_cross_edges(cg_instance, layer_id, chunk_coord):
    layer2_chunks = get_touching_atomic_chunks(cg_instance.meta, layer_id, chunk_coord)

    cg_info = cg_instance.get_serialized_info()
    del cg_info["credentials"]
    
    multi_args = []
    for layer2_chunk in layer2_chunks:
        multi_args.append((cg_info, 2, layer2_chunk))
    children_ids = mu.multithread_func(
        _read_chunk_thread, multi_args, n_threads=len(multi_args)
    )
    return np.concatenate(children_ids)    


def _read_chunk_thread(args):
    cg_info, layer_id, chunk_coord = args
    cg_instance = ChunkedGraph(**cg_info)
    return _read_chunk(cg_instance, layer_id, chunk_coord)


def _read_chunk(cg_instance, layer_id, chunk_coord):
    x, y, z = chunk_coord
    range_read = cg_instance.range_read_chunk(
        layer_id, x, y, z, columns=column_keys.Hierarchy.Child
    )

    # Deserialize row keys and store child with highest id for comparison
    row_ids = np.fromiter(range_read.keys(), dtype=np.uint64)
    segment_ids = np.array([cg_instance.get_segment_id(r_id) for r_id in row_ids])
    max_child_ids = []
    for row_data in range_read.values():
        max_child_ids.append(np.max(row_data[0].value))

    sorting = np.argsort(segment_ids)[::-1]
    row_ids = row_ids[sorting]
    max_child_ids = np.array(max_child_ids, dtype=np.uint64)[sorting]

    counter = defaultdict(int)
    max_child_ids_occ_so_far = np.zeros(len(max_child_ids), dtype=np.int)
    for i_row in range(len(max_child_ids)):
        max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
        counter[max_child_ids[i_row]] += 1
    row_ids = row_ids[max_child_ids_occ_so_far == 0]
    return row_ids


def _read_atomic_chunk_cross_edges(cg_instance, chunk_coord, cross_edge_layer):
    x, y, z = chunk_coord
    range_read = cg_instance.range_read_chunk(
        2, x, y, z, columns=column_keys.Connectivity.CrossChunkEdge[cross_edge_layer]
    )

    cross_edges = [r[0].value for r in range_read.values()]
    cross_edges = np.concatenate(cross_edges)

    sv_ids1 = cross_edges[:,0]
    sv_ids2 = cross_edges[:,1]
    return

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
