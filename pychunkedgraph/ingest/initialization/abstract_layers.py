"""
Functions for creating parents in level 3 and above
"""

import time
import datetime
import multiprocessing as mp
from collections import defaultdict
from collections import abc
from typing import Optional
from typing import Sequence
from typing import List

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from .helpers import get_touching_atomic_chunks
from ...utils.general import chunked
from ...backend import flatgraph_utils
from ...backend.utils import basetypes
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
) -> None:
    x, y, z = parent_coords

    start = time.time()
    children_ids = _read_children_chunks(cg_instance, layer_id, children_coords)
    print(f"_read_children_chunks: {time.time()-start}, count {len(children_ids)}")

    start = time.time()
    edge_ids = _get_cross_edges(cg_instance, layer_id, parent_coords)
    print(f"_get_cross_edges: {time.time()-start}")
    # print(len(children_ids), len(edge_ids))

    # Extract connected components
    isolated_node_mask = ~np.in1d(children_ids, np.unique(edge_ids))
    add_node_ids = children_ids[isolated_node_mask].squeeze()
    add_edge_ids = np.vstack([add_node_ids, add_node_ids]).T
    edge_ids.extend(add_edge_ids)

    graph, _, _, graph_ids = flatgraph_utils.build_gt_graph(
        edge_ids, make_directed=True
    )

    ccs = flatgraph_utils.connected_components(graph)
    start = time.time()
    _write_connected_components(
        cg_instance,
        layer_id,
        cg_instance.get_chunk_id(layer=layer_id, x=x, y=y, z=z),
        ccs,
        graph_ids,
        time_stamp,
    )
    print(f"_write_connected_components: {time.time()-start}")
    return f"{layer_id}_{'_'.join(map(str, (x, y, z)))}"


def _read_children_chunks(cg_instance, layer_id, children_coords):
    with mp.Manager() as manager:
        children_ids_shared = manager.list()
        multi_args = []
        for child_coord in children_coords:
            multi_args.append(
                (
                    children_ids_shared,
                    cg_instance.get_serialized_info(credentials=False),
                    layer_id - 1,
                    child_coord,
                )
            )
        mu.multiprocess_func(
            _read_chunk_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )
        return np.concatenate(children_ids_shared)


def _read_chunk_helper(args):
    children_ids_shared, cg_info, layer_id, chunk_coord = args
    cg_instance = ChunkedGraph(**cg_info)
    _read_chunk(children_ids_shared, cg_instance, layer_id, chunk_coord)


def _filter_latest_ids(row_ids, segment_ids, max_children_ids):
    sorting = np.argsort(segment_ids)[::-1]
    row_ids = row_ids[sorting]
    max_child_ids = np.array(max_children_ids, dtype=np.uint64)[sorting]

    counter = defaultdict(int)
    max_child_ids_occ_so_far = np.zeros(len(max_child_ids), dtype=np.int)
    for i_row in range(len(max_child_ids)):
        max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
        counter[max_child_ids[i_row]] += 1
    return row_ids[max_child_ids_occ_so_far == 0]


def _read_chunk(children_ids_shared, cg_instance, layer_id, chunk_coord):
    x, y, z = chunk_coord
    range_read = cg_instance.range_read_chunk(
        layer_id, x, y, z, columns=column_keys.Hierarchy.Child
    )
    row_ids = []
    max_children_ids = []
    for row_id, row_data in range_read.items():
        row_ids.append(row_id)
        max_children_ids.append(np.max(row_data[0].value))
    row_ids = np.array(row_ids, dtype=basetypes.NODE_ID)
    segment_ids = np.array([cg_instance.get_segment_id(r_id) for r_id in row_ids])

    row_ids = _filter_latest_ids(row_ids, segment_ids, max_children_ids)
    children_ids_shared.append(row_ids)


def _get_cross_edges(cg_instance, layer_id, chunk_coord) -> List:
    start = time.time()
    layer2_chunks = get_touching_atomic_chunks(
        cg_instance.meta, layer_id, chunk_coord, include_both=False
    )
    if not len(layer2_chunks):
        return []
    # print(f"get_touching_atomic_chunks: {time.time()-start}")
    # print(f"touching chunks count (1 side): {len(layer2_chunks)}")

    cg_info = cg_instance.get_serialized_info(credentials=False)

    start = time.time()
    multi_args = []
    for layer2_chunk in layer2_chunks:
        multi_args.append((cg_info, layer2_chunk, layer_id - 1))
    cross_edges = mu.multithread_func(
        _read_atomic_chunk_cross_edges_helper, multi_args, n_threads=4
    )
    # print(f"_read_atomic_chunk_cross_edges: {time.time()-start}")

    cross_edges = np.concatenate(cross_edges)
    if len(cross_edges):
        cross_edges = np.unique(cross_edges, axis=0)
    return list(cross_edges)


def _read_atomic_chunk_cross_edges_helper(args):
    cg_info, layer2_chunk, cross_edge_layer = args
    cg_instance = ChunkedGraph(**cg_info)

    start = time.time()
    cross_edges = _read_atomic_chunk_cross_edges(
        cg_instance, layer2_chunk, cross_edge_layer
    )
    # print(f"single atomic chunk: {time.time()-start}, edges {len(cross_edges)}")
    return cross_edges


def _read_atomic_chunk_cross_edges(cg_instance, chunk_coord, cross_edge_layer):
    x, y, z = chunk_coord
    child_key = column_keys.Hierarchy.Child
    cross_edge_key = column_keys.Connectivity.CrossChunkEdge[cross_edge_layer]
    range_read = cg_instance.range_read_chunk(
        2, x, y, z, columns=[child_key, cross_edge_key]
    )

    row_ids = []
    max_children_ids = []
    for row_id, row_data in range_read.items():
        row_ids.append(row_id)
        max_children_ids.append(np.max(row_data[child_key][0].value))

    row_ids = np.array(row_ids, dtype=basetypes.NODE_ID)
    segment_ids = np.array([cg_instance.get_segment_id(r_id) for r_id in row_ids])
    l2ids = _filter_latest_ids(row_ids, segment_ids, max_children_ids)

    parent_neighboring_chunk_supervoxels_d = defaultdict(list)
    for l2id in l2ids:
        edges = range_read[l2id][cross_edge_key][0].value
        parent_neighboring_chunk_supervoxels_d[l2id] = edges[:, 1]

    parent_ids = cg_instance.get_roots(l2ids, stop_layer=cross_edge_layer)
    cross_edges = []
    for i, l2id in enumerate(parent_neighboring_chunk_supervoxels_d):
        parent_id = parent_ids[i]
        neighboring_parent_ids = cg_instance.get_roots(
            parent_neighboring_chunk_supervoxels_d[l2id], stop_layer=cross_edge_layer
        )
        chunk_parent_ids = np.array(
            [parent_id] * len(neighboring_parent_ids), dtype=basetypes.NODE_ID
        )
        cross_edges.append(np.vstack([chunk_parent_ids, neighboring_parent_ids]).T)

    if cross_edges:
        return np.unique(np.concatenate(cross_edges), axis=0)
    return np.empty([0, 2], dtype=np.uint64)


def _write_connected_components(
    cg_instance, layer_id, parent_chunk_id, ccs, graph_ids, time_stamp
) -> None:
    if not ccs:
        return

    ccs_with_node_ids = []
    for cc in ccs:
        ccs_with_node_ids.append(graph_ids[cc])

    chunked_ccs = chunked(ccs_with_node_ids, len(ccs_with_node_ids) // mp.cpu_count())
    cg_info = cg_instance.get_serialized_info(credentials=False)
    multi_args = []

    for ccs in chunked_ccs:
        multi_args.append((cg_info, layer_id, parent_chunk_id, ccs, time_stamp))
    mu.multiprocess_func(
        _write_components_helper,
        multi_args,
        n_threads=min(len(multi_args), mp.cpu_count()),
    )


def _write_components_helper(args):
    cg_info, layer_id, parent_chunk_id, ccs, time_stamp = args
    _write_components(
        ChunkedGraph(**cg_info), layer_id, parent_chunk_id, ccs, time_stamp
    )


def _write_components(cg_instance, layer_id, parent_chunk_id, ccs, time_stamp):
    time_stamp = get_valid_timestamp(time_stamp)
    cc_connections = {l: [] for l in (layer_id, cg_instance.n_layers)}
    for node_ids in ccs:
        if cg_instance.use_skip_connections and len(node_ids) == 1:
            cc_connections[cg_instance.n_layers].append(node_ids)
        else:
            cc_connections[layer_id].append(node_ids)

    rows = []
    parent_chunk_id_dict = cg_instance.get_parent_chunk_id_dict(parent_chunk_id)
    # Iterate through layers
    for parent_layer_id in (layer_id, cg_instance.n_layers):
        parent_chunk_id = parent_chunk_id_dict[parent_layer_id]
        reserved_parent_ids = cg_instance.get_unique_node_id_range(
            parent_chunk_id, step=len(cc_connections[parent_layer_id])
        )

        for i_cc, node_ids in enumerate(cc_connections[parent_layer_id]):
            parent_id = reserved_parent_ids[i_cc]
            for node_id in node_ids:
                rows.append(
                    cg_instance.mutate_row(
                        serializers.serialize_uint64(node_id),
                        {column_keys.Hierarchy.Parent: parent_id},
                        time_stamp=time_stamp,
                    )
                )

            rows.append(
                cg_instance.mutate_row(
                    serializers.serialize_uint64(parent_id),
                    {column_keys.Hierarchy.Child: node_ids},
                    time_stamp=time_stamp,
                )
            )

            if len(rows) > 100000:
                cg_instance.bulk_write(rows)
                rows = []
    cg_instance.bulk_write(rows)

