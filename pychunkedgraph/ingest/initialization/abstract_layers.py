"""
Functions for creating parents in level 3 and above
"""

import time
import datetime
import multiprocessing as mp
from collections import defaultdict
from typing import Optional
from typing import Sequence
from typing import List

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from ...utils.general import chunked
from ...backend import flatgraph_utils
from ...backend.utils import basetypes
from ...backend.utils import serializers
from ...backend.utils import column_keys
from ...backend.chunkedgraph import ChunkedGraph
from ...backend.chunkedgraph_utils import get_valid_timestamp
from ...backend.chunkedgraph_utils import filter_failed_node_ids
from ...backend.chunks.atomic import get_touching_atomic_chunks
from ...backend.connectivity.cross_edges import get_children_chunk_cross_edges


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
    print(f"_read_children_chunks: {time.time()-start}, id count {len(children_ids)}")

    start = time.time()
    edge_ids = get_children_chunk_cross_edges(cg_instance, layer_id, parent_coords)
    print(f"get_children_chunk_cross_edges: {time.time()-start}, {len(edge_ids)}")
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

    row_ids = filter_failed_node_ids(row_ids, segment_ids, max_children_ids)
    children_ids_shared.append(row_ids)


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

