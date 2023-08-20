# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel

"""
Functions for creating parents in level 3 and above
"""

import math
import datetime
import multiprocessing as mp
from typing import Optional
from typing import Sequence
from collections import defaultdict

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from ...graph import types
from ...graph import attributes
from ...utils.general import chunked
from ...graph.utils import flatgraph
from ...graph.utils import basetypes
from ...graph.utils import serializers
from ...graph.chunkedgraph import ChunkedGraph
from ...graph.utils.generic import get_valid_timestamp
from ...graph.utils.generic import filter_failed_node_ids
from ...graph.chunks.hierarchy import get_children_chunk_coords
from .cross_edges import get_children_chunk_cross_edges
from .cross_edges import get_chunk_nodes_cross_edge_layer


def add_layer(
    cg: ChunkedGraph,
    layer_id: int,
    parent_coords: Sequence[int],
    children_coords: Sequence[Sequence[int]] = np.array([]),
    *,
    time_stamp: Optional[datetime.datetime] = None,
    n_threads: int = 4,
) -> None:
    if not children_coords.size:
        children_coords = get_children_chunk_coords(cg.meta, layer_id, parent_coords)
    children_ids = _read_children_chunks(cg, layer_id, children_coords, n_threads > 1)
    cx_edges = get_children_chunk_cross_edges(
        cg, layer_id, parent_coords, use_threads=n_threads > 1
    )

    node_layers = cg.get_chunk_layers(children_ids)
    edge_layers = cg.get_chunk_layers(np.unique(cx_edges))
    assert np.all(node_layers < layer_id), "invalid node layers"
    assert np.all(edge_layers < layer_id), "invalid edge layers"

    cx_edges = list(cx_edges)
    cx_edges.extend(np.vstack([children_ids, children_ids]).T)  # add self-edges
    graph, _, _, graph_ids = flatgraph.build_gt_graph(cx_edges, make_directed=True)
    raw_ccs = flatgraph.connected_components(graph)  # connected components with indices
    connected_components = [graph_ids[cc] for cc in raw_ccs]

    _write_connected_components(
        cg,
        layer_id,
        parent_coords,
        connected_components,
        cx_edges,
        get_valid_timestamp(time_stamp),
        n_threads > 1,
    )


def _read_children_chunks(
    cg: ChunkedGraph, layer_id, children_coords, use_threads=True
):
    if not use_threads:
        children_ids = [types.empty_1d]
        for child_coord in children_coords:
            children_ids.append(_read_chunk([], cg, layer_id - 1, child_coord))
        return np.concatenate(children_ids)

    with mp.Manager() as manager:
        children_ids_shared = manager.list()
        multi_args = []
        for child_coord in children_coords:
            multi_args.append(
                (
                    children_ids_shared,
                    cg.get_serialized_info(),
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
    cg = ChunkedGraph(**cg_info)
    _read_chunk(children_ids_shared, cg, layer_id, chunk_coord)


def _read_chunk(children_ids_shared, cg: ChunkedGraph, layer_id: int, chunk_coord):
    x, y, z = chunk_coord
    range_read = cg.range_read_chunk(
        cg.get_chunk_id(layer=layer_id, x=x, y=y, z=z),
        properties=attributes.Hierarchy.Child,
    )
    row_ids = []
    max_children_ids = []
    for row_id, row_data in range_read.items():
        row_ids.append(row_id)
        max_children_ids.append(np.max(row_data[0].value))
    row_ids = np.array(row_ids, dtype=basetypes.NODE_ID)
    segment_ids = np.array([cg.get_segment_id(r_id) for r_id in row_ids])

    row_ids = filter_failed_node_ids(row_ids, segment_ids, max_children_ids)
    children_ids_shared.append(row_ids)
    return row_ids


def _write_connected_components(
    cg, layer, pcoords, components, cx_edges, time_stamp, use_threads=True
):
    if len(components) == 0:
        return

    node_layer_d = {}
    if layer < cg.meta.layer_count:
        node_layer_d = get_chunk_nodes_cross_edge_layer(cg, layer, pcoords, use_threads)

    if not use_threads:
        _write(cg, layer, pcoords, components, cx_edges, node_layer_d, time_stamp, use_threads)
        return

    task_size = int(math.ceil(len(components) / mp.cpu_count() / 10))
    chunked_ccs = chunked(components, task_size)
    cg_info = cg.get_serialized_info()
    multi_args = []
    for ccs in chunked_ccs:
        args = (cg_info, layer, pcoords, ccs, cx_edges, node_layer_d, time_stamp)
        multi_args.append(args)
    mu.multiprocess_func(
        _write_components_helper,
        multi_args,
        n_threads=min(len(multi_args), mp.cpu_count()),
    )


def _write_components_helper(args):
    cg_info, layer, pcoords, ccs, cx_edges, node_layer_d, time_stamp = args
    cg = ChunkedGraph(**cg_info)
    _write(cg, layer, pcoords, ccs, cx_edges, node_layer_d, time_stamp)


def _write(
    cg: ChunkedGraph,
    layer_id,
    parent_coords,
    components,
    cx_edges,
    node_layer_d,
    time_stamp,
    use_threads=True,
):
    parent_layer_ids = range(layer_id, cg.meta.layer_count + 1)
    cc_connections = {l: [] for l in parent_layer_ids}
    for node_ids in components:
        layer = layer_id
        if len(node_ids) == 1:
            layer = node_layer_d.get(node_ids[0], cg.meta.layer_count)
        cc_connections[layer].append(node_ids)

    rows = []
    x, y, z = parent_coords
    parent_chunk_id = cg.get_chunk_id(layer=layer_id, x=x, y=y, z=z)
    parent_chunk_id_dict = cg.get_parent_chunk_id_dict(parent_chunk_id)

    cx_edges = np.array(cx_edges, dtype=basetypes.NODE_ID)
    for parent_layer_id in parent_layer_ids:
        if len(cc_connections[parent_layer_id]) == 0:
            continue

        parent_chunk_id = parent_chunk_id_dict[parent_layer_id]
        reserved_parent_ids = cg.id_client.create_node_ids(
            parent_chunk_id,
            size=len(cc_connections[parent_layer_id]),
            root_chunk=parent_layer_id == cg.meta.layer_count and use_threads,
        )

        for i_cc, node_ids in enumerate(cc_connections[parent_layer_id]):
            node_cx_edges_d = defaultdict(lambda: types.empty_2d)
            for node in node_ids:
                mask0 = cx_edges[:, 0] == node
                mask1 = cx_edges[:, 1] == node
                node_cx_edges_d[node] = cx_edges[mask0 | mask1]

            parent_id = reserved_parent_ids[i_cc]
            for node in node_ids:
                row_id = serializers.serialize_uint64(node)
                val_dict = {attributes.Hierarchy.Parent: parent_id}

                node_cx_edges = node_cx_edges_d[node]
                cx_layers = cg.get_cross_chunk_edges_layer(node_cx_edges)
                for layer in set(cx_layers):
                    layer_mask = cx_layers == layer
                    col = attributes.Connectivity.CrossChunkEdge[layer]
                    val_dict[col] = node_cx_edges[layer_mask]
                rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp))

            row_id = serializers.serialize_uint64(parent_id)
            val_dict = {attributes.Hierarchy.Child: node_ids}
            rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp))
            if len(rows) > 100000:
                cg.client.write(rows)
                rows = []
    cg.client.write(rows)
