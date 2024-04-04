# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import math
import multiprocessing as mp

import fastremap
import numpy as np
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.graph.edges.utils import concatenate_cross_edge_dicts
from pychunkedgraph.utils.general import chunked

from .common import exists_as_parent


CHILDREN = {}
TIMESTAMPS = {}


def _populate_nodes_and_children(cg: ChunkedGraph, chunk_id) -> dict:
    global CHILDREN
    response = cg.range_read_chunk(chunk_id, properties=Hierarchy.Child)
    for k, v in response.items():
        CHILDREN[k] = v[0].value


def _populate_edit_timestamps(cg: ChunkedGraph):
    """
    Collect timestamps of edits from children, since we use the same timestamp
    for all IDs involved in an edit, we can use the timestamps of
    when cross edges of children were updated.
    """
    global TIMESTAMPS
    attrs = [Connectivity.CrossChunkEdge[l] for l in range(2, cg.meta.layer_count)]
    all_children = np.concatenate(list(CHILDREN.values()))
    response = cg.client.read_nodes(node_ids=all_children, properties=attrs)
    for node, children in CHILDREN.items():
        TIMESTAMPS[node] = set()
        for child in children:
            if child not in response:
                continue
            for val in response[child].values():
                for cell in val:
                    TIMESTAMPS[node].add(cell.timestamp)


def update_cross_edges(
    cg: ChunkedGraph, layer, node, node_ts, children, earliest_ts
) -> list:
    """
    Helper function to update a single ID.
    Returns a list of mutations with timestamps.
    """

    rows = []
    cx_edges_d = cg.get_cross_chunk_edges(children, time_stamp=node_ts, raw_only=True)
    cx_edges_d = concatenate_cross_edge_dicts(cx_edges_d.values(), unique=True)
    edges = list(cx_edges_d.values())
    if len(edges) == 0:
        # nothing to do
        return rows
    edges = np.concatenate(edges)
    if node_ts > earliest_ts:
        if node != np.unique(cg.get_parents(edges[:, 0], time_stamp=node_ts))[0]:
            # if node is not the parent at this ts, it must be invalid
            assert not exists_as_parent(cg, node, edges[:, 0]), f"{node}, {node_ts}"
            return rows

    for ts in sorted(TIMESTAMPS[node]):
        cx_edges_d = cg.get_cross_chunk_edges(children, time_stamp=ts, raw_only=True)
        cx_edges_d = concatenate_cross_edge_dicts(cx_edges_d.values(), unique=True)
        edges = np.concatenate(list(cx_edges_d.values()))

        val_dict = {}
        nodes = np.unique(edges[:, 1])
        # parents = cg.get_parents(nodes, time_stamp=ts)
        parents = cg.get_roots(nodes, time_stamp=ts, stop_layer=layer, ceil=False)
        edge_parents_d = dict(zip(nodes, parents))
        for layer, layer_edges in cx_edges_d.items():
            layer_edges = fastremap.remap(
                layer_edges, edge_parents_d, preserve_missing_labels=True
            )
            layer_edges[:, 0] = node
            layer_edges = np.unique(layer_edges, axis=0)
            col = Connectivity.CrossChunkEdge[layer]
            val_dict[col] = layer_edges
        row_id = serializers.serialize_uint64(node)
        rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=ts))
    return rows


def _update_cross_edges_helper(args):
    global CHILDREN, TIMESTAMPS
    cg_info, layer, nodes, nodes_ts, earliest_ts = args
    rows = []
    cg = ChunkedGraph(**cg_info)
    parents = cg.get_parents(nodes, fail_to_zero=True)
    for node, parent, node_ts in zip(nodes, parents, nodes_ts):
        if parent == 0:
            # invalid id caused by failed ingest task
            continue
        children = CHILDREN[node]
        _rows = update_cross_edges(cg, layer, node, node_ts, children, earliest_ts)
        rows.extend(_rows)
        CHILDREN.pop(node)
        TIMESTAMPS.pop(node)
    cg.client.write(rows)


def update_chunk(cg: ChunkedGraph, chunk_coords: list[int], layer: int):
    """
    Iterate over all layer IDs in a chunk and update their cross chunk edges.
    """
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)
    _populate_nodes_and_children(cg, chunk_id)
    if not CHILDREN:
        return
    nodes = list(CHILDREN.keys())
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False, normalize=True)
    _populate_edit_timestamps(cg)

    task_size = int(math.ceil(len(nodes) / mp.cpu_count() / 2))
    chunked_nodes = chunked(nodes, task_size)
    chunked_nodes_ts = chunked(nodes_ts, task_size)
    cg_info = cg.get_serialized_info()
    earliest_ts = cg.get_earliest_timestamp()

    multi_args = []
    for chunk, ts_chunk in zip(chunked_nodes, chunked_nodes_ts):
        args = (cg_info, layer, chunk, ts_chunk, earliest_ts)
        multi_args.append(args)

    mu.multiprocess_func(
        _update_cross_edges_helper,
        multi_args,
        n_threads=min(len(multi_args), mp.cpu_count()),
    )
