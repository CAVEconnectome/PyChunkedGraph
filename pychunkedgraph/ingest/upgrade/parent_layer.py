# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import math
import multiprocessing as mp
from collections import defaultdict

import fastremap
import numpy as np
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.graph.types import empty_2d
from pychunkedgraph.utils.general import chunked

from .utils import exists_as_parent


CHILDREN = {}
CX_EDGES = {}


def _populate_nodes_and_children(
    cg: ChunkedGraph, chunk_id: np.uint64, nodes: list = None
) -> dict:
    global CHILDREN
    if nodes:
        CHILDREN = cg.get_children(nodes)
        return
    response = cg.range_read_chunk(chunk_id, properties=Hierarchy.Child)
    for k, v in response.items():
        CHILDREN[k] = v[0].value


def _populate_cx_edges_with_timestamps(
    cg: ChunkedGraph, layer: int, nodes: list, nodes_ts: list
):
    """
    Collect timestamps of edits from children, since we use the same timestamp
    for all IDs involved in an edit, we can use the timestamps of
    when cross edges of children were updated.
    """
    global CX_EDGES
    attrs = [Connectivity.CrossChunkEdge[l] for l in range(layer, cg.meta.layer_count)]
    all_children = np.concatenate(list(CHILDREN.values()))

    response = cg.client.read_nodes(node_ids=all_children, properties=attrs)
    for node, node_ts in zip(nodes, nodes_ts):
        temp = defaultdict(lambda: defaultdict(list))
        for child in CHILDREN[node]:
            if child not in response:
                continue
            for key, val in response[child].items():
                for cell in val:
                    if cell.timestamp < node_ts:
                        # edges from before the node existed, not relevant
                        continue
                    temp[cell.timestamp][key.index].append(cell.value)
        result = {}
        for ts, edges_d in temp.items():
            for _layer, edge_lists in edges_d.items():
                edges = np.concatenate(edge_lists)
                edges = np.unique(edges, axis=0)
                edges_d[_layer] = edges
            result[ts] = edges_d
        CX_EDGES[node] = result


def update_cross_edges(cg: ChunkedGraph, layer, node, node_ts, earliest_ts) -> list:
    """
    Helper function to update a single ID.
    Returns a list of mutations with timestamps.
    """

    rows = []
    cx_edges_d = CX_EDGES[node][node_ts]
    edges = np.concatenate([empty_2d, *cx_edges_d.values()])
    if node_ts > earliest_ts:
        if node != np.unique(cg.get_parents(edges[:, 0], time_stamp=node_ts))[0]:
            # if node is not the parent at this ts, it must be invalid
            assert not exists_as_parent(cg, node, edges[:, 0]), f"{node}, {node_ts}"
            return rows

    for ts, cx_edges_d in CX_EDGES[node].items():
        edges = np.concatenate([empty_2d, *cx_edges_d.values()])
        nodes = np.unique(edges[:, 1])
        parents = cg.get_roots(nodes, time_stamp=ts, stop_layer=layer, ceil=False)
        edge_parents_d = dict(zip(nodes, parents))
        val_dict = {}
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
    global CHILDREN, CX_EDGES
    cg_info, layer, nodes, nodes_ts, earliest_ts = args
    rows = []
    cg = ChunkedGraph(**cg_info)
    parents = cg.get_parents(nodes, fail_to_zero=True)
    for node, parent, node_ts in zip(nodes, parents, nodes_ts):
        if parent == 0:
            # invalid id caused by failed ingest task
            continue
        _rows = update_cross_edges(cg, layer, node, node_ts, earliest_ts)
        rows.extend(_rows)
        CHILDREN.pop(node)
        CX_EDGES.pop(node)
    cg.client.write(rows)


def update_chunk(
    cg: ChunkedGraph, chunk_coords: list[int], layer: int, nodes: list = None
):
    """
    Iterate over all layer IDs in a chunk and update their cross chunk edges.
    """
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)
    _populate_nodes_and_children(cg, chunk_id, nodes=nodes)
    if not CHILDREN:
        return
    nodes = list(CHILDREN.keys())
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False, normalize=True)
    _populate_cx_edges_with_timestamps(cg, layer, nodes, nodes_ts)

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
