# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import math, random, time
import multiprocessing as mp
from collections import defaultdict

import fastremap
import numpy as np
from tqdm import tqdm

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.graph.types import empty_2d
from pychunkedgraph.utils.general import chunked

from .utils import exists_as_parent, get_parent_timestamps


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


def _get_cx_edges_at_timestamp(node, response, ts):
    result = defaultdict(list)
    for child in CHILDREN[node]:
        if child not in response:
            continue
        for key, cells in response[child].items():
            for cell in cells:
                # cells are sorted in descending order of timestamps
                if ts >= cell.timestamp:
                    result[key.index].append(cell.value)
                    break
    for layer, edges in result.items():
        result[layer] = np.concatenate(edges)
    return result


def _populate_cx_edges_with_timestamps(
    cg: ChunkedGraph, layer: int, nodes: list, nodes_ts: list, earliest_ts
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
    timestamps_d = get_parent_timestamps(cg, nodes)
    for node, node_ts in zip(nodes, nodes_ts):
        CX_EDGES[node] = {}
        timestamps = timestamps_d[node]
        cx_edges_d_node_ts = _get_cx_edges_at_timestamp(node, response, node_ts)

        edges = np.concatenate([empty_2d] + list(cx_edges_d_node_ts.values()))
        partner_parent_ts_d = get_parent_timestamps(cg, edges[:, 1])
        for v in partner_parent_ts_d.values():
            timestamps.update(v)
        CX_EDGES[node][node_ts] = cx_edges_d_node_ts

        for ts in sorted(timestamps):
            if ts < earliest_ts:
                ts = earliest_ts
            CX_EDGES[node][ts] = _get_cx_edges_at_timestamp(node, response, ts)


def update_cross_edges(cg: ChunkedGraph, layer, node, node_ts, earliest_ts) -> list:
    """
    Helper function to update a single ID.
    Returns a list of mutations with timestamps.
    """
    rows = []
    if node_ts > earliest_ts:
        try:
            cx_edges_d = CX_EDGES[node][node_ts]
        except KeyError:
            raise KeyError(f"{node}:{node_ts}")
        edges = np.concatenate([empty_2d] + list(cx_edges_d.values()))
        if edges.size:
            parents = cg.get_roots(
                edges[:, 0], time_stamp=node_ts, stop_layer=layer, ceil=False
            )
            uparents = np.unique(parents)
            layers = cg.get_chunk_layers(uparents)
            uparents = uparents[layers == layer]
            assert uparents.size <= 1, f"{node}, {node_ts}, {uparents}"
            if uparents.size == 0 or node != uparents[0]:
                # if node is not the parent at this ts, it must be invalid
                assert not exists_as_parent(cg, node, edges[:, 0]), f"{node}, {node_ts}"
                return rows

    for ts, cx_edges_d in CX_EDGES[node].items():
        edges = np.concatenate([empty_2d] + list(cx_edges_d.values()))
        if edges.size == 0:
            continue
        nodes = np.unique(edges[:, 1])
        svs = cg.get_single_leaf_multiple(nodes)
        parents = cg.get_roots(svs, time_stamp=ts, stop_layer=layer, ceil=False)
        edge_parents_d = dict(zip(nodes, parents))
        val_dict = {}
        for _layer, layer_edges in cx_edges_d.items():
            layer_edges = fastremap.remap(
                layer_edges, edge_parents_d, preserve_missing_labels=True
            )
            layer_edges[:, 0] = node
            layer_edges = np.unique(layer_edges, axis=0)
            col = Connectivity.CrossChunkEdge[_layer]
            val_dict[col] = layer_edges
        row_id = serializers.serialize_uint64(node)
        rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=ts))
    return rows


def _update_cross_edges_helper(args):
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
    cg.client.write(rows)


def update_chunk(
    cg: ChunkedGraph, chunk_coords: list[int], layer: int, nodes: list = None
):
    """
    Iterate over all layer IDs in a chunk and update their cross chunk edges.
    """
    start = time.time()
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)
    earliest_ts = cg.get_earliest_timestamp()
    _populate_nodes_and_children(cg, chunk_id, nodes=nodes)
    if not CHILDREN:
        return
    nodes = list(CHILDREN.keys())
    random.shuffle(nodes)
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False, normalize=True)
    _populate_cx_edges_with_timestamps(cg, layer, nodes, nodes_ts, earliest_ts)

    task_size = int(math.ceil(len(nodes) / mp.cpu_count() / 2))
    chunked_nodes = chunked(nodes, task_size)
    chunked_nodes_ts = chunked(nodes_ts, task_size)
    cg_info = cg.get_serialized_info()

    tasks = []
    for chunk, ts_chunk in zip(chunked_nodes, chunked_nodes_ts):
        args = (cg_info, layer, chunk, ts_chunk, earliest_ts)
        tasks.append(args)

    with mp.Pool(min(mp.cpu_count(), len(tasks))) as pool:
        _ = list(
            tqdm(
                pool.imap_unordered(_update_cross_edges_helper, tasks),
                total=len(tasks),
            )
        )
    print(f"total elaspsed time: {time.time() - start}")
