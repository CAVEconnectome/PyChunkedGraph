# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import logging, math, random, time
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import fastremap
import numpy as np
from tqdm import tqdm

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.edges import get_latest_edges_wrapper
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.graph.types import empty_2d
from pychunkedgraph.utils.general import chunked

from .utils import fix_corrupt_nodes, get_end_timestamps, get_parent_timestamps


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
    cg: ChunkedGraph, layer: int, nodes: list, nodes_ts: list
):
    """
    Collect timestamps of edits from children, since we use the same timestamp
    for all IDs involved in an edit, we can use the timestamps of
    when cross edges of children were updated.
    """

    start = time.time()
    global CX_EDGES
    attrs = [Connectivity.CrossChunkEdge[l] for l in range(layer, cg.meta.layer_count)]
    all_children = np.concatenate(list(CHILDREN.values()))
    response = cg.client.read_nodes(node_ids=all_children, properties=attrs)
    timestamps_d = get_parent_timestamps(cg, nodes)
    end_timestamps = get_end_timestamps(cg, nodes, nodes_ts, CHILDREN, layer=layer)
    logging.info(f"_populate_nodes_and_children init: {time.time() - start}")

    start = time.time()
    partners_map = {}
    for node, node_ts in zip(nodes, nodes_ts):
        CX_EDGES[node] = {}
        cx_edges_d_node_ts = _get_cx_edges_at_timestamp(node, response, node_ts)
        edges = np.concatenate([empty_2d] + list(cx_edges_d_node_ts.values()))
        partners_map[node] = edges[:, 1]
        CX_EDGES[node][node_ts] = cx_edges_d_node_ts

    partners = np.unique(np.concatenate([*partners_map.values()]))
    partner_parent_ts_d = get_parent_timestamps(cg, partners)
    logging.info(f"get partners timestamps init: {time.time() - start}")

    rows = []
    for node, node_ts, node_end_ts in zip(nodes, nodes_ts, end_timestamps):
        timestamps = timestamps_d[node]
        for partner in partners_map[node]:
            timestamps.update(partner_parent_ts_d[partner])

        is_stale = node_end_ts is not None
        node_end_ts = node_end_ts or datetime.now(timezone.utc)
        for ts in sorted(timestamps):
            if ts > node_end_ts:
                break
            CX_EDGES[node][ts] = _get_cx_edges_at_timestamp(node, response, ts)

        if is_stale:
            row_id = serializers.serialize_uint64(node)
            val_dict = {Hierarchy.StaleTimeStamp: 0}
            rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=node_end_ts))

    cg.client.write(rows)


def update_cross_edges(cg: ChunkedGraph, layer, node, node_ts) -> list:
    """
    Helper function to update a single ID.
    Returns a list of mutations with timestamps.
    """
    rows = []
    row_id = serializers.serialize_uint64(node)
    for ts, cx_edges_d in CX_EDGES[node].items():
        if ts < node_ts:
            continue
        cx_edges_d, edge_nodes = get_latest_edges_wrapper(cg, cx_edges_d, parent_ts=ts)
        if edge_nodes.size == 0:
            continue

        parents = cg.get_roots(edge_nodes, time_stamp=ts, stop_layer=layer, ceil=False)
        edge_parents_d = dict(zip(edge_nodes, parents))
        val_dict = {}
        for _layer, layer_edges in cx_edges_d.items():
            layer_edges = fastremap.remap(
                layer_edges, edge_parents_d, preserve_missing_labels=True
            )
            layer_edges[:, 0] = node
            layer_edges = np.unique(layer_edges, axis=0)
            col = Connectivity.CrossChunkEdge[_layer]
            val_dict[col] = layer_edges
        rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=ts))
    return rows


def _update_cross_edges_helper_thread(args):
    cg, layer, node, node_ts = args
    return update_cross_edges(cg, layer, node, node_ts)


def _update_cross_edges_helper(args):
    cg_info, layer, nodes, nodes_ts = args
    rows = []
    cg = ChunkedGraph(**cg_info)
    parents = cg.get_parents(nodes, fail_to_zero=True)

    tasks = []
    corrupt_nodes = []
    earliest_ts = cg.get_earliest_timestamp()
    for node, parent, node_ts in zip(nodes, parents, nodes_ts):
        if parent == 0:
            # ignore invalid nodes from failed ingest tasks, w/o parent column entry
            # retain invalid nodes from edits to fix the hierarchy
            if node_ts > earliest_ts:
                corrupt_nodes.append(node)
        else:
            tasks.append((cg, layer, node, node_ts))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_update_cross_edges_helper_thread, task) for task in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            rows.extend(future.result())

    cg.client.write(rows)
    if len(corrupt_nodes) > 0:
        logging.info(f"found {len(corrupt_nodes)} corrupt nodes {corrupt_nodes[:3]}...")
        fix_corrupt_nodes(cg, corrupt_nodes, CHILDREN)


def update_chunk(
    cg: ChunkedGraph, chunk_coords: list[int], layer: int, nodes: list = None
):
    """
    Iterate over all layer IDs in a chunk and update their cross chunk edges.
    """
    debug = nodes is not None
    start = time.time()
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)

    _populate_nodes_and_children(cg, chunk_id, nodes=nodes)
    logging.info(f"_populate_nodes_and_children: {time.time() - start}")
    if not CHILDREN:
        return
    nodes = list(CHILDREN.keys())
    random.shuffle(nodes)

    start = time.time()
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False, normalize=True)
    logging.info(f"get_node_timestamps: {time.time() - start}")

    start = time.time()
    _populate_cx_edges_with_timestamps(cg, layer, nodes, nodes_ts)
    logging.info(f"_populate_cx_edges_with_timestamps: {time.time() - start}")

    if debug:
        rows = []
        for node, node_ts in zip(nodes, nodes_ts):
            rows.extend(update_cross_edges(cg, layer, node, node_ts))
        logging.info(f"total elaspsed time: {time.time() - start}")
        return

    task_size = int(math.ceil(len(nodes) / mp.cpu_count() / 2))
    chunked_nodes = chunked(nodes, task_size)
    chunked_nodes_ts = chunked(nodes_ts, task_size)
    cg_info = cg.get_serialized_info()

    tasks = []
    for chunk, ts_chunk in zip(chunked_nodes, chunked_nodes_ts):
        args = (cg_info, layer, chunk, ts_chunk)
        tasks.append(args)

    processes = min(mp.cpu_count() * 2, len(tasks))
    logging.info(f"processing {len(nodes)} nodes with {processes} workers.")
    with mp.Pool(processes) as pool:
        _ = list(
            tqdm(
                pool.imap_unordered(_update_cross_edges_helper, tasks),
                total=len(tasks),
            )
        )
    logging.info(f"total elaspsed time: {time.time() - start}")
