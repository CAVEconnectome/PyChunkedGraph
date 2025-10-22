# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging, math, time
from copy import copy

import fastremap
import numpy as np
from tqdm import tqdm
from pychunkedgraph.graph import ChunkedGraph, types
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.utils.general import chunked

from .utils import get_end_timestamps, get_parent_timestamps

CHILDREN = {}


def update_cross_edges(
    cg: ChunkedGraph,
    node,
    cx_edges_d: dict,
    node_ts,
    node_end_ts,
    timestamps_d: defaultdict[int, set],
) -> list:
    """
    Helper function to update a single L2 ID.
    Returns a list of mutations with given timestamps.
    """
    rows = []
    edges = np.concatenate(list(cx_edges_d.values()))
    partners = np.unique(edges[:, 1])

    timestamps = copy(timestamps_d[node])
    for partner in partners:
        timestamps.update(timestamps_d[partner])

    for ts in sorted(timestamps):
        if ts < node_ts:
            continue
        if ts > node_end_ts:
            break
        val_dict = {}

        parents = cg.get_parents(partners, time_stamp=ts)
        edge_parents_d = dict(zip(partners, parents))
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


def update_nodes(cg: ChunkedGraph, nodes, nodes_ts, children_map=None) -> list:
    if children_map is None:
        children_map = CHILDREN
    end_timestamps = get_end_timestamps(cg, nodes, nodes_ts, children_map)

    cx_edges_d = cg.get_atomic_cross_edges(nodes)
    all_cx_edges = [types.empty_2d]
    for _cx_edges_d in cx_edges_d.values():
        if _cx_edges_d:
            all_cx_edges.append(np.concatenate(list(_cx_edges_d.values())))
    all_partners = np.unique(np.concatenate(all_cx_edges)[:, 1])
    timestamps_d = get_parent_timestamps(cg, np.concatenate([nodes, all_partners]))

    rows = []
    for node, node_ts, end_ts in zip(nodes, nodes_ts, end_timestamps):
        _cx_edges_d = cx_edges_d.get(node, {})
        if not _cx_edges_d:
            continue
        _rows = update_cross_edges(cg, node, _cx_edges_d, node_ts, end_ts, timestamps_d)
        rows.extend(_rows)
    return rows


def _update_nodes_helper(args):
    cg, nodes, nodes_ts = args
    return update_nodes(cg, nodes, nodes_ts)


def update_chunk(cg: ChunkedGraph, chunk_coords: list[int], debug: bool = False):
    """
    Iterate over all L2 IDs in a chunk and update their cross chunk edges,
    within the periods they were valid/active.
    """
    global CHILDREN

    start = time.time()
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=2, x=x, y=y, z=z)
    cg.copy_fake_edges(chunk_id)
    rr = cg.range_read_chunk(chunk_id)

    nodes = []
    nodes_ts = []
    earliest_ts = cg.get_earliest_timestamp()
    for k, v in rr.items():
        try:
            _ = v[Hierarchy.Parent]
            nodes.append(k)
            CHILDREN[k] = v[Hierarchy.Child][0].value
            ts = v[Hierarchy.Child][0].timestamp
            nodes_ts.append(earliest_ts if ts < earliest_ts else ts)
        except KeyError:
            continue

    if len(nodes) > 0:
        logging.info(f"processing {len(nodes)} nodes.")
        assert len(CHILDREN) > 0, (nodes, CHILDREN)
    else:
        return

    if debug:
        rows = update_nodes(cg, nodes, nodes_ts)
    else:
        task_size = int(math.ceil(len(nodes) / 16))
        chunked_nodes = chunked(nodes, task_size)
        chunked_nodes_ts = chunked(nodes_ts, task_size)
        tasks = []
        for chunk, ts_chunk in zip(chunked_nodes, chunked_nodes_ts):
            args = (cg, chunk, ts_chunk)
            tasks.append(args)
        logging.info(f"task size {task_size}, count {len(tasks)}.")

        rows = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_update_nodes_helper, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                rows.extend(future.result())

    cg.client.write(rows)
    logging.info(f"total elaspsed time: {time.time() - start}")
