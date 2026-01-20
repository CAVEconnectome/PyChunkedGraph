# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging, time, os
from copy import copy

import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph, types
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.graph.utils.generic import get_parents_at_timestamp

from .utils import fix_corrupt_nodes, get_end_timestamps, get_parent_timestamps

CHILDREN = {}


def update_cross_edges(
    cg: ChunkedGraph,
    node,
    cx_edges_d: dict,
    node_ts,
    node_end_ts,
    timestamps_map: defaultdict[int, set],
    parents_ts_map: defaultdict[int, dict],
) -> list:
    """
    Helper function to update a single L2 ID.
    Returns a list of mutations with given timestamps.
    """
    rows = []
    edges = np.concatenate(list(cx_edges_d.values()))
    partners = np.unique(edges[:, 1])

    timestamps = copy(timestamps_map[node])
    for partner in partners:
        timestamps.update(timestamps_map[partner])

    node_end_ts = node_end_ts or datetime.now(timezone.utc)
    for ts in sorted(timestamps):
        if ts < node_ts:
            continue
        if ts > node_end_ts:
            break

        val_dict = {}
        parents, _ = get_parents_at_timestamp(partners, parents_ts_map, ts)
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
    start = time.time()
    if children_map is None:
        children_map = CHILDREN
    end_timestamps = get_end_timestamps(cg, nodes, nodes_ts, children_map, layer=2)

    cx_edges_d = cg.get_atomic_cross_edges(nodes)
    all_cx_edges = [types.empty_2d]
    for _cx_edges_d in cx_edges_d.values():
        if _cx_edges_d:
            all_cx_edges.append(np.concatenate(list(_cx_edges_d.values())))
    all_partners = np.unique(np.concatenate(all_cx_edges)[:, 1])
    timestamps_d = get_parent_timestamps(cg, np.concatenate([nodes, all_partners]))

    parents_ts_map = defaultdict(dict)
    all_parents = cg.get_parents(all_partners, current=False)
    for partner, parents in zip(all_partners, all_parents):
        for parent, ts in parents:
            parents_ts_map[partner][ts] = parent
    logging.info(f"update_nodes init {len(nodes)}: {time.time() - start}")

    rows = []
    skipped = []
    for node, node_ts, end_ts in zip(nodes, nodes_ts, end_timestamps):
        is_stale = end_ts is not None
        _cx_edges_d = cx_edges_d.get(node, {})
        if is_stale:
            end_ts -= timedelta(milliseconds=1)
            row_id = serializers.serialize_uint64(node)
            val_dict = {Hierarchy.StaleTimeStamp: 0}
            rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=end_ts))

        if not _cx_edges_d:
            skipped.append(node)
            continue

        _rows = update_cross_edges(
            cg, node, _cx_edges_d, node_ts, end_ts, timestamps_d, parents_ts_map
        )
        rows.extend(_rows)
    parents = cg.get_roots(skipped)
    layers = cg.get_chunk_layers(parents)
    assert np.all(layers == cg.meta.layer_count)
    return rows


def update_chunk(cg: ChunkedGraph, chunk_coords: list[int]):
    """
    Iterate over all L2 IDs in a chunk and update their cross chunk edges,
    within the periods they were valid/active.
    """
    global CHILDREN

    start = time.time()
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=2, x=x, y=y, z=z)
    rr = cg.range_read_chunk(chunk_id)

    nodes = []
    nodes_ts = []
    try:
        earliest_ts = os.environ["EARLIEST_TS"]
        earliest_ts = datetime.fromisoformat(earliest_ts)
    except KeyError:
        earliest_ts = cg.get_earliest_timestamp()

    corrupt_nodes = []
    for k, v in rr.items():
        try:
            CHILDREN[k] = v[Hierarchy.Child][0].value
            ts = v[Hierarchy.Child][0].timestamp
            _ = v[Hierarchy.Parent]
            nodes.append(k)
            nodes_ts.append(earliest_ts if ts < earliest_ts else ts)
        except KeyError:
            # ignore invalid nodes from failed ingest tasks, w/o parent column entry
            # retain invalid nodes from edits to fix the hierarchy
            if ts > earliest_ts:
                corrupt_nodes.append(k)

    clean_task = os.environ.get("CLEAN_CHUNKS", "false") == "clean"
    if clean_task:
        logging.info(f"found {len(corrupt_nodes)} corrupt nodes {corrupt_nodes[:3]}...")
        fix_corrupt_nodes(cg, corrupt_nodes, CHILDREN)
        return

    cg.copy_fake_edges(chunk_id)
    if len(nodes) == 0:
        return

    logging.info(f"processing {len(nodes)} nodes.")
    assert len(CHILDREN) > 0, (nodes, CHILDREN)
    rows = update_nodes(cg, nodes, nodes_ts)
    cg.client.write(rows)
    logging.info(f"mutations: {len(rows)}, time: {time.time() - start}")
