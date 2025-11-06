# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging, time
from copy import copy

import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph, types
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers

from .utils import get_end_timestamps, get_parent_timestamps

CHILDREN = {}


def _get_parents_at_timestamp(nodes, parents_ts_map, time_stamp):
    """
    Search for the first parent with ts <= `time_stamp`.
    `parents_ts_map[node]` is a map of ts:parent with sorted timestamps (desc).
    """
    parents = []
    for node in nodes:
        for ts, parent in parents_ts_map[node].items():
            if time_stamp >= ts:
                parents.append(parent)
                break
    return parents


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
        parents = _get_parents_at_timestamp(partners, parents_ts_map, ts)
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

    rows = []
    skipped = []
    for node, node_ts, end_ts in zip(nodes, nodes_ts, end_timestamps):
        is_stale = end_ts is not None
        _cx_edges_d = cx_edges_d.get(node, {})
        if not _cx_edges_d:
            skipped.append(node)
            continue
        if is_stale:
            end_ts -= timedelta(milliseconds=1)

        _rows = update_cross_edges(
            cg, node, _cx_edges_d, node_ts, end_ts, timestamps_d, parents_ts_map
        )
        if is_stale:
            row_id = serializers.serialize_uint64(node)
            val_dict = {Hierarchy.StaleTimeStamp: 0}
            _rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=end_ts))
        rows.extend(_rows)
    parents = cg.get_roots(skipped)
    layers = cg.get_chunk_layers(parents)
    assert np.all(layers == cg.meta.layer_count)
    return rows


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
            # invalid nodes from failed tasks w/o parent column entry
            continue

    if len(nodes) > 0:
        logging.info(f"processing {len(nodes)} nodes.")
        assert len(CHILDREN) > 0, (nodes, CHILDREN)
    else:
        return

    rows = update_nodes(cg, nodes, nodes_ts)
    cg.client.write(rows)
    logging.info(f"mutations: {len(rows)}, time: {time.time() - start}")
