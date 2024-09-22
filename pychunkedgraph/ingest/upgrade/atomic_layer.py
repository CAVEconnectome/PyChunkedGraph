# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

from datetime import timedelta

import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity
from pychunkedgraph.graph.utils import serializers

from .utils import exists_as_parent, get_parent_timestamps


def update_cross_edges(
    cg: ChunkedGraph, node, cx_edges_d, node_ts, timestamps, earliest_ts
) -> list:
    """
    Helper function to update a single L2 ID.
    Returns a list of mutations with given timestamps.
    """
    rows = []
    edges = np.concatenate(list(cx_edges_d.values()))
    uparents = np.unique(cg.get_parents(edges[:, 0], time_stamp=node_ts))
    assert uparents.size <= 1, f"{node}, {node_ts}, {uparents}"
    if uparents.size == 0 or node != uparents[0]:
        # if node is not the parent at this ts, it must be invalid
        assert not exists_as_parent(cg, node, edges[:, 0])
        return rows

    for ts in timestamps:
        if ts < earliest_ts:
            ts = earliest_ts
        val_dict = {}
        svs = edges[:, 1]
        parents = cg.get_parents(svs, time_stamp=ts)
        edge_parents_d = dict(zip(svs, parents))
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


def update_nodes(cg: ChunkedGraph, nodes) -> list:
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False, normalize=True)
    earliest_ts = cg.get_earliest_timestamp()
    timestamps_d = get_parent_timestamps(cg, nodes)
    cx_edges_d = cg.get_atomic_cross_edges(nodes)
    rows = []
    for node, node_ts in zip(nodes, nodes_ts):
        if cg.get_parent(node) is None:
            # invalid id caused by failed ingest task
            continue
        _cx_edges_d = cx_edges_d.get(node, {})
        if not _cx_edges_d:
            continue
        _rows = update_cross_edges(
            cg, node, _cx_edges_d, node_ts, timestamps_d[node], earliest_ts
        )
        rows.extend(_rows)
    return rows


def update_chunk(cg: ChunkedGraph, chunk_coords: list[int], layer: int = 2):
    """
    Iterate over all L2 IDs in a chunk and update their cross chunk edges,
    within the periods they were valid/active.
    """
    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)
    cg.copy_fake_edges(chunk_id)
    rr = cg.range_read_chunk(chunk_id)
    nodes = list(rr.keys())
    rows = update_nodes(cg, nodes)
    cg.client.write(rows)
