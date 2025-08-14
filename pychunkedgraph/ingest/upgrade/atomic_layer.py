# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity, Hierarchy
from pychunkedgraph.graph.utils import serializers

from .utils import exists_as_parent, get_end_timestamps, get_parent_timestamps

CHILDREN = {}


def update_cross_edges(
    cg: ChunkedGraph, node, cx_edges_d: dict, node_ts, node_end_ts, timestamps: set
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

    partner_parent_ts_d = get_parent_timestamps(cg, np.unique(edges[:, 1]))
    for v in partner_parent_ts_d.values():
        timestamps.update(v)

    for ts in sorted(timestamps):
        if ts < node_ts:
            continue
        if ts > node_end_ts:
            break
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


def update_nodes(cg: ChunkedGraph, nodes, nodes_ts, children_map=None) -> list:
    if children_map is None:
        children_map = CHILDREN
    end_timestamps = get_end_timestamps(cg, nodes, nodes_ts, children_map)
    timestamps_d = get_parent_timestamps(cg, nodes)
    cx_edges_d = cg.get_atomic_cross_edges(nodes)
    rows = []
    for node, node_ts, end_ts in zip(nodes, nodes_ts, end_timestamps):
        if cg.get_parent(node) is None:
            # invalid id caused by failed ingest task / edits
            continue
        _cx_edges_d = cx_edges_d.get(node, {})
        if not _cx_edges_d:
            continue
        _rows = update_cross_edges(
            cg, node, _cx_edges_d, node_ts, end_ts, timestamps_d[node]
        )
        rows.extend(_rows)
    return rows


def update_chunk(cg: ChunkedGraph, chunk_coords: list[int], layer: int = 2):
    """
    Iterate over all L2 IDs in a chunk and update their cross chunk edges,
    within the periods they were valid/active.
    """
    global CHILDREN

    x, y, z = chunk_coords
    chunk_id = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)
    cg.copy_fake_edges(chunk_id)
    rr = cg.range_read_chunk(chunk_id)

    nodes = []
    nodes_ts = []
    earliest_ts = cg.get_earliest_timestamp()
    for k, v in rr.items():
        nodes.append(k)
        CHILDREN[k] = v[Hierarchy.Child][0].value
        ts = v[Hierarchy.Child][0].timestamp
        nodes_ts.append(earliest_ts if ts < earliest_ts else ts)

    if len(nodes):
        assert len(CHILDREN) > 0, (nodes, CHILDREN)
    else:
        return

    rows = update_nodes(cg, nodes, nodes_ts)
    cg.client.write(rows)
