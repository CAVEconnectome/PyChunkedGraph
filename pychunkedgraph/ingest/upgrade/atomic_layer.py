# pylint: disable=invalid-name, missing-docstring, c-extension-no-member
from datetime import timedelta

import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity
from pychunkedgraph.graph.attributes import Hierarchy
from pychunkedgraph.graph.utils import serializers

from .utils import exists_as_parent


def get_parent_timestamps(cg, supervoxels, start_time=None, end_time=None) -> set:
    """
    Timestamps of when the given supervoxels were edited, in the given time range.
    """
    response = cg.client.read_nodes(
        node_ids=supervoxels,
        start_time=start_time,
        end_time=end_time,
        end_time_inclusive=False,
    )
    result = set()
    for v in response.values():
        for cell in v[Hierarchy.Parent]:
            valid = cell.timestamp >= start_time or cell.timestamp < end_time
            assert valid, f"{cell.timestamp}, {start_time}"
            result.add(cell.timestamp)
    return result


def get_edit_timestamps(cg: ChunkedGraph, edges_d, start_ts, end_ts) -> list:
    """
    Timestamps of when post-side supervoxels were involved in an edit.
    Post-side - supervoxels in the neighbor chunk.
    This is required because we need to update edges from both sides.
    """
    atomic_cx_edges = np.concatenate(list(edges_d.values()))
    timestamps = get_parent_timestamps(
        cg, atomic_cx_edges[:, 1], start_time=start_ts, end_time=end_ts
    )
    timestamps.add(start_ts)
    return sorted(timestamps)


def update_cross_edges(cg: ChunkedGraph, node, cx_edges_d, node_ts, end_ts) -> list:
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

    timestamps = [node_ts]
    if node_ts != end_ts:
        timestamps = get_edit_timestamps(cg, cx_edges_d, node_ts, end_ts)
    for ts in timestamps:
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
    # get start_ts when node becomes valid
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False, normalize=True)
    cx_edges_d = cg.get_atomic_cross_edges(nodes)
    children_d = cg.get_children(nodes)

    rows = []
    for node, start_ts in zip(nodes, nodes_ts):
        if cg.get_parent(node) is None:
            # invalid id caused by failed ingest task
            continue
        node_cx_edges_d = cx_edges_d.get(node, {})
        if not node_cx_edges_d:
            continue

        # get end_ts when node becomes invalid (bigtable resolution is in ms)
        start = start_ts + timedelta(milliseconds=1)
        _timestamps = get_parent_timestamps(cg, children_d[node], start_time=start)
        try:
            end_ts = sorted(_timestamps)[0]
        except IndexError:
            # start_ts == end_ts means there has been no edit involving this node
            # meaning only one timestamp to update cross edges, start_ts
            end_ts = start_ts
        # for each timestamp until end_ts, update cross chunk edges of node
        _rows = update_cross_edges(cg, node, node_cx_edges_d, start_ts, end_ts)
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
