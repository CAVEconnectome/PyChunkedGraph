# pylint: disable=invalid-name, missing-docstring, c-extension-no-member
from datetime import timedelta

import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity
from pychunkedgraph.graph.attributes import Hierarchy
from pychunkedgraph.graph.utils import serializers


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


def get_edit_timestamps(cg: ChunkedGraph, node, edges_d, start_ts, end_ts) -> list:
    """
    Timestamps of when post-side supervoxels were involved in an edit.
    Post-side - supervoxels in the neighbor chunk.
    This is required because we need to update edges from both sides.
    """
    atomic_cx_edges = np.concatenate(list(edges_d.values()))
    assert node == np.unique(cg.get_parents(atomic_cx_edges[:, 0], time_stamp=start_ts))
    timestamps = get_parent_timestamps(
        cg, atomic_cx_edges[:, 1], start_time=start_ts, end_time=end_ts
    )
    timestamps.add(start_ts)
    return sorted(timestamps)


def update_cross_edges(cg: ChunkedGraph, node, cx_edges_d, node_ts, timestamps) -> list:
    """
    Helper function to update a single L2 ID.
    Returns a list of mutations with given timestamps.
    """
    rows = []
    for ts in timestamps:
        edges = np.concatenate(list(cx_edges_d.values()))
        assert node == np.unique(cg.get_parents(edges[:, 0], time_stamp=node_ts))

        val_dict = {}
        nodes = edges[:, 1]
        parents = cg.get_parents(nodes, time_stamp=ts)
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

    # get start_ts when node becomes valid
    nodes_ts = cg.get_node_timestamps(nodes, return_numpy=False)
    cx_edges_d = cg.get_atomic_cross_edges(nodes)
    children_d = cg.get_children(nodes)

    rows = []
    for node, start_ts in zip(nodes, nodes_ts):
        node_cx_edges_d = cx_edges_d.get(node, {})
        if not node_cx_edges_d:
            continue

        # get end_ts when node becomes invalid (bigtable resolution is in ms)
        start = start_ts + timedelta(microseconds=(start_ts.microsecond % 1000) + 1000)
        _timestamps = get_parent_timestamps(cg, children_d[node], start_time=start)
        try:
            end_ts = sorted(_timestamps)[0]
        except IndexError:
            # start_ts == end_ts means there has been no edit involving this node
            # meaning only one timestamp to update cross edges, start_ts
            _rows = update_cross_edges(cg, node, node_cx_edges_d, start_ts, [start_ts])
            rows.extend(_rows)
            continue

        # for each timestamp until end_ts, update cross chunk edges of node
        valid_times = get_edit_timestamps(cg, node, node_cx_edges_d, start_ts, end_ts)
        _rows = update_cross_edges(cg, node, node_cx_edges_d, start_ts, valid_times)
        rows.extend(_rows)
    cg.client.write(rows)
