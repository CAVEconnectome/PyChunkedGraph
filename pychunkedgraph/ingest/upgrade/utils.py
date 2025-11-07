# pylint: disable=invalid-name, missing-docstring

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Hierarchy
from pychunkedgraph.graph.utils import serializers
from google.cloud.bigtable.row_filters import TimestampRange


def exists_as_parent(cg: ChunkedGraph, parent, nodes) -> bool:
    """
    Check if a given l2 parent is in the history of given nodes.
    """
    response = cg.client.read_nodes(node_ids=nodes, properties=Hierarchy.Parent)
    parents = set()
    for cells in response.values():
        parents.update([cell.value for cell in cells])
    return parent in parents


def get_edit_timestamps(cg: ChunkedGraph, edges_d, start_ts, end_ts) -> list:
    """
    Timestamps of when post-side nodes were involved in an edit.
    Post-side - nodes in the neighbor chunk.
    This is required because we need to update edges from both sides.
    """
    cx_edges = np.concatenate(list(edges_d.values()))
    timestamps = get_parent_timestamps(
        cg, cx_edges[:, 1], start_time=start_ts, end_time=end_ts
    )
    timestamps.add(start_ts)
    return sorted(timestamps)


def _get_end_timestamps_helper(cg: ChunkedGraph, nodes: list) -> defaultdict[int, set]:
    result = defaultdict(set)
    response = cg.client.read_nodes(node_ids=nodes, properties=Hierarchy.StaleTimeStamp)
    for k, v in response.items():
        result[k].add(v[0].timestamp)
    return result


def get_end_timestamps(
    cg: ChunkedGraph, nodes: list, nodes_ts: datetime, children_map: dict, layer: int
):
    """
    Gets the last timestamp for each node at which to update its cross edges.
    For layer 2:
        Get parent timestamps for all children of a node.
        The first timestamp > node_timestamp among these is the last timestamp.
            This is the timestamp at which one of node's children
            got a new parent that superseded the current node.
        These are cached in database.
    For all nodes in each layer > 2:
        Pick the earliest child node_end_ts > node_ts and cache in database.
    """
    result = []
    children = np.concatenate([*children_map.values()])
    if layer == 2:
        timestamps_d = get_parent_timestamps(cg, children)
    else:
        timestamps_d = _get_end_timestamps_helper(cg, children)

    for node, node_ts in zip(nodes, nodes_ts):
        node_children = children_map[node]
        _children_timestamps = []
        for k in node_children:
            if k in timestamps_d:
                _children_timestamps.append(timestamps_d[k])
        _timestamps = set().union(*_children_timestamps)
        _timestamps.add(node_ts)
        try:
            _timestamps = sorted(_timestamps)
            _index = np.searchsorted(_timestamps, node_ts)
            end_ts = _timestamps[_index + 1]
        except IndexError:
            # this node has not been edited, but might have it edges updated
            end_ts = None
        result.append(end_ts)
    return result


def get_parent_timestamps(
    cg: ChunkedGraph, nodes, start_time=None, end_time=None
) -> defaultdict[int, set]:
    """
    Timestamps of when the given nodes were edited.
    """
    earliest_ts = cg.get_earliest_timestamp()
    response = cg.client.read_nodes(
        node_ids=nodes,
        properties=[Hierarchy.Parent],
        start_time=start_time,
        end_time=end_time,
        end_time_inclusive=False,
    )

    result = defaultdict(set)
    for k, v in response.items():
        for cell in v[Hierarchy.Parent]:
            ts = cell.timestamp
            result[k].add(earliest_ts if ts < earliest_ts else ts)
    return result


def _fix_corrupt_node(cg: ChunkedGraph, node: int, children: np.ndarray):
    """
    Removes this node from parent column of its children.
    Then removes the node iteself, effectively erasing it.
    """
    table = cg.client._table
    batcher = table.mutations_batcher(flush_count=500)
    children_d = cg.client.read_nodes(node_ids=children, properties=Hierarchy.Parent)
    for child, parent_cells in children_d.items():
        row = table.direct_row(serializers.serialize_uint64(child))
        for cell in parent_cells:
            if cell.value == node:
                start = cell.timestamp
                end = start + timedelta(microseconds=1)
                row.delete_cell(
                    column_family_id=Hierarchy.Parent.family_id,
                    column=Hierarchy.Parent.key,
                    time_range=TimestampRange(start=start, end=end),
                )
                batcher.mutate(row)

    row = table.direct_row(serializers.serialize_uint64(node))
    row.delete()
    batcher.mutate(row)
    batcher.flush()


def fix_corrupt_nodes(cg: ChunkedGraph, nodes: list, nodes_ts: list, children_d: dict):
    _children_d = {node: children_d[node] for node in nodes}
    end_timestamps = get_end_timestamps(cg, nodes, nodes_ts, _children_d, layer=2)
    for node, end_ts in zip(nodes, end_timestamps):
        assert end_ts is None, f"{node}: {end_ts}"
        _fix_corrupt_node(cg, node, _children_d[node])
