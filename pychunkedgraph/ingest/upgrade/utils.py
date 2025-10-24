# pylint: disable=invalid-name, missing-docstring

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Hierarchy


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
        _timestamps = set().union(*[timestamps_d[k] for k in node_children])
        _timestamps.add(node_ts)
        try:
            _timestamps = sorted(_timestamps)
            _index = np.searchsorted(_timestamps, node_ts)
            assert _timestamps[_index] == node_ts, (_index, node_ts, _timestamps)
            end_ts = _timestamps[_index + 1]
        except IndexError:
            # this node has not been edited, but might have it edges updated
            end_ts = datetime.now(timezone.utc)
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
