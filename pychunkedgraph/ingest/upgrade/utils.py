# pylint: disable=invalid-name, missing-docstring

from collections import defaultdict
from datetime import timedelta

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


def get_end_ts(cg: ChunkedGraph, children, start_ts):
    # get end_ts when node becomes invalid (bigtable resolution is in ms)
    start = start_ts + timedelta(milliseconds=1)
    _timestamps = get_parent_timestamps(cg, children, start_time=start)
    try:
        end_ts = sorted(_timestamps)[0]
    except IndexError:
        # start_ts == end_ts means there has been no edit involving this node
        # meaning only one timestamp to update cross edges, start_ts
        end_ts = start_ts
    return end_ts


def get_parent_timestamps(cg: ChunkedGraph, nodes) -> dict[int, set]:
    """
    Timestamps of when the given nodes were edited.
    """
    response = cg.client.read_nodes(
        node_ids=nodes,
        properties=[Hierarchy.Parent],
        end_time_inclusive=False,
    )

    result = defaultdict(set)
    for k, v in response.items():
        for cell in v[Hierarchy.Parent]:
            result[k].add(cell.timestamp)
    return result
