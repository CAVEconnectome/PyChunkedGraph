"""
Functions for tracking root ID changes over time.
"""
from typing import Optional

import numpy as np
from networkx import DiGraph


def lineage_graph(
    cg,
    node_id: np.uint64,
    timestamp_past: Optional[float] = None,
    timestamp_future: Optional[float] = None,
) -> DiGraph:
    """
    Build lineage graph of a given root ID
    going backwards in time until `timestamp_past`
    and in future until `timestamp_future`
    """
    from time import time
    from .utils.basetypes import NODE_ID
    from .utils.column_keys import Hierarchy
    from .utils.column_keys import OperationLogs

    graph = DiGraph()
    past_ids = np.array([node_id], dtype=np.uint64)
    future_ids = np.array([node_id], dtype=np.uint64)
    if timestamp_past is None:
        timestamp_past = float(0)
    if timestamp_future is None:
        timestamp_future = float(time())

    while past_ids.size or future_ids.size:
        nodes_raw = cg.read_node_id_rows(
            node_ids=np.concatenate([past_ids, future_ids])
        )
        next_past_ids = []
        for k in past_ids:
            val = nodes_raw[k]
            timestamp = val[Hierarchy.Child][0].timestamp.timestamp()
            try:
                operation_id = val[OperationLogs.OperationID][0].value
            except KeyError:
                # if no operation ID, the segment has no edits
                # return single node graph with given ID
                graph.add_node(k, timestamp=timestamp)
                return graph

            graph.add_node(k, operation_id=operation_id, timestamp=timestamp)
            if timestamp < timestamp_past or not Hierarchy.FormerParent in val:
                continue

            former_ids = val[Hierarchy.FormerParent][0].value
            for former in former_ids:
                graph.add_edge(former, k)
            next_past_ids.extend(former_ids)

        next_future_ids = []
        for k in future_ids:
            val = nodes_raw[k]
            operation_id = val[OperationLogs.OperationID][0].value
            timestamp = val[Hierarchy.Child][0].timestamp.timestamp()
            graph.add_node(k, operation_id=operation_id, timestamp=timestamp)
            if timestamp > timestamp_future or not Hierarchy.NewParent in val:
                continue

            new_ids = val[Hierarchy.NewParent][0].value
            for new_id in new_ids:
                graph.add_edge(k, new_id)
            next_future_ids.extend(new_ids)

        past_ids = np.array(next_past_ids, dtype=NODE_ID)
        future_ids = np.array(next_future_ids, dtype=NODE_ID)
    return graph