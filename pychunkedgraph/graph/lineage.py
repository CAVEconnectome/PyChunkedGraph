"""
Functions for tracking root ID changes over time.
"""
from typing import Optional
from typing import Iterable
from datetime import datetime

import numpy as np
from networkx import DiGraph

from . import attributes
from .exceptions import ChunkedGraphError
from .utils.generic import get_min_time
from .utils.generic import get_max_time
from .utils.generic import get_valid_timestamp


def get_latest_root_id(cg, root_id: np.uint64) -> np.ndarray:
    """ Returns the latest root id associated with the provided root id"""
    id_working_set = [root_id]
    latest_root_ids = []
    while len(id_working_set) > 0:
        next_id = id_working_set[0]
        del id_working_set[0]
        node = cg.client.read_node(next_id, properties=attributes.Hierarchy.NewParent)
        # Check if a new root id was attached to this root id
        if node:
            id_working_set.extend(node[0].value)
        else:
            latest_root_ids.append(next_id)
    return np.unique(latest_root_ids)


def get_future_root_ids(
    cg,
    root_id: np.uint64,
    time_stamp: Optional[datetime] = get_max_time(),
) -> np.ndarray:
    """
    Returns all future root ids emerging from this root
    This search happens in a monotic fashion. At no point are past root
    ids of future root ids taken into account.
    """
    id_history = []
    next_ids = [root_id]
    while len(next_ids):
        temp_next_ids = []
        for next_id in next_ids:
            node = cg.client.read_node(
                next_id,
                properties=[attributes.Hierarchy.NewParent, attributes.Hierarchy.Child],
            )
            if attributes.Hierarchy.NewParent in node:
                ids = node[attributes.Hierarchy.NewParent][0].value
                row_time_stamp = node[attributes.Hierarchy.NewParent][0].timestamp
            elif attributes.Hierarchy.Child in node:
                ids = None
                row_time_stamp = node[attributes.Hierarchy.Child][0].timestamp
            else:
                raise ChunkedGraphError(f"Error retrieving future root ID of {next_id}")
            if row_time_stamp < get_valid_timestamp(time_stamp):
                if ids is not None:
                    temp_next_ids.extend(ids)
                if next_id != root_id:
                    id_history.append(next_id)
        next_ids = temp_next_ids
    return np.unique(np.array(id_history, dtype=np.uint64))


def get_past_root_ids(
    cg,
    root_id: np.uint64,
    time_stamp: Optional[datetime] = get_min_time(),
) -> np.ndarray:
    """
    Returns all past root ids emerging from this root.
    This search happens in a monotic fashion. At no point are future root
    ids of past root ids taken into account.
    """
    id_history = []
    next_ids = [root_id]
    while len(next_ids):
        temp_next_ids = []
        for next_id in next_ids:
            node = cg.client.read_node(
                next_id,
                properties=[
                    attributes.Hierarchy.FormerParent,
                    attributes.Hierarchy.Child,
                ],
            )
            if attributes.Hierarchy.FormerParent in node:
                ids = node[attributes.Hierarchy.FormerParent][0].value
                row_time_stamp = node[attributes.Hierarchy.FormerParent][0].timestamp
            elif attributes.Hierarchy.Child in node:
                ids = None
                row_time_stamp = node[attributes.Hierarchy.Child][0].timestamp
            else:
                raise ChunkedGraphError(f"Error retrieving past root ID of {next_id}.")
            if row_time_stamp > get_valid_timestamp(time_stamp):
                if ids is not None:
                    temp_next_ids.extend(ids)
                if next_id != root_id:
                    id_history.append(next_id)
        next_ids = temp_next_ids
    return np.unique(np.array(id_history, dtype=np.uint64))


def get_previous_root_ids(
    cg,
    root_ids: Iterable[np.uint64],
) -> dict:
    """Returns immediate former root IDs (1 step history)"""
    nodes_d = cg.client.read_nodes(
        node_ids=root_ids,
        properties=attributes.Hierarchy.FormerParent,
    )
    result = {}
    for root, val in nodes_d.items():
        result[root] = val[0].value
    return result


def get_root_id_history(
    cg,
    root_id: np.uint64,
    time_stamp_past: Optional[datetime] = get_min_time(),
    time_stamp_future: Optional[datetime] = get_max_time(),
) -> np.ndarray:
    """
    Returns all future root ids emerging from this root
    This search happens in a monotic fashion. At no point are future root
    ids of past root ids or past root ids of future root ids taken into
    account.
    """
    past_ids = get_past_root_ids(cg, root_id, time_stamp=time_stamp_past)
    future_ids = get_future_root_ids(cg, root_id, time_stamp=time_stamp_future)
    return np.concatenate([past_ids, np.array([root_id], dtype=np.uint64), future_ids])


def lineage_graph(
    cg,
    node_id: np.uint64,
    timestamp_past: float = None,
    timestamp_future: float = None,
) -> DiGraph:
    """
    Build lineage graph of a given root ID
    going backwards in time until `timestamp_past`
    and in future until `timestamp_future`
    """
    from time import time
    from .attributes import Hierarchy
    from .attributes import OperationLogs

    graph = DiGraph()
    past_ids = np.array([node_id], dtype=np.uint64)
    future_ids = np.array([node_id], dtype=np.uint64)
    if timestamp_past is None:
        timestamp_past = float(0)
    if timestamp_future is None:
        timestamp_future = float(time())

    while past_ids.size or future_ids.size:
        nodes_raw = cg.client.read_nodes(
            node_ids=np.concatenate([past_ids, future_ids])
        )

        next_past_ids = [np.empty(0, dtype=np.uint64)]
        for k in past_ids:
            val = nodes_raw[k]
            if OperationLogs.OperationID in val:
                operation_id = val[OperationLogs.OperationID][0].value
            else:
                operation_id = 0

            timestamp = val[Hierarchy.Child][0].timestamp.timestamp()
            graph.add_node(k, operation_id=operation_id, timestamp=timestamp)
            if timestamp < timestamp_past or not Hierarchy.FormerParent in val:
                continue

            former_ids = val[Hierarchy.FormerParent][0].value
            for former in former_ids:
                graph.add_edge(former, k)
            next_past_ids.append(former_ids)

        next_future_ids = [np.empty(0, dtype=np.uint64)]
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
            next_future_ids.append(new_ids)

        past_ids = np.concatenate(next_past_ids)
        future_ids = np.concatenate(next_future_ids)
    return graph