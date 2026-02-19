"""
Functions for tracking root ID changes over time.
"""
from typing import Union
from typing import Optional
from typing import Iterable
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from networkx import DiGraph

from . import attributes
from .exceptions import ChunkedGraphError
from .attributes import Hierarchy
from .attributes import OperationLogs
from .utils.basetypes import NODE_ID
from .utils.generic import get_min_time
from .utils.generic import get_max_time
from .utils.generic import get_valid_timestamp


def get_latest_root_id(cg, root_id: NODE_ID.type) -> np.ndarray:
    """Returns the latest root id associated with the provided root id"""
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
    root_id: NODE_ID,
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
    return np.unique(np.array(id_history, dtype=NODE_ID))


def get_past_root_ids(
    cg,
    root_id: NODE_ID,
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
    return np.unique(np.array(id_history, dtype=NODE_ID))


def get_previous_root_ids(
    cg,
    root_ids: Iterable[NODE_ID.type],
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
    root_id: NODE_ID,
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
    return np.concatenate([past_ids, np.array([root_id], dtype=NODE_ID), future_ids])


def _get_node_properties(node_entry: dict) -> dict:
    node_d = {}
    node_d["timestamp"] = node_entry[Hierarchy.Child][0].timestamp.timestamp()
    if OperationLogs.OperationID in node_entry:
        if len(node_entry[OperationLogs.OperationID]) == 2 or (
            len(node_entry[OperationLogs.OperationID]) == 1
            and Hierarchy.NewParent in node_entry
        ):
            node_d["operation_id"] = node_entry[OperationLogs.OperationID][0].value
    return node_d


def lineage_graph(
    cg,
    node_ids: Union[int, Iterable[int]],
    timestamp_past: Optional[datetime] = None,
    timestamp_future: Optional[datetime] = None,
) -> DiGraph:
    """
    Build lineage graph of a given root ID
    going backwards in time until `timestamp_past`
    and in future until `timestamp_future`
    """
    if not isinstance(node_ids, np.ndarray) and not isinstance(node_ids, list):
        node_ids = [node_ids]

    graph = DiGraph()
    past_ids = np.array(node_ids, dtype=NODE_ID)
    future_ids = np.array(node_ids, dtype=NODE_ID)
    timestamp_past = float(0) if timestamp_past is None else timestamp_past.timestamp()
    timestamp_future = (
        datetime.now(timezone.utc).timestamp()
        if timestamp_future is None
        else timestamp_future.timestamp()
    )

    while past_ids.size or future_ids.size:
        nodes_raw = cg.client.read_nodes(
            node_ids=np.unique(np.concatenate([past_ids, future_ids]))
        )
        next_past_ids = []
        for k in past_ids:
            val = nodes_raw[k]
            node_d = _get_node_properties(val)
            graph.add_node(k, **node_d)
            if (
                node_d["timestamp"] < timestamp_past
                or not Hierarchy.FormerParent in val
            ):
                continue
            former_ids = val[Hierarchy.FormerParent][0].value
            next_past_ids.extend(
                [former_id for former_id in former_ids if not former_id in graph.nodes]
            )
            for former in former_ids:
                graph.add_edge(former, k)

        next_future_ids = []
        future_operation_id_dict = defaultdict(list)
        for k in future_ids:
            val = nodes_raw[k]
            node_d = _get_node_properties(val)
            graph.add_node(k, **node_d)
            if node_d["timestamp"] > timestamp_future or not Hierarchy.NewParent in val:
                continue
            try:
                future_operation_id_dict[node_d["operation_id"]].append(k)
            except KeyError:
                pass

        logs_raw = cg.client.read_log_entries(list(future_operation_id_dict.keys()))
        for operation_id in future_operation_id_dict:
            new_ids = logs_raw[operation_id][OperationLogs.RootID]
            next_future_ids.extend(
                [new_id for new_id in new_ids if not new_id in graph.nodes]
            )
            for new_id in new_ids:
                for k in future_operation_id_dict[operation_id]:
                    graph.add_edge(k, new_id)
        past_ids = np.array(np.unique(next_past_ids), dtype=NODE_ID)
        future_ids = np.array(np.unique(next_future_ids), dtype=NODE_ID)
    return graph
