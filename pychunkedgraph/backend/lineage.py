"""
Functions for tracking root ID changes over time.
"""
from typing import Iterable
from typing import Optional

import numpy as np
from networkx import DiGraph

from time import time
from pychunkedgraph.backend.utils.basetypes import NODE_ID
from pychunkedgraph.backend.utils.column_keys import Hierarchy
from pychunkedgraph.backend.utils.column_keys import OperationLogs


def lineage_graph(
    cg,
    node_ids: Iterable[int],
    timestamp_past: Optional[float] = None,
    timestamp_future: Optional[float] = None,
) -> DiGraph:
    """
    Build lineage graph of a given root ID
    going backwards in time until `timestamp_past`
    and in future until `timestamp_future`
    """
    
    if not isinstance(node_ids, np.ndarray) and not isinstance(node_ids, list):
        node_ids = [node_ids]
    
    graph = DiGraph()
    past_ids = np.array(node_ids, dtype=np.uint64)
    future_ids = np.array(node_ids, dtype=np.uint64)
    if timestamp_past is None:
        timestamp_past = float(0)
    else:
        timestamp_past = timestamp_past.timestamp()
    if timestamp_future is None:
        timestamp_future = float(time())
    else:
        timestamp_future = timestamp_future.timestamp()
    
    while past_ids.size or future_ids.size:
        nodes_raw = cg.read_node_id_rows(
            node_ids=np.unique(np.concatenate([past_ids, future_ids]))
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
                continue
            
            if Hierarchy.NewParent in val:
                graph.add_node(k, operation_id=operation_id, timestamp=timestamp)
            else:
                graph.add_node(k, timestamp=timestamp)
                
            if timestamp < timestamp_past or not Hierarchy.FormerParent in val:
                continue
        
            former_ids = val[Hierarchy.FormerParent][0].value
            next_past_ids.extend([former_id for former_id in former_ids if not former_id in graph.nodes ])
            for former in former_ids:
                graph.add_edge(former, k)
        
        next_future_ids = []
        future_operation_id_dict = {}
        for k in future_ids:
            val = nodes_raw[k]
            operation_id = val[OperationLogs.OperationID][0].value
            timestamp = val[Hierarchy.Child][0].timestamp.timestamp()
            
            if Hierarchy.NewParent in val:
                graph.add_node(k, operation_id=operation_id, timestamp=timestamp)
            else:
                graph.add_node(k, timestamp=timestamp)
                
            if timestamp > timestamp_future or not Hierarchy.NewParent in val:
                continue
            
            future_operation_id_dict[operation_id] = k
            
        logs_raw = cg.read_log_rows(list(future_operation_id_dict.keys()))
        for operation_id in future_operation_id_dict:
            new_ids = logs_raw[operation_id][OperationLogs.RootID]
            next_future_ids.extend([new_id for new_id in new_ids if not new_id in graph.nodes])
            for new_id in new_ids:
                graph.add_edge(future_operation_id_dict[operation_id], new_id)
        
        past_ids = np.array(np.unique(next_past_ids), dtype=NODE_ID)
        future_ids = np.array(np.unique(next_future_ids), dtype=NODE_ID)

    return graph