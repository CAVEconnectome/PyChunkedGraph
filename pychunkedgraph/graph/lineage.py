"""
Functions for tracking root ID changes over time.
"""
from typing import Optional
from datetime import datetime

import numpy as np

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
    cg, root_id: np.uint64, time_stamp: Optional[datetime] = get_max_time(),
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
    cg, root_id: np.uint64, time_stamp: Optional[datetime] = get_min_time(),
) -> np.ndarray:
    """
    Returns all future root ids emerging from this root.
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
                columns=[
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


# TODO
# def read_first_log_row
# def get_change_log
# def read_logs

