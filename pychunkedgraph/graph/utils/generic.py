"""
generic helper functions
TODO categorize properly
"""

import datetime
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union
from typing import Sequence
from typing import Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

from ..chunks import utils as chunk_utils


def compute_indices_pandas(data) -> pd.Series:
    """Computes indices of all unique entries
    Make sure to remap your array to a dense range starting at zero
    https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
    :param data: np.ndarray
    :return: pandas dataframe
    """
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)


def log_n(arr, n):
    """Computes log to base n
    :param arr: array or float
    :param n: int
        base
    :return: return log_n(arr)
    """
    if n == 2:
        return np.log2(arr)
    elif n == 10:
        return np.log10(arr)
    else:
        return np.log(arr) / np.log(n)


def compute_bitmasks(n_layers: int, s_bits_atomic_layer: int = 8) -> Dict[int, int]:
    """Computes the bitmasks for each layer. A bitmasks encodes how many bits
    are used to store the chunk id in each dimension. The smallest number of
    bits needed to encode this information is chosen. The layer id is always
    encoded with 8 bits as this information is required a priori.
    Currently, encoding of layer 1 is fixed to 8 bits.
    :param n_layers: int
    :param fan_out: int
    :param s_bits_atomic_layer: int
    :return: dict
        layer -> bits for layer id
    """
    bitmask_dict = {}
    for i_layer in range(n_layers, 1, -1):
        layer_exp = n_layers - i_layer
        n_bits_for_layers = max(1, layer_exp)
        if i_layer == 2:
            if s_bits_atomic_layer < n_bits_for_layers:
                err = f"{s_bits_atomic_layer} bits is not enough for encoding."
                raise ValueError(err)
            n_bits_for_layers = np.max([s_bits_atomic_layer, n_bits_for_layers])

        n_bits_for_layers = int(n_bits_for_layers)
        bitmask_dict[i_layer] = n_bits_for_layers
    bitmask_dict[1] = bitmask_dict[2]
    return bitmask_dict


def get_max_time():
    """Returns the (almost) max time in datetime.datetime
    :return: datetime.datetime
    """
    return datetime.datetime(9999, 12, 31, 23, 59, 59, 0)


def get_min_time():
    """Returns the min time in datetime.datetime
    :return: datetime.datetime
    """
    return datetime.datetime.strptime("01/01/00 00:00", "%d/%m/%y %H:%M")


def time_min():
    """Returns a minimal time stamp that still works with google
    :return: datetime.datetime
    """
    return datetime.datetime.strptime("01/01/00 00:00", "%d/%m/%y %H:%M")


def get_valid_timestamp(timestamp):
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()
    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)
    # Comply to resolution of BigTables TimeRange
    return _get_google_compatible_time_stamp(timestamp, round_up=False)


def get_bounding_box(
    source_coords: Sequence[Sequence[int]],
    sink_coords: Sequence[Sequence[int]],
    bb_offset: Tuple[int, int, int] = (120, 120, 12),
):
    if source_coords is None or sink_coords is None:
        return
    bb_offset = np.array(list(bb_offset))
    source_coords = np.array(source_coords)
    sink_coords = np.array(sink_coords)

    coords = np.concatenate([source_coords, sink_coords])
    bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]
    bounding_box[0] -= bb_offset
    bounding_box[1] += bb_offset
    return bounding_box


def filter_failed_node_ids(row_ids, segment_ids, max_children_ids):
    """filters node ids that were created by failed/in-complete jobs"""
    sorting = np.argsort(segment_ids)[::-1]
    row_ids = row_ids[sorting]
    max_child_ids = np.array(max_children_ids)[sorting]

    counter = defaultdict(int)
    max_child_ids_occ_so_far = np.zeros(len(max_child_ids), dtype=int)
    for i_row in range(len(max_child_ids)):
        max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
        counter[max_child_ids[i_row]] += 1
    return row_ids[max_child_ids_occ_so_far == 0]


def _get_google_compatible_time_stamp(
    time_stamp: datetime.datetime, round_up: bool = False
) -> datetime.datetime:
    """Makes a datetime.datetime time stamp compatible with googles' services.
    Google restricts the accuracy of time stamps to milliseconds. Hence, the
    microseconds are cut of. By default, time stamps are rounded to the lower
    number.
    :param time_stamp: datetime.datetime
    :param round_up: bool
    :return: datetime.datetime
    """
    micro_s_gap = datetime.timedelta(microseconds=time_stamp.microsecond % 1000)
    if micro_s_gap == 0:
        return time_stamp
    if round_up:
        time_stamp += datetime.timedelta(microseconds=1000) - micro_s_gap
    else:
        time_stamp -= micro_s_gap
    return time_stamp


def mask_nodes_by_bounding_box(
    meta,
    nodes: Union[Iterable[np.uint64], np.uint64],
    bounding_box: Optional[Sequence[Sequence[int]]] = None,
) -> Iterable[bool]:
    if bounding_box is None:
        return np.ones(len(nodes), bool)
    else:
        chunk_coordinates = np.array(
            [chunk_utils.get_chunk_coordinates(meta, c) for c in nodes]
        )
        layers = chunk_utils.get_chunk_layers(meta, nodes)
        adapt_layers = layers - 2
        adapt_layers[adapt_layers < 0] = 0
        fanout = meta.graph_config.FANOUT
        bounding_box_layer = bounding_box[None] / (fanout**adapt_layers)[:, None, None]
        bound_check = np.array(
            [
                np.all(chunk_coordinates < bounding_box_layer[:, 1], axis=1),
                np.all(chunk_coordinates + 1 > bounding_box_layer[:, 0], axis=1),
            ]
        ).T

        return np.all(bound_check, axis=1)


def get_parents_at_timestamp(nodes, parents_ts_map, time_stamp, unique: bool = False):
    """
    Search for the first parent with ts <= `time_stamp`.
    `parents_ts_map[node]` is a map of ts:parent with sorted timestamps (desc).
    """
    skipped_nodes = []
    parents = set() if unique else []
    for node in nodes:
        try:
            for ts, parent in parents_ts_map[node].items():
                if time_stamp >= ts:
                    parents.add(parent) if unique else parents.append(parent)
                    break
        except KeyError:
            skipped_nodes.append(node)
    return list(parents), skipped_nodes
