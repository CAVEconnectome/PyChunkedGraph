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
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
from google.cloud import bigtable
from google.cloud.bigtable.row_filters import RowFilter
from cloudvolume import CloudVolume

from .. import types
from ..chunks import utils as chunk_utils
from . import serializers


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
    max_child_ids_occ_so_far = np.zeros(len(max_child_ids), dtype=np.int)
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
) -> Iterable[np.bool]:
    if bounding_box is None:
        return np.ones(len(nodes), np.bool)
    else:
        chunk_coordinates = np.array(
            [chunk_utils.get_chunk_coordinates(meta, c) for c in nodes]
        )
        layers = chunk_utils.get_chunk_layers(meta, nodes)
        adapt_layers = layers - 2
        adapt_layers[adapt_layers < 0] = 0
        fanout = meta.graph_config.FANOUT
        bounding_box_layer = (
            bounding_box[None] / (fanout ** adapt_layers)[:, None, None]
        )
        bound_check = np.array(
            [
                np.all(chunk_coordinates < bounding_box_layer[:, 1], axis=1),
                np.all(chunk_coordinates + 1 > bounding_box_layer[:, 0], axis=1),
            ]
        ).T

        return np.all(bound_check, axis=1)


class SubgraphProgress:
    """
    Helper class to keep track of node relationships
    while calling cg.get_subgraph(node_ids)
    """

    def __init__(self, meta, node_ids, return_layers, serializable):
        self.meta = meta
        self.node_ids = node_ids
        self.return_layers = return_layers
        self.serializable = serializable

        self.node_to_subgraph = {}
        # "Frontier" of nodes that cg.get_children will be called on
        self.cur_nodes = np.array(list(node_ids), dtype=np.uint64)
        # Mapping of current frontier to self.node_ids
        self.cur_nodes_to_original_nodes = dict(zip(self.cur_nodes, self.cur_nodes))
        self.stop_layer = max(1, np.min(return_layers))
        self.create_initial_node_to_subgraph()

    def done_processing(self):
        return self.cur_nodes is None or len(self.cur_nodes) == 0

    def create_initial_node_to_subgraph(self):
        """
        Create initial subgraph. We will incrementally populate after processing
        each batch of children, and return it when there are no more to process.
        """
        for node_id in self.cur_nodes:
            node_key = self.get_dict_key(node_id)
            self.node_to_subgraph[node_key] = {}
            for return_layer in self.return_layers:
                self.node_to_subgraph[node_key][return_layer] = []
            node_layer = chunk_utils.get_chunk_layer(self.meta, node_id)
            if node_layer in self.return_layers:
                self.node_to_subgraph[node_key][node_layer].append([node_id])

    def get_dict_key(self, node_id):
        if self.serializable:
            return str(node_id)
        return node_id

    def process_batch_of_children(self, cur_nodes_children):
        """
        Given children of self.cur_nodes, update subgraph and
        produce next frontier (if any). 
        """
        next_nodes_to_process = []
        next_nodes_to_original_nodes_keys = []
        next_nodes_to_original_nodes_values = []
        for cur_node, children in cur_nodes_children.items():
            children_layers = chunk_utils.get_chunk_layers(self.meta, children)
            continue_mask = children_layers > self.stop_layer
            continue_children = children[continue_mask]
            original_id = self.cur_nodes_to_original_nodes[np.uint64(cur_node)]
            if len(continue_children) > 0:
                # These nodes will be in next frontier
                next_nodes_to_process.append(continue_children)
                next_nodes_to_original_nodes_keys.append(continue_children)
                next_nodes_to_original_nodes_values.append(
                    [original_id] * len(continue_children)
                )
            for return_layer in self.return_layers:
                # Update subgraph for each return_layer
                children_at_layer = children[children_layers == return_layer]
                if len(children_at_layer) > 0:
                    self.node_to_subgraph[self.get_dict_key(original_id)][
                        return_layer
                    ].append(children_at_layer)

        if len(next_nodes_to_process) == 0:
            self.cur_nodes = None
            # We are done, so we can concatenate/flatten each entry in node_to_subgraph
            self.flatten_subgraph()
        else:
            self.cur_nodes = np.concatenate(next_nodes_to_process)
            self.cur_nodes_to_original_nodes = dict(
                zip(
                    np.concatenate(next_nodes_to_original_nodes_keys),
                    np.concatenate(next_nodes_to_original_nodes_values),
                )
            )

    def flatten_subgraph(self):
        # Flatten each entry in node_to_subgraph before returning
        for node_id in self.node_ids:
            for return_layer in self.return_layers:
                node_key = self.get_dict_key(node_id)
                children_at_layer = self.node_to_subgraph[node_key][return_layer]
                if len(children_at_layer) > 0:
                    self.node_to_subgraph[node_key][return_layer] = np.concatenate(
                        children_at_layer
                    )
                else:
                    self.node_to_subgraph[node_key][return_layer] = types.empty_1d
