import datetime
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from google.cloud import bigtable
from google.cloud.bigtable.row_filters import TimestampRange, \
    TimestampRangeFilter, ColumnRangeFilter, RowFilterChain, \
    RowFilterUnion, RowFilter
from pychunkedgraph.backend.utils import column_keys, serializers


def compute_indices_pandas(data) -> pd.Series:
    """ Computes indices of all unique entries

    Make sure to remap your array to a dense range starting at zero

    https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where

    :param data: np.ndarray
    :return: pandas dataframe
    """
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)


def log_n(arr, n):
    """ Computes log to base n

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


def compute_bitmasks(n_layers: int, fan_out: int) -> Dict[int, int]:
    """ Computes the bitmasks for each layer. A bitmasks encodes how many bits
    are used to store the chunk id in each dimension. The smallest number of
    bits needed to encode this information is chosen. The layer id is always
    encoded with 8 bits as this information is required a priori.

    Currently, encoding of layer 1 is fixed to 8 bits.

    :param n_layers: int
    :param fan_out: int
    :return: dict
        layer -> bits for layer id
    """

    bitmask_dict = {}
    for i_layer in range(n_layers, 0, -1):

        if i_layer == 1:
            # Lock this layer to an 8 bit layout to maintain compatibility with
            # the exported segmentation

            # n_bits_for_layers = np.ceil(log_n(fan_out**(n_layers - 2), fan_out))
            n_bits_for_layers = 8
        else:
            layer_exp = n_layers - i_layer
            n_bits_for_layers = max(1, np.ceil(log_n(fan_out**layer_exp, fan_out)))
            # n_bits_for_layers = fan_out ** int(np.ceil(log_n(n_bits_for_layers, fan_out)))

        layer_exp = n_layers - i_layer
        n_bits_for_layers = max(1, np.ceil(log_n(fan_out**layer_exp, fan_out)))

        if i_layer == 1:
            n_bits_for_layers = np.max([8, n_bits_for_layers])

        n_bits_for_layers = int(n_bits_for_layers)

        # assert n_bits_for_layers <= 8

        bitmask_dict[i_layer] = n_bits_for_layers

        # print(f"Bitmasks: {bitmask_dict}")

    return bitmask_dict


def get_google_compatible_time_stamp(time_stamp: datetime.datetime,
                                     round_up: bool =False
                                     ) -> datetime.datetime:
    """ Makes a datetime.datetime time stamp compatible with googles' services.
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
        time_stamp += (datetime.timedelta(microseconds=1000) - micro_s_gap)
    else:
        time_stamp -= micro_s_gap

    return time_stamp


def get_column_filter(
        columns: Union[Iterable[column_keys._Column], column_keys._Column] = None) -> RowFilter:
    """ Generates a RowFilter that accepts the specified columns """

    if isinstance(columns, column_keys._Column):
        return ColumnRangeFilter(columns.family_id,
                                 start_column=columns.key,
                                 end_column=columns.key)
    elif len(columns) == 1:
        return ColumnRangeFilter(columns[0].family_id,
                                 start_column=columns[0].key,
                                 end_column=columns[0].key)

    return RowFilterUnion([ColumnRangeFilter(col.family_id,
                                             start_column=col.key,
                                             end_column=col.key) for col in columns])


def get_time_range_filter(
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_inclusive: bool = True) -> RowFilter:
    """ Generates a TimeStampRangeFilter which is inclusive for start and (optionally) end.

    :param start:
    :param end:
    :return:
    """
    # Comply to resolution of BigTables TimeRange
    if start_time is not None:
        start_time = get_google_compatible_time_stamp(start_time, round_up=False)
    if end_time is not None:
        end_time = get_google_compatible_time_stamp(end_time, round_up=end_inclusive)

    return TimestampRangeFilter(TimestampRange(start=start_time, end=end_time))


def get_time_range_and_column_filter(
        columns: Optional[Union[Iterable[column_keys._Column], column_keys._Column]] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_inclusive: bool = False) -> RowFilter:

    time_filter = get_time_range_filter(start_time=start_time,
                                        end_time=end_time,
                                        end_inclusive=end_inclusive)

    if columns is not None:
        column_filter = get_column_filter(columns)
        return RowFilterChain([column_filter, time_filter])
    else:
        return time_filter


def get_max_time():
    """ Returns the (almost) max time in datetime.datetime

    :return: datetime.datetime
    """
    return datetime.datetime(9999, 12, 31, 23, 59, 59, 0)


def get_min_time():
    """ Returns the min time in datetime.datetime

    :return: datetime.datetime
    """
    return datetime.datetime.strptime("01/01/00 00:00", "%d/%m/%y %H:%M")


def combine_cross_chunk_edge_dicts(d1, d2, start_layer=2):
    """ Combines two cross chunk dictionaries
    Cross chunk dictionaries contain a layer id -> edge list mapping.

    :param d1: dict
    :param d2: dict
    :param start_layer: int
    :return: dict
    """
    assert start_layer >= 2

    new_d = {}

    for l in d2:
        if l < start_layer:
            continue

    layers = np.unique(list(d1.keys()) + list(d2.keys()))
    layers = layers[layers >= start_layer]

    for l in layers:
        if l in d1 and l in d2:
            new_d[l] = np.concatenate([d1[l], d2[l]])
        elif l in d1:
            new_d[l] = d1[l]
        elif l in d2:
            new_d[l] = d2[l]
        else:
            raise Exception()

        edges_flattened_view = new_d[l].view(dtype='u8,u8')
        m = np.unique(edges_flattened_view, return_index=True)[1]
        new_d[l] = new_d[l][m]

    return new_d


def time_min():
    """ Returns a minimal time stamp that still works with google

    :return: datetime.datetime
    """
    return datetime.datetime.strptime("01/01/00 00:00", "%d/%m/%y %H:%M")


def partial_row_data_to_column_dict(partial_row_data: bigtable.row_data.PartialRowData) \
        -> Dict[column_keys._Column, bigtable.row_data.PartialRowData]:
    new_column_dict = {}

    for family_id, column_dict in partial_row_data._cells.items():
        for column_key, column_values in column_dict.items():
            column = column_keys.from_key(family_id, column_key)
            new_column_dict[column] = column_values

    return new_column_dict
