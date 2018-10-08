import collections
import numpy as np
import time
import datetime
import os
import networkx as nx
import pytz
import cloudvolume
import pandas as pd
import re
import itertools

from multiwrapper import multiprocessing_utils as mu
from pychunkedgraph.backend import cutting

from google.api_core.retry import Retry, if_exception_type
from google.api_core.exceptions import Aborted, DeadlineExceeded, \
    ServiceUnavailable
from google.auth import credentials
from google.cloud import bigtable
from google.cloud.bigtable.row_filters import TimestampRange, \
    TimestampRangeFilter, ColumnRangeFilter, ValueRangeFilter, RowFilterChain, \
    ColumnQualifierRegexFilter, RowFilterUnion, ConditionalRowFilter, \
    PassAllFilter, RowFilter, RowKeyRegexFilter, FamilyNameRegexFilter
from google.cloud.bigtable.column_family import MaxVersionsGCRule

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# global variables
from pychunkedgraph.backend import table_info, key_utils

HOME = os.path.expanduser("~")
N_DIGITS_UINT64 = len(str(np.iinfo(np.uint64).max))
LOCK_EXPIRED_TIME_DELTA = datetime.timedelta(minutes=1, seconds=00)
UTC = pytz.UTC

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
           HOME + "/.cloudvolume/secrets/google-secret.json"


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

        n_bits_for_layers = int(n_bits_for_layers)

        assert n_bits_for_layers <= 8

        bitmask_dict[i_layer] = n_bits_for_layers
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


def get_inclusive_time_range_filter(start=None, end=None):
    """ Generates a TimeStampRangeFilter which is inclusive for start and end.

    :param start:
    :param end:
    :return:
    """
    if end is not None:
        end += (datetime.timedelta(microseconds=1000))

    return TimestampRangeFilter(TimestampRange(start=start, end=end))


def get_max_time():
    """ Returns the (almost) max time in datetime.datetime

    :return: datetime.datetime
    """
    return datetime.datetime(9999, 12, 31, 23, 59, 59, 0)


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

class ChunkedGraph(object):
    def __init__(self,
                 table_id: str,
                 instance_id: str = "pychunkedgraph",
                 project_id: str = "neuromancer-seung-import",
                 chunk_size: Tuple[int, int, int] = None,
                 fan_out: Optional[int] = None,
                 n_layers: Optional[int] = None,
                 credentials: Optional[credentials.Credentials] = None,
                 client: bigtable.Client = None,
                 cv_path: str = None,
                 is_new: bool = False) -> None:

        if client is not None:
            self._client = client
        else:
            self._client = bigtable.Client(project=project_id, admin=True,
                                           credentials=credentials)

        self._instance = self.client.instance(instance_id)
        self._table_id = table_id

        self._table = self.instance.table(self.table_id)

        if is_new:
            self.check_and_create_table()

        self._n_layers = self.check_and_write_table_parameters("n_layers",
                                                               n_layers)
        self._fan_out = self.check_and_write_table_parameters("fan_out",
                                                              fan_out)
        self._cv_path = self.check_and_write_table_parameters("cv_path",
                                                              cv_path)
        self._chunk_size = self.check_and_write_table_parameters("chunk_size",
                                                                 chunk_size)

        self._bitmasks = compute_bitmasks(self.n_layers, self.fan_out)

        self._cv = None

        # Hardcoded parameters
        self._n_bits_for_layer_id = 8
        self._cv_mip = 3

    @property
    def client(self) -> bigtable.Client:
        return self._client

    @property
    def instance(self) -> bigtable.instance.Instance:
        return self._instance

    @property
    def table(self) -> bigtable.table.Table:
        return self._table

    @property
    def table_id(self) -> str:
        return self._table_id

    @property
    def instance_id(self):
        return self.instance.instance_id

    @property
    def project_id(self):
        return self.client.project

    @property
    def family_id(self) -> str:
        return "0"

    @property
    def incrementer_family_id(self) -> str:
        return "1"

    @property
    def log_family_id(self) -> str:
        return "2"

    @property
    def cross_edge_family_id(self) -> str:
        return "3"

    @property
    def family_ids(self):
        return [self.family_id, self.incrementer_family_id, self.log_family_id,
                self.cross_edge_family_id]

    @property
    def fan_out(self) -> int:
        return self._fan_out

    @property
    def chunk_size(self) -> np.ndarray:
        return self._chunk_size

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def bitmasks(self) -> Dict[int, int]:
        return self._bitmasks

    @property
    def cv_path(self) -> str:
        return self._cv_path

    @property
    def cv_mip(self) -> int:
        return self._cv_mip

    @property
    def cv(self) -> cloudvolume.CloudVolume:
        if self._cv is None:
            self._cv = cloudvolume.CloudVolume(self.cv_path, mip=self._cv_mip)
        return self._cv

    @property
    def root_chunk_id(self):
        return self.get_chunk_id(layer=int(self.n_layers), x=0, y=0, z=0)

    def check_and_create_table(self) -> None:
        """ Checks if table exists and creates new one if necessary """
        table_ids = [t.table_id for t in self.instance.list_tables()]

        if not self.table_id in table_ids:
            self.table.create()
            f = self.table.column_family(self.family_id)
            f.create()

            f_inc = self.table.column_family(self.incrementer_family_id,
                                             gc_rule=MaxVersionsGCRule(1))
            f_inc.create()

            f_log = self.table.column_family(self.log_family_id)
            f_log.create()

            f_ce = self.table.column_family(self.cross_edge_family_id,
                                            gc_rule=MaxVersionsGCRule(1))
            f_ce.create()

            print("Table created")

    def check_and_write_table_parameters(self, param_key: str,
                                         value: Optional[np.uint64] = None
                                         ) -> np.uint64:
        """ Checks if a parameter already exists in the table. If it already
        exists it returns the stored value, else it stores the given value. It
        raises an exception if no value is passed and the parameter does not
        exist, yet.

        :param param_key: str
        :param value: np.uint64
        :return: np.uint64
            value
        """
        ser_param_key = key_utils.serialize_key(param_key)
        row = self.table.read_row(key_utils.serialize_key("params"))

        if row is None or ser_param_key not in row.cells[self.family_id]:
            assert value is not None

            if param_key in ["fan_out", "n_layers"]:
                val_dict = {param_key: np.array(value,
                                                dtype=np.uint64).tobytes()}
            elif param_key in ["cv_path"]:
                val_dict = {param_key: key_utils.serialize_key(value)}
            elif param_key in ["chunk_size"]:
                val_dict = {param_key: np.array(value,
                                                dtype=np.uint64).tobytes()}
            else:
                raise Exception("Unknown type for parameter")

            row = self.mutate_row(key_utils.serialize_key("params"), self.family_id,
                                  val_dict)

            self.bulk_write([row])
        else:
            value = row.cells[self.family_id][ser_param_key][0].value

            if param_key in ["fan_out", "n_layers"]:
                value = np.frombuffer(value, dtype=np.uint64)[0]
            elif param_key in ["cv_path"]:
                value = key_utils.deserialize_key(value)
            elif param_key in ["chunk_size"]:
                value = np.frombuffer(value, dtype=np.uint64)
            else:
                raise Exception("Unknown key")

        return value

    def clear_changes(self, time_stamp: datetime.datetime):
        """ Clears everything after this time_stamp


        :param time_stamp: datetime.datetime
        :return: bool
            success
        """

        raise NotImplementedError()

        # Read rows with changed cells
        time_filter = get_inclusive_time_range_filter(start=time_stamp)
        rr = self.range_read(start_id=np.uint64(0),
                             end_id=np.iinfo(np.uint64).max - np.uint64(2),
                             time_filter=time_filter, max_block_size=np.iinfo(np.uint64).max)

        rows = list(rr.values())
        for row in rows:
            for fam_id in row.keys():
                row_columns = list(row[fam_id].keys())
                row.delete_cells(column_family_id=fam_id, columns=row_columns,
                                 time_range=TimestampRange(start=time_stamp))

        self.bulk_write(rows)

        return rr

    def get_serialized_info(self):
        """ Rerturns dictionary that can be used to load this ChunkedGraph

        :return: dict
        """
        info = {"table_id": self.table_id,
                "instance_id": self.instance_id,
                "project_id": self.project_id}

        try:
            info["credentials"] = self.client.credentials
        except:
            info["credentials"] = self.client._credentials

        return info

    def get_chunk_layer(self, node_or_chunk_id: np.uint64) -> int:
        """ Extract Layer from Node ID or Chunk ID

        :param node_or_chunk_id: np.uint64
        :return: int
        """
        return int(node_or_chunk_id) >> 64 - self._n_bits_for_layer_id

    def get_chunk_coordinates(self, node_or_chunk_id: np.uint64
                              ) -> np.ndarray:
        """ Extract X, Y and Z coordinate from Node ID or Chunk ID

        :param node_or_chunk_id: np.uint64
        :return: Tuple(int, int, int)
        """
        layer = self.get_chunk_layer(node_or_chunk_id)
        bits_per_dim = self.bitmasks[layer]

        x_offset = 64 - self._n_bits_for_layer_id - bits_per_dim
        y_offset = x_offset - bits_per_dim
        z_offset = y_offset - bits_per_dim

        x = int(node_or_chunk_id) >> x_offset & 2 ** bits_per_dim - 1
        y = int(node_or_chunk_id) >> y_offset & 2 ** bits_per_dim - 1
        z = int(node_or_chunk_id) >> z_offset & 2 ** bits_per_dim - 1
        return np.array([x, y, z])

    def get_chunk_id(self, node_id: Optional[np.uint64] = None,
                     layer: Optional[int] = None,
                     x: Optional[int] = None,
                     y: Optional[int] = None,
                     z: Optional[int] = None) -> np.uint64:
        """ (1) Extract Chunk ID from Node ID
            (2) Build Chunk ID from Layer, X, Y and Z components

        :param node_id: np.uint64
        :param layer: int
        :param x: int
        :param y: int
        :param z: int
        :return: np.uint64
        """
        assert node_id is not None or \
               all(v is not None for v in [layer, x, y, z])

        if node_id is not None:
            layer = self.get_chunk_layer(node_id)
        bits_per_dim = self.bitmasks[layer]

        if node_id is not None:
            chunk_offset = 64 - self._n_bits_for_layer_id - 3 * bits_per_dim
            return np.uint64((int(node_id) >> chunk_offset) << chunk_offset)
        else:

            if not(x < 2 ** bits_per_dim and
                   y < 2 ** bits_per_dim and
                   z < 2 ** bits_per_dim):
                raise Exception("Chunk coordinate is out of range for"
                                "this graph on layer %d with %d bits/dim."
                                "[%d, %d, %d]; max = %d."
                                % (layer, bits_per_dim, x, y, z,
                                   2 ** bits_per_dim))

            layer_offset = 64 - self._n_bits_for_layer_id
            x_offset = layer_offset - bits_per_dim
            y_offset = x_offset - bits_per_dim
            z_offset = y_offset - bits_per_dim
            return np.uint64(layer << layer_offset | x << x_offset |
                             y << y_offset | z << z_offset)

    def get_chunk_ids_from_node_ids(self, node_ids: Iterable[np.uint64]
                                    ) -> np.ndarray:
        """ Extract a list of Chunk IDs from a list of Node IDs

        :param node_ids: np.ndarray(dtype=np.uint64)
        :return: np.ndarray(dtype=np.uint64)
        """
        # TODO: measure and improve performance(?)
        return np.array(list(map(lambda x: self.get_chunk_id(node_id=x),
                                 node_ids)), dtype=np.uint64)

    def get_segment_id_limit(self, node_or_chunk_id: np.uint64) -> np.uint64:
        """ Get maximum possible Segment ID for given Node ID or Chunk ID

        :param node_or_chunk_id: np.uint64
        :return: np.uint64
        """

        layer = self.get_chunk_layer(node_or_chunk_id)
        bits_per_dim = self.bitmasks[layer]
        chunk_offset = 64 - self._n_bits_for_layer_id - 3 * bits_per_dim
        return np.uint64(2 ** chunk_offset - 1)

    def get_segment_id(self, node_id: np.uint64) -> np.uint64:
        """ Extract Segment ID from Node ID

        :param node_id: np.uint64
        :return: np.uint64
        """

        return node_id & self.get_segment_id_limit(node_id)

    def get_node_id(self, segment_id: np.uint64,
                    chunk_id: Optional[np.uint64] = None,
                    layer: Optional[int] = None,
                    x: Optional[int] = None,
                    y: Optional[int] = None,
                    z: Optional[int] = None) -> np.uint64:
        """ (1) Build Node ID from Segment ID and Chunk ID
            (2) Build Node ID from Segment ID, Layer, X, Y and Z components

        :param segment_id: np.uint64
        :param chunk_id: np.uint64
        :param layer: int
        :param x: int
        :param y: int
        :param z: int
        :return: np.uint64
        """

        if chunk_id is not None:
            return chunk_id | segment_id
        else:
            return self.get_chunk_id(layer=layer, x=x, y=y, z=z) | segment_id

    def get_unique_segment_id_range(self, chunk_id: np.uint64, step: int = 1
                                    ) -> np.ndarray:
        """ Return unique Segment ID for given Chunk ID

        atomic counter

        :param chunk_id: np.uint64
        :param step: int
        :return: np.uint64
        """

        counter_key = key_utils.serialize_key('counter')

        # Incrementer row keys start with an "i" followed by the chunk id
        row_key = key_utils.serialize_key("i%s" % key_utils.pad_node_id(chunk_id))
        append_row = self.table.row(row_key, append=True)
        append_row.increment_cell_value(self.incrementer_family_id,
                                        counter_key, step)

        # This increments the row entry and returns the value AFTER incrementing
        latest_row = append_row.commit()
        max_segment_id_b = latest_row[self.incrementer_family_id][counter_key][0][0]
        max_segment_id = int.from_bytes(max_segment_id_b, byteorder="big")

        min_segment_id = max_segment_id + 1 - step
        segment_id_range = np.array(range(min_segment_id, max_segment_id + 1),
                                    dtype=np.uint64)
        return segment_id_range

    def get_unique_segment_id(self, chunk_id: np.uint64) -> np.uint64:
        """ Return unique Segment ID for given Chunk ID

        atomic counter

        :param chunk_id: np.uint64
        :param step: int
        :return: np.uint64
        """

        return self.get_unique_segment_id_range(chunk_id=chunk_id, step=1)[0]

    def get_unique_node_id_range(self, chunk_id: np.uint64, step: int = 1
                                 )  -> np.ndarray:
        """ Return unique Node ID range for given Chunk ID

        atomic counter

        :param chunk_id: np.uint64
        :param step: int
        :return: np.uint64
        """

        segment_ids = self.get_unique_segment_id_range(chunk_id=chunk_id,
                                                       step=step)

        node_ids = np.array([self.get_node_id(segment_id, chunk_id)
                             for segment_id in segment_ids], dtype=np.uint64)
        return node_ids

    def get_unique_node_id(self, chunk_id: np.uint64) -> np.uint64:
        """ Return unique Node ID for given Chunk ID

        atomic counter

        :param chunk_id: np.uint64
        :return: np.uint64
        """

        return self.get_unique_node_id_range(chunk_id=chunk_id, step=1)[0]

    def get_max_seg_id(self, chunk_id: np.uint64) -> np.uint64:
        """  Gets maximal seg id in a chunk based on the atomic counter

        This is an approximation. It is not guaranteed that all ids smaller or
        equal to this id exists. However, it is guaranteed that no larger id
        exist at the time this function is executed.


        :return: uint64
        """

        counter_key = key_utils.serialize_key('counter')

        # Incrementer row keys start with an "i"
        row_key = key_utils.serialize_key("i%s" % key_utils.pad_node_id(chunk_id))
        row = self.table.read_row(row_key)

        # Read incrementer value
        if row is not None:
            max_node_id_b = row.cells[self.incrementer_family_id][counter_key][0].value
            max_node_id = int.from_bytes(max_node_id_b, byteorder="big")
        else:
            max_node_id = 0

        return np.uint64(max_node_id)

    def get_max_node_id(self, chunk_id: np.uint64) -> np.uint64:
        """  Gets maximal node id in a chunk based on the atomic counter

        This is an approximation. It is not guaranteed that all ids smaller or
        equal to this id exists. However, it is guaranteed that no larger id
        exist at the time this function is executed.


        :return: uint64
        """

        max_seg_id = self.get_max_seg_id(chunk_id)
        return self.get_node_id(segment_id=max_seg_id, chunk_id=chunk_id)

    def get_unique_operation_id(self) -> np.uint64:
        """ Finds a unique operation id

        atomic counter

        Operations essentially live in layer 0. Even if segmentation ids might
        live in layer 0 one day, they would not collide with the operation ids
        because we write information belonging to operations in a separate
        family id.

        :return: str
        """

        counter_key = key_utils.serialize_key('counter')

        # Incrementer row keys start with an "i"
        row_key = key_utils.serialize_key("ioperations")
        append_row = self.table.row(row_key, append=True)
        append_row.increment_cell_value(self.incrementer_family_id,
                                        counter_key, 1)

        # This increments the row entry and returns the value AFTER incrementing
        latest_row = append_row.commit()
        operation_id_b = latest_row[self.incrementer_family_id][counter_key][0][0]
        operation_id = int.from_bytes(operation_id_b, byteorder="big")

        return np.uint64(operation_id)

    def get_max_operation_id(self) -> np.uint64:
        """  Gets maximal operation id based on the atomic counter

        This is an approximation. It is not guaranteed that all ids smaller or
        equal to this id exists. However, it is guaranteed that no larger id
        exist at the time this function is executed.


        :return: uint64
        """

        counter_key = key_utils.serialize_key('counter')

        # Incrementer row keys start with an "i"
        row_key = key_utils.serialize_key("ioperations")
        row = self.table.read_row(row_key)

        # Read incrementer value
        if row is not None:
            max_operation_id_b = row.cells[self.incrementer_family_id][counter_key][0].value
            max_operation_id = int.from_bytes(max_operation_id_b,
                                              byteorder="big")
        else:
            max_operation_id = 0

        return np.uint64(max_operation_id)

    def get_cross_chunk_edges_layer(self, cross_edges):
        """ Computes the layer in which a cross chunk edge becomes relevant.

        I.e. if a cross chunk edge links two nodes in layer 4 this function
        returns 3.

        :param cross_edges: n x 2 array
            edges between atomic (level 1) node ids
        :return: array of length n
        """
        if len(cross_edges) == 0:
            return np.array([], dtype=np.int)

        cross_chunk_edge_layers = np.ones(len(cross_edges), dtype=np.int)

        cross_edge_coordinates = []
        for cross_edge in cross_edges:
            cross_edge_coordinates.append(
                [self.get_chunk_coordinates(cross_edge[0]),
                 self.get_chunk_coordinates(cross_edge[1])])

        cross_edge_coordinates = np.array(cross_edge_coordinates, dtype=np.int)

        for layer in range(2, self.n_layers):
            edge_diff = np.sum(np.abs(cross_edge_coordinates[:, 0] -
                                      cross_edge_coordinates[:, 1]), axis=1)
            cross_chunk_edge_layers[edge_diff > 0] += 1
            cross_edge_coordinates = cross_edge_coordinates // self.fan_out

        return cross_chunk_edge_layers

    def get_cross_chunk_edge_dict(self, cross_edges):
        """ Generates a cross chunk edge dict for a list of cross chunk edges

        :param cross_edges: n x 2 array
        :return: dict
        """
        cce_layers = self.get_cross_chunk_edges_layer(cross_edges)
        u_cce_layers = np.unique(cce_layers)
        cross_edge_dict = {}

        for l in range(2, self.n_layers):
            cross_edge_dict[l] = \
                np.array([], dtype=np.uint64).reshape(-1, 2)

        val_dict = {}
        for cc_layer in u_cce_layers:
            layer_cross_edges = cross_edges[cce_layers == cc_layer]

            if len(layer_cross_edges) > 0:
                val_dict[table_info.cross_chunk_edge_keyformat % cc_layer] = \
                    layer_cross_edges.tobytes()
                cross_edge_dict[cc_layer] = layer_cross_edges
        return cross_edge_dict

    def read_row(self, node_id: np.uint64, key: str, idx: int = 0,
                 dtype: type = np.uint64, get_time_stamp: bool = False,
                 fam_id: str = None) -> Any:
        """ Reads row from BigTable and takes care of serializations

        :param node_id: uint64
        :param key: table column
        :param idx: column list index
        :param dtype: np.dtype
        :param get_time_stamp: bool
        :param fam_id: str
        :return: row entry
        """
        key = key_utils.serialize_key(key)

        if fam_id is None:
            fam_id = self.family_id

        row = self.table.read_row(key_utils.serialize_uint64(node_id),
                                  filter_=ColumnQualifierRegexFilter(key))

        if row is None or key not in row.cells[fam_id]:
            if get_time_stamp:
                return None, None
            else:
                return None

        cell_entries = row.cells[fam_id][key]

        if dtype is None:
            cell_value = cell_entries[idx].value
        else:
            cell_value = np.frombuffer(cell_entries[idx].value, dtype=dtype)

        if get_time_stamp:
            return cell_value, cell_entries[idx].timestamp
        else:
            return cell_value

    def read_row_multi_key(self, node_id: np.uint64, keys: str = None,
                           time_stamp: datetime.datetime = get_max_time()
                           ) -> Any:
        """ Reads row from BigTable and takes care of serializations

        :param node_id: uint64
        :param keys: table columns
        :param time_stamp: datetime.datetime
        :return: row entry
        """
        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        # Create filters: time and id range
        time_filter = get_inclusive_time_range_filter(end=time_stamp)

        if keys is not None:
            if not isinstance(keys, list):
                keys = [keys]

            keys_s = [k if isinstance(k, bytes) else key_utils.serialize_key(k)
                      for k in keys]

            filters = [ColumnQualifierRegexFilter(k) for k in keys_s]

            if len(filters) > 1:
                filter_ = RowFilterUnion(filters)
            else:
                filter_ = filters[0]

            filter_ = RowFilterChain([filter_, time_filter])
        else:
            filter_ = time_filter

        row = self.table.read_row(key_utils.serialize_uint64(node_id), filter_=filter_)

        if row is None:
            return {}

        return row.cells

    def read_cross_chunk_edges(self, node_id: np.uint64, start_layer: int = 2,
                               end_layer: int = None) -> Dict:
        """ Reads the cross chunk edge entry from the table for a given node id
        and formats it as cross edge dict

        :param node_id:
        :param start_layer:
        :param end_layer:
        :return:
        """
        if end_layer is None:
            end_layer = self.n_layers

        if start_layer < 2 or start_layer == self.n_layers:
            return {}

        start_layer = np.max([self.get_chunk_layer(node_id), start_layer])

        assert end_layer > start_layer and end_layer <= self.n_layers

        keys = [table_info.cross_chunk_edge_keyformat % l
                for l in range(start_layer, end_layer)]
        row_dict = self.read_row_multi_key(node_id, keys=keys)

        if self.cross_edge_family_id not in row_dict:
            row_dict =  {}
        else:
            row_dict = row_dict[self.cross_edge_family_id]

        cross_edge_dict = {}
        for l in range(start_layer, end_layer):
            cross_edge_dict[l] = np.array([], dtype=table_info.dtype_dict["cross_chunk_edges"]).reshape(-1, 2)

        for k in row_dict.keys():
            l = int(re.findall("[\d]+", key_utils.deserialize_key(k))[-1])
            cross_edge_dict[l] = np.frombuffer(row_dict[k][0].value,
                                               dtype=table_info.dtype_dict["cross_chunk_edges"]).reshape(-1, 2)

        return cross_edge_dict

    def mutate_row(self, row_key: bytes, column_family_id: str, val_dict: dict,
                   time_stamp: Optional[datetime.datetime] = None
                   ) -> bigtable.row.Row:
        """ Mutates a single row

        :param row_key: serialized bigtable row key
        :param column_family_id: str
            serialized column family id
        :param val_dict: dict
        :param time_stamp: None or datetime
        :return: list
        """
        row = self.table.row(row_key)

        for column, value in val_dict.items():
            row.set_cell(column_family_id=column_family_id, column=column,
                         value=value, timestamp=time_stamp)
        return row

    def bulk_write(self, rows: Iterable[bigtable.row.DirectRow],
                   root_ids: Optional[Union[np.uint64,
                                            Iterable[np.uint64]]] = None,
                   operation_id: Optional[np.uint64] = None,
                   slow_retry: bool = True,
                   block_size: int = 2000) -> bool:
        """ Writes a list of mutated rows in bulk

        WARNING: If <rows> contains the same row (same row_key) and column
        key two times only the last one is effectively written to the BigTable
        (even when the mutations were applied to different columns)
        --> no versioning!

        :param rows: list
            list of mutated rows
        :param root_ids: list if uint64
        :param operation_id: uint64 or None
            operation_id (or other unique id) that *was* used to lock the root
            the bulk write is only executed if the root is still locked with
            the same id.
        :param slow_retry: bool
        :param block_size: int
        """
        if slow_retry:
            initial = 5
        else:
            initial = 1

        retry_policy = Retry(
            predicate=if_exception_type((Aborted,
                                         DeadlineExceeded,
                                         ServiceUnavailable)),
            initial=initial,
            maximum=15.0,
            multiplier=2.0,
            deadline=LOCK_EXPIRED_TIME_DELTA.seconds)

        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]

            if not self.check_and_renew_root_locks(root_ids, operation_id):
                return False

        for i_row in range(0, len(rows), block_size):
            status = self.table.mutate_rows(rows[i_row: i_row + block_size],
                                            retry=retry_policy)

            if not all(status):
                raise Exception(status)

        return True

    def _range_read_execution(self, start_id, end_id,
                              row_filter: RowFilter = None,
                              n_retries: int = 100):
        """ Executes predefined range read (read_rows)

        :param start_id: np.uint64
        :param end_id: np.uint64
        :param row_filter: BigTable RowFilter
        :param n_retries: int
        :return: dict
        """
        # Set up read
        range_read = self.table.read_rows(
            start_key=key_utils.serialize_uint64(start_id),
            end_key=key_utils.serialize_uint64(end_id),
            # allow_row_interleaving=True,
            end_inclusive=True,
            filter_=row_filter)

        # Execute read
        consume_success = False

        # Retry reading if any of the writes failed
        i_tries = 0
        while not consume_success and i_tries < n_retries:
            try:
                range_read.consume_all()
                consume_success = True
            except:
                print("FAILURE -- retry")
                time.sleep(i_tries)
            i_tries += 1

        if not consume_success:
            raise Exception("Unable to consume range read: "
                            "%d - %d -- n_retries = %d" %
                            (start_id, end_id, n_retries))

        return range_read.rows

    def range_read(self, start_id: np.uint64, end_id: np.uint64,
                   n_retries: int = 100, max_block_size: int = 50000,
                   row_keys: Optional[Iterable[str]] = None,
                   time_filter: TimestampRangeFilter = None,
                   default_filters: Optional[Iterable[RowFilter]] = [],
                   time_stamp: datetime.datetime = get_max_time()
                   ) -> Union[
                          bigtable.row_data.PartialRowData,
                          Dict[bytes, bigtable.row_data.PartialRowData]]:
        """ Reads all ids within a given range

        :param start_id: np.uint64
        :param end_id: np.uint64
        :param n_retries: int
        :param max_block_size: int
        :param row_keys: list of str
            more efficient read through row filters
        :param time_filter: TimestampRangeFilter
        :param default_filters: list of RowFilters or None
        :param time_stamp: datetime.datetime
        :return: dict
        """

        if time_filter is None:
            # Comply to resolution of BigTables TimeRange
            time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                          round_up=False)

            # Create filters: time and id range
            time_filter = get_inclusive_time_range_filter(end=time_stamp)

        if row_keys is not None:
            row_key_filters = []
            for k in row_keys:
                row_key_filters.append(ColumnQualifierRegexFilter(
                    key_utils.serialize_key(k)))

            if len(row_key_filters) > 1:
                row_key_filter = [RowFilterUnion(row_key_filters)]
            else:
                row_key_filter = [row_key_filters[0]]
        else:
            row_key_filter = []

        filter_list = default_filters + [time_filter] + row_key_filter

        if len(filter_list) > 1:
            row_filter = RowFilterChain(filter_list)
        else:
            row_filter = filter_list[0]

        max_block_size = np.uint64(max_block_size)

        row_dict = {}
        block_start_id = start_id
        while block_start_id < end_id + np.uint64(1):
            block_end_id = np.uint64(block_start_id + max_block_size)
            if block_end_id > end_id:
                block_end_id = end_id

            block_row_dict = self._range_read_execution(start_id=block_start_id,
                                                        end_id=block_end_id,
                                                        row_filter=row_filter,
                                                        n_retries=n_retries)

            row_dict.update(block_row_dict)

            block_start_id += max_block_size

        return row_dict

    def range_read_chunk(self, layer: int = None, x: int = None, y: int = None,
                         z: int = None, chunk_id: np.uint64 = None,
                         n_retries: int = 100, max_block_size: int = 1000000,
                         row_keys: Optional[Iterable[str]] = None,
                         time_stamp: datetime.datetime = get_max_time(),
                         ) -> Union[
                                bigtable.row_data.PartialRowData,
                                Dict[bytes, bigtable.row_data.PartialRowData]]:
        """ Reads all ids within a chunk

        :param layer: int
        :param x: int
        :param y: int
        :param z: int
        :param chunk_id: np.uint64
        :param n_retries: int
        :param max_block_size: int
        :param row_keys: list of str
            more efficient read through row filters
        :param time_stamp: datetime.datetime
        :return: dict
        """
        if chunk_id is not None:
            x, y, z = self.get_chunk_coordinates(chunk_id)
            layer = self.get_chunk_layer(chunk_id)
        elif x is not None and y is not None and z is not None:
            chunk_id = self.get_chunk_id(layer=layer, x=x, y=y, z=z)
        else:
            raise Exception("Either chunk_id or coordinates have to be defined")

        if layer == 1:
            max_segment_id = self.get_segment_id_limit(chunk_id)
            max_block_size = max_segment_id + 1
        else:
            max_segment_id = self.get_max_seg_id(chunk_id=chunk_id)

        # Define BigTable keys
        start_id = self.get_node_id(np.uint64(0), chunk_id=chunk_id)
        end_id = self.get_node_id(max_segment_id, chunk_id=chunk_id)

        try:
            rr = self.range_read(start_id, end_id, n_retries=n_retries,
                                 max_block_size=max_block_size,
                                 row_keys=row_keys,
                                 # row_key_filters=row_key_filters,
                                 time_stamp=time_stamp)
        except:
            raise Exception("Unable to consume range read: "
                            "[%d, %d, %d], l = %d, n_retries = %d" %
                            (x, y, z, layer, n_retries))
        return rr

    def range_read_operations(self,
                              time_start: datetime.datetime = datetime.datetime.min,
                              time_end: datetime.datetime = None,
                              start_id: np.uint64 = 0,
                              end_id: np.uint64 = None,
                              n_retries: int = 100,
                              row_keys: Optional[Iterable[str]] = None
                              ) -> Dict[bytes, bigtable.row_data.PartialRowData]:
        """ Reads all ids within a chunk

        :param time_start: datetime
        :param time_end: datetime
        :param start_id: uint64
        :param end_id: uint64
        :param n_retries: int
        :param row_keys: list of str
            more efficient read through row filters
        :return: list or yield of rows
        """

        # Set defaults
        if end_id is None:
            end_id = self.get_max_operation_id()

        if time_end is None:
            time_end = datetime.datetime.utcnow()

        if end_id < start_id:
            return {}

        # Comply to resolution of BigTables TimeRange
        time_start -= datetime.timedelta(
            microseconds=time_start.microsecond % 1000)

        time_end -= datetime.timedelta(
            microseconds=time_end.microsecond % 1000)

        # Create filters: time and id range
        time_filter = TimestampRangeFilter(TimestampRange(start=time_start,
                                                          end=time_end))

        if row_keys is not None:
            filters = []
            for k in row_keys:
                filters.append(ColumnQualifierRegexFilter(key_utils.serialize_key(k)))

            if len(filters) > 1:
                row_filter = RowFilterUnion(filters)
            else:
                row_filter = filters[0]
        else:
            row_filter = None

        if row_filter is None:
            row_filter = time_filter
        else:
            row_filter = RowFilterChain([time_filter, row_filter])

        # Set up read
        range_read = self.table.read_rows(
            start_key=key_utils.serialize_uint64(start_id),
            end_key=key_utils.serialize_uint64(end_id),
            end_inclusive=False,
            filter_=row_filter)
        range_read.consume_all()

        # Execute read
        consume_success = False

        # Retry reading if any of the writes failed
        i_tries = 0
        while not consume_success and i_tries < n_retries:
            try:
                range_read.consume_all()
                consume_success = True
            except:
                time.sleep(i_tries)
            i_tries += 1

        if not consume_success:
            raise Exception("Unable to consume chunk range read: "
                            "n_retries = %d" % (n_retries))

        return range_read.rows

    def range_read_layer(self, layer_id: int):
        """ Reads all ids within a layer

        This can take a while depending on the size of the graph

        :param layer_id: int
        :return: list of rows
        """
        raise NotImplementedError()

    def test_if_nodes_are_in_same_chunk(self, node_ids: Sequence[np.uint64]
                                        ) -> bool:
        """ Test whether two nodes are in the same chunk

        :param node_ids: list of two ints
        :return: bool
        """
        assert len(node_ids) == 2
        return self.get_chunk_id(node_id=node_ids[0]) == \
            self.get_chunk_id(node_id=node_ids[1])

    def get_chunk_id_from_coord(self, layer: int,
                                x: int, y: int, z: int) -> np.uint64:
        """ Return ChunkID for given chunked graph layer and voxel coordinates.

        :param layer: int -- ChunkedGraph layer
        :param x: int -- X coordinate in voxel
        :param y: int -- Y coordinate in voxel
        :param z: int -- Z coordinate in voxel
        :return: np.uint64 -- ChunkID
        """
        base_chunk_span = int(self.fan_out) ** max(0, layer - 2)

        return self.get_chunk_id(
            layer=layer,
            x=x // (int(self.chunk_size[0]) * base_chunk_span),
            y=y // (int(self.chunk_size[1]) * base_chunk_span),
            z=z // (int(self.chunk_size[2]) * base_chunk_span))

    def get_atomic_id_from_coord(self, x: int, y: int, z: int,
                                 parent_id: np.uint64, n_tries: int=5
                                 ) -> np.uint64:
        """ Determines atomic id given a coordinate

        :param x: int
        :param y: int
        :param z: int
        :param parent_id: np.uint64
        :param n_tries: int
        :return: np.uint64 or None
        """
        if self.get_chunk_layer(parent_id) == 1:
            return parent_id


        x /= 2**self.cv_mip
        y /= 2**self.cv_mip

        x = int(x)
        y = int(y)

        checked = []
        atomic_id = None
        root_id = self.get_root(parent_id)

        for i_try in range(n_tries):

            # Define block size -- increase by one each try
            x_l = x - (i_try - 1)**2
            y_l = y - (i_try - 1)**2
            z_l = z - (i_try - 1)**2

            x_h = x + 1 + (i_try - 1)**2
            y_h = y + 1 + (i_try - 1)**2
            z_h = z + 1 + (i_try - 1)**2

            if x_l < 0:
                x_l = 0

            if y_l < 0:
                y_l = 0

            if z_l < 0:
                z_l = 0

            # Get atomic ids from cloudvolume
            atomic_id_block = self.cv[x_l: x_h, y_l: y_h, z_l: z_h]
            atomic_ids, atomic_id_count = np.unique(atomic_id_block,
                                                    return_counts=True)

            # sort by frequency and discard those ids that have been checked
            # previously
            sorted_atomic_ids = atomic_ids[np.argsort(atomic_id_count)]
            sorted_atomic_ids = sorted_atomic_ids[~np.in1d(sorted_atomic_ids,
                                                           checked)]

            # For each candidate id check whether its root id corresponds to the
            # given root id
            for candidate_atomic_id in sorted_atomic_ids:
                ass_root_id = self.get_root(candidate_atomic_id)

                if ass_root_id == root_id:
                    # atomic_id is not None will be our indicator that the
                    # search was successful

                    atomic_id = candidate_atomic_id
                    break
                else:
                    checked.append(candidate_atomic_id)

            if atomic_id is not None:
                break

        # Returns None if unsuccessful
        return atomic_id

    def _create_split_log_row(self, operation_id: np.uint64, user_id: str,
                              root_ids: Sequence[np.uint64],
                              source_ids: Sequence[np.uint64],
                              sink_ids: Sequence[np.uint64],
                              source_coords: Sequence[Sequence[np.int]],
                              sink_coords: Sequence[Sequence[np.int]],
                              removed_edges: Sequence[np.uint64],
                              bb_offset: Sequence[Sequence[np.int]],
                              time_stamp: datetime.datetime
                              ) -> bigtable.row.Row:
        """ Creates log row for a split

        :param operation_id: np.uint64
        :param user_id: str
        :param root_ids: array of np.uint64
        :param source_ids: array of np.uint64
        :param sink_ids: array of np.uint64
        :param source_coords: array of ints (n x 3)
        :param sink_coords: array of ints (n x 3)
        :param removed_edges: array of np.uint64
        :param bb_offset: array of np.int
        :param time_stamp: datetime.datetime
        :return: row
        """

        val_dict = {key_utils.serialize_key("user"): key_utils.serialize_key(user_id),
                    key_utils.serialize_key("roots"):
                        np.array(root_ids, dtype=np.uint64).tobytes(),
                    key_utils.serialize_key("source_ids"):
                        np.array(source_ids).tobytes(),
                    key_utils.serialize_key("sink_ids"):
                        np.array(sink_ids).tobytes(),
                    key_utils.serialize_key("source_coords"):
                        np.array(source_coords).tobytes(),
                    key_utils.serialize_key("sink_coords"):
                        np.array(sink_coords).tobytes(),
                    key_utils.serialize_key("bb_offset"):
                        np.array(bb_offset).tobytes(),
                    key_utils.serialize_key("removed_edges"):
                        np.array(removed_edges, dtype=np.uint64).tobytes()}

        row = self.mutate_row(key_utils.serialize_uint64(operation_id),
                              self.log_family_id, val_dict, time_stamp)

        return row

    def _create_merge_log_row(self, operation_id: np.uint64, user_id: str,
                              root_ids: Sequence[np.uint64],
                              source_ids: Sequence[np.uint64],
                              sink_ids: Sequence[np.uint64],
                              source_coords: Sequence[Sequence[np.int]],
                              sink_coords: Sequence[Sequence[np.int]],
                              added_edges: Sequence[np.uint64],
                              affinities: Sequence[np.float32],
                              time_stamp: datetime.datetime
                              ) -> bigtable.row.Row:
        """ Creates log row for a split

        :param operation_id: np.uint64
        :param user_id: str
        :param root_ids: array of np.uint64
        :param source_ids: array of np.uint64
        :param sink_ids: array of np.uint64
        :param source_coords: array of ints (n x 3)
        :param sink_coords: array of ints (n x 3)
        :param added_edges: array of np.uint64
        :param affinities: array of np.float32
        :param time_stamp: datetime.datetime
        :return: row
        """

        if affinities is None:
            affinities_b = b''
        else:
            affinities_b = np.array(affinities, dtype=np.float32).tobytes()

        val_dict = {key_utils.serialize_key("user"): key_utils.serialize_key(user_id),
                    key_utils.serialize_key("roots"):
                        np.array(root_ids, dtype=np.uint64).tobytes(),
                    key_utils.serialize_key("source_ids"):
                        np.array(source_ids).tobytes(),
                    key_utils.serialize_key("sink_ids"):
                        np.array(sink_ids).tobytes(),
                    key_utils.serialize_key("source_coords"):
                        np.array(source_coords).tobytes(),
                    key_utils.serialize_key("sink_coords"):
                        np.array(sink_coords).tobytes(),
                    key_utils.serialize_key("added_edges"):
                        np.array(added_edges, dtype=np.uint64).tobytes(),
                    key_utils.serialize_key("affinities"):
                        affinities_b}

        row = self.mutate_row(key_utils.serialize_uint64(operation_id),
                              self.log_family_id, val_dict, time_stamp)

        return row

    def read_log_row(self, operation_id: np.uint64):
        """ Reads a log row (both split and merge)

        :param operation_id: np.uint64
        :return: dict
        """
        keys = ["user", "roots", "sink_ids",
                "source_ids", "source_coords",
                "sink_coords", "added_edges",
                "affinities", "removed_edges",
                "bb_offset"]
        row_dict = self.read_row_multi_key(operation_id, keys=keys)
        return row_dict


    def add_atomic_edges_in_chunks(self, edge_id_dict: dict,
                                   edge_aff_dict: dict, edge_area_dict: dict,
                                   isolated_node_ids: Sequence[np.uint64],
                                   verbose: bool = True,
                                   time_stamp: Optional[datetime.datetime] = None):
        """ Creates atomic nodes in first abstraction layer for a SINGLE chunk
            and all abstract nodes in the second for the same chunk

        Alle edges (edge_ids) need to be from one chunk and no nodes should
        exist for this chunk prior to calling this function. All cross edges
        (cross_edge_ids) have to point out the chunk (first entry is the id
        within the chunk)

        :param edge_id_dict: dict
        :param edge_aff_dict: dict
        :param edge_area_dict: dict
        :param isolated_node_ids: list of uint64s
            ids of nodes that have no edge in the chunked graph
        :param verbose: bool
        :param time_stamp: datetime
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        edge_id_keys = ["in_connected", "in_disconnected", "cross",
                        "between_connected", "between_disconnected"]
        edge_aff_keys = ["in_connected", "in_disconnected", "between_connected",
                         "between_disconnected"]

        # Check if keys exist and include an empty array if not
        n_edge_ids = 0
        chunk_id = None
        for edge_id_key in edge_id_keys:
            if not edge_id_key in edge_id_dict:
                empty_edges = np.array([], dtype=np.uint64).reshape(0, 2)
                edge_id_dict[edge_id_key] = empty_edges
            else:
                n_edge_ids += len(edge_id_dict[edge_id_key])

                if len(edge_id_dict[edge_id_key]) > 0:
                    node_id = edge_id_dict[edge_id_key][0, 0]
                    chunk_id = self.get_chunk_id(node_id)

        for edge_aff_key in edge_aff_keys:
            if not edge_aff_key in edge_aff_dict:
                edge_aff_dict[edge_aff_key] = np.array([], dtype=np.float32)

        time_start = time.time()

        # Catch trivial case
        if n_edge_ids == 0 and len(isolated_node_ids) == 0:
            return 0

        # Make parent id creation easier
        if chunk_id is None:
            chunk_id = self.get_chunk_id(isolated_node_ids[0])

        chunk_id_c = self.get_chunk_coordinates(chunk_id)
        parent_chunk_id = self.get_chunk_id(layer=2, x=chunk_id_c[0],
                                            y=chunk_id_c[1], z=chunk_id_c[2])

        # Get connected component within the chunk
        chunk_node_ids = np.concatenate([
                isolated_node_ids,
                np.unique(edge_id_dict["in_connected"]),
                np.unique(edge_id_dict["in_disconnected"]),
                np.unique(edge_id_dict["cross"][:, 0]),
                np.unique(edge_id_dict["between_connected"][:, 0]),
                np.unique(edge_id_dict["between_disconnected"][:, 0])])

        chunk_node_ids = np.unique(chunk_node_ids)

        node_chunk_ids = np.array([self.get_chunk_id(c)
                                   for c in chunk_node_ids],
                                  dtype=np.uint64)

        u_node_chunk_ids, c_node_chunk_ids = np.unique(node_chunk_ids,
                                                       return_counts=True)
        if len(u_node_chunk_ids) > 1:
            raise Exception("%d: %d chunk ids found in node id list. "
                            "Some edges might be in the wrong order. "
                            "Number of occurences:" %
                            (chunk_id, len(u_node_chunk_ids)), c_node_chunk_ids)

        chunk_g = nx.Graph()
        chunk_g.add_nodes_from(chunk_node_ids)
        chunk_g.add_edges_from(edge_id_dict["in_connected"])

        ccs = list(nx.connected_components(chunk_g))

        # if verbose:
        #     print("CC in chunk: %.3fs" % (time.time() - time_start))

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the print / speed measurement

        n_ccs = len(ccs)

        parent_ids = self.get_unique_node_id_range(parent_chunk_id, step=n_ccs)
        time_start = time.time()

        time_dict = collections.defaultdict(list)

        time_start_1 = time.time()
        sparse_indices = {}
        remapping = {}
        for k in edge_id_dict.keys():
            # Circumvent datatype issues

            u_ids, inv_ids = np.unique(edge_id_dict[k], return_inverse=True)
            mapped_ids = np.arange(len(u_ids), dtype=np.int32)
            remapped_arr = mapped_ids[inv_ids].reshape(edge_id_dict[k].shape)

            sparse_indices[k] = compute_indices_pandas(remapped_arr)
            remapping[k] = dict(zip(u_ids, mapped_ids))

        time_dict["sparse_indices"].append(time.time() - time_start_1)

        rows = []

        for i_cc, cc in enumerate(ccs):
            # if node_c % 1000 == 1 and verbose:
            #     dt = time.time() - time_start
            #     print("%5d at %5d - %.5fs             " %
            #           (i_cc, node_c, dt / node_c), end="\r")

            node_ids = np.array(list(cc))

            u_chunk_ids = np.unique([self.get_chunk_id(n) for n in node_ids])

            if len(u_chunk_ids) > 1:
                print("Found multiple chunk ids:", u_chunk_ids)
                raise Exception()

            # Create parent id
            parent_id = parent_ids[i_cc]
            parent_id_b = np.array(parent_id, dtype=np.uint64).tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # Extract edges relevant to this node

                # in chunk + connected
                time_start_2 = time.time()
                if node_id in remapping["in_connected"]:
                    row_ids, column_ids = sparse_indices["in_connected"][remapping["in_connected"][node_id]]

                    inv_column_ids = (column_ids + 1) % 2

                    connected_ids = edge_id_dict["in_connected"][row_ids, inv_column_ids]
                    connected_affs = edge_aff_dict["in_connected"][row_ids]
                    connected_areas = edge_area_dict["in_connected"][row_ids]
                    time_dict["in_connected"].append(time.time() - time_start_2)
                    time_start_2 = time.time()
                else:
                    connected_ids = np.array([], dtype=np.uint64)
                    connected_affs = np.array([], dtype=np.float32)
                    connected_areas = np.array([], dtype=np.uint64)

                # in chunk + disconnected
                if node_id in remapping["in_disconnected"]:
                    row_ids, column_ids = sparse_indices["in_disconnected"][remapping["in_disconnected"][node_id]]
                    inv_column_ids = (column_ids + 1) % 2

                    disconnected_ids = edge_id_dict["in_disconnected"][row_ids, inv_column_ids]
                    disconnected_affs = edge_aff_dict["in_disconnected"][row_ids]
                    disconnected_areas = edge_area_dict["in_disconnected"][row_ids]
                    time_dict["in_disconnected"].append(time.time() - time_start_2)
                    time_start_2 = time.time()
                else:
                    disconnected_ids = np.array([], dtype=np.uint64)
                    disconnected_affs = np.array([], dtype=np.float32)
                    disconnected_areas = np.array([], dtype=np.uint64)

                # out chunk + connected
                if node_id in remapping["between_connected"]:
                    row_ids, column_ids = sparse_indices["between_connected"][remapping["between_connected"][node_id]]

                    row_ids = row_ids[column_ids == 0]
                    column_ids = column_ids[column_ids == 0]
                    inv_column_ids = (column_ids + 1) % 2
                    time_dict["out_connected_mask"].append(time.time() - time_start_2)
                    time_start_2 = time.time()

                    connected_ids = np.concatenate([connected_ids, edge_id_dict["between_connected"][row_ids, inv_column_ids]])
                    connected_affs = np.concatenate([connected_affs, edge_aff_dict["between_connected"][row_ids]])
                    connected_areas = np.concatenate([connected_areas, edge_area_dict["between_connected"][row_ids]])

                    parent_cross_edges = np.concatenate([parent_cross_edges, edge_id_dict["between_connected"][row_ids]])

                    time_dict["out_connected"].append(time.time() - time_start_2)
                    time_start_2 = time.time()

                # out chunk + disconnected
                if node_id in remapping["between_disconnected"]:
                    row_ids, column_ids = sparse_indices["between_disconnected"][remapping["between_disconnected"][node_id]]

                    row_ids = row_ids[column_ids == 0]
                    column_ids = column_ids[column_ids == 0]
                    inv_column_ids = (column_ids + 1) % 2
                    time_dict["out_disconnected_mask"].append(time.time() - time_start_2)
                    time_start_2 = time.time()

                    disconnected_ids = np.concatenate([disconnected_ids, edge_id_dict["between_disconnected"][row_ids, inv_column_ids]])
                    disconnected_affs = np.concatenate([disconnected_affs, edge_aff_dict["between_disconnected"][row_ids]])
                    disconnected_areas = np.concatenate([disconnected_areas, edge_area_dict["between_disconnected"][row_ids]])

                    time_dict["out_disconnected"].append(time.time() - time_start_2)
                    time_start_2 = time.time()

                # cross
                if node_id in remapping["cross"]:
                    row_ids, column_ids = sparse_indices["cross"][remapping["cross"][node_id]]

                    row_ids = row_ids[column_ids == 0]
                    column_ids = column_ids[column_ids == 0]
                    inv_column_ids = (column_ids + 1) % 2
                    time_dict["cross_mask"].append(time.time() - time_start_2)
                    time_start_2 = time.time()

                    connected_ids = np.concatenate([connected_ids, edge_id_dict["cross"][row_ids, inv_column_ids]])
                    connected_affs = np.concatenate([connected_affs, np.full((len(row_ids)), np.inf, dtype=np.float32)])
                    connected_areas = np.concatenate([connected_areas, np.ones((len(row_ids)), dtype=np.uint64)])

                    parent_cross_edges = np.concatenate([parent_cross_edges, edge_id_dict["cross"][row_ids]])
                    time_dict["cross"].append(time.time() - time_start_2)
                    time_start_2 = time.time()

                # Create node
                partners = np.concatenate([connected_ids, disconnected_ids])
                partners_b = partners.tobytes()
                affinities = np.concatenate([connected_affs, disconnected_affs])
                affinities_b = affinities.tobytes()
                areas = np.concatenate([connected_areas, disconnected_areas])
                areas_b = areas.tobytes()
                connected = np.arange(len(connected_ids), dtype=np.int)
                connected_b = connected.tobytes()

                val_dict = {table_info.partner_key: partners_b,
                            table_info.affinity_key: affinities_b,
                            table_info.area_key: areas_b,
                            table_info.connected_key: connected_b,
                            table_info.parent_key: parent_id_b}

                rows.append(self.mutate_row(key_utils.serialize_uint64(node_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))
                node_c += 1
                time_dict["creating_lv1_row"].append(time.time() - time_start_2)

            time_start_1 = time.time()
            # Create parent node
            rows.append(self.mutate_row(key_utils.serialize_uint64(parent_id),
                                        self.family_id,
                                        {"children": node_ids.tobytes()},
                                        time_stamp=time_stamp))

            time_dict["creating_lv2_row"].append(time.time() - time_start_1)
            time_start_1 = time.time()

            cce_layers = self.get_cross_chunk_edges_layer(parent_cross_edges)
            u_cce_layers = np.unique(cce_layers)

            val_dict = {}
            for cc_layer in u_cce_layers:
                layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

                if len(layer_cross_edges) > 0:
                    val_dict[table_info.cross_chunk_edge_keyformat % cc_layer] = \
                        layer_cross_edges.tobytes()

            if len(val_dict) > 0:
                rows.append(self.mutate_row(key_utils.serialize_uint64(parent_id),
                                            self.cross_edge_family_id, val_dict,
                                            time_stamp=time_stamp))
            node_c += 1

            time_dict["adding_cross_edges"].append(time.time() - time_start_1)

            if len(rows) > 100000:
                time_start_1 = time.time()
                self.bulk_write(rows)
                time_dict["writing"].append(time.time() - time_start_1)

        if len(rows) > 0:
            time_start_1 = time.time()
            self.bulk_write(rows)
            time_dict["writing"].append(time.time() - time_start_1)

        if verbose:
            print("Time creating rows: %.3fs for %d ccs with %d nodes" %
                  (time.time() - time_start, len(ccs), node_c))

            for k in time_dict.keys():
                print("%s -- %.3fms for %d instances -- avg = %.3fms" %
                      (k, np.sum(time_dict[k])*1000, len(time_dict[k]),
                       np.mean(time_dict[k])*1000))

    def add_layer(self, layer_id: int,
                  child_chunk_coords: Sequence[Sequence[int]],
                  time_stamp: Optional[datetime.datetime] = None,
                  verbose: bool = True, n_threads: int = 20) -> None:
        """ Creates the abstract nodes for a given chunk in a given layer

        :param layer_id: int
        :param child_chunk_coords: int array of length 3
            coords in chunk space
        :param time_stamp: datetime
        :param verbose: bool
        :param n_threads: int
        """
        def _read_subchunks_thread(chunk_coord):
            # Get start and end key
            x, y, z = chunk_coord

            row_keys = ["children"] + \
                       [table_info.cross_chunk_edge_keyformat % l
                        for l in range(layer_id - 1, self.n_layers)]
            range_read = self.range_read_chunk(layer_id - 1, x, y, z,
                                               row_keys=row_keys)

            # Due to restarted jobs some parents might be duplicated. We can
            # find these duplicates only by comparing their children because
            # each node has a unique id. However, we can use that more recently
            # created nodes have higher segment ids. We are only interested in
            # the latest version of any duplicated parents.

            # Deserialize row keys and store child with highest id for
            # comparison
            row_cell_dict = {}
            segment_ids = []
            row_ids = []
            max_child_ids = []
            for row_id_b, row_data in range_read.items():
                row_id = key_utils.deserialize_uint64(row_id_b)

                segment_id = self.get_segment_id(row_id)

                cells = row_data.cells

                cell_family = cells[self.family_id]

                if self.cross_edge_family_id in cells:
                    row_cell_dict[row_id] = cells[self.cross_edge_family_id]

                node_child_ids_b = cell_family[children_key][0].value
                node_child_ids = np.frombuffer(node_child_ids_b,
                                               dtype=np.uint64)

                max_child_ids.append(np.max(node_child_ids))
                segment_ids.append(segment_id)
                row_ids.append(row_id)

            segment_ids = np.array(segment_ids, dtype=np.uint64)
            row_ids = np.array(row_ids)
            max_child_ids = np.array(max_child_ids, dtype=np.uint64)

            sorting = np.argsort(segment_ids)[::-1]
            row_ids = row_ids[sorting]
            max_child_ids = max_child_ids[sorting]

            counter = collections.defaultdict(int)
            max_child_ids_occ_so_far = np.zeros(len(max_child_ids),
                                                dtype=np.int)
            for i_row in range(len(max_child_ids)):
                max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
                counter[max_child_ids[i_row]] += 1

            # Filter last occurences (we inverted the list) of each node
            m = max_child_ids_occ_so_far == 0
            row_ids = row_ids[m]
            ll_node_ids.extend(row_ids)

            # Loop through nodes from this chunk
            for row_id in row_ids:
                if row_id in row_cell_dict:
                    cross_edge_dict[row_id] = {}

                    cell_family = row_cell_dict[row_id]

                    for l in range(layer_id - 1, self.n_layers):
                        row_key = key_utils.serialize_key(table_info.cross_chunk_edge_keyformat % l)
                        if row_key in cell_family:
                            cross_edge_dict[row_id][l] = cell_family[row_key][0].value

                    if int(layer_id - 1) in cross_edge_dict[row_id]:
                        atomic_cross_edges_b = cross_edge_dict[row_id][layer_id - 1]
                        atomic_cross_edges = \
                            np.frombuffer(atomic_cross_edges_b,
                                          dtype=np.uint64).reshape(-1, 2)

                        if len(atomic_cross_edges) > 0:
                            atomic_partner_id_dict[row_id] = \
                                atomic_cross_edges[:, 1]

                            new_pairs = zip(atomic_cross_edges[:, 0],
                                            [row_id] * len(atomic_cross_edges))
                            atomic_child_id_dict_pairs.extend(new_pairs)

        def _resolve_cross_chunk_edges_thread(args) -> None:
            start, end = args

            for i_child_key, child_key in\
                    enumerate(atomic_partner_id_dict_keys[start: end]):
                this_atomic_partner_ids = atomic_partner_id_dict[child_key]

                partners = {atomic_child_id_dict[atomic_cross_id]
                            for atomic_cross_id in this_atomic_partner_ids
                            if atomic_child_id_dict[atomic_cross_id] != 0}

                if len(partners) > 0:
                    partners = np.array(list(partners), dtype=np.uint64)[:, None]

                    this_ids = np.array([child_key] * len(partners),
                                        dtype=np.uint64)[:, None]
                    these_edges = np.concatenate([this_ids, partners], axis=1)

                    edge_ids.extend(these_edges)

        def _write_out_connected_components(args) -> None:
            start, end = args

            n_ccs = int(end - start)
            parent_ids = self.get_unique_node_id_range(chunk_id, step=n_ccs)
            rows = []
            for i_cc, cc in enumerate(ccs[start: end]):
                node_ids = np.array(list(cc))

                parent_id = parent_ids[i_cc]
                parent_id_b = np.array(parent_id, dtype=np.uint64).tobytes()

                parent_cross_edges_b = {}
                for l in range(layer_id, self.n_layers):
                    parent_cross_edges_b[l] = b""

                # Add rows for nodes that are in this chunk
                for i_node_id, node_id in enumerate(node_ids):

                    if node_id in cross_edge_dict:
                        # Extract edges relevant to this node
                        for l in range(layer_id, self.n_layers):
                            if l in cross_edge_dict[node_id]:
                                parent_cross_edges_b[l] += \
                                    cross_edge_dict[node_id][l]

                    # Create node
                    val_dict = {"parents": parent_id_b}

                    rows.append(self.mutate_row(key_utils.serialize_uint64(node_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                # Create parent node
                val_dict = {"children": node_ids.tobytes()}

                rows.append(self.mutate_row(key_utils.serialize_uint64(parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

                val_dict = {}

                for l in range(layer_id, self.n_layers):
                    if l in parent_cross_edges_b:
                        val_dict[table_info.cross_chunk_edge_keyformat % l] = \
                            parent_cross_edges_b[l]

                if len(val_dict) > 0:
                    rows.append(self.mutate_row(key_utils.serialize_uint64(parent_id),
                                                self.cross_edge_family_id,
                                                val_dict,
                                                time_stamp=time_stamp))

                if len(rows) > 100000:
                    self.bulk_write(rows)
                    rows = []

            if len(rows) > 0:
                self.bulk_write(rows)

        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        # 1 --------------------------------------------------------------------
        # The first part is concerned with reading data from the child nodes
        # of this layer and pre-processing it for the second part

        time_start = time.time()

        children_key = key_utils.serialize_key("children")
        atomic_partner_id_dict = {}
        cross_edge_dict = {}
        atomic_child_id_dict_pairs = []
        ll_node_ids = []

        multi_args = child_chunk_coords
        n_jobs = np.min([n_threads, len(multi_args)])

        if n_jobs > 0:
            mu.multithread_func(_read_subchunks_thread, multi_args,
                                n_threads=n_jobs)

        d = dict(atomic_child_id_dict_pairs)
        atomic_child_id_dict = collections.defaultdict(np.uint64, d)
        ll_node_ids = np.array(ll_node_ids, dtype=np.uint64)

        if verbose:
            print("Time iterating through subchunks: %.3fs" %
                  (time.time() - time_start))
        time_start = time.time()

        # Extract edges from remaining cross chunk edges
        # and maintain unused cross chunk edges
        edge_ids = []
        # u_atomic_child_ids = np.unique(atomic_child_ids)
        atomic_partner_id_dict_keys = \
            np.array(list(atomic_partner_id_dict.keys()), dtype=np.uint64)

        if n_threads > 1:
            n_jobs = n_threads * 3 # Heuristic
        else:
            n_jobs = 1

        n_jobs = np.min([n_jobs, len(atomic_partner_id_dict_keys)])

        if n_jobs > 0:
            spacing = np.linspace(0, len(atomic_partner_id_dict_keys),
                                  n_jobs+1).astype(np.int)
            starts = spacing[:-1]
            ends = spacing[1:]

            multi_args = list(zip(starts, ends))

            mu.multithread_func(_resolve_cross_chunk_edges_thread, multi_args,
                                n_threads=n_threads)

        if verbose:
            print("Time resolving cross chunk edges: %.3fs" %
                  (time.time() - time_start))
        time_start = time.time()

        # 2 --------------------------------------------------------------------
        # The second part finds connected components, writes the parents to
        # BigTable and updates the childs

        # Make parent id creation easier
        x, y, z = np.min(child_chunk_coords, axis=0) // self.fan_out

        chunk_id = self.get_chunk_id(layer=layer_id, x=x, y=y, z=z)

        # Extract connected components
        chunk_g = nx.from_edgelist(edge_ids)

        # Add single node objects that have no edges
        isolated_node_mask = ~np.in1d(ll_node_ids, np.unique(edge_ids))
        add_ccs = list(ll_node_ids[isolated_node_mask][:, None])

        ccs = list(nx.connected_components(chunk_g)) + add_ccs

        if verbose:
            print("Time connected components: %.3fs" %
                  (time.time() - time_start))
        time_start = time.time()

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        if n_threads > 1:
            n_jobs = n_threads * 3 # Heuristic
        else:
            n_jobs = 1

        n_jobs = np.min([n_jobs, len(ccs)])

        spacing = np.linspace(0, len(ccs), n_jobs+1).astype(np.int)
        starts = spacing[:-1]
        ends = spacing[1:]

        multi_args = list(zip(starts, ends))

        mu.multithread_func(_write_out_connected_components, multi_args,
                            n_threads=n_threads)

        if verbose:
            print("Time writing %d connected components in layer %d: %.3fs" %
                  (len(ccs), layer_id, time.time() - time_start))


    def get_atomic_cross_edge_dict(self, node_id: np.uint64,
                                   layer_ids: Sequence[int] = None,
                                   deserialize_node_ids: bool = False,
                                   reshape: bool = False):
        """ Extracts all atomic cross edges and serves them as a dictionary

        :param node_id: np.uitn64
        :param layer_ids: list of ints
        :param deserialize_node_ids: bool
        :param reshape: bool
            reshapes the list of node ids to an edge list (n x 2)
            Only available when deserializing
        :return: dict
        """
        row = self.table.read_row(key_utils.serialize_uint64(node_id))

        if row is None:
            return {}

        atomic_cross_edges = {}

        if isinstance(layer_ids, int):
            layer_ids = [layer_ids]

        if layer_ids is None:
            layer_ids = range(2, self.n_layers)

        if self.cross_edge_family_id in row.cells:
            for l in layer_ids:
                key = key_utils.serialize_key(table_info.cross_chunk_edge_keyformat % l)
                row_cell = row.cells[self.cross_edge_family_id]

                atomic_cross_edges[l] = []

                if key in row_cell:
                    row_val = row_cell[key][0].value

                    if deserialize_node_ids:
                        atomic_cross_edges[l] = np.frombuffer(row_val,
                                                              dtype=np.uint64)

                        if reshape:
                            atomic_cross_edges[l] = \
                                atomic_cross_edges[l].reshape(-1, 2)
                    else:
                        atomic_cross_edges[l] = row_val


        return atomic_cross_edges

    def get_parent(self, node_id: np.uint64,
                   get_only_relevant_parent: bool = True,
                   time_stamp: Optional[datetime.datetime] = None) -> Union[
                       List[Tuple[np.uint64, datetime.datetime]],
                       np.uint64, None]:
        """ Acquires parent of a node at a specific time stamp

        :param node_id: uint64
        :param get_only_relevant_parent: bool
            True: return single parent according to time_stamp
            False: return n x 2 list of all parents
                   ((parent_id, time_stamp), ...)
        :param time_stamp: datetime or None
        :return: uint64 or None
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        all_parents = []

        p_filter_ = ColumnQualifierRegexFilter(table_info.parent_key)
        row = self.table.read_row(key_utils.serialize_uint64(node_id), filter_=p_filter_)

        if row is not None:
            cells = row.cells[self.family_id]
            if table_info.parent_key_s in cells:
                for parent_entry in cells[table_info.parent_key_s]:
                    if get_only_relevant_parent:
                        if parent_entry.timestamp > time_stamp:
                            continue
                        else:
                            return np.frombuffer(parent_entry.value,
                                                 dtype=np.uint64)[0]
                    else:
                        all_parents.append((np.frombuffer(parent_entry.value,
                                                          dtype=np.uint64)[0],
                                            parent_entry.timestamp))
        else:
            return None

        if len(all_parents) == 0:
            raise Exception("Did not find a valid parent for %d with"
                            " the given time stamp" % node_id)
        else:
            return all_parents

    def get_children(self, node_id: np.uint64) -> np.ndarray:
        """ Returns all children of a node

        :param node_id: np.uint64
        :return: np.ndarray[np.uint64]
        """
        children = self.read_row(node_id, "children", dtype=np.uint64)

        if children is None:
            return np.empty(0, dtype=np.uint64)
        else:
            return children

    def get_latest_roots(self, time_stamp: Optional[datetime.datetime] = get_max_time(),
                         n_threads: int = 1) -> Sequence[np.uint64]:
        """ Reads _all_ root ids

        :param time_stamp: datetime.datetime
        :param n_threads: int
        :return: array of np.uint64
        """
        def _read_root_rows(args) -> None:
            start_seg_id, end_seg_id = args
            start_id = self.get_node_id(segment_id=start_seg_id,
                                        chunk_id=self.root_chunk_id)
            end_id = self.get_node_id(segment_id=end_seg_id,
                                      chunk_id=self.root_chunk_id)
            range_read = self.table.read_rows(
                start_key=key_utils.serialize_uint64(start_id),
                end_key=key_utils.serialize_uint64(end_id),
                # allow_row_interleaving=True,
                end_inclusive=False,
                filter_=time_filter)

            range_read.consume_all()
            rows = range_read.rows
            for row_id, row_data in rows.items():
                row_keys = row_data.cells[self.family_id]
                if not key_utils.serialize_key("new_parents") in row_keys:
                    root_ids.append(key_utils.deserialize_uint64(row_id))


        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        # Create filters: time and id range
        time_filter = get_inclusive_time_range_filter(end=time_stamp)

        max_seg_id = self.get_max_seg_id(self.root_chunk_id) + 1

        root_ids = []

        n_blocks = np.min([n_threads*3+1, max_seg_id])
        seg_id_blocks = np.linspace(1, max_seg_id, n_blocks, dtype=np.uint64)

        multi_args = []
        for i_id_block in range(0, len(seg_id_blocks) - 1):
            multi_args.append([seg_id_blocks[i_id_block],
                               seg_id_blocks[i_id_block + 1]])

        mu.multithread_func(
            _read_root_rows, multi_args, n_threads=n_threads,
            debug=False, verbose=True)

        return root_ids

    def get_root(self, node_id: np.uint64,
                 time_stamp: Optional[datetime.datetime] = None
                 ) -> Union[List[np.uint64], np.uint64]:
        """ Takes a node id and returns the associated agglomeration ids

        :param node_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        early_finish = True

        if self.get_chunk_layer(node_id) == self.n_layers:
            raise Exception("node is already root")

        parent_id = node_id

        while early_finish:
            parent_id = node_id

            early_finish = False

            for i_layer in range(self.get_chunk_layer(node_id)+1,
                                 int(self.n_layers + 1)):

                temp_parent_id = self.get_parent(parent_id, time_stamp=time_stamp)

                if temp_parent_id is None:
                    early_finish = True
                    break
                else:
                    parent_id = temp_parent_id

        return parent_id

    def get_all_parents(self, node_id: np.uint64,
                        time_stamp: Optional[datetime.datetime] = None
                        ) -> Union[List[np.uint64], np.uint64]:
        """ Takes a node id and returns all parents and parents' parents up to
            the top

        :param node_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        early_finish = True
        parent_ids: List[np.uint64] = []

        while early_finish:
            parent_id = node_id
            parent_ids = []

            early_finish = False

            for i_layer in range(self.get_chunk_layer(node_id)+1,
                                 int(self.n_layers + 1)):
                temp_parent_id = self.get_parent(parent_id,
                                                 time_stamp=time_stamp)

                if temp_parent_id is None:
                    early_finish = True
                    break
                else:
                    parent_id = temp_parent_id
                    parent_ids.append(parent_id)

        return parent_ids

    def lock_root_loop(self, root_ids: Sequence[np.uint64],
                       operation_id: np.uint64, max_tries: int = 1,
                       waittime_s: float = 0.5) -> Tuple[bool, np.ndarray]:
        """ Attempts to lock multiple roots at the same time

        :param root_ids: list of uint64
        :param operation_id: uint64
        :param max_tries: int
        :param waittime_s: float
        :return: bool, list of uint64s
            success, latest root ids
        """

        i_try = 0
        while i_try < max_tries:
            lock_acquired = False

            # Collect latest root ids
            new_root_ids: List[np.uint64] = []
            for i_root_id in range(len(root_ids)):
                future_root_ids = self.get_future_root_ids(root_ids[i_root_id])

                if len(future_root_ids) == 0:
                    new_root_ids.append(root_ids[i_root_id])
                else:
                    new_root_ids.extend(future_root_ids)

            # Attempt to lock all latest root ids
            root_ids = np.unique(new_root_ids)

            for i_root_id in range(len(root_ids)):

                print("operation id: %d - root id: %d" %
                      (operation_id, root_ids[i_root_id]))
                lock_acquired = self.lock_single_root(root_ids[i_root_id],
                                                      operation_id)

                # Roll back locks if one root cannot be locked
                if not lock_acquired:
                    for j_root_id in range(len(root_ids)):
                        self.unlock_root(root_ids[j_root_id], operation_id)
                    break

            if lock_acquired:
                return True, root_ids

            time.sleep(waittime_s)
            i_try += 1
            print("Try", i_try)

        return False, root_ids

    def lock_single_root(self, root_id: np.uint64, operation_id: np.uint64
                         ) -> bool:
        """ Attempts to lock the latest version of a root node

        :param root_id: uint64
        :param operation_id: uint64
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """

        operation_id_b = key_utils.serialize_uint64(operation_id)

        lock_key = key_utils.serialize_key("lock")
        new_parents_key = key_utils.serialize_key("new_parents")

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if it is still valid (timestamp younger than
        # LOCK_EXPIRED_TIME_DELTA) and if there is no new parent (== new_parents
        # exists)

        time_cutoff = datetime.datetime.utcnow() - LOCK_EXPIRED_TIME_DELTA

        # Comply to resolution of BigTables TimeRange
        time_cutoff -= datetime.timedelta(
            microseconds=time_cutoff.microsecond % 1000)

        time_filter = TimestampRangeFilter(TimestampRange(start=time_cutoff))

        # lock_key_filter = ColumnQualifierRegexFilter(lock_key)
        # new_parents_key_filter = ColumnQualifierRegexFilter(new_parents_key)

        lock_key_filter = ColumnRangeFilter(
            column_family_id=self.family_id,
            start_column=lock_key,
            end_column=lock_key,
            inclusive_start=True,
            inclusive_end=True)

        new_parents_key_filter = ColumnRangeFilter(
            column_family_id=self.family_id,
            start_column=new_parents_key,
            end_column=new_parents_key,
            inclusive_start=True,
            inclusive_end=True)

        # Combine filters together
        chained_filter = RowFilterChain([time_filter, lock_key_filter])
        combined_filter = ConditionalRowFilter(
            base_filter=chained_filter,
            true_filter=PassAllFilter(True),
            false_filter=new_parents_key_filter)

        # Get conditional row using the chained filter
        root_row = self.table.row(key_utils.serialize_uint64(root_id),
                                  filter_=combined_filter)

        # Set row lock if condition returns no results (state == False)
        time_stamp = datetime.datetime.utcnow()

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        root_row.set_cell(self.family_id, lock_key, operation_id_b, state=False,
                          timestamp=time_stamp)

        # The lock was acquired when set_cell returns False (state)
        lock_acquired = not root_row.commit()

        if not lock_acquired:
            r = self.table.read_row(key_utils.serialize_uint64(root_id))

            l_operation_ids = []
            for cell in r.cells[self.family_id][lock_key]:
                l_operation_id = key_utils.deserialize_uint64(cell.value)
                l_operation_ids.append(l_operation_id)
            print("Locked operation ids:", l_operation_ids)

        return lock_acquired

    def unlock_root(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        """ Unlocks a root

        This is mainly used for cases where multiple roots need to be locked and
        locking was not sucessful for all of them

        :param root_id: np.uint64
        :param operation_id: uint64
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """
        operation_id_b = key_utils.serialize_uint64(operation_id)

        lock_key = key_utils.serialize_key("lock")

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if it is still valid (timestamp younger than
        # LOCK_EXPIRED_TIME_DELTA) and if the given operation_id is still
        # the active lock holder

        time_cutoff = datetime.datetime.utcnow() - LOCK_EXPIRED_TIME_DELTA

        # Comply to resolution of BigTables TimeRange
        time_cutoff -= datetime.timedelta(
            microseconds=time_cutoff.microsecond % 1000)

        time_filter = TimestampRangeFilter(TimestampRange(start=time_cutoff))

        # column_key_filter = ColumnQualifierRegexFilter(lock_key)
        # value_filter = ColumnQualifierRegexFilter(operation_id_b)

        column_key_filter = ColumnRangeFilter(
            column_family_id=self.family_id,
            start_column=lock_key,
            end_column=lock_key,
            inclusive_start=True,
            inclusive_end=True)

        value_filter = ValueRangeFilter(
            start_value=operation_id_b,
            end_value=operation_id_b,
            inclusive_start=True,
            inclusive_end=True)

        # Chain these filters together
        chained_filter = RowFilterChain([time_filter, column_key_filter,
                                         value_filter])

        # Get conditional row using the chained filter
        root_row = self.table.row(key_utils.serialize_uint64(root_id),
                                  filter_=chained_filter)

        # Delete row if conditions are met (state == True)
        root_row.delete_cell(self.family_id, lock_key, state=True)

        return root_row.commit()

    def check_and_renew_root_locks(self, root_ids: Iterable[np.uint64],
                                   operation_id: np.uint64) -> bool:
        """ Tests if the roots are locked with the provided operation_id and
        renews the lock to reset the time_stam

        This is mainly used before executing a bulk write

        :param root_ids: uint64
        :param operation_id: uint64
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """

        for root_id in root_ids:
            if not self.check_and_renew_root_lock_single(root_id, operation_id):
                print("check_and_renew_root_locks failed - %d" % root_id)
                return False

        return True

    def check_and_renew_root_lock_single(self, root_id: np.uint64,
                                         operation_id: np.uint64) -> bool:
        """ Tests if the root is locked with the provided operation_id and
        renews the lock to reset the time_stam

        This is mainly used before executing a bulk write

        :param root_id: uint64
        :param operation_id: uint64
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """
        operation_id_b = key_utils.serialize_uint64(operation_id)

        lock_key = key_utils.serialize_key("lock")
        new_parents_key = key_utils.serialize_key("new_parents")

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if the given operation_id is still the active lock holder
        # and there is no new parent (== new_parents column exists). The latter
        # is not necessary but we include it as a backup to prevent things
        # from going really bad.

        # column_key_filter = ColumnQualifierRegexFilter(lock_key)
        # value_filter = ColumnQualifierRegexFilter(operation_id_b)

        column_key_filter = ColumnRangeFilter(
            column_family_id=self.family_id,
            start_column=lock_key,
            end_column=lock_key,
            inclusive_start=True,
            inclusive_end=True)

        value_filter = ValueRangeFilter(
            start_value=operation_id_b,
            end_value=operation_id_b,
            inclusive_start=True,
            inclusive_end=True)

        new_parents_key_filter = ColumnRangeFilter(
            column_family_id=self.family_id, start_column=new_parents_key,
            end_column=new_parents_key, inclusive_start=True,
            inclusive_end=True)

        # Chain these filters together
        chained_filter = RowFilterChain([column_key_filter, value_filter])
        combined_filter = ConditionalRowFilter(
            base_filter=chained_filter,
            true_filter=new_parents_key_filter,
            false_filter=PassAllFilter(True))

        # Get conditional row using the chained filter
        root_row = self.table.row(key_utils.serialize_uint64(root_id),
                                  filter_=combined_filter)

        # Set row lock if condition returns a result (state == True)
        root_row.set_cell(self.family_id, lock_key, operation_id_b, state=False)

        # The lock was acquired when set_cell returns True (state)
        lock_acquired = not root_row.commit()

        return lock_acquired

    def get_latest_root_id(self, root_id: np.uint64) -> np.ndarray:
        """ Returns the latest root id associated with the provided root id

        :param root_id: uint64
        :return: list of uint64s
        """

        id_working_set = [root_id]
        new_parent_key = key_utils.serialize_key("new_parents")
        latest_root_ids = []

        while len(id_working_set) > 0:

            next_id = id_working_set[0]
            del(id_working_set[0])
            r = self.table.read_row(key_utils.serialize_uint64(next_id))

            # Check if a new root id was attached to this root id
            if new_parent_key in r.cells[self.family_id]:
                id_working_set.extend(
                    np.frombuffer(
                        r.cells[self.family_id][new_parent_key][0].value,
                        dtype=np.uint64))
            else:
                latest_root_ids.append(next_id)

        return np.unique(latest_root_ids)

    def get_future_root_ids(self, root_id: np.uint64,
                            time_stamp: Optional[datetime.datetime] =
                            get_max_time())-> np.ndarray:
        """ Returns all future root ids emerging from this root

        This search happens in a monotic fashion. At no point are past root
        ids of future root ids taken into account.

        :param root_id: np.uint64
        :param time_stamp: None or datetime
            restrict search to ids created before this time_stamp
            None=search whole future
        :return: array of uint64
        """
        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        id_history = []

        next_ids = [root_id]
        while len(next_ids):
            temp_next_ids = []

            for next_id in next_ids:
                ids, row_time_stamp = self.read_row(next_id,
                                                    key="new_parents",
                                                    dtype=np.uint64,
                                                    get_time_stamp=True)
                if ids is None:
                    r, row_time_stamp = self.read_row(next_id,
                                                      key="children",
                                                      dtype=np.uint64,
                                                      get_time_stamp=True)

                    if row_time_stamp is None:
                        raise Exception("Something went wrong...")

                if row_time_stamp < time_stamp:
                    if ids is not None:
                        temp_next_ids.extend(ids)

                    if next_id != root_id:
                        id_history.append(next_id)

            next_ids = temp_next_ids

        return np.unique(np.array(id_history, dtype=np.uint64))

    def get_past_root_ids(self, root_id: np.uint64,
                          time_stamp: Optional[datetime.datetime] =
                          datetime.datetime.min) -> np.ndarray:
        """ Returns all future root ids emerging from this root

        This search happens in a monotic fashion. At no point are future root
        ids of past root ids taken into account.

        :param root_id: np.uint64
        :param time_stamp: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole future
        :return: array of uint64
        """
        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        id_history = []

        next_ids = [root_id]
        while len(next_ids):
            temp_next_ids = []

            for next_id in next_ids:
                ids, row_time_stamp = self.read_row(next_id,
                                                    key="former_parents",
                                                    dtype=np.uint64,
                                                    get_time_stamp=True)
                if ids is None:
                    _, row_time_stamp = self.read_row(next_id,
                                                      key="children",
                                                      dtype=np.uint64,
                                                      get_time_stamp=True)

                    if row_time_stamp is None:
                        raise Exception("Something went wrong...")

                if row_time_stamp > time_stamp:
                    if ids is not None:
                        temp_next_ids.extend(ids)

                    if next_id != root_id:
                        id_history.append(next_id)

            next_ids = temp_next_ids

        return np.unique(np.array(id_history, dtype=np.uint64))

    def get_root_id_history(self, root_id: np.uint64,
                            time_stamp_past:
                            Optional[datetime.datetime] = datetime.datetime.min,
                            time_stamp_future:
                            Optional[datetime.datetime] = get_max_time()
                            ) -> np.ndarray:
        """ Returns all future root ids emerging from this root

        This search happens in a monotic fashion. At no point are future root
        ids of past root ids or past root ids of future root ids taken into
        account.

        :param root_id: np.uint64
        :param time_stamp_past: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole future
        :param time_stamp_future: None or datetime
            restrict search to ids created before this time_stamp
            None=search whole future
        :return: array of uint64
        """
        past_ids = self.get_past_root_ids(root_id=root_id,
                                          time_stamp=time_stamp_past)
        future_ids = self.get_future_root_ids(root_id=root_id,
                                              time_stamp=time_stamp_future)

        history_ids = np.concatenate([past_ids,
                                      np.array([root_id], dtype=np.uint64),
                                      future_ids])

        return history_ids

    def get_subgraph(self, agglomeration_id: np.uint64,
                     bounding_box: Optional[Sequence[Sequence[int]]] = None,
                     bb_is_coordinate: bool = False,
                     stop_lvl: int = 1,
                     get_edges: bool = False, verbose: bool = True
                     ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """ Returns all edges between supervoxels belonging to the specified
            agglomeration id within the defined bouning box

        :param agglomeration_id: int
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param stop_lvl: int
        :param get_edges: bool
        :param verbose: bool
        :return: edge list
        """
        # Helper functions for multithreading
        def _handle_subgraph_children_layer2_edges_thread(
                child_ids: Iterable[np.uint64]) -> Tuple[List[np.ndarray],
                                                         List[np.float32]]:

            _edges = []
            _affinities = []
            _areas = []
            for child_id in child_ids:
                this_edges, this_affinities, this_areas = \
                    self.get_subgraph_chunk(child_id, time_stamp=time_stamp)
                _edges.extend(this_edges)
                _affinities.extend(this_affinities)
                _areas.extend(this_areas)
            return _edges, _affinities, _areas

        def _handle_subgraph_children_layer2_thread(
                child_ids: Iterable[np.uint64]) -> None:

            for child_id in child_ids:
                atomic_ids.extend(self.get_children(child_id))

        def _handle_subgraph_children_higher_layers_thread(
                child_ids: Iterable[np.uint64]) -> None:

            for child_id in child_ids:
                _children = self.get_children(child_id)

                if bounding_box is not None:
                    chunk_ids = self.get_chunk_ids_from_node_ids(_children)
                    chunk_ids = np.array([self.get_chunk_coordinates(c)
                                          for c in chunk_ids])
                    chunk_ids = np.array(chunk_ids)

                    bounding_box_layer = bounding_box / self.fan_out ** np.max([0, (layer - 3)])

                    bound_check = np.array([
                        np.all(chunk_ids < bounding_box_layer[1], axis=1),
                        np.all(chunk_ids + 1 > bounding_box_layer[0], axis=1)]).T

                    bound_check_mask = np.all(bound_check, axis=1)
                    _children = _children[bound_check_mask]

                new_childs.extend(_children)

        # Make sure that edges are not requested if we should stop on an
        # intermediate level
        assert stop_lvl == 1 or not get_edges

        if get_edges:
            time_stamp = self.read_row(agglomeration_id, "children",
                                       get_time_stamp=True)[1]

        if bounding_box is not None:
            if bb_is_coordinate:
                bounding_box = np.array(bounding_box,
                                        dtype=np.float32) / self.chunk_size
                bounding_box[0] = np.floor(bounding_box[0])
                bounding_box[1] = np.ceil(bounding_box[1])
                bounding_box = bounding_box.astype(np.int)
            else:
                bounding_box = np.array(bounding_box, dtype=np.int)

        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        areas = np.array([], dtype=np.uint64)
        atomic_ids = []
        affinities = np.array([], dtype=np.float32)
        child_ids = [agglomeration_id]

        time_start = time.time()
        while len(child_ids) > 0:
            new_childs = []
            layer = self.get_chunk_layer(child_ids[0])

            if stop_lvl == layer:
                atomic_ids = child_ids
                break

            # Use heuristic to guess the optimal number of threads
            n_child_ids = len(child_ids)
            this_n_threads = np.min([int(n_child_ids // 20) + 1, mu.n_cpus])

            if layer == 2:
                if get_edges:
                    child_ids = np.array(child_ids, dtype=np.uint64)
                    child_chunk_ids = self.get_chunk_ids_from_node_ids(child_ids)
                    u_ccids = np.unique(child_chunk_ids)

                    # this_n_threads = 1 # ----------------------------------------------------------------------------------------------------------------------------------

                    child_blocks = []
                    # Make blocks of child ids that are in the same chunk
                    for u_ccid in u_ccids:
                        child_blocks.append(child_ids[child_chunk_ids == u_ccid])

                    edge_infos = mu.multithread_func(
                        _handle_subgraph_children_layer2_edges_thread,
                        child_blocks,
                        n_threads=this_n_threads, debug=this_n_threads == 1)

                    for edge_info in edge_infos:
                        _edges, _affinities, _areas = edge_info
                        areas = np.concatenate([areas, _areas])
                        affinities = np.concatenate([affinities, _affinities])
                        edges = np.concatenate([edges, _edges])
                else:
                    mu.multithread_func(
                        _handle_subgraph_children_layer2_thread,
                        np.array_split(child_ids, this_n_threads),
                        n_threads=this_n_threads, debug=this_n_threads == 1)
            else:
                mu.multithread_func(
                    _handle_subgraph_children_higher_layers_thread,
                    np.array_split(child_ids, this_n_threads),
                    n_threads=this_n_threads, debug=this_n_threads == 1)

            child_ids = new_childs

            if verbose:
                print("Layer %d: %.3fms for %d children with %d threads" %
                      (layer, (time.time() - time_start) * 1000, n_child_ids,
                       this_n_threads))

            time_start = time.time()

        atomic_ids = np.array(atomic_ids, np.uint64)

        if get_edges:
            return edges, affinities, areas
        else:
            return atomic_ids

    def flatten_row_dict(self, row_dict_fam) -> Dict:
        """ Flattens multiple entries to columns by appending them

        :param row_dict_fam: dict
            family key has to be resolved
        :return:
        """

        flattened_row_dict = {}
        for key_s in row_dict_fam.keys():
            if key_s in self.family_ids:
                raise Exception("Need to resolve family id first before "
                                "flattening")

            key = key_utils.deserialize_key(key_s)
            col = row_dict_fam[key_s]
            flattened_row_dict[key] = []

            for col_entry in col[::-1]:
                deserialized_entry = np.frombuffer(col_entry.value,
                                                   dtype=table_info.dtype_dict[key])
                flattened_row_dict[key].extend(deserialized_entry)

            flattened_row_dict[key] = np.array(flattened_row_dict[key],
                                               dtype=table_info.dtype_dict[key])

            if key_s == table_info.connected_key_s:
                u_ids, c_ids = np.unique(flattened_row_dict[key],
                                         return_counts=True)
                flattened_row_dict[key] = u_ids[(c_ids % 2) == 1].astype(np.int)
        return flattened_row_dict

    def get_atomic_node_partners(self, atomic_id: np.uint64,
                                 time_stamp: datetime.datetime = get_max_time()
                                 ) -> Dict:
        """ Reads register partner ids

        :param atomic_id: np.uint64
        :param time_stamp: datetime.datetime
        :return: dict
        """
        keys = [table_info.partner_key_s, table_info.connected_key_s]
        row_dict = self.read_row_multi_key(atomic_id, keys=keys,
                                           time_stamp=time_stamp)
        flattened_row_dict = self.flatten_row_dict(row_dict[self.family_id])
        return flattened_row_dict[table_info.partner_key]

    def _get_atomic_node_info_core(self, row_dict) -> Dict:
        """ Reads connectivity information for a single node

        :param atomic_id: np.uint64
        :param time_stamp: datetime.datetime
        :return: dict
        """
        flattened_row_dict = self.flatten_row_dict(row_dict[self.family_id])
        all_ids = np.arange(len(flattened_row_dict[table_info.partner_key]), dtype=np.int)
        disconnected_m = ~np.in1d(all_ids, flattened_row_dict[table_info.connected_key])
        flattened_row_dict[table_info.disconnected_key] = all_ids[disconnected_m]

        return flattened_row_dict

    def get_atomic_node_info(self, atomic_id: np.uint64,
                             time_stamp: datetime.datetime = get_max_time()
                             ) -> Dict:
        """ Reads connectivity information for a single node

        :param atomic_id: np.uint64
        :param time_stamp: datetime.datetime
        :return: dict
        """
        keys = [table_info.connected_key_s, table_info.affinity_key_s,
                table_info.area_key_s, table_info.partner_key_s,
                table_info.parent_key_s]
        row_dict = self.read_row_multi_key(atomic_id, keys=keys,
                                           time_stamp=time_stamp)

        return self._get_atomic_node_info_core(row_dict)

    def _get_atomic_partners_core(self, flattened_row_dict: Dict,
                                  include_connected_partners=True,
                                  include_disconnected_partners=False
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Extracts the atomic partners and affinities for a given timestamp

        :param flattened_row_dict: dict
        :param include_connected_partners: bool
        :param include_disconnected_partners: bool
        :return: list of np.ndarrays
        """
        conn_keys = []
        if include_connected_partners:
            conn_keys.append(table_info.connected_key)
        if include_disconnected_partners:
            conn_keys.append(table_info.disconnected_key)

        included_ids = []
        for key in conn_keys:
            included_ids.extend(flattened_row_dict[key])

        included_ids = np.array(included_ids, dtype=np.int)

        areas = flattened_row_dict[table_info.area_key][included_ids]
        affinities = flattened_row_dict[table_info.affinity_key][included_ids]
        partners = flattened_row_dict[table_info.partner_key][included_ids]

        return partners, affinities, areas

    def get_atomic_partners(self, atomic_id: np.uint64,
                            include_connected_partners=True,
                            include_disconnected_partners=False,
                            time_stamp: Optional[datetime.datetime] = get_max_time()
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Extracts the atomic partners and affinities for a given timestamp

        :param atomic_id: np.uint64
        :param include_connected_partners: bool
        :param include_disconnected_partners: bool
        :param time_stamp: None or datetime
        :return: list of np.ndarrays
        """
        assert include_connected_partners or include_disconnected_partners

        flattened_row_dict = self.get_atomic_node_info(atomic_id,
                                                       time_stamp=time_stamp)

        return self._get_atomic_partners_core(flattened_row_dict,
                                              include_connected_partners,
                                              include_disconnected_partners)

    def get_subgraph_chunk(self, parent_ids: Iterable[np.uint64],
                           make_unique: bool = True,
                           time_stamp: Optional[datetime.datetime] = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Takes an atomic id and returns the associated agglomeration ids

        :param parent_ids: array of np.uint64
        :param make_unique: bool
        :param time_stamp: None or datetime
        :return: edge list
        """
        def _read_atomic_partners(child_id_block: Iterable[np.uint64]
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            thread_edges = np.array([], dtype=np.uint64).reshape(0, 2)
            thread_affinities = np.array([], dtype=np.float32)
            thread_areas = np.array([], dtype=np.uint64)

            for child_id in child_id_block:
                child_id_s = key_utils.serialize_uint64(child_id)
                flattened_row_dict = self.flatten_row_dict(row_dict[child_id_s].cells[self.family_id])
                node_edges, node_affinities, node_areas = \
                    self._get_atomic_partners_core(flattened_row_dict,
                                                   include_connected_partners=True,
                                                   include_disconnected_partners=False)

                # If we have edges add them to the chunk global edge list
                if len(node_edges) > 0:
                    # Build n x 2 edge list from partner list
                    node_edges = \
                        np.concatenate([np.ones((len(node_edges), 1),
                                                dtype=np.uint64) * child_id,
                                        node_edges[:, None]], axis=1)

                    thread_edges = np.concatenate([thread_edges,
                                                   node_edges])
                    thread_affinities = np.concatenate([thread_affinities,
                                                        node_affinities])
                    thread_areas = np.concatenate([thread_areas,
                                                   node_areas])

            return thread_edges, thread_affinities, thread_areas

        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        if not isinstance(parent_ids, list):
            parent_ids = [parent_ids]

        child_ids = []
        for parent_id in parent_ids:
            child_ids.extend(self.get_children(parent_id))

        child_ids = np.sort(np.array(child_ids, dtype=np.uint64))

        # Iterate through all children of this parent and retrieve their edges
        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        affinities = np.array([], dtype=np.float32)
        areas = np.array([], dtype=np.uint64)

        # Range read (optimize reading procedure)
        family_filter = FamilyNameRegexFilter(self.family_id)

        # We need to block the reading procedure as we could hit the
        # filter max size
        n_blocks = int(np.ceil(len(child_ids) / 500))
        row_dict = {}
        for child_id_block in np.array_split(child_ids, n_blocks):
            start_id = child_ids.min()
            end_id = child_ids.max()

            key_filter = RowKeyRegexFilter(
                key_utils.serialize_uint64s_to_regex(child_id_block))

            filters = [key_filter, family_filter]
            row_dict.update(self.range_read(start_id=start_id, end_id=end_id,
                                            default_filters=filters,
                                            time_stamp=time_stamp))

        n_child_ids = len(child_ids)
        this_n_threads = np.min([int(n_child_ids // 20) + 1, mu.n_cpus])

        child_id_blocks = np.array_split(child_ids, this_n_threads)
        edges_and_affinities = mu.multithread_func(_read_atomic_partners,
                                                   child_id_blocks,
                                                   n_threads=this_n_threads,
                                                   debug=this_n_threads == 1)

        for edges_and_affinities_pairs in edges_and_affinities:
            this_edges, this_affinities, this_areas = edges_and_affinities_pairs
            edges = np.concatenate([edges, this_edges])
            affinities = np.concatenate([affinities, this_affinities])
            areas = np.concatenate([areas, this_areas])

        # If requested, remove duplicate edges. Every edge is stored in each
        # participating node. Hence, we have many edge pairs that look
        # like [x, y], [y, x]. We solve this by sorting and calling np.unique
        # row-wise
        if make_unique:
            edges, idx = np.unique(np.sort(edges, axis=1), axis=0,
                                   return_index=True)
            affinities = affinities[idx]
            areas = areas[idx]

        return edges, affinities, areas

    def add_edges(self, user_id: str, atomic_edges: Sequence[np.uint64],
                  affinities: Sequence[np.float32] = None,
                  source_coord: Sequence[int] = None,
                  sink_coord: Sequence[int] = None,
                  n_tries: int = 60) -> np.uint64:
        """ Adds an edge to the chunkedgraph

            Multi-user safe through locking of the root node

            This function acquires a lock and ensures that it still owns the
            lock before executing the write.

        :param user_id: str
            unique id - do not just make something up, use the same id for the
            same user every time
        :param atomic_edges: list of two uint64s
            have to be from the same two root ids!
        :param affinities: list of np.float32 or None
            will eventually be set to 1 if None
        :param source_coord: list of int (n x 3)
        :param sink_coord: list of int (n x 3)
        :param n_tries: int
        :return: uint64
            if successful the new root id is send
            else None
        """
        if not (isinstance(atomic_edges[0], list) or isinstance(atomic_edges[0], np.ndarray)):
            atomic_edges = [atomic_edges]

        atomic_edges = np.array(atomic_edges)

        if affinities is not None:
            if not (isinstance(affinities, list) or
                    isinstance(affinities, np.ndarray)):
                affinities = [affinities]

        root_ids = []
        for atomic_edge in atomic_edges:
            # Sanity Checks
            if atomic_edge[0] == atomic_edge[1]:
                return None

            if self.get_chunk_layer(atomic_edge[0]) != \
                    self.get_chunk_layer(atomic_edge[1]):
                return None

            # Lookup root ids
            root_ids.append([self.get_root(atomic_edge[0]),
                             self.get_root(atomic_edge[1])])

        u_root_ids = np.unique(root_ids)

        # Get a unique id for this operation
        operation_id = self.get_unique_operation_id()

        i_try = 0
        lock_root_ids = u_root_ids
        while i_try < n_tries:
            # Try to acquire lock and only continue if successful
            lock_acquired, lock_root_ids = \
                self.lock_root_loop(root_ids=lock_root_ids,
                                    operation_id=operation_id)

            if lock_acquired:
                rows = []
                new_root_ids = []
                time_stamp = datetime.datetime.utcnow()

                # Add edge and change hierarchy
                # for atomic_edge in atomic_edges:
                new_root_id, new_rows = \
                    self._add_edges(operation_id=operation_id,
                                    atomic_edges=atomic_edges,
                                    time_stamp=time_stamp,
                                    affinities=affinities)
                rows.extend(new_rows)
                new_root_ids.append(new_root_id)

                # Add a row to the log
                rows.append(self._create_merge_log_row(operation_id,
                                                       user_id,
                                                       new_root_ids,
                                                       atomic_edges[:, 0],
                                                       atomic_edges[:, 1],
                                                       [source_coord],
                                                       [sink_coord],
                                                       atomic_edges,
                                                       affinities,
                                                       time_stamp))

                # Execute write (makes sure that we are still owning the lock)
                if self.bulk_write(rows, lock_root_ids,
                                   operation_id=operation_id,
                                   slow_retry=False):
                    return new_root_id

            for lock_root_id in lock_root_ids:
                self.unlock_root(lock_root_id, operation_id)

            i_try += 1

            print("Waiting - %d" % i_try)
            time.sleep(1)

        return None


    def _add_edges(self, operation_id: np.uint64,
                   atomic_edges: Sequence[Sequence[np.uint64]],
                   time_stamp=datetime.datetime,
                   affinities: Optional[Sequence[np.float32]] = None
                   ):
        """ Adds multiple edges to a graph

        :param operation_id: np.uint64
        :param atomic_edges: list of two uint64s
        :param time_stamp: datetime.datetime
        :param affinities: list of np.float32
        :return: list of np.uint64, rows
        """

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        if affinities is None:
            affinities = np.ones(len(atomic_edges),
                                 dtype=table_info.dtype_dict[table_info.affinity_key])

        assert len(affinities) == len(atomic_edges)

        rows = []

        # Create node_id to parent look up for later
        node_ids = np.unique(atomic_edges)
        node_id_parent_dict = {}
        parent_node_id_dict = collections.defaultdict(list)
        for node_id in node_ids:
            parent_ids = self.get_all_parents(node_id)
            node_id_parent_dict[node_id] = parent_ids

            for parent_id in parent_ids:
                parent_node_id_dict[parent_id].append(node_id)

        # Layer 1 --------------------------------------------------------------

        # Look up when the edges in question "become relevant"
        edge_layers = self.get_cross_chunk_edges_layer(atomic_edges)

        parental_edges = []
        cross_chunk_edges = {}
        future_links = collections.defaultdict(list)

        for atomic_edge, edge_layer in zip(atomic_edges, edge_layers):
            # Test if edge is cross chunk edge
            if edge_layer == 1:
                # Level 1 edge
                old_parent_ids = [node_id_parent_dict[atomic_edge[0]][0],
                                  node_id_parent_dict[atomic_edge[1]][0]]

                parental_edges.append(old_parent_ids)

                cross_chunk_edges[atomic_edge[0]] = {}
                cross_chunk_edges[atomic_edge[1]] = {}
            else:
                # Add self edges to have these ids in the graph
                parental_edges.append([node_id_parent_dict[atomic_edge[0]][0],
                                       node_id_parent_dict[atomic_edge[0]][0]])
                parental_edges.append([node_id_parent_dict[atomic_edge[1]][0],
                                       node_id_parent_dict[atomic_edge[1]][0]])

                # Cross chunk edge
                cross_chunk_edges[atomic_edge[0]] = {edge_layer: np.array([atomic_edge])}
                cross_chunk_edges[atomic_edge[1]] = {edge_layer: np.array([atomic_edge[::-1]])}

                future_links[edge_layer].append([node_id_parent_dict[atomic_edge[0]][edge_layer - 1],
                                                 node_id_parent_dict[atomic_edge[1]][edge_layer - 1]])
        # resolve edges with same parent
        G = nx.Graph()
        G.add_edges_from(parental_edges)

        ccs = nx.connected_components(G)

        next_cc_storage = [] # Data passed on from layer to layer
        old_parent_mapping = collections.defaultdict(list)
        for cc in ccs:
            old_parent_ids = list(cc)

            atomic_ids = []
            involved_atomic_ids = []
            cross_chunk_edge_dict = {}

            # for l in range(2, self.n_layers):
            #     cross_chunk_edge_dict[l] = np.array([], dtype=table_info.dtype_dict["cross_chunk_edges"])

            for old_parent_id in old_parent_ids:
                involved_atomic_ids.extend(parent_node_id_dict[old_parent_id])

                child_ids = self.get_children(old_parent_id)
                atomic_ids.extend(child_ids)

                cross_chunk_edge_dict = \
                    combine_cross_chunk_edge_dicts(cross_chunk_edge_dict,
                                                   self.read_cross_chunk_edges(old_parent_id))

            involved_atomic_ids = np.unique(involved_atomic_ids)
            for involved_atomic_id in involved_atomic_ids:
                if involved_atomic_id in cross_chunk_edges:
                    cross_chunk_edge_dict = combine_cross_chunk_edge_dicts(cross_chunk_edge_dict, cross_chunk_edges[involved_atomic_id])

            atomic_ids = np.array(atomic_ids, dtype=np.uint64)

            chunk_id = self.get_chunk_id(node_id=old_parent_ids[0])
            new_parent_id = self.get_unique_node_id(chunk_id)
            new_parent_id_b = np.array(new_parent_id).tobytes()

            for atomic_id in atomic_ids:
                val_dict = {"parents": new_parent_id_b}
                rows.append(self.mutate_row(key_utils.serialize_uint64(atomic_id),
                                            self.family_id,
                                            val_dict,
                                            time_stamp=time_stamp))

            val_dict = {"children": atomic_ids.tobytes()}

            rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                        self.family_id, val_dict,
                                        time_stamp=time_stamp))

            val_dict = {}
            for l in range(2, self.n_layers):
                if len(cross_chunk_edge_dict[l]) > 0:
                    val_dict[table_info.cross_chunk_edge_keyformat % l] = cross_chunk_edge_dict[l].tobytes()

            if len(val_dict):
                rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                            self.cross_edge_family_id, val_dict,
                                            time_stamp=time_stamp))

            next_old_parents = [self.get_parent(n) for n in old_parent_ids]
            for p in next_old_parents:
                old_parent_mapping[p].append(len(next_cc_storage)) # Save storage ids for each future parent

            next_cc_storage.append([new_parent_id,
                                    next_old_parents,
                                    old_parent_ids,
                                    cross_chunk_edge_dict])

        # Higher Layers --------------------------------------------------------

        new_root_ids = []
        if self.n_layers == 2:
            # Special case
            for cc_storage in next_cc_storage:
                new_root_ids.append(cc_storage[0])

        for i_layer in range(2, self.n_layers):
            cc_storage = list(next_cc_storage) # copy
            next_cc_storage = []

            # combine what belongs together
            parental_edges = []
            for p in old_parent_mapping.keys():
                # building edges between parents
                parental_edges.extend(
                    list(itertools.product(old_parent_mapping[p],
                                           old_parent_mapping[p])))

            for e in future_links[i_layer]:
                parental_edges.extend(list(itertools.product(old_parent_mapping[e[0]],
                                                             old_parent_mapping[e[1]])))

            old_parent_mapping = collections.defaultdict(list)

            G = nx.Graph()
            G.add_edges_from(parental_edges)
            ccs = list(nx.connected_components(G))

            for cc in ccs:
                cc_storage_ids = list(cc)

                new_child_ids = [] # new_parent_id
                old_parent_ids = [] # next_old_parents
                old_child_ids = [] # old_parent_ids
                cross_chunk_edge_dict = {}

                for cc_storage_id in cc_storage_ids:
                    cc_storage_entry = cc_storage[cc_storage_id]

                    new_child_ids.append(cc_storage_entry[0])
                    old_parent_ids.extend(cc_storage_entry[1])
                    old_child_ids.extend(cc_storage_entry[2])
                    cross_chunk_edge_dict = \
                        combine_cross_chunk_edge_dicts(cross_chunk_edge_dict,
                                                       cc_storage_entry[3])

                new_child_ids = np.array(new_child_ids,
                                         dtype=table_info.dtype_dict[table_info.partner_key])

                chunk_id = self.get_chunk_id(node_id=old_parent_ids[0])
                new_parent_id = self.get_unique_node_id(chunk_id)
                new_parent_id_b = np.array(new_parent_id).tobytes()

                maintained_child_ids = []
                for old_parent_id in old_parent_ids:
                    old_parent_child_ids = self.get_children(old_parent_id)
                    maintained_child_ids.extend(old_parent_child_ids)

                maintained_child_ids = np.array(maintained_child_ids,
                                                table_info.dtype_dict[table_info.partner_key])

                m = ~np.in1d(maintained_child_ids, old_child_ids)
                maintained_child_ids = maintained_child_ids[m]
                child_ids = np.concatenate([new_child_ids,
                                            maintained_child_ids])

                for child_id in child_ids:
                    child_cross_chunk_edges = self.read_cross_chunk_edges(child_id)
                    cross_chunk_edge_dict = combine_cross_chunk_edge_dicts(cross_chunk_edge_dict,
                                                                           child_cross_chunk_edges)
                # ----------------------------------------------------------------------------------------------

                for child_id in child_ids:
                    val_dict = {"parents": new_parent_id_b}
                    rows.append(self.mutate_row(key_utils.serialize_uint64(child_id),
                                                self.family_id,
                                                val_dict,
                                                time_stamp=time_stamp))

                val_dict = {"children": child_ids.tobytes()}

                rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))
                val_dict = {}

                for l in range(i_layer, self.n_layers):
                    if len(cross_chunk_edge_dict[l]) > 0:
                        print(cross_chunk_edge_dict[l])
                        val_dict[table_info.cross_chunk_edge_keyformat % l] = \
                            cross_chunk_edge_dict[l].tobytes()

                if len(val_dict):
                    rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                                self.cross_edge_family_id, val_dict,
                                                time_stamp=time_stamp))

                if i_layer < self.n_layers - 1:
                    next_old_parents = np.unique([self.get_parent(n)
                                                  for n in old_parent_ids]) # ---------------------------------

                    for p in next_old_parents:
                        old_parent_mapping[p].append(len(next_cc_storage))

                    next_cc_storage.append([new_parent_id,
                                            next_old_parents,
                                            old_parent_ids,
                                            cross_chunk_edge_dict])
                else:
                    val_dict = {}
                    val_dict["former_parents"] = np.array(old_parent_ids).tobytes()
                    val_dict["operation_id"] = key_utils.serialize_uint64(operation_id)

                    rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                    new_root_ids.append(new_parent_id)

                    for p in old_parent_ids:
                        rows.append(self.mutate_row(key_utils.serialize_uint64(p),
                                                    self.family_id,
                                                    {"new_parents": new_parent_id_b},
                                                    time_stamp=time_stamp))

        # Atomic edge
        for i_atomic_edge, atomic_edge in enumerate(atomic_edges):
            affinity = affinities[i_atomic_edge]

            for i_atomic_id in range(2):
                atomic_id = atomic_edge[i_atomic_id]
                edge_partner = atomic_edge[(i_atomic_id + 1) % 2]

                atomic_node_info = self.get_atomic_node_info(atomic_id)

                if edge_partner in atomic_node_info[table_info.partner_key]:
                    partner_id = np.where(atomic_node_info[
                                              table_info.partner_key] == edge_partner)[0]

                    if partner_id in atomic_node_info[table_info.disconnected_key]:
                        partner_id_b = np.array(partner_id, dtype=table_info.dtype_dict[
                            table_info.connected_key]).tobytes()
                        val_dict = {table_info.connected_key: partner_id_b}
                    else:
                        val_dict = {}
                else:
                    affinity_b = np.array(affinity, dtype=table_info.dtype_dict[
                        table_info.affinity_key]).tobytes()
                    area_b = np.array(0, dtype=table_info.dtype_dict[table_info.area_key]).tobytes()
                    partner_id_b = np.array(len(atomic_node_info[table_info.partner_key]), dtype=
                    table_info.dtype_dict[
                        table_info.connected_key]).tobytes()
                    edge_partner_b = np.array(edge_partner, dtype=table_info.dtype_dict[
                        table_info.partner_key]).tobytes()
                    val_dict = {table_info.affinity_key: affinity_b,
                                table_info.area_key: area_b,
                                table_info.connected_key: partner_id_b,
                                table_info.partner_key: edge_partner_b}

                if len(val_dict) > 0:
                    rows.append(self.mutate_row(key_utils.serialize_uint64(
                        atomic_edge[i_atomic_id]), self.family_id, val_dict,
                        time_stamp=time_stamp))

        return new_root_ids, rows

    def remove_edges(self,
                     user_id: str,
                     source_id: np.uint64,
                     sink_id: np.uint64,
                     source_coord: Optional[Sequence[int]] = None,
                     sink_coord: Optional[Sequence[int]] = None,
                     mincut: bool = True,
                     bb_offset: Tuple[int, int, int] = (240, 240, 24),
                     root_ids: Optional[Sequence[np.uint64]] = None,
                     n_tries: int = 20) -> Sequence[np.uint64]:
        """ Removes edges - either directly or after applying a mincut

            Multi-user safe through locking of the root node

            This function acquires a lock and ensures that it still owns the
            lock before executing the write.

        :param user_id: str
            unique id - do not just make something up, use the same id for the
            same user every time
        :param source_id: uint64
        :param sink_id: uint64
        :param source_coord: list of 3 ints
            [x, y, z] coordinate of source supervoxel
        :param sink_coord: list of 3 ints
            [x, y, z] coordinate of sink supervoxel
        :param mincut:
        :param bb_offset: list of 3 ints
            [x, y, z] bounding box padding beyond box spanned by coordinates
        :param root_ids: list of uint64s
        :param n_tries: int
        :return: list of uint64s or None if no split was performed
        """

        # Sanity Checks
        if source_id == sink_id:
            print("source == sink")
            return None

        if self.get_chunk_layer(source_id) != \
                self.get_chunk_layer(sink_id):
            print("layer(source) !== layer(sink)")
            return None

        if mincut:
            assert source_coord is not None
            assert sink_coord is not None

        if root_ids is None:
            root_ids = [self.get_root(source_id),
                        self.get_root(sink_id)]

        if root_ids[0] != root_ids[1]:
            print("root(source) != root(sink):", root_ids)
            return None

        # Get a unique id for this operation
        operation_id = self.get_unique_operation_id()

        i_try = 0

        while i_try < n_tries:
            # Try to acquire lock and only continue if successful
            lock_root_ids = np.unique(root_ids)

            lock_acquired, lock_root_ids = \
                self.lock_root_loop(root_ids=lock_root_ids,
                                    operation_id=operation_id)

            if lock_acquired:
                # (run mincut) and remove edges + update hierarchy
                if mincut:
                    success, result = \
                        self._remove_edges_mincut(operation_id=operation_id,
                                                  source_id=source_id,
                                                  sink_id=sink_id,
                                                  source_coord=source_coord,
                                                  sink_coord=sink_coord,
                                                  bb_offset=bb_offset)
                    if success:
                        new_root_ids, rows, removed_edges, time_stamp = result
                    else:
                        for lock_root_id in lock_root_ids:
                            self.unlock_root(lock_root_id,
                                             operation_id=operation_id)
                        return None
                else:
                    success, result = \
                        self._remove_edges(operation_id=operation_id,
                                           atomic_edges=[(source_id, sink_id)])
                    if success:
                        new_root_ids, rows, time_stamp = result
                        removed_edges = [[source_id, sink_id]]
                    else:
                        for lock_root_id in lock_root_ids:
                            self.unlock_root(lock_root_id,
                                             operation_id=operation_id)
                        return None

                # Add a row to the log
                rows.append(self._create_split_log_row(operation_id,
                                                       user_id,
                                                       new_root_ids,
                                                       [source_id],
                                                       [sink_id],
                                                       [source_coord],
                                                       [sink_coord],
                                                       removed_edges,
                                                       bb_offset,
                                                       time_stamp))

                # Execute write (makes sure that we are still owning the lock)
                if self.bulk_write(rows, lock_root_ids,
                                   operation_id=operation_id, slow_retry=False):
                    return new_root_ids

                for lock_root_id in lock_root_ids:
                    self.unlock_root(lock_root_id, operation_id=operation_id)

            i_try += 1

            print("Waiting - %d" % i_try)
            time.sleep(1)

        return None

    def _remove_edges_mincut(self, operation_id: np.uint64, source_id: np.uint64,
                             sink_id: np.uint64, source_coord: Sequence[int],
                             sink_coord: Sequence[int],
                             bb_offset: Tuple[int, int, int] = (120, 120, 12)
                             ) -> Tuple[
                                 bool,                         # success
                                 Optional[Tuple[
                                    List[np.uint64],           # new_roots
                                    List[bigtable.row.Row],    # rows
                                    np.ndarray,                # removed_edges
                                    datetime.datetime]]]:      # timestamp
        """ Computes mincut and removes edges accordingly

        :param operation_id: uint64
        :param source_id: uint64
        :param sink_id: uint64
        :param source_coord: list of 3 ints
            [x, y, z] coordinate of source supervoxel
        :param sink_coord: list of 3 ints
            [x, y, z] coordinate of sink supervoxel
        :param bb_offset: list of 3 ints
            [x, y, z] bounding box padding beyond box spanned by coordinates
        :return: list of uint64s if successful, or None if no valid split
            new root ids
        """

        time_start = time.time()  # ------------------------------------------

        bb_offset = np.array(list(bb_offset))
        source_coord = np.array(source_coord)
        sink_coord = np.array(sink_coord)

        # Decide a reasonable bounding box (NOT guaranteed to be successful!)
        coords = np.concatenate([source_coord[:, None],
                                 sink_coord[:, None]], axis=1).T
        bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]

        bounding_box[0] -= bb_offset
        bounding_box[1] += bb_offset

        root_id_source = self.get_root(source_id)
        root_id_sink = self.get_root(source_id)

        # Verify that sink and source are from the same root object
        if root_id_source != root_id_sink:
            print("root(source) != root(sink)")
            return False, None

        print("Get roots and check: %.3fms" %
              ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        root_id = root_id_source

        # Get edges between local supervoxels
        n_chunks_affected = np.product((np.ceil(bounding_box[1] / self.chunk_size)).astype(np.int) -
                                       (np.floor(bounding_box[0] / self.chunk_size)).astype(np.int))
        print("Number of affected chunks: %d" % n_chunks_affected)
        print("Bounding box:", bounding_box)
        print("Bounding box padding:", bb_offset)
        print("Atomic ids: %d - %d" % (source_id, sink_id))
        print("Root id:", root_id)

        edges, affs, areas = self.get_subgraph(root_id, get_edges=True,
                                               bounding_box=bounding_box,
                                               bb_is_coordinate=True)

        print("Get edges and affs: %.3fms" %
              ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        # Compute mincut
        atomic_edges = cutting.mincut(edges, affs, source_id, sink_id)

        print("Mincut: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        if len(atomic_edges) == 0:
            print("WARNING: Mincut failed. Try again...")
            return False, None

        # Check if any edge in the cutset is infinite (== between chunks)
        # We would prevent such a cut

        atomic_edges_flattened_view = atomic_edges.view(dtype='u8,u8')
        edges_flattened_view = edges.view(dtype='u8,u8')

        cutset_mask = np.in1d(edges_flattened_view, atomic_edges_flattened_view)
        if np.any(np.isinf(affs[cutset_mask])):
            print("inf in cutset")
            return False, None

        # Remove edgesc
        success, result = self._remove_edges(operation_id, atomic_edges)

        if not success:
            print("remove edges failed")
            return False, None

        new_roots, rows, time_stamp = result

        print("Remove edges: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        return True, (new_roots, rows, atomic_edges, time_stamp)

    def _remove_edges(self, operation_id: np.uint64,
                      atomic_edges: Sequence[Tuple[np.uint64, np.uint64]]
                      ) -> Tuple[bool,                          # success
                                 Optional[Tuple[
                                     List[np.uint64],           # new_roots
                                     List[bigtable.row.Row],    # rows
                                     datetime.datetime]]]:      # timestamp
        """ Removes atomic edges from the ChunkedGraph

        :param operation_id: uint64
        :param atomic_edges: list of two uint64s
        :return: list of uint64s
            new root ids
        """
        time_stamp = datetime.datetime.utcnow()

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        # Make sure that we have a list of edges
        if isinstance(atomic_edges[0], np.uint64):
            atomic_edges = [atomic_edges]

        atomic_edges = np.array(atomic_edges)
        u_atomic_ids = np.unique(atomic_edges)

        # Get number of layers and the original root
        original_parent_ids = self.get_all_parents(atomic_edges[0, 0])
        original_root = original_parent_ids[-1]

        # Find lowest level chunks that might have changed
        chunk_ids = self.get_chunk_ids_from_node_ids(u_atomic_ids)
        u_chunk_ids, u_chunk_ids_idx = np.unique(chunk_ids,
                                                 return_index=True)

        involved_chunk_id_dict = dict(zip(u_chunk_ids,
                                          u_atomic_ids[u_chunk_ids_idx]))

        # Note: After removing the atomic edges, we basically need to build the
        # ChunkedGraph for these chunks from the ground up.
        # involved_chunk_id_dict stores a representative for each chunk that we
        # can use to acquire the parent that knows about all atomic nodes in the
        # chunk.

        rows = []

        # Remove atomic edges

        # Removing edges nodewise. We cannot remove edges edgewise because that
        # would add up multiple changes to each node (row). Unfortunately,
        # the batch write (mutate_rows) from BigTable cannot handle multiple
        # changes to the same row-col within a batch write and only executes
        # one of them.
        for u_atomic_id in np.unique(atomic_edges):
            atomic_node_info = self.get_atomic_node_info(u_atomic_id)

            partners = np.concatenate([atomic_edges[atomic_edges[:, 0] == u_atomic_id][:, 1],
                                       atomic_edges[atomic_edges[:, 1] == u_atomic_id][:, 0]])

            partner_ids = np.where(np.in1d(atomic_node_info[table_info.partner_key], partners))[0]
            partner_ids_b = np.array(partner_ids, dtype=table_info.dtype_dict[
                table_info.connected_key]).tobytes()

            val_dict = {table_info.connected_key: partner_ids_b}

            rows.append(self.mutate_row(key_utils.serialize_uint64(u_atomic_id),
                                        self.family_id, val_dict,
                                        time_stamp=time_stamp))

        # Dictionaries keeping temporary information about the ChunkedGraph
        # while updates are not written to BigTable yet
        new_layer_parent_dict = {}
        cross_edge_dict = {}
        old_id_dict = collections.defaultdict(list)

        # This view of the to be removed edges helps us to compute the mask
        # of the retained edges in each chunk
        double_atomic_edges = np.concatenate([atomic_edges,
                                              atomic_edges[:, ::-1]],
                                             axis=0)
        double_atomic_edges_view = double_atomic_edges.view(dtype='u8,u8')
        n_edges = double_atomic_edges.shape[0]
        double_atomic_edges_view = double_atomic_edges_view.reshape(n_edges)
        nodes_in_removed_edges = np.unique(atomic_edges)

        # For each involved chunk we need to compute connected components
        for chunk_id in involved_chunk_id_dict.keys():
            # Get the local subgraph
            node_id = involved_chunk_id_dict[chunk_id]
            old_parent_id = self.get_parent(node_id)
            chunk_edges, _, _ = self.get_subgraph_chunk(old_parent_id,
                                                  make_unique=False)

            # These edges still contain the removed edges.
            # For consistency reasons we can only write to BigTable one time.
            # Hence, we have to evict the to be removed "atomic_edges" from the
            # queried edges.
            retained_edges_mask =\
                ~np.in1d(chunk_edges.view(dtype='u8,u8').reshape(chunk_edges.shape[0]),
                         double_atomic_edges_view)

            chunk_edges = chunk_edges[retained_edges_mask]

            # The cross chunk edges are passed on to the parents to compute
            # connected components in higher layers.
            terminal_chunk_ids = self.get_chunk_ids_from_node_ids(np.ascontiguousarray(chunk_edges[:, 1]))
            cross_edge_mask = terminal_chunk_ids != chunk_id

            cross_edges = chunk_edges[cross_edge_mask]
            chunk_edges = chunk_edges[~cross_edge_mask]

            isolated_nodes = list(filter(
                lambda x: x not in chunk_edges and self.get_chunk_id(x) == chunk_id,
                nodes_in_removed_edges))

            # Build the local subgraph and compute connected components
            G = nx.from_edgelist(chunk_edges)
            G.add_nodes_from(isolated_nodes)
            ccs = nx.connected_components(G)

            # For each connected component we create one new parent
            for cc in ccs:
                cc_node_ids = np.array(list(cc), dtype=np.uint64)

                # Get the associated cross edges
                cc_cross_edges = cross_edges[np.in1d(cross_edges[:, 0], cc_node_ids)]

                # Get a new parent id
                new_parent_id = self.get_unique_node_id(
                    self.get_chunk_id(node_id=old_parent_id))

                new_parent_id_b = np.array(new_parent_id).tobytes()
                new_parent_id = new_parent_id

                # Temporarily storing information on how the parents of this cc
                # are changed by the split. We need this information when
                # processing the next layer
                new_layer_parent_dict[new_parent_id] = old_parent_id
                old_id_dict[old_parent_id].append(new_parent_id)

                # Make changes to the rows of the lowest layer
                val_dict = {"children": cc_node_ids.tobytes()}

                rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(self.mutate_row(key_utils.serialize_uint64(cc_node_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                cce_layers = self.get_cross_chunk_edges_layer(cc_cross_edges)
                u_cce_layers = np.unique(cce_layers)

                cross_edge_dict[new_parent_id] = {}

                for l in range(2, self.n_layers):
                    empty_edges = np.array([], dtype=np.uint64).reshape(-1, 2)
                    cross_edge_dict[new_parent_id][l] = empty_edges

                val_dict = {}
                for cc_layer in u_cce_layers:
                    layer_cross_edges = cc_cross_edges[cce_layers == cc_layer]

                    if len(layer_cross_edges) > 0:
                        val_dict[table_info.cross_chunk_edge_keyformat % cc_layer] = \
                            layer_cross_edges.tobytes()
                        cross_edge_dict[new_parent_id][cc_layer] = layer_cross_edges

                if len(val_dict) > 0:
                    rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                                self.cross_edge_family_id,
                                                val_dict,
                                                time_stamp=time_stamp))

        # Now that the lowest layer has been updated, we need to walk through
        # all layers and move our new parents forward
        # new_layer_parent_dict stores all newly created parents. We first
        # empty it and then fill it with the new parents in the next layer
        if self.n_layers == 2:
            return True, (list(new_layer_parent_dict.keys()), rows, time_stamp)

        new_roots = []
        for i_layer in range(2, self.n_layers):
            old_parent_dict = {}

            edges = []
            leftover_old_parents = set()
            for new_layer_parent in new_layer_parent_dict.keys():
                old_parent_id = new_layer_parent_dict[new_layer_parent]
                cross_edges = cross_edge_dict[new_layer_parent]

                # Using the old parent's parents: get all nodes in the
                # neighboring chunks (go one up and one down in all directions)
                old_next_layer_parent = self.get_parent(old_parent_id)
                old_chunk_neighbors = self.get_children(old_next_layer_parent)
                old_chunk_neighbors = \
                    old_chunk_neighbors[old_chunk_neighbors != old_parent_id]

                old_parent_dict[new_layer_parent] = old_next_layer_parent

                # In analogy to `add_layer`, we need to compare
                # cross_chunk_edges among potential neighbors. Here, we know
                # that all future neighbors are among the old neighbors
                # (old_chunk_neighbors) or their new replacements due to this
                # split.

                atomic_children = cross_edges[i_layer][:, 0]

                for old_chunk_neighbor in old_chunk_neighbors:
                    # For each neighbor we need to check whether this neighbor
                    # was affected by a split as well (and was updated):
                    # neighbor_id in old_id_dict. If so, we take the new atomic
                    # cross edges (temporary data) into account, else, we load
                    # the atomic_cross_edges from BigTable
                    if old_chunk_neighbor in old_id_dict:
                        for new_neighbor in old_id_dict[old_chunk_neighbor]:
                            neigh_cross_edges = cross_edge_dict[new_neighbor][i_layer]

                            if np.any(np.in1d(neigh_cross_edges[:, 1], atomic_children)):
                                edges.append([new_neighbor, new_layer_parent])
                    else:
                        cross_edge_dict_neigh = self.read_cross_chunk_edges(old_chunk_neighbor)

                        cross_edge_dict[old_chunk_neighbor] = cross_edge_dict_neigh
                        neigh_cross_edges = cross_edge_dict_neigh[i_layer]

                        if np.any(np.in1d(neigh_cross_edges[:, 1],
                                          atomic_children)):
                            edges.append([old_chunk_neighbor, new_layer_parent])

                        leftover_old_parents.add(old_chunk_neighbor)

            for old_chunk_neighbor in leftover_old_parents:
                atomic_ids = cross_edge_dict[old_chunk_neighbor][i_layer][:, 0]

                for old_chunk_neighbor_partner in leftover_old_parents:
                    neigh_cross_edges = cross_edge_dict[old_chunk_neighbor_partner][i_layer]

                    if np.any(np.in1d(atomic_ids, neigh_cross_edges[:, 1])):
                        edges.append([old_chunk_neighbor,
                                      old_chunk_neighbor_partner])

            # Create graph and run connected components
            chunk_g = nx.from_edgelist(edges)
            chunk_g.add_nodes_from(np.array(list(new_layer_parent_dict.keys()),
                                            dtype=np.uint64))
            ccs = list(nx.connected_components(chunk_g))

            new_layer_parent_dict = {}
            # Filter the connected component that is relevant to the
            # current new_layer_parent
            for cc in ccs:
                partners = list(cc)

                # Create the new_layer_parent_dict for the next layer and write
                # nodes (lazy)

                old_next_layer_parent = None
                for partner in partners:
                    if partner in old_parent_dict:
                        old_next_layer_parent = old_parent_dict[partner]

                if old_next_layer_parent is None:
                    print("No old parents for any member of the cc")
                    lop = np.unique(list(leftover_old_parents))
                    llop = lop[~np.in1d(lop, np.unique(edges))]
                    raise()
                    return False, None

                partners = np.array(partners, dtype=np.uint64)

                this_chunk_id = self.get_chunk_id(
                    node_id=old_next_layer_parent)
                new_parent_id = self.get_unique_node_id(this_chunk_id)
                new_parent_id_b = np.array(new_parent_id).tobytes()

                new_layer_parent_dict[new_parent_id] = old_next_layer_parent
                old_id_dict[old_next_layer_parent].append(new_parent_id)

                cross_edge_dict[new_parent_id] = {}
                for partner in partners:
                    cross_edge_dict[new_parent_id] = combine_cross_chunk_edge_dicts(cross_edge_dict[new_parent_id],
                                                                                    cross_edge_dict[partner], start_layer=i_layer+1)

                for partner in partners:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(
                        self.mutate_row(key_utils.serialize_uint64(partner),
                                        self.family_id, val_dict,
                                        time_stamp=time_stamp))

                val_dict = {"children": partners.tobytes()}

                if i_layer == self.n_layers - 1:
                    new_roots.append(new_parent_id)
                    val_dict["former_parents"] = \
                        np.array(original_root).tobytes()
                    val_dict["operation_id"] = \
                        key_utils.serialize_uint64(operation_id)

                rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

                if i_layer < self.n_layers - 1:
                    val_dict = {}
                    for l in range(i_layer + 1, self.n_layers):
                        val_dict[table_info.cross_chunk_edge_keyformat % l] = \
                            cross_edge_dict[new_parent_id][l].tobytes()

                    if len(val_dict) == 0:
                        print("Cross chunk edges are missing")
                        return False, None

                    rows.append(self.mutate_row(key_utils.serialize_uint64(new_parent_id),
                                                self.cross_edge_family_id,
                                                val_dict,
                                                time_stamp=time_stamp))

            if i_layer == self.n_layers - 1:
                val_dict = {"new_parents": np.array(new_roots,
                                                    dtype=np.uint64).tobytes()}
                rows.append(self.mutate_row(key_utils.serialize_uint64(original_root),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

        # print("MADE IT")
        # raise()
        return True, (new_roots, rows, time_stamp)
