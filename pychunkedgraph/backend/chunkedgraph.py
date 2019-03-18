import collections
import numpy as np
import time
import datetime
import os
import sys
import networkx as nx
import pytz
import cloudvolume
import re
import itertools
import logging

from itertools import chain
from multiwrapper import multiprocessing_utils as mu
from pychunkedgraph.backend import cutting, chunkedgraph_comp
from pychunkedgraph.backend.chunkedgraph_utils import compute_indices_pandas, \
    compute_bitmasks, get_google_compatible_time_stamp, \
    get_time_range_filter, get_time_range_and_column_filter, get_max_time, \
    combine_cross_chunk_edge_dicts, get_min_time, partial_row_data_to_column_dict
from pychunkedgraph.backend.utils import serializers, column_keys, row_keys, basetypes
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.meshing import meshgen

from google.api_core.retry import Retry, if_exception_type
from google.api_core.exceptions import Aborted, DeadlineExceeded, \
    ServiceUnavailable
from google.auth import credentials
from google.cloud import bigtable
from google.cloud.bigtable.row_filters import TimestampRange, \
    TimestampRangeFilter, ColumnRangeFilter, ValueRangeFilter, RowFilterChain, \
    ColumnQualifierRegexFilter, RowFilterUnion, ConditionalRowFilter, \
    PassAllFilter, RowFilter, RowKeyRegexFilter, FamilyNameRegexFilter
from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.column_family import MaxVersionsGCRule

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, NamedTuple


HOME = os.path.expanduser("~")
N_DIGITS_UINT64 = len(str(np.iinfo(np.uint64).max))
LOCK_EXPIRED_TIME_DELTA = datetime.timedelta(minutes=1, seconds=00)
UTC = pytz.UTC

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
           HOME + "/.cloudvolume/secrets/google-secret.json"


class ChunkedGraph(object):
    def __init__(self,
                 table_id: str,
                 instance_id: str = "pychunkedgraph",
                 project_id: str = "neuromancer-seung-import",
                 chunk_size: Tuple[np.uint64, np.uint64, np.uint64] = None,
                 fan_out: Optional[np.uint64] = None,
                 n_layers: Optional[np.uint64] = None,
                 credentials: Optional[credentials.Credentials] = None,
                 client: bigtable.Client = None,
                 dataset_info: Optional[object] = None,
                 is_new: bool = False,
                 logger: Optional[logging.Logger] = None) -> None:

        if logger is None:
            self.logger = logging.getLogger(f"{project_id}/{instance_id}/{table_id}")
            self.logger.setLevel(logging.WARNING)
            if not self.logger.handlers:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(logging.WARNING)
                self.logger.addHandler(sh)
        else:
            self.logger = logger

        if client is not None:
            self._client = client
        else:
            self._client = bigtable.Client(project=project_id, admin=True,
                                           credentials=credentials)

        self._instance = self.client.instance(instance_id)
        self._table_id = table_id

        self._table = self.instance.table(self.table_id)

        if is_new:
            self._check_and_create_table()

        self._dataset_info = self.check_and_write_table_parameters(
            column_keys.GraphSettings.DatasetInfo, dataset_info,
            required=True, is_new=is_new)

        self._cv_path = self._dataset_info["data_dir"]         # required
        self._mesh_dir = self._dataset_info.get("mesh", None)  # optional

        self._n_layers = self.check_and_write_table_parameters(
            column_keys.GraphSettings.LayerCount, n_layers,
            required=True, is_new=is_new)
        self._fan_out = self.check_and_write_table_parameters(
            column_keys.GraphSettings.FanOut, fan_out,
            required=True, is_new=is_new)
        self._chunk_size = self.check_and_write_table_parameters(
            column_keys.GraphSettings.ChunkSize, chunk_size,
            required=True, is_new=is_new)

        self._dataset_info["graph"] = {"chunk_size": self.chunk_size}

        self._bitmasks = compute_bitmasks(self.n_layers, self.fan_out)

        self._cv = None

        # Hardcoded parameters
        self._n_bits_for_layer_id = 8
        self._cv_mip = 0

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
    def fan_out(self) -> np.uint64:
        return self._fan_out

    @property
    def chunk_size(self) -> np.ndarray:
        return self._chunk_size

    @property
    def segmentation_chunk_size(self) -> np.ndarray:
        return self.cv.scale["chunk_sizes"][0]

    @property
    def segmentation_resolution(self) -> np.ndarray:
        return np.array(self.cv.scale["resolution"])

    @property
    def segmentation_bounds(self) -> np.ndarray:
        return np.array(self.cv.bounds.to_list()).reshape(2, 3)

    @property
    def n_layers(self) -> np.uint64:
        return self._n_layers

    @property
    def bitmasks(self) -> Dict[int, int]:
        return self._bitmasks

    @property
    def cv_mesh_path(self) -> str:
        return "%s/%s" % (self._cv_path, self._mesh_dir)

    @property
    def dataset_info(self) -> object:
        return self._dataset_info

    @property
    def cv_mip(self) -> int:
        return self._cv_mip

    @property
    def cv(self) -> cloudvolume.CloudVolume:
        if self._cv is None:
            self._cv = cloudvolume.CloudVolume(self._cv_path, mip=self._cv_mip,
                                               info=self.dataset_info)

        return self._cv

    @property
    def root_chunk_id(self):
        return self.get_chunk_id(layer=int(self.n_layers), x=0, y=0, z=0)

    def _check_and_create_table(self) -> None:
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

            self.logger.info(f"Table {self.table_id} created")

    def check_and_write_table_parameters(self, column: column_keys._Column,
                                         value: Optional[Union[str, np.uint64]] = None,
                                         required: bool = True,
                                         is_new: bool = False
                                         ) -> Union[str, np.uint64]:
        """ Checks if a parameter already exists in the table. If it already
        exists it returns the stored value, else it stores the given value.
        Storing the given values can be enforced with `is_new`. The function
        raises an exception if no value is passed and the parameter does not
        exist, yet.

        :param column: column_keys._Column
        :param value: Union[str, np.uint64]
        :param required: bool
        :param is_new: bool
        :return: Union[str, np.uint64]
            value
        """
        setting = self.read_byte_row(row_key=row_keys.GraphSettings,
                                     columns=column)

        if (not setting or is_new) and value is not None:
            row = self.mutate_row(row_keys.GraphSettings, {column: value})
            self.bulk_write([row])
        elif not setting and value is None:
            assert not required
            return None
        else:
            value = setting[0].value

        return value

    def is_in_bounds(self, coordinate: Sequence[int]):
        """ Checks whether a coordinate is within the segmentation bounds

        :param coordinate: [int, int, int]
        :return bool
        """
        coordinate = np.array(coordinate)

        if np.any(coordinate < self.segmentation_bounds[0]):
            return False
        elif np.any(coordinate > self.segmentation_bounds[1]):
            return False
        else:
            return True

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

        column = column_keys.Concurrency.CounterID

        # Incrementer row keys start with an "i" followed by the chunk id
        row_key = serializers.serialize_key("i%s" % serializers.pad_node_id(chunk_id))
        append_row = self.table.row(row_key, append=True)
        append_row.increment_cell_value(column.family_id, column.key, step)

        # This increments the row entry and returns the value AFTER incrementing
        latest_row = append_row.commit()
        max_segment_id = column.deserialize(latest_row[column.family_id][column.key][0][0])

        min_segment_id = max_segment_id + 1 - step
        segment_id_range = np.arange(min_segment_id, max_segment_id + 1,
                                     dtype=basetypes.SEGMENT_ID)
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

        # Incrementer row keys start with an "i"
        row_key = serializers.serialize_key("i%s" % serializers.pad_node_id(chunk_id))
        row = self.read_byte_row(row_key, columns=column_keys.Concurrency.CounterID)

        # Read incrementer value (default to 0) and interpret is as Segment ID
        return basetypes.SEGMENT_ID.type(row[0].value if row else 0)

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
        column = column_keys.Concurrency.CounterID

        append_row = self.table.row(row_keys.OperationID, append=True)
        append_row.increment_cell_value(column.family_id, column.key, 1)

        # This increments the row entry and returns the value AFTER incrementing
        latest_row = append_row.commit()
        operation_id_b = latest_row[column.family_id][column.key][0][0]
        operation_id = column.deserialize(operation_id_b)

        return np.uint64(operation_id)

    def get_max_operation_id(self) -> np.int64:
        """  Gets maximal operation id based on the atomic counter

        This is an approximation. It is not guaranteed that all ids smaller or
        equal to this id exists. However, it is guaranteed that no larger id
        exist at the time this function is executed.


        :return: int64
        """
        column = column_keys.Concurrency.CounterID
        row = self.read_byte_row(row_keys.OperationID, columns=column)

        return row[0].value if row else column.basetype(0)

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
            cross_edge_dict[l] = column_keys.Connectivity.CrossChunkEdge.deserialize(b'')

        val_dict = {}
        for cc_layer in u_cce_layers:
            layer_cross_edges = cross_edges[cce_layers == cc_layer]

            if len(layer_cross_edges) > 0:
                val_dict[column_keys.Connectivity.CrossChunkEdge[cc_layer]] = \
                    layer_cross_edges
                cross_edge_dict[cc_layer] = layer_cross_edges
        return cross_edge_dict

    def read_byte_rows(
            self,
            start_key: Optional[bytes] = None,
            end_key: Optional[bytes] = None,
            end_key_inclusive: bool = False,
            row_keys: Optional[Iterable[bytes]] = None,
            columns: Optional[Union[Iterable[column_keys._Column], column_keys._Column]] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            end_time_inclusive: bool = False) -> Dict[bytes, Union[
                Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                List[bigtable.row_data.Cell]
            ]]:
        """Main function for reading a row range or non-contiguous row sets from Bigtable using
        `bytes` keys.

        Keyword Arguments:
            start_key {Optional[bytes]} -- The first row to be read, ignored if `row_keys` is set.
                If None, no lower boundary is used. (default: {None})
            end_key {Optional[bytes]} -- The end of the row range, ignored if `row_keys` is set.
                If None, no upper boundary is used. (default: {None})
            end_key_inclusive {bool} -- Whether or not `end_key` itself should be included in the
                request, ignored if `row_keys` is set or `end_key` is None. (default: {False})
            row_keys {Optional[Iterable[bytes]]} -- An `Iterable` containing possibly
                non-contiguous row keys. Takes precedence over `start_key` and `end_key`.
                (default: {None})
            columns {Optional[Union[Iterable[column_keys._Column], column_keys._Column]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime.datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime.datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})

        Returns:
            Dict[bytes, Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                              List[bigtable.row_data.Cell]]] --
                Returns a dictionary of `byte` rows as keys. Their value will be a mapping of
                columns to a List of cells (one cell per timestamp). Each cell has a `value`
                property, which returns the deserialized field, and a `timestamp` property, which
                returns the timestamp as `datetime.datetime` object.
                If only a single `column_keys._Column` was requested, the List of cells will be
                attached to the row dictionary directly (skipping the column dictionary).
        """

        # Create filters: Column and Time
        filter_ = get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive)

        # Create filters: Rows
        row_set = RowSet()

        if row_keys is not None:
            for row_key in row_keys:
                row_set.add_row_key(row_key)
        elif start_key is not None and end_key is not None:
            row_set.add_row_range_from_keys(
                start_key=start_key,
                start_inclusive=True,
                end_key=end_key,
                end_inclusive=end_key_inclusive)
        else:
            raise cg_exceptions.PreconditionError("Need to either provide a valid set of rows, or"
                                                  " both, a start row and an end row.")

        # Bigtable read with retries
        rows = self._execute_read(row_set=row_set, row_filter=filter_)

        # Deserialize cells
        for row_key, column_dict in rows.items():
            for column, cell_entries in column_dict.items():
                for cell_entry in cell_entries:
                    cell_entry.value = column.deserialize(cell_entry.value)
            # If no column array was requested, reattach single column's values directly to the row
            if isinstance(columns, column_keys._Column):
                rows[row_key] = cell_entries

        return rows

    def read_byte_row(
            self,
            row_key: bytes,
            columns: Optional[Union[Iterable[column_keys._Column], column_keys._Column]] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            end_time_inclusive: bool = False) -> \
                Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                      List[bigtable.row_data.Cell]]:
        """Convenience function for reading a single row from Bigtable using its `bytes` keys.

        Arguments:
            row_key {bytes} -- The row to be read.

        Keyword Arguments:
            columns {Optional[Union[Iterable[column_keys._Column], column_keys._Column]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime.datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime.datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})

        Returns:
            Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                  List[bigtable.row_data.Cell]] --
                Returns a mapping of columns to a List of cells (one cell per timestamp). Each cell
                has a `value` property, which returns the deserialized field, and a `timestamp`
                property, which returns the timestamp as `datetime.datetime` object.
                If only a single `column_keys._Column` was requested, the List of cells is returned
                directly.
        """
        row = self.read_byte_rows(row_keys=[row_key], columns=columns, start_time=start_time,
                                  end_time=end_time, end_time_inclusive=end_time_inclusive)

        if isinstance(columns, column_keys._Column):
            return row.get(row_key, [])
        else:
            return row.get(row_key, {})

    def read_node_id_rows(
            self,
            start_id: Optional[np.uint64] = None,
            end_id: Optional[np.uint64] = None,
            end_id_inclusive: bool = False,
            node_ids: Optional[Iterable[np.uint64]] = None,
            columns: Optional[Union[Iterable[column_keys._Column], column_keys._Column]] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            end_time_inclusive: bool = False) -> Dict[np.uint64, Union[
                Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                List[bigtable.row_data.Cell]
            ]]:
        """Convenience function for reading a row range or non-contiguous row sets from Bigtable
        representing NodeIDs.

        Keyword Arguments:
            start_id {Optional[np.uint64]} -- The first row to be read, ignored if `node_ids` is
                set. If None, no lower boundary is used. (default: {None})
            end_id {Optional[np.uint64]} -- The end of the row range, ignored if `node_ids` is set.
                If None, no upper boundary is used. (default: {None})
            end_id_inclusive {bool} -- Whether or not `end_id` itself should be included in the
                request, ignored if `node_ids` is set or `end_id` is None. (default: {False})
            node_ids {Optional[Iterable[np.uint64]]} -- An `Iterable` containing possibly
                non-contiguous row keys. Takes precedence over `start_id` and `end_id`.
                (default: {None})
            columns {Optional[Union[Iterable[column_keys._Column], column_keys._Column]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime.datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime.datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})

        Returns:
            Dict[np.uint64, Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                                  List[bigtable.row_data.Cell]]] --
                Returns a dictionary of NodeID rows as keys. Their value will be a mapping of
                columns to a List of cells (one cell per timestamp). Each cell has a `value`
                property, which returns the deserialized field, and a `timestamp` property, which
                returns the timestamp as `datetime.datetime` object.
                If only a single `column_keys._Column` was requested, the List of cells will be
                attached to the row dictionary directly (skipping the column dictionary).
        """
        to_bytes = serializers.serialize_uint64
        from_bytes = serializers.deserialize_uint64

        # Read rows (convert Node IDs to row_keys)
        rows = self.read_byte_rows(
            start_key=to_bytes(start_id) if start_id is not None else None,
            end_key=to_bytes(end_id) if end_id is not None else None,
            end_key_inclusive=end_id_inclusive,
            row_keys=(to_bytes(node_id) for node_id in node_ids) if node_ids is not None else None,
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive)

        # Convert row_keys back to Node IDs
        return {from_bytes(row_key): data for (row_key, data) in rows.items()}

    def read_node_id_row(
            self,
            node_id: np.uint64,
            columns: Optional[Union[Iterable[column_keys._Column], column_keys._Column]] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            end_time_inclusive: bool = False) -> \
                Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                      List[bigtable.row_data.Cell]]:
        """Convenience function for reading a single row from Bigtable, representing a NodeID.

        Arguments:
            node_id {np.uint64} -- the NodeID of the row to be read.

        Keyword Arguments:
            columns {Optional[Union[Iterable[column_keys._Column], column_keys._Column]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime.datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime.datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})

        Returns:
            Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                  List[bigtable.row_data.Cell]] --
                Returns a mapping of columns to a List of cells (one cell per timestamp). Each cell
                has a `value` property, which returns the deserialized field, and a `timestamp`
                property, which returns the timestamp as `datetime.datetime` object.
                If only a single `column_keys._Column` was requested, the List of cells is returned
                directly.
        """
        return self.read_byte_row(row_key=serializers.serialize_uint64(node_id), columns=columns,
                                  start_time=start_time, end_time=end_time,
                                  end_time_inclusive=end_time_inclusive)

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

        columns = [column_keys.Connectivity.CrossChunkEdge[l]
                   for l in range(start_layer, end_layer)]
        row_dict = self.read_node_id_row(node_id, columns=columns)

        cross_edge_dict = {}
        for l in range(start_layer, end_layer):
            col = column_keys.Connectivity.CrossChunkEdge[l]
            if col in row_dict:
                cross_edge_dict[l] = row_dict[col][0].value
            else:
                cross_edge_dict[l] = col.deserialize(b'')

        return cross_edge_dict

    def mutate_row(self, row_key: bytes,
                   val_dict: Dict[column_keys._Column, Any],
                   time_stamp: Optional[datetime.datetime] = None,
                   isbytes: bool = False
                   ) -> bigtable.row.Row:
        """ Mutates a single row

        :param row_key: serialized bigtable row key
        :param val_dict: Dict[column_keys._TypedColumn: bytes]
        :param time_stamp: None or datetime
        :return: list
        """
        row = self.table.row(row_key)

        for column, value in val_dict.items():
            if not isbytes:
                value = column.serialize(value)

            row.set_cell(column_family_id=column.family_id,
                         column=column.key,
                         value=value,
                         timestamp=time_stamp)
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

    def _execute_read_thread(self, row_set_and_filter: Tuple[RowSet, RowFilter]):
        row_set, row_filter = row_set_and_filter
        if not row_set.row_keys and not row_set.row_ranges:
            # Check for everything falsy, because Bigtable considers even empty
            # lists of row_keys as no upper/lower bound!
            return {}

        range_read = self.table.read_rows(row_set=row_set, filter_=row_filter)
        res = {v.row_key: partial_row_data_to_column_dict(v)
               for v in range_read}
        return res

    def _execute_read(self, row_set: RowSet, row_filter: RowFilter = None) \
            -> Dict[bytes, Dict[column_keys._Column, bigtable.row_data.PartialRowData]]:
        """ Core function to read rows from Bigtable. Uses standard Bigtable retry logic
        :param row_set: BigTable RowSet
        :param row_filter: BigTable RowFilter
        :return: Dict[bytes, Dict[column_keys._Column, bigtable.row_data.PartialRowData]]
        """

        # FIXME: Bigtable limits the length of the serialized request to 512 KiB. We should
        # calculate this properly (range_read.request.SerializeToString()), but this estimate is
        # good enough for now
        max_row_key_count = 20000
        n_subrequests = max(1, int(np.ceil(len(row_set.row_keys) /
                                           max_row_key_count)))
        n_threads = min(n_subrequests, 2 * mu.n_cpus)

        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[i * max_row_key_count:
                                          (i + 1) * max_row_key_count]
            row_sets.append(r)

        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges

        responses = mu.multithread_func(self._execute_read_thread,
                                        params=((r, row_filter)
                                                for r in row_sets),
                                        debug=n_threads == 1,
                                        n_threads=n_threads)

        combined_response = {}
        for resp in responses:
            combined_response.update(resp)

        return combined_response

    def range_read_chunk(
            self,
            layer: Optional[int] = None,
            x: Optional[int] = None,
            y: Optional[int] = None,
            z: Optional[int] = None,
            chunk_id: Optional[np.uint64] = None,
            columns: Optional[Union[Iterable[column_keys._Column], column_keys._Column]] = None,
            time_stamp: Optional[datetime.datetime] = None) -> Dict[np.uint64, Union[
                Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                List[bigtable.row_data.Cell]
            ]]:
        """Convenience function for reading all NodeID rows of a single chunk from Bigtable.
        Chunk can either be specified by its (layer, x, y, and z coordinate), or by the chunk ID.

        Keyword Arguments:
            layer {Optional[int]} -- The layer of the chunk within the graph (default: {None})
            x {Optional[int]} -- The xth chunk in x dimension within the graph, within `layer`.
                (default: {None})
            y {Optional[int]} -- The yth chunk in y dimension within the graph, within `layer`.
                (default: {None})
            z {Optional[int]} -- The zth chunk in z dimension within the graph, within `layer`.
                (default: {None})
            chunk_id {Optional[np.uint64]} -- Alternative way to specify the chunk, if the Chunk ID
                is already known. (default: {None})
            columns {Optional[Union[Iterable[column_keys._Column], column_keys._Column]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            time_stamp {Optional[datetime.datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})

        Returns:
            Dict[np.uint64, Union[Dict[column_keys._Column, List[bigtable.row_data.Cell]],
                                  List[bigtable.row_data.Cell]]] --
                Returns a dictionary of NodeID rows as keys. Their value will be a mapping of
                columns to a List of cells (one cell per timestamp). Each cell has a `value`
                property, which returns the deserialized field, and a `timestamp` property, which
                returns the timestamp as `datetime.datetime` object.
                If only a single `column_keys._Column` was requested, the List of cells will be
                attached to the row dictionary directly (skipping the column dictionary).
        """
        if chunk_id is not None:
            x, y, z = self.get_chunk_coordinates(chunk_id)
            layer = self.get_chunk_layer(chunk_id)
        elif layer is not None and x is not None and y is not None and z is not None:
            chunk_id = self.get_chunk_id(layer=layer, x=x, y=y, z=z)
        else:
            raise Exception("Either chunk_id or layer and coordinates have to be defined")

        if layer == 1:
            max_segment_id = self.get_segment_id_limit(chunk_id)
        else:
            max_segment_id = self.get_max_seg_id(chunk_id=chunk_id)

        # Define BigTable keys
        start_id = self.get_node_id(np.uint64(0), chunk_id=chunk_id)
        end_id = self.get_node_id(max_segment_id, chunk_id=chunk_id)

        try:
            rr = self.read_node_id_rows(
                start_id=start_id,
                end_id=end_id,
                end_id_inclusive=True,
                columns=columns,
                end_time=time_stamp,
                end_time_inclusive=True)
        except Exception as err:
            raise Exception("Unable to consume chunk read: "
                            "[%d, %d, %d], l = %d: %s" %
                            (x, y, z, layer, err))
        return rr

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
                              bb_offset: Sequence[np.int],
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

        val_dict = {column_keys.OperationLogs.UserID: user_id,
                    column_keys.OperationLogs.RootID:
                        np.array(root_ids, dtype=np.uint64),
                    column_keys.OperationLogs.SourceID:
                        np.array(source_ids),
                    column_keys.OperationLogs.SinkID:
                        np.array(sink_ids),
                    column_keys.OperationLogs.SourceCoordinate:
                        np.array(source_coords),
                    column_keys.OperationLogs.SinkCoordinate:
                        np.array(sink_coords),
                    column_keys.OperationLogs.BoundingBoxOffset:
                        np.array(bb_offset),
                    column_keys.OperationLogs.RemovedEdge:
                        np.array(removed_edges, dtype=np.uint64)}

        row = self.mutate_row(serializers.serialize_uint64(operation_id),
                              val_dict, time_stamp)

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
            affinities = np.array([], dtype=column_keys.Connectivity.Affinity.basetype)
        else:
            affinities = np.array(affinities, dtype=column_keys.Connectivity.Affinity.basetype)

        val_dict = {column_keys.OperationLogs.UserID: user_id,
                    column_keys.OperationLogs.RootID:
                        np.array(root_ids, dtype=np.uint64),
                    column_keys.OperationLogs.SourceID:
                        np.array(source_ids),
                    column_keys.OperationLogs.SinkID:
                        np.array(sink_ids),
                    column_keys.OperationLogs.SourceCoordinate:
                        np.array(source_coords),
                    column_keys.OperationLogs.SinkCoordinate:
                        np.array(sink_coords),
                    column_keys.OperationLogs.AddedEdge:
                        np.array(added_edges, dtype=np.uint64),
                    column_keys.OperationLogs.Affinity: affinities}

        row = self.mutate_row(serializers.serialize_uint64(operation_id),
                              val_dict, time_stamp)

        return row

    def read_log_row(self, operation_id: np.uint64):
        """ Reads a log row (both split and merge)

        :param operation_id: np.uint64
        :return: dict
        """
        columns = [column_keys.OperationLogs.UserID, column_keys.OperationLogs.RootID,
                   column_keys.OperationLogs.SinkID, column_keys.OperationLogs.SourceID,
                   column_keys.OperationLogs.SourceCoordinate,
                   column_keys.OperationLogs.SinkCoordinate, column_keys.OperationLogs.AddedEdge,
                   column_keys.OperationLogs.Affinity, column_keys.OperationLogs.RemovedEdge,
                   column_keys.OperationLogs.BoundingBoxOffset]
        row_dict = self.read_node_id_row(operation_id, columns=columns)
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
        #     self.logger.debug("CC in chunk: %.3fs" % (time.time() - time_start))

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the log / speed measurement

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
            #     self.logger.debug("%5d at %5d - %.5fs             " %
            #                       (i_cc, node_c, dt / node_c), end="\r")

            node_ids = np.array(list(cc))

            u_chunk_ids = np.unique([self.get_chunk_id(n) for n in node_ids])

            if len(u_chunk_ids) > 1:
                self.logger.error(f"Found multiple chunk ids: {u_chunk_ids}")
                raise Exception()

            # Create parent id
            parent_id = parent_ids[i_cc]

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
                affinities = np.concatenate([connected_affs, disconnected_affs])
                areas = np.concatenate([connected_areas, disconnected_areas])
                connected = np.arange(len(connected_ids), dtype=np.int)

                val_dict = {column_keys.Connectivity.Partner: partners,
                            column_keys.Connectivity.Affinity: affinities,
                            column_keys.Connectivity.Area: areas,
                            column_keys.Connectivity.Connected: connected,
                            column_keys.Hierarchy.Parent: parent_id}

                rows.append(self.mutate_row(serializers.serialize_uint64(node_id),
                                            val_dict, time_stamp=time_stamp))
                node_c += 1
                time_dict["creating_lv1_row"].append(time.time() - time_start_2)

            time_start_1 = time.time()
            # Create parent node
            rows.append(self.mutate_row(serializers.serialize_uint64(parent_id),
                                        {column_keys.Hierarchy.Child: node_ids},
                                        time_stamp=time_stamp))

            time_dict["creating_lv2_row"].append(time.time() - time_start_1)
            time_start_1 = time.time()

            cce_layers = self.get_cross_chunk_edges_layer(parent_cross_edges)
            u_cce_layers = np.unique(cce_layers)

            val_dict = {}
            for cc_layer in u_cce_layers:
                layer_cross_edges = parent_cross_edges[cce_layers == cc_layer]

                if len(layer_cross_edges) > 0:
                    val_dict[column_keys.Connectivity.CrossChunkEdge[cc_layer]] = \
                        layer_cross_edges

            if len(val_dict) > 0:
                rows.append(self.mutate_row(serializers.serialize_uint64(parent_id),
                                            val_dict, time_stamp=time_stamp))
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
            self.logger.debug("Time creating rows: %.3fs for %d ccs with %d nodes" %
                              (time.time() - time_start, len(ccs), node_c))

            for k in time_dict.keys():
                self.logger.debug("%s -- %.3fms for %d instances -- avg = %.3fms" %
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

            columns = [column_keys.Hierarchy.Child] + \
                      [column_keys.Connectivity.CrossChunkEdge[l]
                       for l in range(layer_id - 1, self.n_layers)]
            range_read = self.range_read_chunk(layer_id - 1, x, y, z,
                                               columns=columns)

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
            for row_id, row_data in range_read.items():
                segment_id = self.get_segment_id(row_id)

                cross_edge_columns = {k: v for (k, v) in row_data.items()
                                      if k.family_id == self.cross_edge_family_id}
                if cross_edge_columns:
                    row_cell_dict[row_id] = cross_edge_columns

                node_child_ids = row_data[column_keys.Hierarchy.Child][0].value

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
                        row_key = column_keys.Connectivity.CrossChunkEdge[l]
                        if row_key in cell_family:
                            cross_edge_dict[row_id][l] = cell_family[row_key][0].value

                    if int(layer_id - 1) in cross_edge_dict[row_id]:
                        atomic_cross_edges = cross_edge_dict[row_id][layer_id - 1]

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
                parent_cross_edges = {l: [] for l in range(layer_id, self.n_layers)}

                # Add rows for nodes that are in this chunk
                for node_id in node_ids:
                    if node_id in cross_edge_dict:
                        # Extract edges relevant to this node
                        for l in range(layer_id, self.n_layers):
                            if l in cross_edge_dict[node_id] and len(cross_edge_dict[node_id][l]) > 0:
                                parent_cross_edges[l].append(cross_edge_dict[node_id][l])

                    # Create node
                    val_dict = {column_keys.Hierarchy.Parent: parent_id}

                    rows.append(self.mutate_row(serializers.serialize_uint64(node_id),
                                                val_dict, time_stamp=time_stamp))

                # Create parent node
                val_dict = {column_keys.Hierarchy.Child: node_ids}

                rows.append(self.mutate_row(serializers.serialize_uint64(parent_id),
                                            val_dict, time_stamp=time_stamp))

                val_dict = {}

                for l in range(layer_id, self.n_layers):
                    if l in parent_cross_edges and len(parent_cross_edges[l]) > 0:
                        val_dict[column_keys.Connectivity.CrossChunkEdge[l]] = \
                            np.concatenate(parent_cross_edges[l])

                if len(val_dict) > 0:
                    rows.append(self.mutate_row(serializers.serialize_uint64(parent_id),
                                                val_dict, time_stamp=time_stamp))

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
            self.logger.debug("Time iterating through subchunks: %.3fs" %
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
            self.logger.debug("Time resolving cross chunk edges: %.3fs" %
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
            self.logger.debug("Time connected components: %.3fs" %
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
            self.logger.debug("Time writing %d connected components in layer %d: %.3fs" %
                              (len(ccs), layer_id, time.time() - time_start))

    def get_atomic_cross_edge_dict(self, node_id: np.uint64,
                                   layer_ids: Sequence[int] = None):
        """ Extracts all atomic cross edges and serves them as a dictionary

        :param node_id: np.uint64
        :param layer_ids: list of ints
        :return: dict
        """
        if isinstance(layer_ids, int):
            layer_ids = [layer_ids]

        if layer_ids is None:
            layer_ids = list(range(2, self.n_layers))

        if not layer_ids:
            return {}

        columns = [column_keys.Connectivity.CrossChunkEdge[l] for l in layer_ids]

        row = self.read_node_id_row(node_id, columns=columns)

        if not row:
            return {}

        atomic_cross_edges = {}

        for l in layer_ids:
            column = column_keys.Connectivity.CrossChunkEdge[l]

            atomic_cross_edges[l] = []

            if column in row:
                atomic_cross_edges[l] = row[column][0].value

        return atomic_cross_edges

    def get_parent(self, node_id: np.uint64,
                   get_only_relevant_parent: bool = True,
                   time_stamp: Optional[datetime.datetime] = None) -> Union[
                       List[Tuple[np.uint64, datetime.datetime]], np.uint64]:
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

        parents = self.read_node_id_row(node_id,
                                        columns=column_keys.Hierarchy.Parent,
                                        end_time=time_stamp,
                                        end_time_inclusive=True)

        if not parents:
            return None

        if get_only_relevant_parent:
            return parents[0].value

        return [(p.value, p.timestamp) for p in parents]

    def get_children(self, node_id: Union[Iterable[np.uint64], np.uint64],
                     flatten: bool = False) -> Union[Dict[np.uint64, np.ndarray], np.ndarray]:
        """Returns children for the specified NodeID or NodeIDs

        :param node_id: The NodeID or NodeIDs for which to retrieve children
        :type node_id: Union[Iterable[np.uint64], np.uint64]
        :param flatten: If True, combine all children into a single array, else generate a map
            of input ``node_id`` to their respective children.
        :type flatten: bool, default is True
        :return: Children for each requested NodeID. The return type depends on the ``flatten``
            parameter.
        :rtype: Union[Dict[np.uint64, np.ndarray], np.ndarray]
        """
        if np.isscalar(node_id):
            children = self.read_node_id_row(node_id=node_id, columns=column_keys.Hierarchy.Child)
            if not children:
                return np.empty(0, dtype=basetypes.NODE_ID)
            return children[0].value
        else:
            children = self.read_node_id_rows(node_ids=node_id, columns=column_keys.Hierarchy.Child)
            if flatten:
                if not children:
                    return np.empty(0, dtype=basetypes.NODE_ID)
                return np.concatenate([x[0].value for x in children.values()])
            return {x: children[x][0].value
                       if x in children else np.empty(0, dtype=basetypes.NODE_ID)
                    for x in node_id}

    def get_latest_roots(self, time_stamp: Optional[datetime.datetime] = get_max_time(),
                         n_threads: int = 1) -> Sequence[np.uint64]:
        """ Reads _all_ root ids

        :param time_stamp: datetime.datetime
        :param n_threads: int
        :return: array of np.uint64
        """

        return chunkedgraph_comp.get_latest_roots(self, time_stamp=time_stamp,
                                                  n_threads=n_threads)

    def get_delta_roots(self,
                        time_stamp_start: datetime.datetime,
                        time_stamp_end: Optional[datetime.datetime] = None,
                        min_seg_id: int =1,
                        n_threads: int = 1) -> Sequence[np.uint64]:
        """ Returns root ids that have expired or have been created between two timestamps

        :param time_stamp_start: datetime.datetime
            starting timestamp to return deltas from
        :param time_stamp_end: datetime.datetime
            ending timestamp to return deltasfrom
        :param min_seg_id: int (default=1)
            only search from this seg_id and higher (note not a node_id.. use get_seg_id)
        :param n_threads: int (default=1)
            number of threads to use in performing search
        :return new_ids, expired_ids: np.arrays of np.uint64
            new_ids is an array of root_ids for roots that were created after time_stamp_start
            and are still current as of time_stamp_end.
            expired_ids is list of node_id's for roots the expired after time_stamp_start
            but before time_stamp_end.
        """

        return chunkedgraph_comp.get_delta_roots(self, time_stamp_start=time_stamp_start,
                                                  time_stamp_end=time_stamp_end,
                                                  min_seg_id=min_seg_id,
                                                  n_threads=n_threads)

    def get_root(self, node_id: np.uint64,
                 time_stamp: Optional[datetime.datetime] = None,
                 n_tries: int = 1) -> Union[List[np.uint64], np.uint64]:
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

        parent_id = node_id

        for i_try in range(n_tries):
            parent_id = node_id

            for i_layer in range(self.get_chunk_layer(node_id)+1,
                                 int(self.n_layers + 1)):

                temp_parent_id = self.get_parent(parent_id,
                                                 time_stamp=time_stamp)

                if temp_parent_id is None:
                    break
                else:
                    parent_id = temp_parent_id

            if self.get_chunk_layer(parent_id) == self.n_layers:
                break
            else:
                time.sleep(.5)

        if self.get_chunk_layer(parent_id) != self.n_layers:
            raise Exception("Cannot find root id {}, {}".format(node_id,
                                                                time_stamp))

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

                self.logger.debug("operation id: %d - root id: %d" %
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
            self.logger.debug(f"Try {i_try}")

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

        operation_id_b = serializers.serialize_uint64(operation_id)

        lock_column = column_keys.Concurrency.Lock
        new_parents_column = column_keys.Hierarchy.NewParent

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if it is still valid (timestamp younger than
        # LOCK_EXPIRED_TIME_DELTA) and if there is no new parent (== new_parents
        # exists)

        time_cutoff = datetime.datetime.utcnow() - LOCK_EXPIRED_TIME_DELTA

        # Comply to resolution of BigTables TimeRange
        time_cutoff -= datetime.timedelta(
            microseconds=time_cutoff.microsecond % 1000)

        time_filter = TimestampRangeFilter(TimestampRange(start=time_cutoff))

        # lock_key_filter = ColumnQualifierRegexFilter(lock_column.key)
        # new_parents_key_filter = ColumnQualifierRegexFilter(new_parents_column.key)

        lock_key_filter = ColumnRangeFilter(
            column_family_id=lock_column.family_id,
            start_column=lock_column.key,
            end_column=lock_column.key,
            inclusive_start=True,
            inclusive_end=True)

        new_parents_key_filter = ColumnRangeFilter(
            column_family_id=new_parents_column.family_id,
            start_column=new_parents_column.key,
            end_column=new_parents_column.key,
            inclusive_start=True,
            inclusive_end=True)

        # Combine filters together
        chained_filter = RowFilterChain([time_filter, lock_key_filter])
        combined_filter = ConditionalRowFilter(
            base_filter=chained_filter,
            true_filter=PassAllFilter(True),
            false_filter=new_parents_key_filter)

        # Get conditional row using the chained filter
        root_row = self.table.row(serializers.serialize_uint64(root_id),
                                  filter_=combined_filter)

        # Set row lock if condition returns no results (state == False)
        time_stamp = datetime.datetime.utcnow()

        # Comply to resolution of BigTables TimeRange
        time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                      round_up=False)

        root_row.set_cell(lock_column.family_id, lock_column.key, operation_id_b, state=False,
                          timestamp=time_stamp)

        # The lock was acquired when set_cell returns False (state)
        lock_acquired = not root_row.commit()

        if not lock_acquired:
            row = self.read_node_id_row(root_id, columns=lock_column)

            l_operation_ids = [cell.value for cell in row]
            self.logger.debug(f"Locked operation ids: {l_operation_ids}")

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
        lock_column = column_keys.Concurrency.Lock
        operation_id_b = lock_column.serialize(operation_id)

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if it is still valid (timestamp younger than
        # LOCK_EXPIRED_TIME_DELTA) and if the given operation_id is still
        # the active lock holder

        time_cutoff = datetime.datetime.utcnow() - LOCK_EXPIRED_TIME_DELTA

        # Comply to resolution of BigTables TimeRange
        time_cutoff -= datetime.timedelta(
            microseconds=time_cutoff.microsecond % 1000)

        time_filter = TimestampRangeFilter(TimestampRange(start=time_cutoff))

        # column_key_filter = ColumnQualifierRegexFilter(lock_column.key)
        # value_filter = ColumnQualifierRegexFilter(operation_id_b)

        column_key_filter = ColumnRangeFilter(
            column_family_id=lock_column.family_id,
            start_column=lock_column.key,
            end_column=lock_column.key,
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
        root_row = self.table.row(serializers.serialize_uint64(root_id),
                                  filter_=chained_filter)

        # Delete row if conditions are met (state == True)
        root_row.delete_cell(lock_column.family_id, lock_column.key, state=True)

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
                self.logger.warning(f"check_and_renew_root_locks failed - {root_id}")
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
        lock_column = column_keys.Concurrency.Lock
        new_parents_column = column_keys.Hierarchy.NewParent

        operation_id_b = lock_column.serialize(operation_id)

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if the given operation_id is still the active lock holder
        # and there is no new parent (== new_parents column exists). The latter
        # is not necessary but we include it as a backup to prevent things
        # from going really bad.

        # column_key_filter = ColumnQualifierRegexFilter(lock_column.key)
        # value_filter = ColumnQualifierRegexFilter(operation_id_b)

        column_key_filter = ColumnRangeFilter(
            column_family_id=lock_column.family_id,
            start_column=lock_column.key,
            end_column=lock_column.key,
            inclusive_start=True,
            inclusive_end=True)

        value_filter = ValueRangeFilter(
            start_value=operation_id_b,
            end_value=operation_id_b,
            inclusive_start=True,
            inclusive_end=True)

        new_parents_key_filter = ColumnRangeFilter(
            column_family_id=self.family_id,
            start_column=new_parents_column.key,
            end_column=new_parents_column.key,
            inclusive_start=True,
            inclusive_end=True)

        # Chain these filters together
        chained_filter = RowFilterChain([column_key_filter, value_filter])
        combined_filter = ConditionalRowFilter(
            base_filter=chained_filter,
            true_filter=new_parents_key_filter,
            false_filter=PassAllFilter(True))

        # Get conditional row using the chained filter
        root_row = self.table.row(serializers.serialize_uint64(root_id),
                                  filter_=combined_filter)

        # Set row lock if condition returns a result (state == True)
        root_row.set_cell(lock_column.family_id, lock_column.key, operation_id_b, state=False)

        # The lock was acquired when set_cell returns True (state)
        lock_acquired = not root_row.commit()

        return lock_acquired

    def get_latest_root_id(self, root_id: np.uint64) -> np.ndarray:
        """ Returns the latest root id associated with the provided root id

        :param root_id: uint64
        :return: list of uint64s
        """

        id_working_set = [root_id]
        column = column_keys.Hierarchy.NewParent
        latest_root_ids = []

        while len(id_working_set) > 0:
            next_id = id_working_set[0]
            del(id_working_set[0])
            row = self.read_node_id_row(next_id, columns=column)

            # Check if a new root id was attached to this root id
            if row:
                id_working_set.extend(row[0].value)
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
                row = self.read_node_id_row(next_id, columns=[column_keys.Hierarchy.NewParent,
                                                              column_keys.Hierarchy.Child])
                if column_keys.Hierarchy.NewParent in row:
                    ids = row[column_keys.Hierarchy.NewParent][0].value
                    row_time_stamp = row[column_keys.Hierarchy.NewParent][0].timestamp
                elif column_keys.Hierarchy.Child in row:
                    ids = None
                    row_time_stamp = row[column_keys.Hierarchy.Child][0].timestamp
                else:
                    raise cg_exceptions.ChunkedGraphError("Error retrieving future root ID of %s" % next_id)

                if row_time_stamp < time_stamp:
                    if ids is not None:
                        temp_next_ids.extend(ids)

                    if next_id != root_id:
                        id_history.append(next_id)

            next_ids = temp_next_ids

        return np.unique(np.array(id_history, dtype=np.uint64))

    def get_past_root_ids(self, root_id: np.uint64,
                          time_stamp: Optional[datetime.datetime] =
                          get_min_time()) -> np.ndarray:
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
                row = self.read_node_id_row(next_id, columns=[column_keys.Hierarchy.FormerParent,
                                                              column_keys.Hierarchy.Child])
                if column_keys.Hierarchy.FormerParent in row:
                    ids = row[column_keys.Hierarchy.FormerParent][0].value
                    row_time_stamp = row[column_keys.Hierarchy.FormerParent][0].timestamp
                elif column_keys.Hierarchy.Child in row:
                    ids = None
                    row_time_stamp = row[column_keys.Hierarchy.Child][0].timestamp
                else:
                    raise cg_exceptions.ChunkedGraphError("Error retrieving past root ID of %s" % next_id)

                if row_time_stamp > time_stamp:
                    if ids is not None:
                        temp_next_ids.extend(ids)

                    if next_id != root_id:
                        id_history.append(next_id)

            next_ids = temp_next_ids

        return np.unique(np.array(id_history, dtype=np.uint64))

    def get_root_id_history(self, root_id: np.uint64,
                            time_stamp_past:
                            Optional[datetime.datetime] = get_min_time(),
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

    def get_change_log(self, root_id: np.uint64,
                       correct_for_wrong_coord_type: bool = True,
                       time_stamp_past: Optional[datetime.datetime] = get_min_time()
                       ) -> dict:
        """ Returns all past root ids for this root

        This search happens in a monotic fashion. At no point are future root
        ids of past root ids taken into account.

        :param root_id: np.uint64
        :param correct_for_wrong_coord_type: bool
            pinky100? --> True
        :param time_stamp_past: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole past
        :return: past ids, merge sv ids, merge edge coords, split sv ids
        """
        if time_stamp_past.tzinfo is None:
            time_stamp_past = UTC.localize(time_stamp_past)

        id_history = []
        merge_history = []
        merge_history_edges = []
        split_history = []

        next_ids = [root_id]
        while len(next_ids):
            temp_next_ids = []
            former_parent_col = column_keys.Hierarchy.FormerParent
            row_dict = self.read_node_id_rows(node_ids=next_ids,
                                              columns=[former_parent_col])

            for row in row_dict.values():
                if column_keys.Hierarchy.FormerParent in row:
                    if time_stamp_past > row[former_parent_col][0].timestamp:
                        continue

                    ids = row[former_parent_col][0].value

                    lock_col = column_keys.Concurrency.Lock
                    former_row = self.read_node_id_row(ids[0],
                                                       columns=[lock_col])
                    operation_id = former_row[lock_col][0].value
                    log_row = self.read_log_row(operation_id)
                    is_merge = column_keys.OperationLogs.AddedEdge in log_row

                    for id_ in ids:
                        if id_ in id_history:
                            continue

                        id_history.append(id_)
                        temp_next_ids.append(id_)

                    if is_merge:
                        added_edges = log_row[column_keys.OperationLogs.AddedEdge][0].value
                        merge_history.append(added_edges)

                        coords = [log_row[column_keys.OperationLogs.SourceCoordinate][0].value,
                                  log_row[column_keys.OperationLogs.SinkCoordinate][0].value]

                        if correct_for_wrong_coord_type:
                            # A little hack because we got the datatype wrong...
                            coords = [np.frombuffer(coords[0]),
                                      np.frombuffer(coords[1])]
                            coords *= self.segmentation_resolution

                        merge_history_edges.append(coords)

                    if not is_merge:
                        removed_edges = log_row[column_keys.OperationLogs.RemovedEdge][0].value
                        split_history.append(removed_edges)
                else:
                    continue

            next_ids = temp_next_ids

        return {"past_ids": np.unique(np.array(id_history, dtype=np.uint64)),
                "merge_edges": np.array(merge_history),
                "merge_edge_coords": np.array(merge_history_edges),
                "split_edges": np.array(split_history)}

    def normalize_bounding_box(self,
                               bounding_box: Optional[Sequence[Sequence[int]]],
                               bb_is_coordinate: bool) -> \
            Union[Sequence[Sequence[int]], None]:
        if bounding_box is None:
            return None

        if bb_is_coordinate:
            bounding_box = np.array(bounding_box,
                                    dtype=np.float32) / self.chunk_size
            bounding_box[0] = np.floor(bounding_box[0])
            bounding_box[1] = np.ceil(bounding_box[1])
            return bounding_box.astype(np.int)
        else:
            return np.array(bounding_box, dtype=np.int)

    def _get_subgraph_higher_layer_nodes(
            self, node_id: np.uint64,
            bounding_box: Optional[Sequence[Sequence[int]]],
            return_layers: Sequence[int],
            verbose: bool):

        layer = self.get_chunk_layer(node_id)
        assert layer > 1

        def _get_subgraph_higher_layer_nodes_threaded(
                node_ids: Iterable[np.uint64]) -> List[np.uint64]:
            children = self.get_children(node_ids, flatten=True)

            if len(children) > 0 and bounding_box is not None:
                chunk_coordinates = np.array([self.get_chunk_coordinates(c) for c in children])

                bounding_box_layer = bounding_box / self.fan_out ** np.max([0, (layer - 3)])

                bound_check = np.array([
                    np.all(chunk_coordinates < bounding_box_layer[1], axis=1),
                    np.all(chunk_coordinates + 1 > bounding_box_layer[0], axis=1)]).T

                bound_check_mask = np.all(bound_check, axis=1)
                children = children[bound_check_mask]

            return children

        nodes_per_layer = {}
        child_ids = np.array([node_id], dtype=np.uint64)
        stop_layer = max(2, np.min(return_layers))

        if layer in return_layers:
            nodes_per_layer[layer] = child_ids

        if verbose:
            time_start = time.time()

        while layer > stop_layer:
            # Use heuristic to guess the optimal number of threads
            n_child_ids = len(child_ids)
            this_n_threads = np.min([int(n_child_ids // 50000) + 1, mu.n_cpus])

            child_ids = np.fromiter(chain.from_iterable(mu.multithread_func(
                _get_subgraph_higher_layer_nodes_threaded,
                np.array_split(child_ids, this_n_threads),
                n_threads=this_n_threads, debug=this_n_threads == 1)), np.uint64)

            if verbose:
                self.logger.debug("Layer %d: %.3fms for %d children with %d threads" %
                                  (layer, (time.time() - time_start) * 1000, n_child_ids,
                                   this_n_threads))
                time_start = time.time()

            layer -= 1
            if layer in return_layers:
                nodes_per_layer[layer] = child_ids

        return nodes_per_layer

    def get_subgraph_edges(self, agglomeration_id: np.uint64,
                           bounding_box: Optional[Sequence[Sequence[int]]] = None,
                           bb_is_coordinate: bool = False, verbose: bool = True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return all atomic edges between supervoxels belonging to the
            specified agglomeration ID within the defined bounding box

        :param agglomeration_id: int
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param verbose: bool
        :return: edge list
        """

        def _get_subgraph_layer2_edges(node_ids) -> \
                Tuple[List[np.ndarray], List[np.float32], List[np.uint64]]:
            return self.get_subgraph_chunk(node_ids, time_stamp=time_stamp)

        time_stamp = self.read_node_id_row(agglomeration_id,
                                           columns=column_keys.Hierarchy.Child)[0].timestamp

        bounding_box = self.normalize_bounding_box(bounding_box, bb_is_coordinate)

        # Layer 3+
        child_ids = self._get_subgraph_higher_layer_nodes(
            node_id=agglomeration_id, bounding_box=bounding_box,
            return_layers=[2], verbose=verbose)[2]

        # Layer 2
        if verbose:
            time_start = time.time()

        child_chunk_ids = self.get_chunk_ids_from_node_ids(child_ids)
        u_ccids = np.unique(child_chunk_ids)

        child_blocks = []
        # Make blocks of child ids that are in the same chunk
        for u_ccid in u_ccids:
            child_blocks.append(child_ids[child_chunk_ids == u_ccid])

        n_child_ids = len(child_ids)
        this_n_threads = np.min([int(n_child_ids // 50000) + 1, mu.n_cpus])

        edge_infos = mu.multithread_func(
            _get_subgraph_layer2_edges,
            np.array_split(child_ids, this_n_threads),
            n_threads=this_n_threads, debug=this_n_threads == 1)

        affinities = np.array([], dtype=np.float32)
        areas = np.array([], dtype=np.uint64)
        edges = np.array([], dtype=np.uint64).reshape(0, 2)

        for edge_info in edge_infos:
            _edges, _affinities, _areas = edge_info
            areas = np.concatenate([areas, _areas])
            affinities = np.concatenate([affinities, _affinities])
            edges = np.concatenate([edges, _edges])

        if verbose:
            self.logger.debug("Layer %d: %.3fms for %d children with %d threads" %
                              (2, (time.time() - time_start) * 1000, n_child_ids,
                               this_n_threads))

        return edges, affinities, areas

    def get_subgraph_nodes(self, agglomeration_id: np.uint64,
                           bounding_box: Optional[Sequence[Sequence[int]]] = None,
                           bb_is_coordinate: bool = False,
                           return_layers: List[int] = [1],
                           verbose: bool = True) -> \
            Union[Dict[int, np.ndarray], np.ndarray]:
        """ Return all nodes belonging to the specified agglomeration ID within
            the defined bounding box and requested layers.

        :param agglomeration_id: np.uint64
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param return_layers: List[int]
        :param verbose: bool
        :return: np.array of atomic IDs if single layer is requested,
                 Dict[int, np.array] if multiple layers are requested
        """

        def _get_subgraph_layer2_nodes(node_ids: Iterable[np.uint64]) -> np.ndarray:
            return self.get_children(node_ids, flatten=True)

        stop_layer = np.min(return_layers)
        bounding_box = self.normalize_bounding_box(bounding_box, bb_is_coordinate)

        # Layer 3+
        if stop_layer >= 2:
            nodes_per_layer = self._get_subgraph_higher_layer_nodes(
                node_id=agglomeration_id, bounding_box=bounding_box,
                return_layers=return_layers, verbose=verbose)
        else:
            # Need to retrieve layer 2 even if the user doesn't require it
            nodes_per_layer = self._get_subgraph_higher_layer_nodes(
                node_id=agglomeration_id, bounding_box=bounding_box,
                return_layers=return_layers+[2], verbose=verbose)

            # Layer 2
            if verbose:
                time_start = time.time()

            child_ids = nodes_per_layer[2]
            if 2 not in return_layers:
                del nodes_per_layer[2]

            # Use heuristic to guess the optimal number of threads
            n_child_ids = len(child_ids)
            this_n_threads = np.min([int(n_child_ids // 50000) + 1, mu.n_cpus])

            child_ids = np.fromiter(chain.from_iterable(mu.multithread_func(
                _get_subgraph_layer2_nodes,
                np.array_split(child_ids, this_n_threads),
                n_threads=this_n_threads, debug=this_n_threads == 1)), dtype=np.uint64)

            if verbose:
                self.logger.debug("Layer 2: %.3fms for %d children with %d threads" %
                                  ((time.time() - time_start) * 1000, n_child_ids,
                                   this_n_threads))

            nodes_per_layer[1] = child_ids

        if len(nodes_per_layer) == 1:
            return list(nodes_per_layer.values())[0]
        else:
            return nodes_per_layer

    def flatten_row_dict(self, row_dict: Dict[column_keys._Column,
                                              List[bigtable.row_data.Cell]]) -> Dict:
        """ Flattens multiple entries to columns by appending them

        :param row_dict: dict
            family key has to be resolved
        :return: dict
        """

        flattened_row_dict = {}
        for column, column_entries in row_dict.items():
            flattened_row_dict[column] = []

            if len(column_entries) > 0:
                for column_entry in column_entries[::-1]:
                    flattened_row_dict[column].append(column_entry.value)

                if np.isscalar(column_entry.value):
                    flattened_row_dict[column] = np.array(flattened_row_dict[column])
                else:
                    flattened_row_dict[column] = np.concatenate(flattened_row_dict[column])
            else:
                flattened_row_dict[column] = column.deserialize(b'')

            if column == column_keys.Connectivity.Connected:
                u_ids, c_ids = np.unique(flattened_row_dict[column],
                                         return_counts=True)
                flattened_row_dict[column] = u_ids[(c_ids % 2) == 1].astype(column.basetype)

        return flattened_row_dict

    def get_chunk_split_partners(self, atomic_id: np.uint64):
        """ Finds all atomic nodes beloning to the same supervoxel before
            chunking (affs == inf)

        :param atomic_id: np.uint64
        :return: list of np.uint64
        """

        chunk_split_partners = [atomic_id]
        atomic_ids = [atomic_id]

        while len(atomic_ids) > 0:
            atomic_id = atomic_ids[0]
            del atomic_ids[0]

            partners, affs, _ = self.get_atomic_partners(atomic_id,
                                                         include_connected_partners=True,
                                                         include_disconnected_partners=False)

            m = np.isinf(affs)

            inf_partners = partners[m]
            new_chunk_split_partners = inf_partners[~np.in1d(inf_partners, chunk_split_partners)]
            atomic_ids.extend(new_chunk_split_partners)
            chunk_split_partners.extend(new_chunk_split_partners)

        return chunk_split_partners

    def get_all_original_partners(self, atomic_id: np.uint64):
        """ Finds all partners from the unchunked region graph
            Merges split supervoxels over chunk boundaries first (affs == inf)

        :param atomic_id: np.uint64
        :return: dict np.uint64 -> np.uint64
        """

        atomic_ids = [atomic_id]
        partner_dict = {}

        while len(atomic_ids) > 0:
            atomic_id = atomic_ids[0]
            del atomic_ids[0]

            partners, affs, _ = self.get_atomic_partners(atomic_id,
                                                         include_connected_partners=True,
                                                         include_disconnected_partners=False)

            m = np.isinf(affs)
            partner_dict[atomic_id] = partners[~m]

            inf_partners = partners[m]
            new_chunk_split_partners = inf_partners[
                ~np.in1d(inf_partners, list(partner_dict.keys()))]
            atomic_ids.extend(new_chunk_split_partners)

        return partner_dict

    def get_atomic_node_partners(self, atomic_id: np.uint64,
                                 time_stamp: datetime.datetime = get_max_time()
                                 ) -> Dict:
        """ Reads register partner ids

        :param atomic_id: np.uint64
        :param time_stamp: datetime.datetime
        :return: dict
        """
        col_partner = column_keys.Connectivity.Partner
        col_connected = column_keys.Connectivity.Connected
        columns = [col_partner, col_connected]
        row_dict = self.read_node_id_row(atomic_id, columns=columns,
                                         end_time=time_stamp, end_time_inclusive=True)
        flattened_row_dict = self.flatten_row_dict(row_dict)
        return flattened_row_dict[col_partner][flattened_row_dict[col_connected]]

    def _get_atomic_node_info_core(self, row_dict) -> Dict:
        """ Reads connectivity information for a single node

        :param atomic_id: np.uint64
        :param time_stamp: datetime.datetime
        :return: dict
        """
        flattened_row_dict = self.flatten_row_dict(row_dict)
        all_ids = np.arange(len(flattened_row_dict[column_keys.Connectivity.Partner]),
                            dtype=column_keys.Connectivity.Partner.basetype)
        disconnected_m = ~np.in1d(all_ids,
                                  flattened_row_dict[column_keys.Connectivity.Connected])
        flattened_row_dict[column_keys.Connectivity.Disconnected] = all_ids[disconnected_m]

        return flattened_row_dict

    def get_atomic_node_info(self, atomic_id: np.uint64,
                             time_stamp: datetime.datetime = get_max_time()
                             ) -> Dict:
        """ Reads connectivity information for a single node

        :param atomic_id: np.uint64
        :param time_stamp: datetime.datetime
        :return: dict
        """
        columns = [column_keys.Connectivity.Connected, column_keys.Connectivity.Affinity,
                   column_keys.Connectivity.Area, column_keys.Connectivity.Partner,
                   column_keys.Hierarchy.Parent]
        row_dict = self.read_node_id_row(atomic_id, columns=columns,
                                         end_time=time_stamp, end_time_inclusive=True)

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
        columns = []
        if include_connected_partners:
            columns.append(column_keys.Connectivity.Connected)
        if include_disconnected_partners:
            columns.append(column_keys.Connectivity.Disconnected)

        included_ids = []
        for column in columns:
            included_ids.extend(flattened_row_dict[column])

        included_ids = np.array(included_ids, dtype=column_keys.Connectivity.Connected.basetype)

        areas = flattened_row_dict[column_keys.Connectivity.Area][included_ids]
        affinities = flattened_row_dict[column_keys.Connectivity.Affinity][included_ids]
        partners = flattened_row_dict[column_keys.Connectivity.Partner][included_ids]

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

    def _retrieve_connectivity(self, dict_item: Tuple[np.uint64, Dict[column_keys._Column, List[bigtable.row_data.Cell]]]):
        node_id, row = dict_item

        tmp = set()
        for x in itertools.chain.from_iterable(
                generation.value for generation in row[column_keys.Connectivity.Connected][::-1]):
            tmp.remove(x) if x in tmp else tmp.add(x)

        connected_indices = np.fromiter(tmp, np.uint64)

        if column_keys.Connectivity.Partner in row:
            edges = np.fromiter(itertools.chain.from_iterable(
                (node_id, partner_id)
                for generation in row[column_keys.Connectivity.Partner][::-1]
                for partner_id in generation.value),
                dtype=basetypes.NODE_ID).reshape((-1, 2))[connected_indices]
        else:
            edges = np.empty((0, 2), basetypes.NODE_ID)

        if column_keys.Connectivity.Affinity in row:
            affinities = np.fromiter(itertools.chain.from_iterable(
                generation.value for generation in row[column_keys.Connectivity.Affinity][::-1]),
                dtype=basetypes.EDGE_AFFINITY)[connected_indices]
        else:
            edges = np.empty(0, basetypes.EDGE_AFFINITY)

        if column_keys.Connectivity.Area in row:
            areas = np.fromiter(itertools.chain.from_iterable(
                generation.value for generation in row[column_keys.Connectivity.Area][::-1]),
                dtype=basetypes.EDGE_AREA)[connected_indices]
        else:
            areas = np.empty(0, basetypes.EDGE_AREA)

        return edges, affinities, areas

    def get_subgraph_chunk(self, node_ids: Iterable[np.uint64],
                           make_unique: bool = True,
                           time_stamp: Optional[datetime.datetime] = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Takes an atomic id and returns the associated agglomeration ids

        :param node_ids: array of np.uint64
        :param make_unique: bool
        :param time_stamp: None or datetime
        :return: edge list
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        child_ids = self.get_children(node_ids, flatten=True)

        row_dict = self.read_node_id_rows(node_ids=child_ids,
                                          columns=[column_keys.Connectivity.Area,
                                                   column_keys.Connectivity.Affinity,
                                                   column_keys.Connectivity.Partner,
                                                   column_keys.Connectivity.Connected,
                                                   column_keys.Connectivity.Disconnected],
                                          end_time=time_stamp,
                                          end_time_inclusive=True)

        tmp_edges, tmp_affinites, tmp_areas = [], [], []
        for row_dict_item in row_dict.items():
            edges, affinities, areas = self._retrieve_connectivity(row_dict_item)
            tmp_edges.append(edges)
            tmp_affinites.append(affinities)
            tmp_areas.append(areas)

        edges = np.concatenate(tmp_edges) if tmp_edges \
            else np.empty((0, 2), dtype=basetypes.NODE_ID)
        affinities = np.concatenate(tmp_affinites) if tmp_affinites \
            else np.empty(0, dtype=basetypes.AFFINITY)
        areas = np.concatenate(tmp_areas) if tmp_areas \
            else np.empty(0, dtype=basetypes.AREA)

        # If requested, remove duplicate edges. Every edge is stored in each
        # participating node. Hence, we have many edge pairs that look
        # like [x, y], [y, x]. We solve this by sorting and calling np.unique
        # row-wise
        if make_unique and len(edges) > 0:
            edges, idx = np.unique(np.sort(edges, axis=1), axis=0,
                                   return_index=True)
            affinities = affinities[idx]
            areas = areas[idx]

        return edges, affinities, areas

    def add_edges(self, user_id: str, atomic_edges: Sequence[np.uint64],
                  affinities: Sequence[np.float32] = None,
                  source_coord: Sequence[int] = None,
                  sink_coord: Sequence[int] = None,
                  remesh_preview: bool = False,
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
                new_root_id, new_rows, lvl2_node_mapping = \
                    self._add_edges(operation_id=operation_id,
                                    atomic_edges=atomic_edges,
                                    time_stamp=time_stamp,
                                    affinities=affinities)
                rows.extend(new_rows)
                new_root_ids.append(new_root_id)

                # Add a row to the log
                log_row = self._create_merge_log_row(operation_id,
                                                     user_id,
                                                     new_root_ids,
                                                     atomic_edges[:, 0],
                                                     atomic_edges[:, 1],
                                                     [source_coord],
                                                     [sink_coord],
                                                     atomic_edges,
                                                     affinities,
                                                     time_stamp)

                # Put log row first!
                rows = [log_row] + rows

                # Execute write (makes sure that we are still owning the lock)
                if self.bulk_write(rows, lock_root_ids,
                                   operation_id=operation_id,
                                   slow_retry=False):
                    if remesh_preview:
                        meshgen.mesh_lvl2_previews(self, list(
                            lvl2_node_mapping.keys()))

                    return new_root_id

            for lock_root_id in lock_root_ids:
                self.unlock_root(lock_root_id, operation_id)

            i_try += 1

            self.logger.debug(f"Waiting - {i_try}")
            time.sleep(1)

        self.logger.warning("Could not acquire root object lock.")
        raise cg_exceptions.LockingError(
            f"Could not acquire root object lock."
        )

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
                                 dtype=column_keys.Connectivity.Affinity.basetype)

        assert len(affinities) == len(atomic_edges)

        rows = []

        # Create node_id to parent look up for later
        lvl2_node_mapping = {} # fore remeshing
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
            #     cross_chunk_edge_dict[l] = \
            #         np.array([], dtype=column_keys.Connectivity.CrossChunkEdge.basetype)

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

            for atomic_id in atomic_ids:
                val_dict = {column_keys.Hierarchy.Parent: new_parent_id}
                rows.append(self.mutate_row(serializers.serialize_uint64(atomic_id),
                                            val_dict, time_stamp=time_stamp))

            val_dict = {column_keys.Hierarchy.Child: atomic_ids}

            rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                        val_dict, time_stamp=time_stamp))
            lvl2_node_mapping[new_parent_id] = atomic_ids

            val_dict = {}
            for l in range(2, self.n_layers):
                if len(cross_chunk_edge_dict[l]) > 0:
                    val_dict[column_keys.Connectivity.CrossChunkEdge[l]] = \
                        cross_chunk_edge_dict[l]

            if len(val_dict):
                rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                            val_dict, time_stamp=time_stamp))

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
                                         dtype=column_keys.Connectivity.Partner.basetype)

                chunk_id = self.get_chunk_id(node_id=old_parent_ids[0])
                new_parent_id = self.get_unique_node_id(chunk_id)

                maintained_child_ids = []
                for old_parent_id in old_parent_ids:
                    old_parent_child_ids = self.get_children(old_parent_id)
                    maintained_child_ids.extend(old_parent_child_ids)

                maintained_child_ids = np.array(maintained_child_ids,
                                                dtype=column_keys.Connectivity.Partner.basetype)

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
                    val_dict = {column_keys.Hierarchy.Parent: new_parent_id}
                    rows.append(self.mutate_row(serializers.serialize_uint64(child_id),
                                                val_dict, time_stamp=time_stamp))

                val_dict = {column_keys.Hierarchy.Child: child_ids}

                rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                            val_dict, time_stamp=time_stamp))
                val_dict = {}

                for l in range(i_layer, self.n_layers):
                    if len(cross_chunk_edge_dict[l]) > 0:
                        val_dict[column_keys.Connectivity.CrossChunkEdge[l]] = \
                            cross_chunk_edge_dict[l]

                if len(val_dict):
                    rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                                val_dict, time_stamp=time_stamp))

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
                    val_dict[column_keys.Hierarchy.FormerParent] = \
                        np.array(old_parent_ids)
                    val_dict[column_keys.OperationLogs.OperationID] = operation_id

                    rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                                val_dict, time_stamp=time_stamp))

                    new_root_ids.append(new_parent_id)

                    for p in old_parent_ids:
                        rows.append(self.mutate_row(serializers.serialize_uint64(p),
                                                    {column_keys.Hierarchy.NewParent: new_parent_id},
                                                    time_stamp=time_stamp))

        # Atomic edge
        for i_atomic_edge, atomic_edge in enumerate(atomic_edges):
            affinity = affinities[i_atomic_edge]

            for i_atomic_id in range(2):
                atomic_id = atomic_edge[i_atomic_id]
                edge_partner = atomic_edge[(i_atomic_id + 1) % 2]

                atomic_node_info = self.get_atomic_node_info(atomic_id)

                if edge_partner in atomic_node_info[column_keys.Connectivity.Partner]:
                    partner_id = np.where(atomic_node_info[
                                              column_keys.Connectivity.Partner] == edge_partner)[0]

                    if partner_id in atomic_node_info[column_keys.Connectivity.Disconnected]:
                        partner_id = \
                            np.array(partner_id,
                                     dtype=column_keys.Connectivity.Connected.basetype)
                        val_dict = {column_keys.Connectivity.Connected: partner_id}
                    else:
                        val_dict = {}
                else:
                    affinity = \
                        np.array(affinity, dtype=column_keys.Connectivity.Affinity.basetype)

                    area = \
                        np.array(0, dtype=column_keys.Connectivity.Area.basetype)

                    partner_id = \
                        np.array(len(atomic_node_info[column_keys.Connectivity.Partner]),
                                 dtype=column_keys.Connectivity.Connected.basetype)

                    edge_partner = \
                        np.array(edge_partner, dtype=column_keys.Connectivity.Partner.basetype)

                    val_dict = {column_keys.Connectivity.Affinity: affinity,
                                column_keys.Connectivity.Area: area,
                                column_keys.Connectivity.Connected: partner_id,
                                column_keys.Connectivity.Partner: edge_partner}

                if len(val_dict) > 0:
                    rows.append(
                        self.mutate_row(serializers.serialize_uint64(atomic_edge[i_atomic_id]),
                                        val_dict, time_stamp=time_stamp))

        return new_root_ids, rows, lvl2_node_mapping

    def shatter_nodes_bbox(self, user_id: str,
                           bounding_box: Optional[Sequence[Sequence[int]]]):
        """ Removes all edges (except inf edges) of supervoxels (partly)
            inside the bounding box

        :param user_id: str
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
            voxels
        :return: list of uint64s or None if no split was performed
        """

        vol = self.cv[bounding_box[0][0]: bounding_box[1][0],
                      bounding_box[0][1]: bounding_box[1][1],
                      bounding_box[0][2]: bounding_box[1][2]]

        atomic_node_ids = np.unique(vol)
        return self. shatter_nodes(user_id=user_id,
                                   atomic_node_ids=atomic_node_ids,
                                   radius=1)

    def shatter_nodes(self, user_id: str, atomic_node_ids: Sequence[np.uint64],
                      radius: int = 1, remesh_preview: bool = False):
        """ Removes all edges (except inf edges) in radius around nodes

        :param user_id: str
        :param atomic_node_ids: list of np.uint64
        :param radius: int
        :return: list of uint64s or None if no split was performed
        """
        shattered_edges = []

        # kepp track of which partners were visited already
        visited_partners = list(atomic_node_ids)

        for i_neighbors in range(radius):

            next_partners = []
            for atomic_node_id in atomic_node_ids:
                partner_dict = self.get_all_original_partners(atomic_node_id)

                # Iterate over inf partners
                for k in partner_dict:
                    partners = partner_dict[k]

                    edges = np.zeros([len(partners), 2], dtype=np.uint64)
                    edges[:, 0] = k
                    edges[:, 1] = partners
                    shattered_edges.extend(edges)

                    unvisited_partners = partners[~np.in1d(partners, visited_partners)]
                    next_partners.extend(unvisited_partners)
                    visited_partners.extend(unvisited_partners)

            atomic_node_ids = list(next_partners)

        shattered_edges = np.array(shattered_edges)
        shattered_edges = np.unique(np.sort(shattered_edges, axis=1), axis=0)

        return self.remove_edges(user_id=user_id, atomic_edges=shattered_edges,
                                 mincut=False, remesh_preview=remesh_preview)

    def remove_edges(self,
                     user_id: str,
                     source_ids: Sequence[np.uint64] = None,
                     sink_ids: Sequence[np.uint64] = None,
                     source_coords: Sequence[Sequence[int]] = None,
                     sink_coords: Sequence[Sequence[int]] = None,
                     atomic_edges: Sequence[Tuple[np.uint64, np.uint64]] = None,
                     mincut: bool = True,
                     bb_offset: Tuple[int, int, int] = (240, 240, 24),
                     remesh_preview: bool = False,
                     root_ids: Optional[Sequence[np.uint64]] = None,
                     n_tries: int = 20) -> Sequence[np.uint64]:
        """ Removes edges - either directly or after applying a mincut

            Multi-user safe through locking of the root node

            This function acquires a lock and ensures that it still owns the
            lock before executing the write.

        :param user_id: str
            unique id - do not just make something up, use the same id for the
            same user every time
        :param source_ids: uint64
        :param sink_ids: uint64
        :param atomic_edges: list of 2 uint64
        :param source_coords: list of 3 ints
            [x, y, z] coordinate of source supervoxel
        :param sink_coords: list of 3 ints
            [x, y, z] coordinate of sink supervoxel
        :param mincut:
        :param bb_offset: list of 3 ints
            [x, y, z] bounding box padding beyond box spanned by coordinates
        :param remesh_preview: bool
        :param root_ids: list of uint64s
        :param n_tries: int
        :return: list of uint64s or None if no split was performed
        """

        if source_ids is not None and sink_ids is not None:
            if not (isinstance(source_ids, list) or isinstance(source_ids,
                                                               np.ndarray)):
                source_ids = [source_ids]

            if not (isinstance(sink_ids, list) or isinstance(sink_ids,
                                                             np.ndarray)):
                sink_ids = [sink_ids]

            # Sanity Checks
            if np.any(np.in1d(sink_ids, source_ids)):
                raise cg_exceptions.PreconditionError(
                    f"One or more supervoxel exists as both, sink and source."
                )

            for source_id in source_ids:
                layer = self.get_chunk_layer(source_id)
                if layer != 1:
                    raise cg_exceptions.PreconditionError(
                        f"Supervoxel expected, but {source_id} is a layer {layer} node."
                    )

            for sink_id in sink_ids:
                layer = self.get_chunk_layer(sink_id)
                if layer != 1:
                    raise cg_exceptions.PreconditionError(
                        f"Supervoxel expected, but {sink_id} is a layer {layer} node."
                    )

            root_ids = set()
            for source_id in source_ids:
                root_ids.add(self.get_root(source_id))
            for sink_id in sink_ids:
                root_ids.add(self.get_root(sink_id))

        if mincut:
            assert source_coords is not None
            assert sink_coords is not None
            assert sink_ids is not None
            assert source_ids is not None

            root_ids = set()
            for source_id in source_ids:
                root_ids.add(self.get_root(source_id))
            for sink_id in sink_ids:
                root_ids.add(self.get_root(sink_id))
        else:
            if atomic_edges is None:
                assert source_ids is not None
                assert sink_ids is not None

                atomic_edges = np.array(list(itertools.product(source_ids,
                                                               sink_ids)))

            root_ids = set()
            for atomic_edge in atomic_edges:
                root_ids.add(self.get_root(atomic_edge[0]))
                root_ids.add(self.get_root(atomic_edge[1]))

        if len(root_ids) > 1:
            raise cg_exceptions.PreconditionError(
                f"All supervoxel must belong to the same object. Already split?"
            )

        root_ids = list(root_ids)

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
                                                  source_ids=source_ids,
                                                  sink_ids=sink_ids,
                                                  source_coords=source_coords,
                                                  sink_coords=sink_coords,
                                                  bb_offset=bb_offset)
                    if success:
                        new_root_ids, rows, removed_edges, time_stamp, \
                            lvl2_node_mapping = result
                    else:
                        for lock_root_id in lock_root_ids:
                            self.unlock_root(lock_root_id,
                                             operation_id=operation_id)
                        return None
                else:
                    success, result = \
                        self._remove_edges(operation_id=operation_id,
                                           atomic_edges=atomic_edges)
                    if success:
                        new_root_ids, rows, time_stamp, \
                            lvl2_node_mapping = result
                        removed_edges = atomic_edges
                    else:
                        for lock_root_id in lock_root_ids:
                            self.unlock_root(lock_root_id,
                                             operation_id=operation_id)
                        return None

                # Add a row to the log
                log_row = self._create_split_log_row(operation_id,
                                                     user_id,
                                                     new_root_ids,
                                                     source_ids,
                                                     sink_ids,
                                                     source_coords,
                                                     sink_coords,
                                                     removed_edges,
                                                     bb_offset,
                                                     time_stamp)
                # Put log row first!
                rows = [log_row] + rows

                # Execute write (makes sure that we are still owning the lock)
                # if len(sink_ids) > 1 or len(source_ids) > 1:
                #     self.logger.debug(removed_edges)
                # else:
                if self.bulk_write(rows, lock_root_ids,
                                   operation_id=operation_id, slow_retry=False):
                    if remesh_preview:
                        meshgen.mesh_lvl2_previews(self, list(
                            lvl2_node_mapping.keys()))

                    self.logger.debug(f"new root ids: {new_root_ids}")
                    return new_root_ids

                for lock_root_id in lock_root_ids:
                    self.unlock_root(lock_root_id, operation_id=operation_id)

            i_try += 1

            self.logger.debug(f"Waiting - {i_try}")
            time.sleep(1)

        self.logger.warning("Could not acquire root object lock.")
        raise cg_exceptions.LockingError(
            f"Could not acquire root object lock."
        )

    def _remove_edges_mincut(self, operation_id: np.uint64,
                             source_ids: Sequence[np.uint64],
                             sink_ids: Sequence[np.uint64],
                             source_coords: Sequence[Sequence[int]],
                             sink_coords: Sequence[Sequence[int]],
                             bb_offset: Tuple[int, int, int] = (120, 120, 12)
                             ) -> Tuple[
                                 bool,                         # success
                                 Optional[Tuple[
                                    List[np.uint64],           # new_roots
                                    List[bigtable.row.Row],    # rows
                                    np.ndarray,                # removed_edges
                                    datetime.datetime,
                                    dict]]]:      # timestamp
        """ Computes mincut and removes edges accordingly

        :param operation_id: uint64
        :param source_ids: uint64
        :param sink_ids: uint64
        :param source_coords: list of 3 ints
            [x, y, z] coordinate of source supervoxel
        :param sink_coords: list of 3 ints
            [x, y, z] coordinate of sink supervoxel
        :param bb_offset: list of 3 ints
            [x, y, z] bounding box padding beyond box spanned by coordinates
        :return: list of uint64s if successful, or None if no valid split
            new root ids
        """

        time_start = time.time()

        bb_offset = np.array(list(bb_offset))
        source_coords = np.array(source_coords)
        sink_coords = np.array(sink_coords)

        # Decide a reasonable bounding box (NOT guaranteed to be successful!)
        coords = np.concatenate([source_coords, sink_coords])
        bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]

        bounding_box[0] -= bb_offset
        bounding_box[1] += bb_offset

        # Verify that sink and source are from the same root object
        root_ids = set()
        for source_id in source_ids:
            root_ids.add(self.get_root(source_id))
        for sink_id in sink_ids:
            root_ids.add(self.get_root(sink_id))

        if len(root_ids) > 1:
            raise cg_exceptions.PreconditionError(
                f"All supervoxel must belong to the same object. Already split?"
            )

        self.logger.debug("Get roots and check: %.3fms" %
                          ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        root_id = root_ids.pop()

        # Get edges between local supervoxels
        n_chunks_affected = np.product((np.ceil(bounding_box[1] / self.chunk_size)).astype(np.int) -
                                       (np.floor(bounding_box[0] / self.chunk_size)).astype(np.int))
        self.logger.debug("Number of affected chunks: %d" % n_chunks_affected)
        self.logger.debug(f"Bounding box: {bounding_box}")
        self.logger.debug(f"Bounding box padding: {bb_offset}")
        self.logger.debug(f"Source ids: {source_ids}")
        self.logger.debug(f"Sink ids: {sink_ids}")
        self.logger.debug(f"Root id: {root_id}")

        edges, affs, areas = self.get_subgraph_edges(root_id,
                                                     bounding_box=bounding_box,
                                                     bb_is_coordinate=True)

        self.logger.debug("Get edges and affs: %.3fms" %
                          ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        # Compute mincut
        atomic_edges = cutting.mincut(edges, affs, source_ids, sink_ids)

        self.logger.debug("Mincut: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        if len(atomic_edges) == 0:
            self.logger.warning("Mincut failed. Try again...")
            return False, None

        # Check if any edge in the cutset is infinite (== between chunks)
        # We would prevent such a cut

        atomic_edges_flattened_view = atomic_edges.view(dtype='u8,u8')
        edges_flattened_view = edges.view(dtype='u8,u8')

        cutset_mask = np.in1d(edges_flattened_view, atomic_edges_flattened_view)
        if np.any(np.isinf(affs[cutset_mask])):
            self.logger.error("inf in cutset")
            return False, None

        # Remove edgesc
        success, result = self._remove_edges(operation_id, atomic_edges)

        if not success:
            self.logger.error("remove edges failed")
            return False, None

        new_roots, rows, time_stamp, lvl2_node_mapping = result

        self.logger.debug("Remove edges: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        return True, (new_roots, rows, atomic_edges, time_stamp, lvl2_node_mapping)

    def _remove_edges(self, operation_id: np.uint64,
                      atomic_edges: Sequence[Tuple[np.uint64, np.uint64]]
                      ) -> Tuple[bool,                          # success
                                 Optional[Tuple[
                                     List[np.uint64],           # new_roots
                                     List[bigtable.row.Row],    # rows
                                     datetime.datetime,
                                     dict]]]:      # timestamp
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

            partner_ids = np.where(
                np.in1d(atomic_node_info[column_keys.Connectivity.Partner], partners))[0]

            partner_ids = \
                np.array(partner_ids, dtype=column_keys.Connectivity.Connected.basetype)

            val_dict = {column_keys.Connectivity.Connected: partner_ids}

            rows.append(self.mutate_row(serializers.serialize_uint64(u_atomic_id),
                                        val_dict, time_stamp=time_stamp))

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

        lvl2_node_mapping = {} # Needed for instant remeshing

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

                # Temporarily storing information on how the parents of this cc
                # are changed by the split. We need this information when
                # processing the next layer
                new_layer_parent_dict[new_parent_id] = old_parent_id
                old_id_dict[old_parent_id].append(new_parent_id)

                # Make changes to the rows of the lowest layer
                val_dict = {column_keys.Hierarchy.Child: cc_node_ids}

                segment_ids = [self.get_segment_id(cc_node_id)
                               for cc_node_id in cc_node_ids]
                lvl2_node_mapping[new_parent_id] = segment_ids

                rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                            val_dict, time_stamp=time_stamp))

                for cc_node_id in cc_node_ids:
                    val_dict = {column_keys.Hierarchy.Parent: new_parent_id}

                    rows.append(self.mutate_row(serializers.serialize_uint64(cc_node_id),
                                                val_dict, time_stamp=time_stamp))

                cce_layers = self.get_cross_chunk_edges_layer(cc_cross_edges)
                u_cce_layers = np.unique(cce_layers)

                cross_edge_dict[new_parent_id] = {}

                for l in range(2, self.n_layers):
                    empty_edges = column_keys.Connectivity.CrossChunkEdge.deserialize(b'')
                    cross_edge_dict[new_parent_id][l] = empty_edges

                val_dict = {}
                for cc_layer in u_cce_layers:
                    layer_cross_edges = cc_cross_edges[cce_layers == cc_layer]

                    if len(layer_cross_edges) > 0:
                        val_dict[column_keys.Connectivity.CrossChunkEdge[cc_layer]] = \
                            layer_cross_edges
                        cross_edge_dict[new_parent_id][cc_layer] = layer_cross_edges

                if len(val_dict) > 0:
                    rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                                val_dict, time_stamp=time_stamp))

        # Now that the lowest layer has been updated, we need to walk through
        # all layers and move our new parents forward
        # new_layer_parent_dict stores all newly created parents. We first
        # empty it and then fill it with the new parents in the next layer
        if self.n_layers == 2:
            return True, (list(new_layer_parent_dict.keys()), rows, time_stamp,
                          lvl2_node_mapping)

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
                    self.logger.debug("No old parents for any member of the cc")
                    lop = np.unique(list(leftover_old_parents))
                    llop = lop[~np.in1d(lop, np.unique(edges))]
                    raise()
                    return False, None

                partners = np.array(partners, dtype=np.uint64)

                this_chunk_id = self.get_chunk_id(
                    node_id=old_next_layer_parent)
                new_parent_id = self.get_unique_node_id(this_chunk_id)

                new_layer_parent_dict[new_parent_id] = old_next_layer_parent
                old_id_dict[old_next_layer_parent].append(new_parent_id)

                cross_edge_dict[new_parent_id] = {}
                for partner in partners:
                    cross_edge_dict[new_parent_id] = \
                        combine_cross_chunk_edge_dicts(cross_edge_dict[new_parent_id],
                                                       cross_edge_dict[partner],
                                                       start_layer=i_layer+1)

                for partner in partners:
                    val_dict = {column_keys.Hierarchy.Parent: new_parent_id}

                    rows.append(
                        self.mutate_row(serializers.serialize_uint64(partner),
                                        val_dict, time_stamp=time_stamp))

                val_dict = {column_keys.Hierarchy.Child: partners}

                if i_layer == self.n_layers - 1:
                    new_roots.append(new_parent_id)
                    val_dict[column_keys.Hierarchy.FormerParent] = \
                        np.array(original_root)
                    val_dict[column_keys.OperationLogs.OperationID] = operation_id

                rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                            val_dict, time_stamp=time_stamp))

                if i_layer < self.n_layers - 1:
                    val_dict = {}
                    for l in range(i_layer + 1, self.n_layers):
                        val_dict[column_keys.Connectivity.CrossChunkEdge[l]] = \
                            cross_edge_dict[new_parent_id][l]

                    if len(val_dict) == 0:
                        self.logger.error("Cross chunk edges are missing")
                        return False, None

                    rows.append(self.mutate_row(serializers.serialize_uint64(new_parent_id),
                                                val_dict, time_stamp=time_stamp))

            if i_layer == self.n_layers - 1:
                val_dict = {column_keys.Hierarchy.NewParent:
                            np.array(new_roots,
                                     dtype=column_keys.Hierarchy.NewParent.basetype)}
                rows.append(self.mutate_row(serializers.serialize_uint64(original_root),
                                            val_dict, time_stamp=time_stamp))

        return True, (new_roots, rows, time_stamp, lvl2_node_mapping)
