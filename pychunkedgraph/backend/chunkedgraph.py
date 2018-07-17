import asyncio
from concurrent.futures import ThreadPoolExecutor
import collections
import numpy as np
import time
import datetime
import os
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import pytz

from . import multiprocessing_utils as mu
from google.api_core.retry import Retry, if_exception_type
from google.api_core.exceptions import Aborted, DeadlineExceeded, \
    ServiceUnavailable
from google.auth import credentials
from google.cloud import bigtable
from google.cloud.bigtable.row_filters import TimestampRange, \
    TimestampRangeFilter, ColumnRangeFilter, ValueRangeFilter, RowFilterChain, \
    ColumnQualifierRegexFilter, RowFilterUnion, ConditionalRowFilter, \
    PassAllFilter, BlockAllFilter
from google.cloud.bigtable.column_family import MaxVersionsGCRule

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# global variables
HOME = os.path.expanduser("~")
N_DIGITS_UINT64 = len(str(np.iinfo(np.uint64).max))
# LOCK_EXPIRED_TIME_DELTA = datetime.timedelta(minutes=3, seconds=00)
LOCK_EXPIRED_TIME_DELTA = datetime.timedelta(minutes=3, seconds=0)
UTC = pytz.UTC

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
    HOME + "/.cloudvolume/secrets/google-secret.json"


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


def pad_node_id(node_id: np.uint64) -> str:
    """ Pad node id to 20 digits

    :param node_id: int
    :return: str
    """
    return "%.20d" % node_id


def serialize_node_id(node_id: np.uint64) -> bytes:
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    return serialize_key(pad_node_id(node_id))  # type: ignore


def deserialize_node_id(node_id: bytes) -> np.uint64:
    """ De-serializes a node id from a BigTable row

    :param node_id: bytes
    :return: np.uint64
    """
    return np.uint64(node_id.decode())  # type: ignore


def serialize_key(key: str) -> bytes:
    """ Serializes a key to be ingested by a bigtable table row

    :param key: str
    :return: bytes
    """
    return key.encode("utf-8")


def deserialize_key(key: bytes) -> str:
    """ Deserializes a row key

    :param key: bytes
    :return: str
    """
    return key.decode()


def row_to_byte_dict(row: bigtable.row.Row, f_id: str = None, idx: int = None
                     ) -> Dict[int, Dict]:
    """ Reads row entries to a dictionary

    :param row: row
    :param f_id: str
    :param idx: int
    :return: dict
    """
    row_dict = {}

    for fam_id in row.cells.keys():
        row_dict[fam_id] = {}

        for row_k in row.cells[fam_id].keys():
            if idx is None:
                row_dict[fam_id][deserialize_key(row_k)] = \
                    [c.value for c in row.cells[fam_id][row_k]]
            else:
                row_dict[fam_id][deserialize_key(row_k)] = \
                    row.cells[fam_id][row_k][idx].value

    if f_id is not None and f_id in row_dict:
        return row_dict[f_id]
    elif f_id is None:
        return row_dict
    else:
        raise Exception("Family id not found")


def compute_bitmasks(n_layers: int, fan_out: int) -> Dict[int, int]:
    """

    :param n_layers: int
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
            n_bits_for_layers = max(1,
                                    np.ceil(log_n(fan_out**(n_layers - i_layer),
                                                  fan_out)))
            # n_bits_for_layers = fan_out ** int(np.ceil(log_n(n_bits_for_layers, fan_out)))

        n_bits_for_layers = int(n_bits_for_layers)

        assert n_bits_for_layers <= 8

        bitmask_dict[i_layer] = n_bits_for_layers
    return bitmask_dict


def mincut(edges: Iterable[Sequence[np.uint64]], affs: Sequence[np.uint64],
           source: np.uint64, sink: np.uint64) -> np.ndarray:
    """ Computes the min cut on a local graph

    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :param source: uint64
    :param sink: uint64
    :return: m x 2 array of uint64s
        edges that should be removed
    """

    time_start = time.time()

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['weight'] = affs[i_edge]

    dt = time.time() - time_start
    print("Graph creation: %.2fms" % (dt * 1000))
    time_start = time.time()

    # cutset = nx.minimum_edge_cut(weighted_graph, source, sink)
    cutset = nx.minimum_edge_cut(weighted_graph, source, sink,
                                 flow_func=shortest_augmenting_path)

    dt = time.time() - time_start
    print("Mincut: %.2fms" % (dt * 1000))

    if cutset is None:
        return []

    time_start = time.time()

    weighted_graph.remove_edges_from(cutset)
    print("Graph split up in %d parts" %
          (len(list(nx.connected_components(weighted_graph)))))

    dt = time.time() - time_start
    print("Test: %.2fms" % (dt * 1000))

    return np.array(list(cutset), dtype=np.uint64)


class ChunkedGraph(object):
    def __init__(self,
                 table_id: str,
                 instance_id: str = "pychunkedgraph",
                 project_id: str = "neuromancer-seung-import",
                 chunk_size: Tuple[int, int, int] = (512, 512, 64),
                 fan_out: Optional[int] = None,
                 n_layers: Optional[int] = None,
                 credentials: Optional[credentials.Credentials] = None,
                 client: bigtable.Client = None,
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
        self._chunk_size = np.array(chunk_size)
        self._bitmasks = compute_bitmasks(self.n_layers, self.fan_out)

        self._n_bits_for_layer_id = 8

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
        ser_param_key = serialize_key(param_key)
        row = self.table.read_row(serialize_key("params"))

        if row is None or ser_param_key not in row.cells[self.family_id]:
            assert value is not None

            val_dict = {param_key: np.array(value, dtype=np.uint64).tobytes()}
            row = self.mutate_row(serialize_key("params"), self.family_id,
                                  val_dict)

            self.bulk_write([row])
        else:
            value = row.cells[self.family_id][ser_param_key][0].value
            value = np.frombuffer(value, dtype=np.uint64)[0]

        return value

    def get_serialized_info(self):
        """ Rerturns dictionary that can be used to load this AnnotationMetaDB

        :return: dict
        """
        amdb_info = {"table_id": self.table_id,
                     "instance_id": self.instance_id,
                     "project_id": self.project_id,
                     "credentials": self.client.credentials}

        return amdb_info

    def get_chunk_layer(self, node_or_chunk_id: np.uint64) -> int:
        """ Extract Layer from Node ID or Chunk ID

        :param node_or_chunk_id: np.uint64
        :return: int
        """
        return int(node_or_chunk_id) >> 64 - self._n_bits_for_layer_id

    def get_chunk_coordinates(self, node_or_chunk_id: np.uint64
                              ) -> Tuple[int, int, int]:
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
        return x, y, z

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
        assert node_id is not None or all(v is not None
                                          for v in [layer, x, y, z])

        if node_id is not None:
            layer = self.get_chunk_layer(node_id)
        bits_per_dim = self.bitmasks[layer]

        if node_id is not None:
            chunk_offset = 64 - self._n_bits_for_layer_id - 3 * bits_per_dim
            return np.uint64((int(node_id) >> chunk_offset) << chunk_offset)
        else:
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

    def get_unique_node_id(self, chunk_id: np.uint64) -> np.uint64:
        """ Return unique Node ID for given Chunk ID

        atomic counter

        :param chunk_id: np.uint64
        :return: np.uint64
        """

        counter_key = serialize_key('counter')

        # Incrementer row keys start with an "i" followed by the chunk id
        row_key = serialize_key("i%s" % pad_node_id(chunk_id))
        append_row = self.table.row(row_key, append=True)
        append_row.increment_cell_value(self.incrementer_family_id,
                                        counter_key, 1)

        # This increments the row entry and returns the value AFTER incrementing
        latest_row = append_row.commit()
        segment_id_b = latest_row[self.incrementer_family_id][counter_key][0][0]
        segment_id = int.from_bytes(segment_id_b, byteorder="big")

        return self.get_node_id(np.uint64(segment_id), chunk_id=chunk_id)

    def get_max_node_id(self, chunk_id: np.uint64) -> np.uint64:
        """  Gets maximal node id in a chunk based on the atomic counter

        This is an approximation. It is not guaranteed that all ids smaller or
        equal to this id exists. However, it is guaranteed that no larger id
        exist at the time this function is executed.


        :return: uint64
        """

        counter_key = serialize_key('counter')

        # Incrementer row keys start with an "i"
        row_key = serialize_key("i%s" % pad_node_id(chunk_id))
        row = self.table.read_row(row_key)

        # Read incrementer value
        if row is not None:
            max_node_id_b = row.cells[self.incrementer_family_id][counter_key][0].value
            max_node_id = int.from_bytes(max_node_id_b, byteorder="big")
        else:
            max_node_id = 0

        return np.uint64(max_node_id)

    def get_unique_operation_id(self) -> str:
        """ Finds a unique operation id

        atomic counter

        :return: str
        """

        counter_key = serialize_key('counter')

        # Incrementer row keys start with an "i"
        row_key = serialize_key("ioperations")
        append_row = self.table.row(row_key, append=True)
        append_row.increment_cell_value(self.incrementer_family_id,
                                        counter_key, 1)

        # This increments the row entry and returns the value AFTER incrementing
        latest_row = append_row.commit()
        operation_id_b = latest_row[self.incrementer_family_id][counter_key][0][0]

        operation_id = "op%d" % int.from_bytes(operation_id_b, byteorder="big")

        return operation_id

    def get_max_operation_id(self) -> str:
        """  Gets maximal operation id based on the atomic counter

        This is an approximation. It is not guaranteed that all ids smaller or
        equal to this id exists. However, it is guaranteed that no larger id
        exist at the time this function is executed.


        :return: uint64
        """

        counter_key = serialize_key('counter')

        # Incrementer row keys start with an "i"
        row_key = serialize_key("ioperations")
        row = self.table.read_row(row_key)

        # Read incrementer value
        if row is not None:
            max_operation_id_b = row.cells[self.incrementer_family_id][counter_key][0].value
            # max_operation_id = deserialize_
        else:
            max_operation_id_b = b"op0"

        return max_operation_id_b

    def read_row(self, node_id: np.uint64, key: str, idx: int = 0,
                 dtype: type = np.uint64, get_time_stamp: bool = False) -> Any:
        """ Reads row from BigTable and takes care of serializations

        :param node_id: uint64
        :param key: table column
        :param idx: column list index
        :param dtype: np.dtype
        :param get_time_stamp: bool
        :return: row entry
        """
        key = serialize_key(key)

        row = self.table.read_row(serialize_node_id(node_id),
                                  filter_=ColumnQualifierRegexFilter(key))

        if row is None or key not in row.cells[self.family_id]:
            return None

        cell_entries = row.cells[self.family_id][key]

        if dtype is None:
            cell_value = cell_entries[idx].value
        else:
            cell_value = np.frombuffer(cell_entries[idx].value, dtype=dtype)

        if get_time_stamp:
            return cell_value, cell_entries[idx].timestamp
        else:
            return cell_value

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
                   operation_id: Optional[str] = None,
                   slow_retry: bool = True) -> bool:
        """ Writes a list of mutated rows in bulk

        WARNING: If <rows> contains the same row (same row_key) and column
        key two times only the last one is effectively written to the BigTable
        (even when the mutations were applied to different columns)
        --> no versioning!

        :param rows: list
            list of mutated rows
        :param root_ids: list if uint64
        :param operation_id: str or None
            operation_id (or other unique id) that *was* used to lock the root
            the bulk write is only executed if the root is still locked with
            the same id.
        :param slow_retry: bool
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

        status = self.table.mutate_rows(rows, retry=retry_policy)

        if not any(status):
            raise Exception(status)

        return True

    def range_read_chunk(self, layer: int, x: int, y: int, z: int,
                         n_retries: int = 100,
                         row_keys: Optional[Iterable[str]] = None,
                         yield_rows: bool = False) -> Union[
                                bigtable.row_data.PartialRowData,
                                Dict[bytes, bigtable.row_data.PartialRowData]]:
        """ Reads all ids within a chunk

        :param layer: int
        :param x: int
        :param y: int
        :param z: int
        :param n_retries: int
        :param row_keys: list of str
            more efficient read through row filters
        :param yield_rows: bool
        :return: list or yield of rows
        """
        if row_keys is not None:
            filters = []
            for k in row_keys:
                filters.append(ColumnQualifierRegexFilter(serialize_key(k)))

            if len(filters) > 1:
                row_filter = RowFilterUnion(filters)
            else:
                row_filter = filters[0]
        else:
            row_filter = None

        chunk_id = self.get_chunk_id(layer=layer, x=x, y=y, z=z)
        max_segment_id = self.get_segment_id_limit(chunk_id)

        # Define BigTable keys
        start_id = self.get_node_id(np.uint64(0), chunk_id=chunk_id)
        end_id = self.get_node_id(max_segment_id, chunk_id=chunk_id)

        if yield_rows:
            range_read_yield = self.table.yield_rows(
                start_key=serialize_node_id(start_id),
                end_key=serialize_node_id(end_id),
                filter_=row_filter)
            return range_read_yield
        else:
            # Set up read
            range_read = self.table.read_rows(
                start_key=serialize_node_id(start_id),
                end_key=serialize_node_id(end_id),
                # allow_row_interleaving=True,
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
                                "[%d, %d, %d], l = %d, n_retries = %d" %
                                (x, y, z, layer, n_retries))

            return range_read.rows

    def range_read_layer(self, layer_id):
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

    def _create_split_log_row(self, operation_id: str, user_id: str,
                              root_ids: Sequence[np.uint64],
                              selected_atomic_ids: Sequence[np.uint64],
                              removed_edges: Sequence[np.uint64],
                              time_stamp: datetime.datetime
                              ) -> bigtable.row.Row:

        val_dict = {serialize_key("user"): serialize_key(user_id),
                    serialize_key("roots"):
                        np.array(root_ids, dtype=np.uint64).tobytes(),
                    serialize_key("atomic_ids"):
                        np.array(selected_atomic_ids).tobytes(),
                    serialize_key("removed_edges"):
                        np.array(removed_edges, dtype=np.uint64).tobytes()}

        row = self.mutate_row(serialize_key(operation_id),
                              self.family_id, val_dict, time_stamp)

        return row

    def _create_merge_log_row(self, operation_id: str, user_id: str,
                              root_ids: Sequence[np.uint64],
                              selected_atomic_ids: Sequence[np.uint64],
                              time_stamp: datetime.datetime
                              ) -> bigtable.row.Row:

        val_dict = {serialize_key("user"):
                        serialize_key(user_id),
                    serialize_key("roots"):
                        np.array(root_ids, dtype=np.uint64).tobytes(),
                    serialize_key("atomic_ids"):
                        np.array(selected_atomic_ids).tobytes()}

        row = self.mutate_row(serialize_key(operation_id),
                              self.family_id, val_dict, time_stamp)

        return row

    def add_atomic_edges_in_chunks(self, edge_ids: np.ndarray,
                                   cross_edge_ids: np.ndarray,
                                   edge_affs: Sequence[np.float32],
                                   cross_edge_affs: Sequence[np.float32],
                                   isolated_node_ids: Sequence[np.uint64],
                                   verbose: bool = False,
                                   time_stamp: Optional[datetime.datetime] = None):
        """ Creates atomic nodes in first abstraction layer for a SINGLE chunk

        Alle edges (edge_ids) need to be from one chunk and no nodes should
        exist for this chunk prior to calling this function. All cross edges
        (cross_edge_ids) have to point out the chunk (first entry is the id
        within the chunk)

        :param edge_ids: n x 2 array of uint64s
        :param cross_edge_ids: m x 2 array of uint64s
        :param edge_affs: float array of length n
        :param cross_edge_affs: float array of length m
        :param isolated_node_ids: list of uint64s
            ids of nodes that have no edge in the chunked graph
        :param verbose: bool
        :param time_stamp: datetime
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Catch trivial case
        if edge_ids.size == 0 and cross_edge_ids.size == 0 and \
                len(isolated_node_ids) == 0:
            return 0

        # Make parent id creation easier
        if edge_ids.size > 0:
            chunk_id_c = self.get_chunk_coordinates(edge_ids[0, 0])
        elif cross_edge_ids.size > 0:
            chunk_id_c = self.get_chunk_coordinates(cross_edge_ids[0, 0])
        else:
            chunk_id_c = self.get_chunk_coordinates(isolated_node_ids[0])

        parent_chunk_id = self.get_chunk_id(layer=2, x=chunk_id_c[0],
                                            y=chunk_id_c[1], z=chunk_id_c[2])

        # Get connected component within the chunk
        chunk_g = nx.from_edgelist(edge_ids)

        isolated_node_mask = ~np.in1d(cross_edge_ids[:, 0], np.unique(edge_ids))
        chunk_g.add_nodes_from(cross_edge_ids[:, 0][isolated_node_mask])
        chunk_g.add_nodes_from(isolated_node_ids)
        ccs = list(nx.connected_components(chunk_g))

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the print / speed measurement
        time_start = time.time()
        for i_cc, cc in enumerate(ccs):
            if verbose and node_c > 0:
                dt = time.time() - time_start
                print("%5d at %5d - %.5fs             " %
                      (i_cc, node_c, dt / node_c), end="\r")

            rows = []

            node_ids = np.array(list(cc))

            # Create parent id
            parent_id = self.get_unique_node_id(parent_chunk_id)
            parent_id_b = np.array(parent_id, dtype=np.uint64).tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # Extract edges relevant to this node
                edge_col1_mask = edge_ids[:, 0] == node_id
                edge_col2_mask = edge_ids[:, 1] == node_id

                # Cross edges are ordered to always point OUT of the chunk
                cross_edge_mask = cross_edge_ids[:, 0] == node_id

                parent_cross_edges =\
                    np.concatenate([parent_cross_edges,
                                    cross_edge_ids[cross_edge_mask]])

                connected_partner_ids = \
                    np.concatenate([edge_ids[edge_col1_mask][:, 1],
                                    edge_ids[edge_col2_mask][:, 0],
                                    cross_edge_ids[cross_edge_mask][:, 1]]).tobytes()

                connected_partner_affs = \
                    np.concatenate([
                        edge_affs[np.logical_or(edge_col1_mask,
                                                edge_col2_mask)],
                        cross_edge_affs[cross_edge_mask]]).tobytes()

                # Create node
                val_dict = {"atomic_partners": connected_partner_ids,
                            "atomic_affinities": connected_partner_affs,
                            "parents": parent_id_b}

                rows.append(self.mutate_row(serialize_node_id(node_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))
                node_c += 1

            # Create parent node
            val_dict = {"children": node_ids.tobytes(),
                        "atomic_cross_edges": parent_cross_edges.tobytes()}

            rows.append(self.mutate_row(serialize_node_id(parent_id),
                                        self.family_id, val_dict,
                                        time_stamp=time_stamp))

            node_c += 1

            self.bulk_write(rows)

        if verbose:
            try:
                dt = time.time() - time_start
                print("Average time: %.5fs / node; %.5fs / edge - "
                      "Number of edges: %6d, %6d" %
                      (dt / node_c, dt / len(edge_ids), len(edge_ids),
                       len(cross_edge_ids)))
            except:
                print("WARNING: NOTHING HAPPENED")

    def add_layer(self, layer_id: int,
                  child_chunk_coords: Sequence[Sequence[int]],
                  time_stamp: Optional[datetime.datetime] = None,
                  n_threads: int = 20) -> None:
        """ Creates the abstract nodes for a given chunk in a given layer

        :param layer_id: int
        :param child_chunk_coords: int array of length 3
            coords in chunk space
        :param time_stamp: datetime
        :param n_threads: int
        """
        def _resolve_cross_chunk_edges_thread(args) -> None:
            start, end = args
            for i_child_key, child_key in\
                    enumerate(atomic_partner_id_dict_keys[start: end]):
                this_atomic_partner_ids = atomic_partner_id_dict[child_key]
                this_atomic_child_ids = atomic_child_id_dict[child_key]

                leftover_mask = ~np.in1d(this_atomic_partner_ids,
                                         u_atomic_child_ids)
                leftover_atomic_edges[child_key] = \
                    np.concatenate([this_atomic_child_ids[leftover_mask, None],
                                    this_atomic_partner_ids[leftover_mask, None]],
                                   axis=1)

                partners = np.unique(child_ids[np.in1d(atomic_child_ids,
                                                       this_atomic_partner_ids)])

                if len(partners) > 0:
                    these_edges =\
                        np.concatenate([np.array([child_key] * len(partners),
                                                 dtype=np.uint64)[:, None],
                                        partners[:, None]], axis=1)
                    edge_ids.extend(these_edges)

        def _write_out_connected_components(args) -> None:
            start, end = args
            for i_cc, cc in enumerate(ccs[start: end]):
                    rows = []

                    node_ids = np.array(list(cc))

                    parent_id = self.get_unique_node_id(chunk_id)
                    parent_id_b = np.array(parent_id, dtype=np.uint64).tobytes()

                    parent_cross_edges = np.array([],
                                                  dtype=np.uint64).reshape(0, 2)

                    # Add rows for nodes that are in this chunk
                    for i_node_id, node_id in enumerate(node_ids):

                        # Extract edges relevant to this node
                        parent_cross_edges =\
                            np.concatenate([parent_cross_edges,
                                            leftover_atomic_edges[node_id]])

                        # Create node
                        val_dict = {"parents": parent_id_b}

                        rows.append(self.mutate_row(serialize_node_id(node_id),
                                                    self.family_id, val_dict,
                                                    time_stamp=time_stamp))

                    # Create parent node
                    val_dict = {"children":
                                    node_ids.tobytes(),
                                "atomic_cross_edges":
                                    parent_cross_edges.tobytes()}

                    rows.append(self.mutate_row(serialize_node_id(parent_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                    self.bulk_write(rows)

        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # 1 ----------
        # The first part is concerned with reading data from the child nodes
        # of this layer and pre-processing it for the second part

        time_start = time.time()

        # ids in lowest layer
        atomic_child_ids = np.array([], dtype=np.uint64)

        # ids in layer one below this one
        child_ids = np.array([], dtype=np.uint64)
        atomic_partner_id_dict = {}
        atomic_child_id_dict = {}

        leftover_atomic_edges = {}

        for chunk_coord in child_chunk_coords:
            # Get start and end key
            x, y, z = chunk_coord

            range_read = self.range_read_chunk(layer_id - 1, x, y, z,
                                               row_keys=["atomic_cross_edges"],
                                               yield_rows=False)

            # Loop through nodes from this chunk
            for row_key, row_data in range_read.items():
                row_key = deserialize_node_id(row_key)

                cell = row_data.cells[self.family_id][serialize_key("atomic_cross_edges")]
                atomic_edges_b = cell[0].value
                atomic_edges = np.frombuffer(atomic_edges_b,
                                             dtype=np.uint64).reshape(-1, 2)

                atomic_partner_id_dict[int(row_key)] = atomic_edges[:, 1]
                atomic_child_id_dict[int(row_key)] = atomic_edges[:, 0]

                atomic_child_ids = np.concatenate([atomic_child_ids,
                                                   atomic_edges[:, 0]])
                child_ids =\
                    np.concatenate([child_ids,
                                    np.array([row_key] * len(atomic_edges[:, 0]),
                                             dtype=np.uint64)])

        print("Time iterating through subchunks: %.3fs" %
              (time.time() - time_start))
        time_start = time.time()

        # Extract edges from remaining cross chunk edges
        # and maintain unused cross chunk edges
        edge_ids = []
        u_atomic_child_ids = np.unique(atomic_child_ids)
        atomic_partner_id_dict_keys = \
            np.array(list(atomic_partner_id_dict.keys()), dtype=np.uint64)

        if n_threads > 1:
            n_jobs = n_threads * 3 # Heuristic
        else:
            n_jobs = 1

        spacing = np.linspace(0, len(atomic_partner_id_dict_keys),
                              n_jobs+1).astype(np.int)
        starts = spacing[:-1]
        ends = spacing[1:]

        multi_args = list(zip(starts, ends))

        mu.multithread_func(_resolve_cross_chunk_edges_thread, multi_args,
                            n_threads=n_threads)

        print("Time resolving cross chunk edges: %.3fs" %
              (time.time() - time_start))
        time_start = time.time()

        # 2 ----------
        # The second part finds connected components, writes the parents to
        # BigTable and updates the childs

        # Make parent id creation easier
        x, y, z = np.min(child_chunk_coords, axis=0) // self.fan_out
        chunk_id = self.get_chunk_id(layer=layer_id, x=x, y=y, z=z)

        # Extract connected components
        chunk_g = nx.from_edgelist(edge_ids)
        # chunk_g.add_nodes_from(atomic_partner_id_dict_keys)

        # Add single node objects that have no edges
        add_ccs = []

        isolated_node_mask = ~np.in1d(atomic_partner_id_dict_keys,
                                      np.unique(edge_ids))
        for node_id in atomic_partner_id_dict_keys[isolated_node_mask]:
            add_ccs.append([node_id])

        ccs = list(nx.connected_components(chunk_g)) + add_ccs

        # Add rows for nodes that are in this chunk
        # a connected component at a time

        spacing = np.linspace(0, len(ccs), n_jobs+1).astype(np.int)
        starts = spacing[:-1]
        ends = spacing[1:]

        multi_args = list(zip(starts, ends))

        mu.multithread_func(_write_out_connected_components, multi_args,
                            n_threads=n_threads)

        print("Time connected components: %.3fs" % (time.time() - time_start))

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
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        parent_key = serialize_key("parents")
        all_parents = []

        p_filter_ = ColumnQualifierRegexFilter(parent_key)
        row = self.table.read_row(serialize_node_id(node_id), filter_=p_filter_)

        if row and parent_key in row.cells[self.family_id]:
            for parent_entry in row.cells[self.family_id][parent_key]:
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

    def get_root(self, node_id: np.uint64,
                 time_stamp: Optional[datetime.datetime] = None
                 ) -> Union[List[np.uint64], np.uint64]:
        """ Takes a node id and returns the associated agglomeration ids

        :param atomic_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

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

        :param atomic_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

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
                temp_parent_id = self.get_parent(parent_id, time_stamp=time_stamp)

                if temp_parent_id is None:
                    early_finish = True
                    break
                else:
                    parent_id = temp_parent_id
                    parent_ids.append(parent_id)

        return parent_ids

    def lock_root_loop(self, root_ids: Sequence[np.uint64], operation_id: str,
                       max_tries: int = 1, waittime_s: float = 0.5
                       ) -> Tuple[bool, np.ndarray]:
        """ Attempts to lock multiple roots at the same time

        :param root_ids: list of uint64
        :param operation_id: str
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
                latest_root_ids = self.get_latest_root_id(root_ids[i_root_id])

                new_root_ids.extend(latest_root_ids)

            # Attempt to lock all latest root ids
            root_ids = np.unique(new_root_ids)
            for i_root_id in range(len(root_ids)):

                print(i_root_id, root_ids[i_root_id])
                lock_acquired = self.lock_single_root(root_ids[i_root_id],
                                                      operation_id)

                # Roll back locks if one root cannot be locked
                if not lock_acquired:
                    for j_root_id in range(i_root_id):
                        self.unlock_root(root_ids[j_root_id], operation_id)
                    break

            if lock_acquired:
                return True, root_ids

            time.sleep(waittime_s)
            i_try += 1
            print(i_try)

        return False, root_ids

    def lock_single_root(self, root_id: np.uint64, operation_id: str) -> bool:
        """ Attempts to lock the latest version of a root node

        :param root_id: uint64
        :param operation_id: str
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """

        operation_id_b = serialize_key(operation_id)

        lock_key = serialize_key("lock")
        new_parents_key = serialize_key("new_parents")

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
        root_row = self.table.row(serialize_node_id(root_id),
                                  filter_=combined_filter)

        # Set row lock if condition returns no results (state == False)
        time_stamp = datetime.datetime.utcnow()
        root_row.set_cell(self.family_id, lock_key, operation_id_b, state=False,
                          timestamp=time_stamp)

        # The lock was acquired when set_cell returns False (state)
        lock_acquired = not root_row.commit()

        return lock_acquired

    def unlock_root(self, root_id: np.uint64, operation_id: str) -> bool:
        """ Unlocks a root

        This is mainly used for cases where multiple roots need to be locked and
        locking was not sucessful for all of them

        :param root_id: np.uint64
        :param operation_id: str
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """
        operation_id_b = serialize_key(operation_id)

        lock_key = serialize_key("lock")

        # Build a column filter which tests if a lock was set (== lock column
        # exists) and if it is still valid (timestamp younger than
        # LOCK_EXPIRED_TIME_DELTA) and if the given operation_id is still
        # the active lock holder

        time_cutoff = datetime.datetime.now(UTC) - LOCK_EXPIRED_TIME_DELTA

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
        root_row = self.table.row(serialize_node_id(root_id),
                                  filter_=chained_filter)

        # Delete row if conditions are met (state == True)
        root_row.delete_cell(self.family_id, lock_key, state=True)

        return root_row.commit()

    def check_and_renew_root_locks(self, root_ids: Iterable[np.uint64],
                                   operation_id: str) -> bool:
        """ Tests if the roots are locked with the provided operation_id and
        renews the lock to reset the time_stam

        This is mainly used before executing a bulk write

        :param root_ids: uint64
        :param operation_id: str
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """

        for root_id in root_ids:
            if not self.check_and_renew_root_lock_single(root_id, operation_id):
                return False

        return True

    def check_and_renew_root_lock_single(self, root_id: np.uint64,
                                         operation_id: str) -> bool:
        """ Tests if the root is locked with the provided operation_id and
        renews the lock to reset the time_stam

        This is mainly used before executing a bulk write

        :param root_id: uint64
        :param operation_id: str
            an id that is unique to the process asking to lock the root node
        :return: bool
            success
        """
        operation_id_b = serialize_key(operation_id)

        lock_key = serialize_key("lock")
        new_parents_key = serialize_key("new_parents")

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
        root_row = self.table.row(serialize_node_id(root_id),
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
        new_parent_key = serialize_key("new_parents")
        latest_root_ids = []

        while len(id_working_set) > 0:

            next_id = id_working_set[0]
            del(id_working_set[0])
            r = self.table.read_row(serialize_node_id(next_id))

            # Check if a new root id was attached to this root id
            if new_parent_key in r.cells[self.family_id]:
                id_working_set.extend(
                    np.frombuffer(
                        r.cells[self.family_id][new_parent_key][0].value,
                        dtype=np.uint64))
            else:
                latest_root_ids.append(next_id)

        return np.unique(latest_root_ids)

    def read_agglomeration_id_history(self, agglomeration_id: np.uint64,
                                      time_stamp:
                                      Optional[datetime.datetime] = None
                                      ) -> np.ndarray:
        """ Returns all agglomeration ids agglomeration_id was part of

        :param agglomeration_id: np.uint64
        :param time_stamp: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole history
        :return: array of int
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.min

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        id_working_set = np.array([agglomeration_id], dtype=np.uint64)
        visited_ids = []
        id_history = [agglomeration_id]

        former_parent_key = serialize_key("former_parents")
        new_parent_key = serialize_key("new_parents")

        i = 0
        while len(id_working_set) > 0:
            i += 1

            next_id = id_working_set[0]
            visited_ids.append(id_working_set[0])

            # Get current row
            r = self.table.read_row(serialize_node_id(next_id))

            # Check if there is a newer parent and append
            if new_parent_key in r.cells[self.family_id]:
                new_parent_ids_b = \
                    r.cells[self.family_id][new_parent_key][0].value
                new_parent_ids = np.frombuffer(new_parent_ids_b,
                                               dtype=np.uint64)

                id_working_set = np.concatenate([id_working_set,
                                                 new_parent_ids])
                id_history.extend(new_parent_ids)

            # Check if there is an older parent and append if not too old
            if former_parent_key in r.cells[self.family_id]:
                cell = r.cells[self.family_id][former_parent_key][0]
                if time_stamp < cell.timestamp:
                    former_parent_ids_b = \
                        r.cells[self.family_id][former_parent_key][0].value
                    former_parent_ids = np.frombuffer(former_parent_ids_b,
                                                      dtype=np.uint64)

                    id_working_set = np.concatenate([id_working_set,
                                                     former_parent_ids])
                    id_history.extend(former_parent_ids)

            id_working_set = id_working_set[~np.in1d(id_working_set, visited_ids)]

        return np.unique(id_history)

    def get_subgraph(self, agglomeration_id: np.uint64,
                     bounding_box: Optional[Sequence[Sequence[int]]] = None,
                     bb_is_coordinate: bool = False, stop_lvl: int = 1,
                     get_edges: bool = False
                     ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """ Returns all edges between supervoxels belonging to the specified
            agglomeration id within the defined bouning box

        :param agglomeration_id: int
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param stop_lvl: int
        :param get_edges: bool
        :return: edge list
        """
        # Helper functions for multithreading
        def _handle_subgraph_children_layer2_edges_thread(
                child_ids: Iterable[np.uint64]) -> Tuple[List[np.ndarray],
                                                         List[np.float32]]:

            _edges = []
            _affinities = []
            for child_id in child_ids:
                this_edges, this_affinities = self.get_subgraph_chunk(
                    child_id, time_stamp=time_stamp)
                _edges.extend(this_edges)
                _affinities.extend(this_affinities)
            return _edges, _affinities

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

                    scaled_chunk_ids = chunk_ids * self.fan_out ** np.max([0, (layer - 3)])

                    chunk_id_bounds = np.array([scaled_chunk_ids,
                                                scaled_chunk_ids +
                                                self.fan_out **
                                                np.max([0, (layer - 3)])])

                    bound_check = np.array([
                        np.all(chunk_id_bounds[0] < bounding_box[1], axis=1),
                        np.all(chunk_id_bounds[1] > bounding_box[0], axis=1)]).T

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
            this_n_threads = int(n_child_ids // 20) + 1

            if layer == 2:
                if get_edges:
                    edges_and_affinities = mu.multithread_func(
                        _handle_subgraph_children_layer2_edges_thread,
                        np.array_split(child_ids, this_n_threads),
                        n_threads=this_n_threads, debug=this_n_threads == 1)

                    for edges_and_affinities_pair in edges_and_affinities:
                        _edges, _affinities = edges_and_affinities_pair
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

            print("Layer %d: %.3fms for %d chunks with %d threads" %
                  (layer, (time.time() - time_start) * 1000, n_child_ids,
                   this_n_threads))
            time_start = time.time()

        atomic_ids = np.array(atomic_ids, np.uint64)

        if get_edges:
            return edges, affinities
        else:
            return atomic_ids

    def get_atomic_partners(self, atomic_id: np.uint64,
                            time_stamp: Optional[datetime.datetime] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """ Extracts the atomic partners and affinities for a given timestamp

        :param atomic_id: np.uint64
        :param time_stamp: None or datetime
        :return: list of uint64, list of float32
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        edge_key = serialize_key("atomic_partners")
        affinity_key = serialize_key("atomic_affinities")

        filters = [ColumnQualifierRegexFilter(edge_key),
                   ColumnQualifierRegexFilter(affinity_key)]

        filter_ = RowFilterUnion(filters)

        partners = np.array([], dtype=np.uint64)
        affinities = np.array([], dtype=np.float32)

        r = self.table.read_row(serialize_node_id(atomic_id),
                                filter_=filter_)

        # Shortcut for the trivial case that there have been no changes to
        # the edges of this child:
        if len(r.cells[self.family_id][edge_key]) == 0:
            partners = np.frombuffer(
                r.cells[self.family_id][edge_key][0].value, dtype=np.uint64)
            affinities = np.frombuffer(
                r.cells[self.family_id][affinity_key][0].value,
                dtype=np.float32)

        # From new to old: Add partners that are not
        # in the edge list of this child. This assures that more recent
        # changes are prioritized. For each, check if the time_stamp
        # is satisfied.
        # Note: The creator writes one list of partners (edges) and
        # affinities. Each edit makes only small edits (yet), hence,
        # all but the oldest entry are short lists of length ~ 1-10

        for i_edgelist in range(len(r.cells[self.family_id][edge_key])):
            cell = r.cells[self.family_id][edge_key][i_edgelist]
            if time_stamp >= cell.timestamp:
                partner_batch_b = \
                    r.cells[self.family_id][edge_key][i_edgelist].value
                partner_batch = np.frombuffer(partner_batch_b,
                                              dtype=np.uint64)

                affinity_batch_b = \
                    r.cells[self.family_id][affinity_key][i_edgelist].value
                affinity_batch = np.frombuffer(affinity_batch_b,
                                               dtype=np.float32)
                partner_batch_m = ~np.in1d(partner_batch, partners)

                partners = np.concatenate([partners,
                                           partner_batch[partner_batch_m]])
                affinities = np.concatenate([affinities,
                                             affinity_batch[partner_batch_m]])

        # Take care of removed edges (affinity == 0)
        partners_m = affinities > 0
        partners = partners[partners_m]
        affinities = affinities[partners_m]

        return partners, affinities

    def get_subgraph_chunk(self, parent_id: np.uint64, make_unique: bool = True,
                           time_stamp: Optional[datetime.datetime] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """ Takes an atomic id and returns the associated agglomeration ids

        :param parent_id: np.uint64
        :param time_stamp: None or datetime
        :return: edge list
        """
        def _read_atomic_partners(child_id_block: Iterable[np.uint64]
                                  ) -> Tuple[np.ndarray, np.ndarray]:
            thread_edges = np.array([], dtype=np.uint64).reshape(0, 2)
            thread_affinities = np.array([], dtype=np.float32)

            for child_id in child_id_block:
                node_edges, node_affinities = \
                    self.get_atomic_partners(child_id, time_stamp=time_stamp)

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

            return thread_edges, thread_affinities

        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        child_ids = self.get_children(parent_id)

        # Iterate through all children of this parent and retrieve their edges
        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        affinities = np.array([], dtype=np.float32)

        n_child_ids = len(child_ids)
        this_n_threads = int(n_child_ids // 20) + 1

        child_id_blocks = np.array_split(child_ids, this_n_threads)
        edges_and_affinities = mu.multithread_func(_read_atomic_partners,
                                                   child_id_blocks,
                                                   n_threads=this_n_threads,
                                                   debug=this_n_threads == 1)

        for edges_and_affinities_pairs in edges_and_affinities:
            edges = np.concatenate([edges,
                                    edges_and_affinities_pairs[0]])
            affinities = np.concatenate([affinities,
                                         edges_and_affinities_pairs[1]])

        # If requested, remove duplicate edges. Every edge is stored in each
        # participating node. Hence, we have many edge pairs that look
        # like [x, y], [y, x]. We solve this by sorting and calling np.unique
        # row-wise
        if make_unique:
            edges, idx = np.unique(np.sort(edges, axis=1), axis=0,
                                   return_index=True)
            affinities = affinities[idx]

        return edges, affinities

    def add_edge(self, user_id: str, atomic_edge: Sequence[np.uint64],
                 affinity: Optional[np.float32] = None,
                 root_ids: Optional[Sequence[np.uint64]] = None,
                 n_tries: int = 20) -> np.uint64:
        """ Adds an edge to the chunkedgraph

            Multi-user safe through locking of the root node

            This function acquires a lock and ensures that it still owns the
            lock before executing the write.

        :param user_id: str
            unique id - do not just make something up, use the same id for the
            same user every time
        :param atomic_edge: list of two uint64s
        :param affinity: float or None
            will eventually be set to 1 if None
        :param root_ids: list of uint64s
            avoids reading the root ids again if already computed
        :param n_tries: int
        :return: uint64
            if successful the new root id is send
            else None
        """

        print(atomic_edge)

        # Sanity Checks
        if atomic_edge[0] == atomic_edge[1]:
            return None

        if self.get_chunk_layer(atomic_edge[0]) != \
                self.get_chunk_layer(atomic_edge[1]):
            return None

        # Lookup root ids
        if root_ids is None:
            root_ids = [self.get_root(atomic_edge[0]),
                        self.get_root(atomic_edge[1])]

        # Get a unique id for this operation
        operation_id = self.get_unique_operation_id()

        i_try = 0

        while i_try < n_tries:
            # Try to acquire lock and only continue if successful
            if self.lock_root_loop(root_ids=root_ids,
                                   operation_id=operation_id)[0]:

                # Add edge and change hierarchy
                new_root_id, rows, time_stamp = \
                    self._add_edge(operation_id=operation_id,
                                   atomic_edge=atomic_edge, affinity=affinity)

                # Add a row to the log
                rows.append(self._create_merge_log_row(operation_id, user_id,
                                                       [new_root_id],
                                                       atomic_edge, time_stamp))

                # Execute write (makes sure that we are still owning the lock)
                if self.bulk_write(rows, root_ids,
                                   operation_id=operation_id, slow_retry=False):
                    return new_root_id

            i_try += 1

            print("Waiting - %d" % i_try)
            time.sleep(1)

        return None

    def _add_edge(self, operation_id: str, atomic_edge: Sequence[np.uint64],
                  affinity: Optional[np.float32] = None
                  ) -> Tuple[np.uint64, List[bigtable.row.Row],
                             datetime.datetime]:
        """ Adds an atomic edge to the ChunkedGraph

        :param operation_id: str
        :param atomic_edge: list of two ints
        :param affinity: float
        :return: int
            new root id
        """
        time_stamp = datetime.datetime.utcnow()

        if affinity is None:
            affinity = np.float32(1.0)

        rows = []

        # Walk up the hierarchy until a parent in the same chunk is found
        original_parent_ids = [self.get_all_parents(atomic_edge[0]),
                               self.get_all_parents(atomic_edge[1])]

        original_parent_ids = np.array(original_parent_ids).T

        merge_layer = None
        for i_layer in range(len(original_parent_ids)):
            if self.test_if_nodes_are_in_same_chunk(original_parent_ids[i_layer]):
                merge_layer = i_layer
                break

        if merge_layer is None:
            raise Exception("No parents found. Did you set is_cg_id correctly?")

        original_root = original_parent_ids[-1]

        # Find a new node id and update all children
        # circumvented_nodes = current_parent_ids.copy()
        # chunk_id = self.get_chunk_id(node_id=original_parent_ids[merge_layer][0])

        new_parent_id = self.get_unique_node_id(
            self.get_chunk_id(node_id=original_parent_ids[merge_layer][0]))
        new_parent_id_b = np.array(new_parent_id).tobytes()
        current_node_id = None

        for i_layer in range(merge_layer, len(original_parent_ids)):
            current_parent_ids = original_parent_ids[i_layer]

            # Collect child ids of all nodes --> childs of new node
            if current_node_id is None:
                combined_child_ids = np.array([], dtype=np.uint64)
            else:
                combined_child_ids = np.array([current_node_id],
                                              dtype=np.uint64).flatten()

            for prior_parent_id in current_parent_ids:
                child_ids = self.get_children(prior_parent_id)

                # Exclude parent nodes from old hierarchy path
                if i_layer > merge_layer:
                    child_ids = child_ids[~np.in1d(child_ids,
                                                   original_parent_ids)]

                combined_child_ids = np.concatenate([combined_child_ids,
                                                     child_ids])

                # Append new parent entry for all children
                for child_id in child_ids:
                    val_dict = {"parents": new_parent_id_b}
                    rows.append(self.mutate_row(serialize_node_id(child_id),
                                                self.family_id,
                                                val_dict,
                                                time_stamp=time_stamp))

            # Create new parent node
            val_dict = {"children": combined_child_ids.tobytes()}
            current_node_id = new_parent_id  # Store for later

            if i_layer < len(original_parent_ids) - 1:

                new_parent_id = self.get_unique_node_id(
                    self.get_chunk_id(
                        node_id=original_parent_ids[i_layer + 1][0]))
                new_parent_id_b = np.array(new_parent_id).tobytes()

                val_dict["parents"] = new_parent_id_b
            else:
                val_dict["former_parents"] = np.array(original_root).tobytes()
                val_dict["operation_id"] = serialize_key(operation_id)

                rows.append(self.mutate_row(serialize_node_id(original_root[0]),
                                            self.family_id,
                                            {"new_parents": new_parent_id_b},
                                            time_stamp=time_stamp))

                rows.append(self.mutate_row(serialize_node_id(original_root[1]),
                                            self.family_id,
                                            {"new_parents": new_parent_id_b},
                                            time_stamp=time_stamp))

            # Read original cross chunk edges
            atomic_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)
            for original_parent_id in original_parent_ids[i_layer]:
                this_atomic_cross_edges = \
                    self.read_row(original_parent_id,
                                  "atomic_cross_edges").reshape(-1, 2)
                atomic_cross_edges = np.concatenate([atomic_cross_edges,
                                                     this_atomic_cross_edges])

            val_dict["atomic_cross_edges"] = atomic_cross_edges.tobytes()

            rows.append(self.mutate_row(serialize_node_id(current_node_id),
                                        self.family_id, val_dict,
                                        time_stamp=time_stamp))

        # Atomic edge
        for i_atomic_id in range(2):
            val_dict = \
                {"atomic_partners":
                     np.array([atomic_edge[(i_atomic_id + 1) % 2]]).tobytes(),
                 "atomic_affinities":
                     np.array([affinity], dtype=np.float32).tobytes()}

            rows.append(self.mutate_row(serialize_node_id(
                atomic_edge[i_atomic_id]), self.family_id, val_dict,
                time_stamp=time_stamp))

        return new_parent_id, rows, time_stamp

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
            return None

        if self.get_chunk_layer(source_id) != \
                self.get_chunk_layer(sink_id):
            return None

        if mincut:
            assert source_coord is not None
            assert sink_coord is not None

        if root_ids is None:
            root_ids = [self.get_root(source_id),
                        self.get_root(sink_id)]

        if root_ids[0] != root_ids[1]:
            return None

        # Get a unique id for this operation
        operation_id = self.get_unique_operation_id()

        i_try = 0

        while i_try < n_tries:
            # Try to acquire lock and only continue if successful
            if self.lock_root_loop(root_ids=root_ids[:1],
                                   operation_id=operation_id)[0]:

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
                        return None
                else:
                    success, result = \
                        self._remove_edges(operation_id=operation_id,
                                           atomic_edges=[[source_id, sink_id]])
                    if success:
                        new_root_ids, rows, time_stamp = result
                        removed_edges = [[source_id, sink_id]]
                    else:
                        return None

                # Add a row to the log
                rows.append(self._create_split_log_row(operation_id, user_id,
                                                       new_root_ids,
                                                       [source_id, sink_id],
                                                       removed_edges,
                                                       time_stamp))

                # Execute write (makes sure that we are still owning the lock)
                if self.bulk_write(rows, root_ids[:1],
                                   operation_id=operation_id, slow_retry=False):
                    return new_root_ids

            i_try += 1

            print("Waiting - %d" % i_try)
            time.sleep(1)

        return root_ids[:1]

    def _remove_edges_mincut(self, operation_id: str, source_id: np.uint64,
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

        :param operation_id: str
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
        coords = np.concatenate([source_coord[:, None], sink_coord[:, None]],
                                axis=1).T
        bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]

        bounding_box[0] -= bb_offset
        bounding_box[1] += bb_offset

        root_id_source = self.get_root(source_id)
        root_id_sink = self.get_root(source_id)

        # Verify that sink and source are from the same root object
        if root_id_source != root_id_sink:
            return False, None

        print(
            "Get roots and check: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        root_id = root_id_source

        # Get edges between local supervoxels
        edges, affs = self.get_subgraph(root_id, get_edges=True,
                                        bounding_box=bounding_box,
                                        bb_is_coordinate=True)

        print(
            "Get edges and affs: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        # Compute mincut
        atomic_edges = mincut(edges, affs, source_id, sink_id)

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
            return False, None

        # Remove edges
        success, result = self._remove_edges(operation_id, atomic_edges)

        if not success:
            return False, None

        new_roots, rows, time_stamp = result

        print("Remove edges: %.3fms" % ((time.time() - time_start) * 1000))
        time_start = time.time()  # ------------------------------------------

        return True, (new_roots, rows, atomic_edges, time_stamp)

    def _remove_edges(self, operation_id: str,
                      atomic_edges: Sequence[Tuple[np.uint64, np.uint64]]
                      ) -> Tuple[bool,                          # success
                                 Optional[Tuple[
                                     List[np.uint64],           # new_roots
                                     List[bigtable.row.Row],    # rows
                                     datetime.datetime]]]:      # timestamp
        """ Removes atomic edges from the ChunkedGraph

        :param operation_id: str
        :param atomic_edges: list of two uint64s
        :return: list of uint64s
            new root ids
        """
        time_stamp = datetime.datetime.utcnow()

        # Make sure that we have a list of edges
        if isinstance(atomic_edges[0], np.uint64):
            atomic_edges = [atomic_edges]

        atomic_edges = np.array(atomic_edges)
        u_atomic_ids = np.unique(atomic_edges)

        # Get number of layers and the original root
        original_parent_ids = self.get_all_parents(atomic_edges[0, 0])
        n_layers = len(original_parent_ids)
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
        # changes to the same row within a batch write and only executes
        # one of them.
        for u_atomic_id in np.unique(atomic_edges):
            partners = np.concatenate([atomic_edges[atomic_edges[:, 0] ==
                                                    u_atomic_id][:, 1],
                                       atomic_edges[atomic_edges[:, 1] ==
                                                    u_atomic_id][:, 0]])

            val_dict = {"atomic_partners":
                            partners.tobytes(),
                        "atomic_affinities":
                            np.zeros(len(partners), dtype=np.float32).tobytes()}

            rows.append(self.mutate_row(serialize_node_id(u_atomic_id),
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
        double_atomic_edges_view = \
            double_atomic_edges_view.reshape(double_atomic_edges.shape[0])
        nodes_in_removed_edges = np.unique(atomic_edges)

        # For each involved chunk we need to compute connected components
        for chunk_id in involved_chunk_id_dict.keys():
            # Get the local subgraph
            node_id = involved_chunk_id_dict[chunk_id]
            old_parent_id = self.get_parent(node_id)
            edges, _ = self.get_subgraph_chunk(old_parent_id, make_unique=False)

            # These edges still contain the removed edges.
            # For consistency reasons we can only write to BigTable one time.
            # Hence, we have to evict the to be removed "atomic_edges" from the
            # queried edges.
            retained_edges_mask =\
                ~np.in1d(edges.view(dtype='u8,u8').reshape(edges.shape[0]),
                         double_atomic_edges_view)

            edges = edges[retained_edges_mask]

            # The cross chunk edges are passed on to the parents to compute
            # connected components in higher layers.

            cross_edge_mask = self.get_chunk_ids_from_node_ids(
                np.ascontiguousarray(edges[:, 1])) != \
                              self.get_chunk_id(node_id=node_id)

            cross_edges = edges[cross_edge_mask]
            edges = edges[~cross_edge_mask]
            isolated_nodes = list(filter(
                lambda x: x not in edges and self.get_chunk_id(x) == chunk_id,
                nodes_in_removed_edges))

            # Build the local subgraph and compute connected components
            G = nx.from_edgelist(edges)
            G.add_nodes_from(isolated_nodes)
            ccs = nx.connected_components(G)

            # For each connected component we create one new parent
            for cc in ccs:
                cc_node_ids = np.array(list(cc), dtype=np.uint64)

                # Get the associated cross edges
                cc_cross_edges = cross_edges[np.in1d(cross_edges[:, 0],
                                                     cc_node_ids)]

                # Get a new parent id
                new_parent_id = self.get_unique_node_id(
                    self.get_chunk_id(node_id=old_parent_id))

                new_parent_id_b = np.array(new_parent_id).tobytes()
                new_parent_id = new_parent_id

                # Temporarily storing information on how the parents of this cc
                # are changed by the split. We need this information when
                # processing the next layer
                new_layer_parent_dict[new_parent_id] = old_parent_id
                cross_edge_dict[new_parent_id] = cc_cross_edges
                old_id_dict[old_parent_id].append(new_parent_id)

                # Make changes to the rows of the lowest layer
                val_dict = {"children": cc_node_ids.tobytes(),
                            "atomic_cross_edges": cc_cross_edges.tobytes()}

                rows.append(self.mutate_row(serialize_node_id(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(self.mutate_row(serialize_node_id(cc_node_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

        # Now that the lowest layer has been updated, we need to walk through
        # all layers and move our new parents forward
        # new_layer_parent_dict stores all newly created parents. We first
        # empty it and then fill it with the new parents in the next layer
        if n_layers == 1:
            return True, (list(new_layer_parent_dict.keys()), rows, time_stamp)

        new_roots = []
        for i_layer in range(n_layers - 1):

            parent_cc_list = []
            parent_cc_old_parent_list = []
            parent_cc_mapping = {}
            leftover_edges = {}
            old_parent_dict = {}

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
                atomic_children = cross_edges[:, 0]
                atomic_id_map = np.ones(len(cross_edges), dtype=np.uint64) * \
                                new_layer_parent
                partner_cross_edges = {new_layer_parent: cross_edges}

                for old_chunk_neighbor in old_chunk_neighbors:
                    # For each neighbor we need to check whether this neighbor
                    # was affected by a split as well (and was updated):
                    # neighbor_id in old_id_dict. If so, we take the new atomic
                    # cross edges (temporary data) into account, else, we load
                    # the atomic_cross_edges from BigTable
                    if old_chunk_neighbor in old_id_dict:
                        for new_neighbor in old_id_dict[old_chunk_neighbor]:
                            neigh_cross_edges = cross_edge_dict[new_neighbor]
                            atomic_children = \
                                np.concatenate([atomic_children,
                                                neigh_cross_edges[:, 0]])

                            partner_cross_edges[new_neighbor] = \
                                neigh_cross_edges

                            ps = np.ones(len(neigh_cross_edges),
                                         dtype=np.uint64) * new_neighbor
                            atomic_id_map =  np.concatenate([atomic_id_map, ps])
                    else:
                        neigh_cross_edges = self.read_row(old_chunk_neighbor,
                                                          "atomic_cross_edges")
                        neigh_cross_edges = neigh_cross_edges.reshape(-1, 2)

                        atomic_children = \
                            np.concatenate([atomic_children,
                                            neigh_cross_edges[:, 0]])

                        partner_cross_edges[old_chunk_neighbor] = \
                            neigh_cross_edges

                        ps = np.ones(len(neigh_cross_edges),
                                     dtype=np.uint64) * old_chunk_neighbor
                        atomic_id_map = np.concatenate([atomic_id_map, ps])

                u_atomic_children = np.unique(atomic_children)
                edge_ids = np.array([], dtype=np.uint64).reshape(-1, 2)

                # raise()

                # For each potential neighbor (now, adjusted for changes in
                # neighboring chunks), compare cross edges and extract edges
                # (edge_ids) between them
                for pot_partner in partner_cross_edges.keys():
                    this_atomic_partner_ids = partner_cross_edges[pot_partner]
                    this_atomic_partner_ids = this_atomic_partner_ids[:, 1]

                    this_atomic_child_ids = partner_cross_edges[pot_partner]
                    this_atomic_child_ids = this_atomic_child_ids[:, 0]

                    leftover_mask = ~np.in1d(this_atomic_partner_ids,
                                             u_atomic_children)

                    leftover_edges[pot_partner] = np.concatenate(
                        [this_atomic_child_ids[leftover_mask, None],
                         this_atomic_partner_ids[leftover_mask, None]], axis=1)

                    partner_mask = np.in1d(atomic_children,
                                           this_atomic_partner_ids)
                    partners = np.unique(atomic_id_map[partner_mask])

                    ps = np.array([pot_partner] * len(partners),
                                  dtype=np.uint64)
                    these_edges = np.concatenate([ps[:, None],
                                                  partners[:, None]], axis=1)

                    edge_ids = np.concatenate([edge_ids, these_edges])

                # Create graph and run connected components
                chunk_g = nx.from_edgelist(edge_ids)
                chunk_g.add_nodes_from(np.array([new_layer_parent],
                                                dtype=np.uint64))
                ccs = list(nx.connected_components(chunk_g))

                # Filter the connected component that is relevant to the
                # current new_layer_parent
                partners = []
                for cc in ccs:
                    if new_layer_parent in cc:
                        partners = cc
                        break

                # Check if the parent has already been "created"
                if new_layer_parent in parent_cc_mapping:
                    parent_cc_id = parent_cc_mapping[new_layer_parent]
                    parent_cc_list[parent_cc_id].extend(partners)
                    parent_cc_list[parent_cc_id].append(new_layer_parent)
                else:
                    parent_cc_id = len(parent_cc_list)
                    parent_cc_list.append(list(partners))
                    parent_cc_list[parent_cc_id].append(new_layer_parent)
                    parent_cc_old_parent_list.append(old_next_layer_parent)

                # Inverse mapping
                for partner_id in partners:
                    parent_cc_mapping[partner_id] = parent_cc_id

            # Create the new_layer_parent_dict for the next layer and write
            # nodes (lazy)
            new_layer_parent_dict = {}
            for i_cc, parent_cc in enumerate(parent_cc_list):
                old_next_layer_parent = None
                for parent_id in parent_cc:
                    if parent_id in old_parent_dict:
                        old_next_layer_parent = old_parent_dict[parent_id]

                assert old_next_layer_parent is not None

                cc_node_ids = np.array(list(parent_cc), dtype=np.uint64)
                cc_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

                for parent_id in parent_cc:
                    cc_cross_edges = np.concatenate([cc_cross_edges,
                                                     leftover_edges[parent_id]])

                this_chunk_id = self.get_chunk_id(node_id=old_next_layer_parent)
                new_parent_id = self.get_unique_node_id(this_chunk_id)
                new_parent_id_b = np.array(new_parent_id).tobytes()
                new_parent_id = new_parent_id

                new_layer_parent_dict[new_parent_id] = \
                    parent_cc_old_parent_list[i_cc]
                cross_edge_dict[new_parent_id] = cc_cross_edges

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(self.mutate_row(serialize_node_id(cc_node_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                val_dict = {"children": cc_node_ids.tobytes(),
                            "atomic_cross_edges": cc_cross_edges.tobytes()}

                if i_layer == n_layers - 2:
                    new_roots.append(new_parent_id)
                    val_dict["former_parents"] = \
                        np.array(original_root).tobytes()
                    val_dict["operation_id"] = \
                        serialize_key(operation_id)

                rows.append(self.mutate_row(serialize_node_id(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

            if i_layer == n_layers - 2:
                val_dict = {"new_parents": np.array(new_roots,
                                                    dtype=np.uint64).tobytes()}
                rows.append(self.mutate_row(serialize_node_id(original_root),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

        return True, (new_roots, rows, time_stamp)
