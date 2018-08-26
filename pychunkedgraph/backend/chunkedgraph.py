import collections
import numpy as np
import time
import itertools
import datetime
import os
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
import pytz
import cloudvolume

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
LOCK_EXPIRED_TIME_DELTA = datetime.timedelta(minutes=2, seconds=00)
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


def serialize_uint64(node_id: np.uint64) -> bytes:
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    return serialize_key(pad_node_id(node_id))  # type: ignore


def deserialize_uint64(node_id: bytes) -> np.uint64:
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


def merge_cross_chunk_edges(edges: Iterable[Sequence[np.uint64]],
                            affs: Sequence[np.uint64]):
    """ Merges cross chunk edges

    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :return:
    """

    cross_chunk_edge_mask = np.isinf(affs)

    cross_chunk_graph = nx.Graph()
    cross_chunk_graph.add_edges_from(edges[cross_chunk_edge_mask])

    ccs = nx.connected_components(cross_chunk_graph)

    remapping = {}
    mapping = np.array([], dtype=np.uint64).reshape(-1, 2)

    for cc in ccs:
        nodes = np.array(list(cc))
        rep_node = np.min(nodes)

        remapping[rep_node] = nodes

        rep_nodes = np.ones(len(nodes), dtype=np.uint64).reshape(-1, 1) * rep_node
        m = np.concatenate([nodes.reshape(-1, 1), rep_nodes], axis=1)

        mapping = np.concatenate([mapping, m], axis=0)

    u_nodes = np.unique(edges)
    u_unmapped_nodes = u_nodes[~np.in1d(u_nodes, mapping)]

    unmapped_mapping = np.concatenate([u_unmapped_nodes.reshape(-1, 1),
                                       u_unmapped_nodes.reshape(-1, 1)], axis=1)
    mapping = np.concatenate([mapping, unmapped_mapping], axis=0)

    sort_idx = np.argsort(mapping[:, 0])
    idx = np.searchsorted(mapping[:, 0], edges, sorter=sort_idx)
    remapped_edges = np.asarray(mapping[:, 1])[sort_idx][idx]

    remapped_edges = remapped_edges[~cross_chunk_edge_mask]
    remapped_affs = affs[~cross_chunk_edge_mask]

    return remapped_edges, remapped_affs, mapping, remapping


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

    original_edges = edges.copy()
    original_affs = affs.copy()

    edges, affs, mapping, remapping = merge_cross_chunk_edges(edges.copy(), affs.copy())

    sink_map = np.where(mapping[:, 0] == sink)[0]
    source_map = np.where(mapping[:, 0] == source)[0]

    print(sink, source)

    if len(sink_map) == 0:
        pass
    elif len(sink_map) == 1:
        sink = mapping[sink_map[0]][1]
    else:
        raise Exception("Sink appears to be overmerged")

    if len(source_map) == 0:
        pass
    elif len(source_map) == 1:
        source = mapping[source_map[0]][1]
    else:
        raise Exception("Source appears to be overmerged")

    print(sink, source)

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = affs[i_edge]
        weighted_graph[edge[0]][edge[1]]['weight'] = affs[i_edge]

    mst_weighted_graph = nx.minimum_spanning_tree(weighted_graph, weight="weight")

    dt = time.time() - time_start
    print("Graph creation: %.2fms" % (dt * 1000))
    time_start = time.time()

    ccs = list(nx.connected_components(mst_weighted_graph))
    for cc in ccs:
        if not (source in cc and sink in cc):
            mst_weighted_graph.remove_nodes_from(cc)

    # cutset = nx.minimum_edge_cut(weighted_graph, source, sink)
    min_cut_set = nx.minimum_edge_cut(mst_weighted_graph, source, sink,
                                  flow_func=shortest_augmenting_path)

    dt = time.time() - time_start
    print("Mincut: %.2fms" % (dt * 1000))

    if min_cut_set is None:
        return []

    if len(min_cut_set) != 1:
        raise  Exception("Too many or too few cuts: %d" %
                         len(min_cut_set))

    time_start = time.time()

    edge_cut = list(list(min_cut_set)[0])

    print(edge_cut)

    mst_weighted_graph.remove_edges_from([edge_cut])
    # mst_weighted_graph.add_nodes_from(edge_cut)
    ccs = list(nx.connected_components(mst_weighted_graph))

    for cc in ccs:
        print("CC size = %d" % len(cc))

    if len(ccs) != 2:
        raise  Exception("Too many or too few connected components: %d" %
                         len(ccs))

    flat_edges = edges.flatten()

    cc0 = np.array(list(ccs[0]), dtype=np.uint64)
    cc0_edge_mask = np.sum(np.in1d(flat_edges, cc0).reshape(-1, 2), axis=1) == 1

    cc1 = np.array(list(ccs[1]), dtype=np.uint64)
    cc1_edge_mask = np.sum(np.in1d(flat_edges, cc1).reshape(-1, 2), axis=1) == 1

    cutset = edges[np.where(np.logical_and(cc0_edge_mask, cc1_edge_mask))]

    dt = time.time() - time_start
    print("Splitting: %.2fms" % (dt * 1000))

    remapped_cutset = []
    for cut in cutset:
        if cut[0] in remapping:
            pre_cut = remapping[cut[0]]
        else:
            pre_cut = [cut[0]]

        if cut[1] in remapping:
            post_cut = remapping[cut[1]]
        else:
            post_cut = [cut[1]]

        remapped_cutset.extend(list(itertools.product(pre_cut, post_cut)))
        remapped_cutset.extend(list(itertools.product(post_cut, pre_cut)))

    remapped_cutset = np.array(remapped_cutset, dtype=np.uint64)

    remapped_cutset_flattened_view = remapped_cutset.view(dtype='u8,u8')
    edges_flattened_view = original_edges.view(dtype='u8,u8')

    cutset_mask = np.in1d(remapped_cutset_flattened_view, edges_flattened_view)

    return remapped_cutset[cutset_mask]


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

            if param_key in ["fan_out", "n_layers"]:
                val_dict = {param_key: np.array(value,
                                                dtype=np.uint64).tobytes()}
            elif param_key in ["cv_path"]:
                val_dict = {param_key: serialize_key(value)}
            elif param_key in ["chunk_size"]:
                val_dict = {param_key: np.array(value,
                                                dtype=np.uint64).tobytes()}
            else:
                raise Exception("Unknown type for parameter")

            row = self.mutate_row(serialize_key("params"), self.family_id,
                                  val_dict)

            self.bulk_write([row])
        else:
            value = row.cells[self.family_id][ser_param_key][0].value

            if param_key in ["fan_out", "n_layers"]:
                value = np.frombuffer(value, dtype=np.uint64)[0]
            elif param_key in ["cv_path"]:
                value = deserialize_key(value)
            elif param_key in ["chunk_size"]:
                value = np.frombuffer(value, dtype=np.uint64)
            else:
                raise Exception("Unknown key")

        return value

    def get_serialized_info(self):
        """ Rerturns dictionary that can be used to load this ChunkedGraph

        :return: dict
        """
        info = {"table_id": self.table_id,
                "instance_id": self.instance_id,
                "project_id": self.project_id,
                "credentials": self.client.credentials}

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
        assert node_id is not None or all(v is not None
                                          for v in [layer, x, y, z])

        if node_id is not None:
            layer = self.get_chunk_layer(node_id)
        bits_per_dim = self.bitmasks[layer]

        if node_id is not None:
            chunk_offset = 64 - self._n_bits_for_layer_id - 3 * bits_per_dim
            return np.uint64((int(node_id) >> chunk_offset) << chunk_offset)
        else:

            assert x < 2 ** bits_per_dim
            assert y < 2 ** bits_per_dim
            assert z < 2 ** bits_per_dim

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

    def get_unique_operation_id(self) -> np.uint64:
        """ Finds a unique operation id

        atomic counter

        Operations essentially live in layer 0. Even if segmentation ids might
        live in layer 0 one day, they would not collide with the operation ids
        because we write information belonging to operations in a separate
        family id.

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
        operation_id = int.from_bytes(operation_id_b, byteorder="big")

        return np.uint64(operation_id)

    def get_max_operation_id(self) -> np.uint64:
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
            max_operation_id = int.from_bytes(max_operation_id_b,
                                              byteorder="big")
        else:
            max_operation_id = 0

        return np.uint64(max_operation_id)

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

        row = self.table.read_row(serialize_uint64(node_id),
                                  filter_=ColumnQualifierRegexFilter(key))

        if row is None or key not in row.cells[self.family_id]:
            if get_time_stamp:
                return None, None
            else:
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
                   operation_id: Optional[np.uint64] = None,
                   slow_retry: bool = True) -> bool:
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
                         row_key_filters: Optional[Iterable[str]] = None,
                         time_stamp: datetime.datetime = datetime.datetime.max,
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
        :param row_key_filters: list of str
            rows *with* this column will be ignored
        :param time_stamp: datetime.datetime
        :param yield_rows: bool
        :return: list or yield of rows
        """
        # Comply to resolution of BigTables TimeRange
        time_stamp -= datetime.timedelta(
            microseconds=time_stamp.microsecond % 1000)

        # Create filters: time and id range
        time_filter = TimestampRangeFilter(TimestampRange(end=time_stamp))

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

        if row_filter is None:
            row_filter = time_filter
        else:
            row_filter = RowFilterChain([time_filter, row_filter])

        if row_key_filters is not None:
            for row_key in row_key_filters:
                key_filter = ColumnRangeFilter(
                    column_family_id=self.family_id,
                    start_column=row_key,
                    end_column=row_key,
                    inclusive_start=True,
                    inclusive_end=True)

                row_filter = ConditionalRowFilter(base_filter=key_filter,
                                                  false_filter=row_filter,
                                                  true_filter=BlockAllFilter(True))


        chunk_id = self.get_chunk_id(layer=layer, x=x, y=y, z=z)
        max_segment_id = self.get_segment_id_limit(chunk_id)

        # Define BigTable keys
        start_id = self.get_node_id(np.uint64(0), chunk_id=chunk_id)
        end_id = self.get_node_id(max_segment_id, chunk_id=chunk_id)

        if yield_rows:
            range_read_yield = self.table.yield_rows(
                start_key=serialize_uint64(start_id),
                end_key=serialize_uint64(end_id),
                filter_=row_filter)
            return range_read_yield
        else:
            # Set up read
            range_read = self.table.read_rows(
                start_key=serialize_uint64(start_id),
                end_key=serialize_uint64(end_id),
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
                filters.append(ColumnQualifierRegexFilter(serialize_key(k)))

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
            start_key=serialize_uint64(start_id),
            end_key=serialize_uint64(end_id),
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

        row = self.mutate_row(serialize_uint64(operation_id),
                              self.log_family_id, val_dict, time_stamp)

        return row

    def _create_merge_log_row(self, operation_id: np.uint64, user_id: str,
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

        row = self.mutate_row(serialize_uint64(operation_id),
                              self.log_family_id, val_dict, time_stamp)

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
            time_stamp = datetime.datetime.utcnow()

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

                rows.append(self.mutate_row(serialize_uint64(node_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))
                node_c += 1

            # Create parent node
            val_dict = {"children": node_ids.tobytes(),
                        "atomic_cross_edges": parent_cross_edges.tobytes()}

            rows.append(self.mutate_row(serialize_uint64(parent_id),
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

                        rows.append(self.mutate_row(serialize_uint64(node_id),
                                                    self.family_id, val_dict,
                                                    time_stamp=time_stamp))

                    # Create parent node
                    val_dict = {"children":
                                    node_ids.tobytes(),
                                "atomic_cross_edges":
                                    parent_cross_edges.tobytes()}

                    rows.append(self.mutate_row(serialize_uint64(parent_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                    self.bulk_write(rows)

        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

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
                                               row_keys=["atomic_cross_edges",
                                                         "children"],
                                               yield_rows=False)

            # Due to restarted jobs some parents might be duplicated. We can
            # find these duplicates only by comparing their children because
            # each node has a unique id. However, we can use that more recently
            # created nodes have higher segment ids. We are only interested in
            # the latest version of any duplicated parents.

            # Deserialize row keys and store child with highest id for
            # comparison

            segment_ids = np.array([], dtype=np.uint64)
            row_ids_b = np.array([])
            max_child_ids = np.array([], dtype=np.uint64)
            for row_id_b, row_data in range_read.items():
                row_id = deserialize_uint64(row_id_b)
                segment_id = self.get_segment_id(row_id)

                cell = row_data.cells[self.family_id]

                node_child_ids_b = cell[serialize_key("children")][0].value
                node_child_ids = np.frombuffer(node_child_ids_b,
                                               dtype=np.uint64)

                max_child_ids = np.concatenate([max_child_ids,
                                                [np.max(node_child_ids)]])
                segment_ids = np.concatenate([segment_ids, [segment_id]])
                row_ids_b = np.concatenate([row_ids_b, [row_id_b]])

            sorting = np.argsort(segment_ids)[::-1]
            row_ids_b = row_ids_b[sorting]
            max_child_ids = max_child_ids[sorting]

            counter = collections.defaultdict(int)

            max_child_ids_occ_so_far = np.zeros(len(max_child_ids),
                                                dtype=np.int)
            for i_row in range(len(max_child_ids)):
                max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
                counter[max_child_ids[i_row]] += 1

            # Filter last occurences (we inverted the list) of each node
            m = max_child_ids_occ_so_far == 0
            row_ids_b = row_ids_b[m]

            # Loop through nodes from this chunk
            for row_id_b in row_ids_b:
                row_id = deserialize_uint64(row_id_b)

                cell = range_read[row_id_b].cells[self.family_id][
                    serialize_key("atomic_cross_edges")]
                atomic_edges_b = cell[0].value
                atomic_edges = np.frombuffer(atomic_edges_b,
                                             dtype=np.uint64).reshape(-1, 2)

                atomic_partner_id_dict[row_id] = atomic_edges[:, 1]
                atomic_child_id_dict[row_id] = atomic_edges[:, 0]

                atomic_child_ids = np.concatenate([atomic_child_ids,
                                                   atomic_edges[:, 0]])
                child_ids =\
                    np.concatenate([child_ids,
                                    np.array([row_id] * len(atomic_edges[:, 0]),
                                             dtype=np.uint64)])

        # print("Time iterating through subchunks: %.3fs" %
        #       (time.time() - time_start))
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

        # print("Time resolving cross chunk edges: %.3fs" %
        #       (time.time() - time_start))
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

        # print("Time connected components: %.3fs" % (time.time() - time_start))

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

        parent_key = serialize_key("parents")
        all_parents = []

        p_filter_ = ColumnQualifierRegexFilter(parent_key)
        row = self.table.read_row(serialize_uint64(node_id), filter_=p_filter_)

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

    def get_latest_edge_affinity(self, atomic_edge: [np.uint64, np.uint64],
                                 check: bool = False) -> np.float32:
        """ Looks up the LATEST affinity of an edge

        Future work should add a timestamp option

        :param atomic_edge: [uint64, uint64]
        :param check: bool
            whether to look up affinity from both sides and compare
        :return: float32
        """
        edge_affinities = []

        if check:
            iter_max = 2
        else:
            iter_max = 1

        for i in range(iter_max):
            atomic_partners, atomic_affinities = \
                self.get_atomic_partners(atomic_edge[i % 2])

            edge_mask = atomic_partners == atomic_edge[(i + 1) % 2]

            if len(edge_mask) == 0:
                raise Exception("Edge does not exist")

            edge_affinities.append(atomic_affinities[edge_mask][0])

        if len(np.unique(edge_affinities)) == 1:
            return edge_affinities[0]
        else:
            raise Exception("Different edge affinities found... Something went "
                            "horribly wrong.")

    def get_latest_roots(self, time_stamp: Optional[datetime.datetime] = datetime.datetime.max,
                         n_threads: int = 1):
        """

        :param time_stamp:
        :return:
        """

        def _read_root_rows(args) -> None:
            start_seg_id, end_seg_id = args

            start_id = self.get_node_id(segment_id=start_seg_id,
                                        chunk_id=self.root_chunk_id)
            end_id = self.get_node_id(segment_id=end_seg_id,
                                      chunk_id=self.root_chunk_id)

            range_read = self.table.read_rows(
                start_key=serialize_uint64(start_id),
                end_key=serialize_uint64(end_id),
                # allow_row_interleaving=True,
                end_inclusive=False,
                filter_=time_filter)

            range_read.consume_all()

            rows = range_read.rows

            for row_id, row_data in rows.items():
                row_keys = row_data.cells[self.family_id]

                if not serialize_key("new_parents") in row_keys:
                    root_ids.append(deserialize_uint64(row_id))


        time_stamp -= datetime.timedelta(microseconds=time_stamp.microsecond % 1000)

        time_filter = TimestampRangeFilter(TimestampRange(end=time_stamp))

        max_seg_id = self.get_max_node_id(self.root_chunk_id) + 1

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

        :param atomic_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.utcnow()

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
            print(i_try)

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

        operation_id_b = serialize_uint64(operation_id)

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
        root_row = self.table.row(serialize_uint64(root_id),
                                  filter_=combined_filter)

        # Set row lock if condition returns no results (state == False)
        time_stamp = datetime.datetime.utcnow()
        root_row.set_cell(self.family_id, lock_key, operation_id_b, state=False,
                          timestamp=time_stamp)

        # The lock was acquired when set_cell returns False (state)
        lock_acquired = not root_row.commit()

        if not lock_acquired:
            r = self.table.read_row(serialize_uint64(root_id))

            l_operation_ids = []
            for cell in r.cells[self.family_id][lock_key]:
                l_operation_id = deserialize_uint64(cell.value)
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
        operation_id_b = serialize_uint64(operation_id)

        lock_key = serialize_key("lock")

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
        root_row = self.table.row(serialize_uint64(root_id),
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
        operation_id_b = serialize_uint64(operation_id)

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
        root_row = self.table.row(serialize_uint64(root_id),
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
            r = self.table.read_row(serialize_uint64(next_id))

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
                            datetime.datetime.max)-> np.ndarray:
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
                    _, row_time_stamp = self.read_row(next_id,
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
                            Optional[datetime.datetime] = datetime.datetime.max
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
                     bb_is_coordinate: bool = False, stop_lvl: int = 1,
                     get_edges: bool = False, verbose: bool = True
                     ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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

            if verbose:
                print("Layer %d: %.3fms for %d children with %d threads" %
                      (layer, (time.time() - time_start) * 1000, n_child_ids,
                       this_n_threads))

            # if len(child_ids) != len(np.unique(child_ids)):
            #     print("N children %d - %d" % (len(child_ids), len(np.unique(child_ids))))
            #     # print(agglomeration_id, child_ids)
            #
            # assert len(child_ids) == len(np.unique(child_ids))

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

        r = self.table.read_row(serialize_uint64(atomic_id),
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
        lock_root_ids = np.unique(root_ids)
        while i_try < n_tries:
            # Try to acquire lock and only continue if successful
            lock_acquired, lock_root_ids = \
                self.lock_root_loop(root_ids=lock_root_ids,
                                    operation_id=operation_id)

            if lock_acquired:
                # Add edge and change hierarchy
                new_root_id, rows, time_stamp = \
                    self._add_edge(operation_id=operation_id,
                                   atomic_edge=atomic_edge, affinity=affinity)

                # Add a row to the log
                rows.append(self._create_merge_log_row(operation_id, user_id,
                                                       [new_root_id],
                                                       atomic_edge, time_stamp))

                # Execute write (makes sure that we are still owning the lock)
                if self.bulk_write(rows, lock_root_ids,
                                   operation_id=operation_id, slow_retry=False):
                    return new_root_id

            for lock_root_id in lock_root_ids:
                self.unlock_root(lock_root_id, operation_id)

            i_try += 1

            print("Waiting - %d" % i_try)
            time.sleep(1)

        return None

    def _add_edge(self, operation_id: np.uint64,
                  atomic_edge: Sequence[np.uint64],
                  affinity: Optional[np.float32] = None
                  ) -> Tuple[np.uint64, List[bigtable.row.Row],
                             datetime.datetime]:
        """ Adds an atomic edge to the ChunkedGraph

        :param operation_id: uint64
        :param atomic_edge: list of two ints
        :param affinity: float
        :return: int
            new root id
        """
        time_stamp = datetime.datetime.utcnow()

        if affinity is None:
            affinity = np.float32(1.0)

        rows = []

        assert len(atomic_edge) == 2

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
            # If an edge connects two supervoxel that were already conntected
            # through another path, we will reach a point where we find the same
            # parent twice.
            current_parent_ids = np.unique(original_parent_ids[i_layer])

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
                    rows.append(self.mutate_row(serialize_uint64(child_id),
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
                val_dict["operation_id"] = serialize_uint64(operation_id)

                rows.append(self.mutate_row(serialize_uint64(original_root[0]),
                                            self.family_id,
                                            {"new_parents": new_parent_id_b},
                                            time_stamp=time_stamp))

                rows.append(self.mutate_row(serialize_uint64(original_root[1]),
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

            rows.append(self.mutate_row(serialize_uint64(current_node_id),
                                        self.family_id, val_dict,
                                        time_stamp=time_stamp))

        # Atomic edge
        for i_atomic_id in range(2):
            val_dict = \
                {"atomic_partners":
                     np.array([atomic_edge[(i_atomic_id + 1) % 2]]).tobytes(),
                 "atomic_affinities":
                     np.array([affinity], dtype=np.float32).tobytes()}

            rows.append(self.mutate_row(serialize_uint64(
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
                rows.append(self._create_split_log_row(operation_id, user_id,
                                                       new_root_ids,
                                                       [source_id, sink_id],
                                                       removed_edges,
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

        print(
            "Get roots and check: %.3fms" % ((time.time() - time_start) * 1000))
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
            print("inf in cutset")
            return False, None

        # Remove edges
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

        # Make sure that we have a list of edges
        if isinstance(atomic_edges[0], np.uint64):
            atomic_edges = [atomic_edges]

        for atomic_edge in atomic_edges:
            if np.isinf(self.get_latest_edge_affinity(atomic_edge)):
                return False, None

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

            rows.append(self.mutate_row(serialize_uint64(u_atomic_id),
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

                rows.append(self.mutate_row(serialize_uint64(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(self.mutate_row(serialize_uint64(cc_node_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

        # Now that the lowest layer has been updated, we need to walk through
        # all layers and move our new parents forward
        # new_layer_parent_dict stores all newly created parents. We first
        # empty it and then fill it with the new parents in the next layer
        if self.n_layers == 2:
            return True, (list(new_layer_parent_dict.keys()), rows, time_stamp)

        new_roots = []
        for i_layer in range(2, self.n_layers):

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
                            atomic_id_map = np.concatenate([atomic_id_map, ps])
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
                    parent_cc_list[parent_cc_id] = \
                        np.unique(parent_cc_list[parent_cc_id] + list(partners))
                else:
                    parent_cc_id = len(parent_cc_list)
                    parent_cc_list.append(list(partners))
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

                if old_next_layer_parent is None:
                    return False, None

                cc_node_ids = np.array(list(parent_cc), dtype=np.uint64)
                cc_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

                for parent_id in parent_cc:
                    cc_cross_edges = np.concatenate([cc_cross_edges,
                                                     leftover_edges[parent_id]])

                this_chunk_id = self.get_chunk_id(node_id=old_next_layer_parent)
                new_parent_id = self.get_unique_node_id(this_chunk_id)
                new_parent_id_b = np.array(new_parent_id).tobytes()

                new_layer_parent_dict[new_parent_id] = \
                    parent_cc_old_parent_list[i_cc]
                cross_edge_dict[new_parent_id] = cc_cross_edges

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(self.mutate_row(serialize_uint64(cc_node_id),
                                                self.family_id, val_dict,
                                                time_stamp=time_stamp))

                val_dict = {"children": cc_node_ids.tobytes(),
                            "atomic_cross_edges": cc_cross_edges.tobytes()}

                if i_layer == self.n_layers-1:
                    new_roots.append(new_parent_id)
                    val_dict["former_parents"] = \
                        np.array(original_root).tobytes()
                    val_dict["operation_id"] = \
                        serialize_uint64(operation_id)

                rows.append(self.mutate_row(serialize_uint64(new_parent_id),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

            if i_layer == self.n_layers-1:
                val_dict = {"new_parents": np.array(new_roots,
                                                    dtype=np.uint64).tobytes()}
                rows.append(self.mutate_row(serialize_uint64(original_root),
                                            self.family_id, val_dict,
                                            time_stamp=time_stamp))

        return True, (new_roots, rows, time_stamp)
