import datetime
from typing import Any
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Iterable
from typing import Optional

import numpy as np
from multiwrapper import multiprocessing_utils as mu
from google.auth import credentials
from google.cloud import bigtable
from google.api_core.retry import Retry
from google.api_core.retry import if_exception_type
from google.api_core.exceptions import Aborted
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.bigtable.table import Table
from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.row_filters import TimestampRange
from google.cloud.bigtable.row_filters import TimestampRangeFilter
from google.cloud.bigtable.row_filters import ColumnRangeFilter
from google.cloud.bigtable.row_filters import ValueRangeFilter
from google.cloud.bigtable.row_filters import RowFilterChain
from google.cloud.bigtable.row_filters import PassAllFilter
from google.cloud.bigtable.row_filters import RowFilter
from google.cloud.bigtable.row_filters import ConditionalRowFilter
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter
from google.cloud.bigtable.column_family import MaxVersionsGCRule

from . import attributes
from ..base import ClientWithIDGen
from ...meta import ChunkedGraphMeta


class BigTableClient(bigtable.Client, ClientWithIDGen):
    def __init__(
        self,
        project: str = None,
        read_only: bool = False,
        admin: bool = False,
        graph_meta: ChunkedGraphMeta = None,
    ):
        super(BigTableClient, self).__init__(
            project=project, read_only=read_only, admin=admin
        )
        self._graph_meta = graph_meta
        self._instance = self.instance(graph_meta.bigtable_config.INSTANCE)
        self._table = self._instance.table(graph_meta.graph_config.ID)

    @property
    def graph_meta(self):
        # TODO
        # read meta from table if None
        return self._graph_meta

    def create_graph(self) -> None:
        """Initialize the graph and store associated meta."""
        config = self._graph_meta.graph_config
        if not config.overwrite and self._table.exists():
            ValueError(f"{config.graph_id} already exists.")
        self._table.create()
        self._create_column_families()
        # TODO
        # store meta in the table

    def read_nodes(
        self,
        start_id=None,
        end_id=None,
        node_ids=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """
        Read nodes and their properties.
        Accepts a range of node IDs or specific node IDs.
        """
        pass

    def read_node(
        self,
        node_id: np.uint64,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single node and it's properties."""
        pass

    def write_nodes(self, nodes, root_ids, operation_id):
        """
        Writes/updates nodes (IDs along with properties)
        by locking root nodes until changes are written.
        """
        pass

    def create_segment_ids(self):
        """Generate a range of unique segment IDs."""
        pass

    def create_segment_id(self):
        """Generate a unique segment ID."""
        pass

    def get_max_segment_id(self, chunk_id: np.uint64):
        """Gets the current maximum segment ID in the chunk."""
        pass

    def create_operation_id(self):
        """Generate a unique operation ID."""
        pass

    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""
        pass

    def _create_column_families(self):
        f = self._table.column_family("0")
        f.create()

        f = self._table.column_family("1", gc_rule=MaxVersionsGCRule(1))
        f.create()

        f = self._table.column_family("2")
        f.create()

        f = self._table.column_family("3", gc_rule=MaxVersionsGCRule(1))
        f.create()

    def _read(
        self, row_set: RowSet, row_filter: RowFilter = None
    ) -> Dict[bytes, Dict[attributes._Attribute, bigtable.row_data.PartialRowData]]:
        """ Core function to read rows from Bigtable. Uses standard Bigtable retry logic
        :param row_set: BigTable RowSet
        :param row_filter: BigTable RowFilter
        :return: Dict[bytes, Dict[column_keys._Column, bigtable.row_data.PartialRowData]]
        """

        def _execute_read_thread(args: Tuple[Table, RowSet, RowFilter]):
            table, row_set, row_filter = args
            if not row_set.row_keys and not row_set.row_ranges:
                # Check for everything falsy, because Bigtable considers even empty
                # lists of row_keys as no upper/lower bound!
                return {}

            range_read = table.read_rows(row_set=row_set, filter_=row_filter)
            res = {v.row_key: partial_row_data_to_column_dict(v) for v in range_read}
            return res

        # FIXME: Bigtable limits the length of the serialized request to 512 KiB. We should
        # calculate this properly (range_read.request.SerializeToString()), but this estimate is
        # good enough for now
        max_row_key_count = 20000
        n_subrequests = max(1, int(np.ceil(len(row_set.row_keys) / max_row_key_count)))
        n_threads = min(n_subrequests, 2 * mu.n_cpus)

        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[
                i * max_row_key_count : (i + 1) * max_row_key_count
            ]
            row_sets.append(r)

        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges
        responses = mu.multithread_func(
            _execute_read_thread,
            params=((self._table, r, row_filter) for r in row_sets),
            n_threads=n_threads,
        )

        combined_response = {}
        for resp in responses:
            combined_response.update(resp)
        return combined_response

    def _write(
        self,
        rows: Iterable[bigtable.row.DirectRow],
        root_ids: Optional[Union[np.uint64, Iterable[np.uint64]]] = None,
        operation_id: Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
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
            predicate=if_exception_type(
                (Aborted, DeadlineExceeded, ServiceUnavailable)
            ),
            initial=initial,
            maximum=15.0,
            multiplier=2.0,
            deadline=self.graph_meta.graph_config.ROOT_LOCK_EXPIRY.seconds,
        )

        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]
            if not self.check_and_renew_root_locks(root_ids, operation_id):
                raise cg_exceptions.LockError(
                    f"Root lock renewal failed for operation ID {operation_id}"
                )

        for i_row in range(0, len(rows), block_size):
            status = self.table.mutate_rows(
                rows[i_row : i_row + block_size], retry=retry_policy
            )
            if not all(status):
                raise cg_exceptions.ChunkedGraphError(
                    f"Bulk write failed for operation ID {operation_id}"
                )

    def _mutate_row(
        self,
        row_key: bytes,
        val_dict: Dict[column_keys._Column, Any],
        time_stamp: Optional[datetime.datetime] = None,
    ) -> bigtable.row.Row:
        """ Mutates a single row
        :param row_key: serialized bigtable row key
        :param val_dict: Dict[column_keys._TypedColumn: bytes]
        :param time_stamp: None or datetime
        :return: list
        """
        row = self._table.row(row_key)
        for column, value in val_dict.items():
            row.set_cell(
                column_family_id=column.family_id,
                column=column.key,
                value=column.serialize(value),
                timestamp=time_stamp,
            )
        return row


a = BigTableClient()


# def _get_unique_range(self, row_key, step):
#     column = column_keys.Concurrency.CounterID
#     # Incrementer row keys start with an "i" followed by the chunk id
#     append_row = self.table.row(row_key, append=True)
#     append_row.increment_cell_value(column.family_id, column.key, step)

#     # This increments the row entry and returns the value AFTER incrementing
#     latest_row = append_row.commit()
#     max_segment_id = column.deserialize(latest_row[column.family_id][column.key][0][0])
#     min_segment_id = max_segment_id + np.uint64(1) - step
#     return min_segment_id, max_segment_id


# def get_unique_segment_id_root_row(
#     self, step: int = 1, counter_id: int = None
# ) -> np.ndarray:
#     """ Return unique Segment ID for the Root Chunk
#     atomic counter
#     :param step: int
#     :param counter_id: np.uint64
#     :return: np.uint64
#     """
#     if self.n_bits_root_counter == 0:
#         return self.get_unique_segment_id_range(self.root_chunk_id, step=step)

#     n_counters = np.uint64(2 ** self._n_bits_root_counter)
#     if counter_id is None:
#         counter_id = np.uint64(np.random.randint(0, n_counters))
#     else:
#         counter_id = np.uint64(counter_id % n_counters)

#     row_key = serializers.serialize_key(
#         f"i{serializers.pad_node_id(self.root_chunk_id)}_{counter_id}"
#     )
#     min_segment_id, max_segment_id = self._get_unique_range(row_key=row_key, step=step)

#     segment_id_range = np.arange(
#         min_segment_id * n_counters + counter_id,
#         max_segment_id * n_counters + np.uint64(1) + counter_id,
#         n_counters,
#         dtype=basetypes.SEGMENT_ID,
#     )
#     return segment_id_range


# def get_unique_segment_id_range(self, chunk_id: np.uint64, step: int = 1) -> np.ndarray:
#     """ Return unique Segment ID for given Chunk ID
#     atomic counter
#     :param chunk_id: np.uint64
#     :param step: int
#     :return: np.uint64
#     """
#     if self.n_layers == self.get_chunk_layer(chunk_id) and self.n_bits_root_counter > 0:
#         return self.get_unique_segment_id_root_row(step=step)

#     row_key = serializers.serialize_key("i%s" % serializers.pad_node_id(chunk_id))
#     min_segment_id, max_segment_id = self._get_unique_range(row_key=row_key, step=step)
#     segment_id_range = np.arange(
#         min_segment_id, max_segment_id + np.uint64(1), dtype=basetypes.SEGMENT_ID
#     )
#     return segment_id_range


# def get_unique_node_id_range(self, chunk_id: np.uint64, step: int = 1) -> np.ndarray:
#     """ Return unique Node ID range for given Chunk ID atomic counter
#     :param chunk_id: np.uint64
#     :param step: int
#     :return: np.uint64
#     """
#     segment_ids = self.get_unique_segment_id_range(chunk_id=chunk_id, step=step)
#     node_ids = np.array(
#         [self.get_node_id(segment_id, chunk_id) for segment_id in segment_ids],
#         dtype=np.uint64,
#     )
#     return node_ids


# def get_unique_node_id(self, chunk_id: np.uint64) -> np.uint64:
#     """ Return unique Node ID for given Chunk ID atomic counter
#     :param chunk_id: np.uint64
#     :return: np.uint64
#     """
#     return self.get_unique_node_id_range(chunk_id=chunk_id, step=1)[0]


# def get_max_seg_id_root_chunk(self) -> np.uint64:
#     """  Gets maximal root id based on the atomic counter
#     This is an approximation. It is not guaranteed that all ids smaller or
#     equal to this id exists. However, it is guaranteed that no larger id
#     exist at the time this function is executed.
#     :return: uint64
#     """
#     if self.n_bits_root_counter == 0:
#         return self.get_max_seg_id(self.root_chunk_id)

#     n_counters = np.uint64(2 ** self.n_bits_root_counter)
#     max_value = 0
#     for counter_id in range(n_counters):
#         row_key = serializers.serialize_key(
#             f"i{serializers.pad_node_id(self.root_chunk_id)}_{counter_id}"
#         )
#         row = self.read_byte_row(row_key, columns=column_keys.Concurrency.CounterID)
#         counter = basetypes.SEGMENT_ID.type(row[0].value if row else 0) * n_counters
#         if counter > max_value:
#             max_value = counter
#     return max_value


# def get_max_seg_id(self, chunk_id: np.uint64) -> np.uint64:
#     """  Gets maximal seg id in a chunk based on the atomic counter
#     This is an approximation. It is not guaranteed that all ids smaller or
#     equal to this id exists. However, it is guaranteed that no larger id
#     exist at the time this function is executed.
#     :return: uint64
#     """
#     if self.n_layers == self.get_chunk_layer(chunk_id) and self.n_bits_root_counter > 0:
#         return self.get_max_seg_id_root_chunk()

#     # Incrementer row keys start with an "i"
#     row_key = serializers.serialize_key("i%s" % serializers.pad_node_id(chunk_id))
#     row = self.read_byte_row(row_key, columns=column_keys.Concurrency.CounterID)

#     # Read incrementer value (default to 0) and interpret is as Segment ID
#     return basetypes.SEGMENT_ID.type(row[0].value if row else 0)


# def get_max_node_id(self, chunk_id: np.uint64) -> np.uint64:
#     """  Gets maximal node id in a chunk based on the atomic counter
#     This is an approximation. It is not guaranteed that all ids smaller or
#     equal to this id exists. However, it is guaranteed that no larger id
#     exist at the time this function is executed.
#     :return: uint64
#     """
#     max_seg_id = self.get_max_seg_id(chunk_id)
#     return self.get_node_id(segment_id=max_seg_id, chunk_id=chunk_id)


# def get_unique_operation_id(self) -> np.uint64:
#     """ Finds a unique operation id atomic counter
#     Operations essentially live in layer 0. Even if segmentation ids might
#     live in layer 0 one day, they would not collide with the operation ids
#     because we write information belonging to operations in a separate
#     family id.
#     :return: str
#     """
#     column = column_keys.Concurrency.CounterID
#     append_row = self.table.row(row_keys.OperationID, append=True)
#     append_row.increment_cell_value(column.family_id, column.key, 1)

#     # This increments the row entry and returns the value AFTER incrementing
#     latest_row = append_row.commit()
#     operation_id_b = latest_row[column.family_id][column.key][0][0]
#     operation_id = column.deserialize(operation_id_b)
#     return np.uint64(operation_id)


# def get_max_operation_id(self) -> np.int64:
#     """  Gets maximal operation id based on the atomic counter
#         This is an approximation. It is not guaranteed that all ids smaller or
#         equal to this id exists. However, it is guaranteed that no larger id
#         exist at the time this function is executed.
#         :return: int64
#         """
#     column = column_keys.Concurrency.CounterID
#     row = self.read_byte_row(row_keys.OperationID, columns=column)
#     return row[0].value if row else column.basetype(0)

