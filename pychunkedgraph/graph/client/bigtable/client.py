# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel, line-too-long, protected-access, arguments-differ, arguments-renamed, logging-fstring-interpolation, too-many-arguments

import sys
import time
import typing
import logging
from datetime import datetime
from datetime import timedelta

import numpy as np
from multiwrapper import multiprocessing_utils as mu
from google.cloud import bigtable
from google.api_core.retry import Retry
from google.api_core.retry import if_exception_type
from google.api_core.exceptions import Aborted
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.bigtable.column_family import MaxAgeGCRule
from google.cloud.bigtable.column_family import MaxVersionsGCRule
from google.cloud.bigtable.table import Table
from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.row_data import PartialRowData
from google.cloud.bigtable.row_filters import RowFilter

from . import utils
from . import BigTableConfig
from ..base import ClientWithIDGen
from ..base import OperationLogger
from ... import attributes
from ... import exceptions
from ...utils import basetypes
from ...utils.serializers import pad_node_id
from ...utils.serializers import serialize_key
from ...utils.serializers import serialize_uint64
from ...utils.serializers import deserialize_uint64
from ...meta import ChunkedGraphMeta
from ...utils.generic import get_valid_timestamp


class Client(bigtable.Client, ClientWithIDGen, OperationLogger):
    def __init__(
        self,
        table_id: str,
        config: BigTableConfig = BigTableConfig(),
        graph_meta: ChunkedGraphMeta = None,
    ):
        if config.CREDENTIALS:
            super(Client, self).__init__(
                project=config.PROJECT,
                read_only=config.READ_ONLY,
                admin=config.ADMIN,
                credentials=config.CREDENTIALS,
            )
        else:
            super(Client, self).__init__(
                project=config.PROJECT,
                read_only=config.READ_ONLY,
                admin=config.ADMIN,
            )
        self._instance = self.instance(config.INSTANCE)
        self._table = self._instance.table(table_id)

        self.logger = logging.getLogger(
            f"{config.PROJECT}/{config.INSTANCE}/{table_id}"
        )
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.WARNING)
            self.logger.addHandler(sh)
        self._graph_meta = graph_meta
        self._version = None
        self._max_row_key_count = config.MAX_ROW_KEY_COUNT

    @property
    def graph_meta(self):
        return self._graph_meta

    def create_graph(self, meta: ChunkedGraphMeta, version: str) -> None:
        """Initialize the graph and store associated meta."""
        if self._table.exists():
            raise ValueError(f"{self._table.table_id} already exists.")
        self._table.create()
        self._create_column_families()
        self.add_graph_version(version)
        self.update_graph_meta(meta)

    def add_graph_version(self, version: str):
        assert self.read_graph_version() is None, "Graph has already been versioned."
        self._version = version
        row = self.mutate_row(
            attributes.GraphVersion.key,
            {attributes.GraphVersion.Version: version},
        )
        self.write([row])

    def read_graph_version(self) -> str:
        try:
            row = self._read_byte_row(attributes.GraphVersion.key)
            self._version = row[attributes.GraphVersion.Version][0].value
            return self._version
        except KeyError:
            return None

    def _delete_meta(self):
        # temprorary fix, use new column with GCRule for permanent fix
        # delete existing meta before update, but compatibilty issues
        meta_row = self._table.direct_row(attributes.GraphMeta.key)
        meta_row.delete()
        meta_row.commit()

    def update_graph_meta(
        self, meta: ChunkedGraphMeta, overwrite: typing.Optional[bool] = False
    ):
        if overwrite:
            self._delete_meta()
        self._graph_meta = meta
        row = self.mutate_row(
            attributes.GraphMeta.key,
            {attributes.GraphMeta.Meta: meta},
        )
        self.write([row])

    def read_graph_meta(self) -> ChunkedGraphMeta:
        row = self._read_byte_row(attributes.GraphMeta.key)
        self._graph_meta = row[attributes.GraphMeta.Meta][0].value
        return self._graph_meta

    def read_nodes(
        self,
        start_id=None,
        end_id=None,
        end_id_inclusive=False,
        user_id=None,
        node_ids=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
    ):
        """
        Read nodes and their properties.
        Accepts a range of node IDs or specific node IDs.
        """
        if node_ids is not None and len(node_ids) > self._max_row_key_count:
            # bigtable reading is faster
            # when all IDs in a block are within a range
            node_ids = np.sort(node_ids)
        rows = self._read_byte_rows(
            start_key=serialize_uint64(start_id, fake_edges=fake_edges)
            if start_id is not None
            else None,
            end_key=serialize_uint64(end_id, fake_edges=fake_edges)
            if end_id is not None
            else None,
            end_key_inclusive=end_id_inclusive,
            row_keys=(
                serialize_uint64(node_id, fake_edges=fake_edges) for node_id in node_ids
            )
            if node_ids is not None
            else None,
            columns=properties,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
            user_id=user_id,
        )
        return {
            deserialize_uint64(row_key, fake_edges=fake_edges): data
            for (row_key, data) in rows.items()
        }

    def read_node(
        self,
        node_id: np.uint64,
        properties: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
    ) -> typing.Union[
        typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
        typing.List[bigtable.row_data.Cell],
    ]:
        """Convenience function for reading a single node from Bigtable.
        Arguments:
            node_id {np.uint64} -- the NodeID of the row to be read.
        Keyword Arguments:
            columns {typing.Optional[typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]]} --
                typing.Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {typing.Optional[datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {typing.Optional[datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})
        Returns:
            typing.Union[typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
                  typing.List[bigtable.row_data.Cell]] --
                Returns a mapping of columns to a typing.List of cells (one cell per timestamp). Each cell
                has a `value` property, which returns the deserialized field, and a `timestamp`
                property, which returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the typing.List of cells is returned
                directly.
        """
        return self._read_byte_row(
            row_key=serialize_uint64(node_id, fake_edges=fake_edges),
            columns=properties,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )

    def write_nodes(self, nodes, root_ids=None, operation_id=None):
        """
        Writes/updates nodes (IDs along with properties)
        by locking root nodes until changes are written.
        """

    def read_log_entry(
        self, operation_id: np.uint64
    ) -> typing.Tuple[typing.Dict, datetime]:
        log_record = self.read_node(
            operation_id, properties=attributes.OperationLogs.all()
        )
        if len(log_record) == 0:
            return {}, None
        try:
            timestamp = log_record[attributes.OperationLogs.OperationTimeStamp][0].value
        except KeyError:
            timestamp = log_record[attributes.OperationLogs.RootID][0].timestamp
        log_record.update((column, v[0].value) for column, v in log_record.items())
        return log_record, timestamp

    def read_log_entries(
        self,
        operation_ids: typing.Optional[typing.Iterable] = None,
        user_id: typing.Optional[str] = None,
        properties: typing.Optional[typing.Iterable[attributes._Attribute]] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ):
        if properties is None:
            properties = attributes.OperationLogs.all()

        if operation_ids is None:
            logs_d = self.read_nodes(
                start_id=np.uint64(0),
                end_id=self.get_max_operation_id(),
                end_id_inclusive=True,
                user_id=user_id,
                properties=properties,
                start_time=start_time,
                end_time=end_time,
                end_time_inclusive=end_time_inclusive,
            )
        else:
            logs_d = self.read_nodes(
                node_ids=operation_ids,
                properties=properties,
                start_time=start_time,
                end_time=end_time,
                end_time_inclusive=end_time_inclusive,
                user_id=user_id,
            )
        if not logs_d:
            return {}
        for operation_id in logs_d:
            log_record = logs_d[operation_id]
            try:
                timestamp = log_record[attributes.OperationLogs.OperationTimeStamp][
                    0
                ].value
            except KeyError:
                timestamp = log_record[attributes.OperationLogs.RootID][0].timestamp
            log_record.update((column, v[0].value) for column, v in log_record.items())
            log_record["timestamp"] = timestamp
        return logs_d

    # Helpers
    def write(
        self,
        rows: typing.Iterable[bigtable.row.DirectRow],
        root_ids: typing.Optional[
            typing.Union[np.uint64, typing.Iterable[np.uint64]]
        ] = None,
        operation_id: typing.Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
        """Writes a list of mutated rows in bulk
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

        exception_types = (Aborted, DeadlineExceeded, ServiceUnavailable)
        retry = Retry(
            predicate=if_exception_type(exception_types),
            initial=initial,
            maximum=15.0,
            multiplier=2.0,
            deadline=self.graph_meta.graph_config.ROOT_LOCK_EXPIRY.seconds,
        )

        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]
            if not self.renew_locks(root_ids, operation_id):
                raise exceptions.LockingError(
                    f"Root lock renewal failed: operation {operation_id}"
                )

        for i in range(0, len(rows), block_size):
            status = self._table.mutate_rows(rows[i : i + block_size], retry=retry)
            if not all(status):
                raise exceptions.ChunkedGraphError(
                    f"Bulk write failed: operation {operation_id}"
                )

    def mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> bigtable.row.Row:
        """Mutates a single row (doesn't write to big table)."""
        row = self._table.direct_row(row_key)
        for column, value in val_dict.items():
            row.set_cell(
                column_family_id=column.family_id,
                column=column.key,
                value=column.serialize(value),
                timestamp=time_stamp,
            )
        return row

    # Locking
    def lock_root(
        self,
        root_id: np.uint64,
        operation_id: np.uint64,
    ) -> bool:
        """Attempts to lock the latest version of a root node."""
        lock_expiry = self.graph_meta.graph_config.ROOT_LOCK_EXPIRY
        lock_column = attributes.Concurrency.Lock
        indefinite_lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_root_lock_filter(
            lock_column, lock_expiry, indefinite_lock_column
        )

        root_row = self._table.conditional_row(
            serialize_uint64(root_id), filter_=filter_
        )
        # Set row lock if condition returns no results (state == False)
        root_row.set_cell(
            lock_column.family_id,
            lock_column.key,
            serialize_uint64(operation_id),
            state=False,
            timestamp=get_valid_timestamp(None),
        )

        # The lock was acquired when set_cell returns False (state)
        lock_acquired = not root_row.commit()
        if not lock_acquired:
            row = self._read_byte_row(serialize_uint64(root_id), columns=lock_column)
            l_operation_ids = [cell.value for cell in row]
            self.logger.debug(f"Locked operation ids: {l_operation_ids}")
        return lock_acquired

    def lock_root_indefinitely(
        self,
        root_id: np.uint64,
        operation_id: np.uint64,
    ) -> bool:
        """Attempts to indefinitely lock the latest version of a root node."""
        lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_indefinite_root_lock_filter(lock_column)
        root_row = self._table.conditional_row(
            serialize_uint64(root_id), filter_=filter_
        )
        # Set row lock if condition returns no results (state == False)
        root_row.set_cell(
            lock_column.family_id,
            lock_column.key,
            serialize_uint64(operation_id),
            state=False,
            timestamp=get_valid_timestamp(None),
        )

        # The lock was acquired when set_cell returns False (state)
        lock_acquired = not root_row.commit()
        if not lock_acquired:
            row = self._read_byte_row(serialize_uint64(root_id), columns=lock_column)
            l_operation_ids = [cell.value for cell in row]
            self.logger.debug(f"Indefinitely locked operation ids: {l_operation_ids}")
        return lock_acquired

    def lock_roots(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: typing.Dict,
        max_tries: int = 1,
        waittime_s: float = 0.5,
    ) -> typing.Tuple[bool, typing.Iterable]:
        """Attempts to lock multiple nodes with same operation id"""
        i_try = 0
        while i_try < max_tries:
            lock_acquired = False
            # Collect latest root ids
            new_root_ids: typing.List[np.uint64] = []
            for root_id in root_ids:
                future_root_ids = future_root_ids_d[root_id]
                if not future_root_ids.size:
                    new_root_ids.append(root_id)
                else:
                    new_root_ids.extend(future_root_ids)

            # Attempt to lock all latest root ids
            root_ids = np.unique(new_root_ids)
            for root_id in root_ids:
                lock_acquired = self.lock_root(root_id, operation_id)
                # Roll back locks if one root cannot be locked
                if not lock_acquired:
                    for id_ in root_ids:
                        self.unlock_root(id_, operation_id)
                    break

            if lock_acquired:
                return True, root_ids
            time.sleep(waittime_s)
            i_try += 1
            self.logger.debug(f"Try {i_try}")
        return False, root_ids

    def lock_roots_indefinitely(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: typing.Dict,
    ) -> typing.Tuple[bool, typing.Iterable]:
        """Attempts to indefinitely lock multiple nodes with same operation id"""
        lock_acquired = False
        # Collect latest root ids
        new_root_ids: typing.List[np.uint64] = []
        for _id in root_ids:
            future_root_ids = future_root_ids_d.get(_id)
            if not future_root_ids.size:
                new_root_ids.append(_id)
            else:
                new_root_ids.extend(future_root_ids)

        # Attempt to lock all latest root ids
        failed_to_lock_id = None
        root_ids = np.unique(new_root_ids)
        for _id in root_ids:
            self.logger.debug(f"operation {operation_id} root_id {_id}")
            lock_acquired = self.lock_root_indefinitely(_id, operation_id)
            # Roll back locks if one root cannot be locked
            if not lock_acquired:
                failed_to_lock_id = _id
                for id_ in root_ids:
                    self.unlock_indefinitely_locked_root(id_, operation_id)
                break
        if lock_acquired:
            return True, root_ids, failed_to_lock_id
        return False, root_ids, failed_to_lock_id

    def unlock_root(self, root_id: np.uint64, operation_id: np.uint64):
        """Unlocks root node that is locked with operation_id."""
        lock_column = attributes.Concurrency.Lock
        expiry = self.graph_meta.graph_config.ROOT_LOCK_EXPIRY
        root_row = self._table.conditional_row(
            serialize_uint64(root_id),
            filter_=utils.get_unlock_root_filter(lock_column, expiry, operation_id),
        )
        # Delete row if conditions are met (state == True)
        root_row.delete_cell(lock_column.family_id, lock_column.key, state=True)
        return root_row.commit()

    def unlock_indefinitely_locked_root(
        self, root_id: np.uint64, operation_id: np.uint64
    ):
        """Unlocks root node that is indefinitely locked with operation_id."""
        lock_column = attributes.Concurrency.IndefiniteLock
        # Get conditional row using the chained filter
        root_row = self._table.conditional_row(
            serialize_uint64(root_id),
            filter_=utils.get_indefinite_unlock_root_filter(lock_column, operation_id),
        )
        # Delete row if conditions are met (state == True)
        root_row.delete_cell(lock_column.family_id, lock_column.key, state=True)
        return root_row.commit()

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        """Renews existing root node lock with operation_id to extend time."""
        lock_column = attributes.Concurrency.Lock
        root_row = self._table.conditional_row(
            serialize_uint64(root_id),
            filter_=utils.get_renew_lock_filter(lock_column, operation_id),
        )
        # Set row lock if condition returns a result (state == True)
        root_row.set_cell(
            lock_column.family_id,
            lock_column.key,
            lock_column.serialize(operation_id),
            state=False,
        )
        # The lock was acquired when set_cell returns True (state)
        return not root_row.commit()

    def renew_locks(self, root_ids: np.uint64, operation_id: np.uint64) -> bool:
        """Renews existing root node locks with operation_id to extend time."""
        for root_id in root_ids:
            if not self.renew_lock(root_id, operation_id):
                self.logger.warning(f"renew_lock failed - {root_id}")
                return False
        return True

    def get_lock_timestamp(
        self, root_id: np.uint64, operation_id: np.uint64
    ) -> typing.Union[datetime, None]:
        """Lock timestamp for a Root ID operation."""
        row = self.read_node(root_id, properties=attributes.Concurrency.Lock)
        if len(row) == 0:
            self.logger.warning(f"No lock found for {root_id}")
            return None
        if row[0].value != operation_id:
            self.logger.warning(f"{root_id} not locked with {operation_id}")
            return None
        return row[0].timestamp

    def get_consolidated_lock_timestamp(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_ids: typing.Sequence[np.uint64],
    ) -> typing.Union[datetime, None]:
        """Minimum of multiple lock timestamps."""
        time_stamps = []
        for root_id, operation_id in zip(root_ids, operation_ids):
            time_stamp = self.get_lock_timestamp(root_id, operation_id)
            if time_stamp is None:
                return None
            time_stamps.append(time_stamp)
        if len(time_stamps) == 0:
            return None
        return np.min(time_stamps)

    # IDs
    def create_node_ids(
        self, chunk_id: np.uint64, size: int, root_chunk=False
    ) -> np.ndarray:
        """Generates a list of unique node IDs for the given chunk."""
        if root_chunk:
            new_ids = self._get_root_segment_ids_range(chunk_id, size)
        else:
            low, high = self._get_ids_range(
                serialize_uint64(chunk_id, counter=True), size
            )
            low, high = basetypes.SEGMENT_ID.type(low), basetypes.SEGMENT_ID.type(high)
            new_ids = np.arange(low, high + np.uint64(1), dtype=basetypes.SEGMENT_ID)
        return new_ids | chunk_id

    def create_node_id(
        self, chunk_id: np.uint64, root_chunk=False
    ) -> basetypes.NODE_ID:
        """Generate a unique node ID in the chunk."""
        return self.create_node_ids(chunk_id, 1, root_chunk=root_chunk)[0]

    def get_max_node_id(
        self, chunk_id: basetypes.CHUNK_ID, root_chunk=False
    ) -> basetypes.NODE_ID:
        """Gets the current maximum segment ID in the chunk."""
        if root_chunk:
            n_counters = np.uint64(2**8)
            max_value = 0
            for counter in range(n_counters):
                row = self._read_byte_row(
                    serialize_key(f"i{pad_node_id(chunk_id)}_{counter}"),
                    columns=attributes.Concurrency.Counter,
                )
                val = (
                    basetypes.SEGMENT_ID.type(row[0].value if row else 0) * n_counters
                    + counter
                )
                max_value = val if val > max_value else max_value
            return chunk_id | basetypes.SEGMENT_ID.type(max_value)
        column = attributes.Concurrency.Counter
        row = self._read_byte_row(
            serialize_uint64(chunk_id, counter=True), columns=column
        )
        return chunk_id | basetypes.SEGMENT_ID.type(row[0].value if row else 0)

    def create_operation_id(self):
        """Generate a unique operation ID."""
        return self._get_ids_range(attributes.OperationLogs.key, 1)[1]

    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""
        column = attributes.Concurrency.Counter
        row = self._read_byte_row(attributes.OperationLogs.key, columns=column)
        return row[0].value if row else column.basetype(0)

    def get_compatible_timestamp(
        self, time_stamp: datetime, round_up: bool = False
    ) -> datetime:
        return utils.get_google_compatible_time_stamp(time_stamp, round_up=round_up)

    # PRIVATE METHODS
    def _create_column_families(self):
        f = self._table.column_family("0")
        f.create()
        f = self._table.column_family("1", gc_rule=MaxVersionsGCRule(1))
        f.create()
        f = self._table.column_family("2")
        f.create()
        f = self._table.column_family("3", gc_rule=MaxAgeGCRule(timedelta(days=365)))
        f.create()
        f = self._table.column_family("4")
        f.create()

    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        """Returns a range (min, max) of IDs for a given `key`."""
        column = attributes.Concurrency.Counter
        row = self._table.append_row(key)
        row.increment_cell_value(column.family_id, column.key, size)
        row = row.commit()
        high = column.deserialize(row[column.family_id][column.key][0][0])
        return high + np.uint64(1) - size, high

    def _get_root_segment_ids_range(
        self, chunk_id: basetypes.CHUNK_ID, size: int = 1, counter: int = None
    ) -> np.ndarray:
        """Return unique segment ID for the root chunk."""
        n_counters = np.uint64(2**8)
        counter = (
            np.uint64(counter % n_counters)
            if counter
            else np.uint64(np.random.randint(0, n_counters))
        )
        key = serialize_key(f"i{pad_node_id(chunk_id)}_{counter}")
        min_, max_ = self._get_ids_range(key=key, size=size)
        return np.arange(
            min_ * n_counters + counter,
            max_ * n_counters + np.uint64(1) + counter,
            n_counters,
            dtype=basetypes.SEGMENT_ID,
        )

    def _read_byte_rows(
        self,
        start_key: typing.Optional[bytes] = None,
        end_key: typing.Optional[bytes] = None,
        end_key_inclusive: bool = False,
        row_keys: typing.Optional[typing.Iterable[bytes]] = None,
        columns: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
        user_id: typing.Optional[str] = None,
    ) -> typing.Dict[
        bytes,
        typing.Union[
            typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
            typing.List[bigtable.row_data.Cell],
        ],
    ]:
        """Main function for reading a row range or non-contiguous row sets from Bigtable using
        `bytes` keys.

        Keyword Arguments:
            start_key {typing.Optional[bytes]} -- The first row to be read, ignored if `row_keys` is set.
                If None, no lower boundary is used. (default: {None})
            end_key {typing.Optional[bytes]} -- The end of the row range, ignored if `row_keys` is set.
                If None, no upper boundary is used. (default: {None})
            end_key_inclusive {bool} -- Whether or not `end_key` itself should be included in the
                request, ignored if `row_keys` is set or `end_key` is None. (default: {False})
            row_keys {typing.Optional[typing.Iterable[bytes]]} -- An `typing.Iterable` containing possibly
                non-contiguous row keys. Takes precedence over `start_key` and `end_key`.
                (default: {None})
            columns {typing.Optional[typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]]} --
                typing.Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {typing.Optional[datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {typing.Optional[datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})
            user_id {typing.Optional[str]} -- Only return cells with userID equal to this

        Returns:
            typing.Dict[bytes, typing.Union[typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
                              typing.List[bigtable.row_data.Cell]]] --
                Returns a dictionary of `byte` rows as keys. Their value will be a mapping of
                columns to a typing.List of cells (one cell per timestamp). Each cell has a `value`
                property, which returns the deserialized field, and a `timestamp` property, which
                returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the typing.List of cells will be
                attached to the row dictionary directly (skipping the column dictionary).
        """

        # Create filters: Rows
        row_set = RowSet()
        if row_keys is not None:
            row_set.row_keys = list(row_keys)
        elif start_key is not None and end_key is not None:
            row_set.add_row_range_from_keys(
                start_key=start_key,
                start_inclusive=True,
                end_key=end_key,
                end_inclusive=end_key_inclusive,
            )
        else:
            raise exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or"
                " both, a start row and an end row."
            )
        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
            user_id=user_id,
        )
        # Bigtable read with retries
        rows = self._read(row_set=row_set, row_filter=filter_)

        # Deserialize cells
        for row_key, column_dict in rows.items():
            for column, cell_entries in column_dict.items():
                for cell_entry in cell_entries:
                    cell_entry.value = column.deserialize(cell_entry.value)
            # If no column array was requested, reattach single column's values directly to the row
            if isinstance(columns, attributes._Attribute):
                rows[row_key] = cell_entries
        return rows

    def _read_byte_row(
        self,
        row_key: bytes,
        columns: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ) -> typing.Union[
        typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
        typing.List[bigtable.row_data.Cell],
    ]:
        """Convenience function for reading a single row from Bigtable using its `bytes` keys.

        Arguments:
            row_key {bytes} -- The row to be read.

        Keyword Arguments:
            columns {typing.Optional[typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]]} --
                typing.Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {typing.Optional[datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {typing.Optional[datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})

        Returns:
            typing.Union[typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
                  typing.List[bigtable.row_data.Cell]] --
                Returns a mapping of columns to a typing.List of cells (one cell per timestamp). Each cell
                has a `value` property, which returns the deserialized field, and a `timestamp`
                property, which returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the typing.List of cells is returned
                directly.
        """
        row = self._read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return (
            row.get(row_key, [])
            if isinstance(columns, attributes._Attribute)
            else row.get(row_key, {})
        )

    def _execute_read_thread(self, args: typing.Tuple[Table, RowSet, RowFilter]):
        table, row_set, row_filter = args
        if not row_set.row_keys and not row_set.row_ranges:
            # Check for everything falsy, because Bigtable considers even empty
            # lists of row_keys as no upper/lower bound!
            return {}
        range_read = table.read_rows(row_set=row_set, filter_=row_filter)
        res = {v.row_key: utils.partial_row_data_to_column_dict(v) for v in range_read}
        return res

    def _read(
        self, row_set: RowSet, row_filter: RowFilter = None
    ) -> typing.Dict[bytes, typing.Dict[attributes._Attribute, PartialRowData]]:
        """Core function to read rows from Bigtable. Uses standard Bigtable retry logic
        :param row_set: BigTable RowSet
        :param row_filter: BigTable RowFilter
        :return: typing.Dict[bytes, typing.Dict[attributes._Attribute, bigtable.row_data.PartialRowData]]
        """
        # FIXME: Bigtable limits the length of the serialized request to 512 KiB. We should
        # calculate this properly (range_read.request.SerializeToString()), but this estimate is
        # good enough for now

        n_subrequests = max(
            1, int(np.ceil(len(row_set.row_keys) / self._max_row_key_count))
        )
        n_threads = min(n_subrequests, 2 * mu.n_cpus)

        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[
                i * self._max_row_key_count : (i + 1) * self._max_row_key_count
            ]
            row_sets.append(r)

        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges
        responses = mu.multithread_func(
            self._execute_read_thread,
            params=((self._table, r, row_filter) for r in row_sets),
            debug=n_threads == 1,
            n_threads=n_threads,
        )

        combined_response = {}
        for resp in responses:
            combined_response.update(resp)
        return combined_response
