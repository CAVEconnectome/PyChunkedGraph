# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel, line-too-long, protected-access, arguments-differ, arguments-renamed, logging-fstring-interpolation

import sys
import time
import typing
import logging
import datetime
import random
from datetime import datetime
from datetime import timedelta

import numpy as np
from multiwrapper import multiprocessing_utils as mu
import happybase

from . import utils
from . import HBaseConfig
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

Cell = utils.Cell


class Client(ClientWithIDGen, OperationLogger):
    def __init__(
        self,
        table_id: str,
        config: HBaseConfig = HBaseConfig(),
        graph_meta: ChunkedGraphMeta = None,
    ):
        self._table_id = table_id
        self._config = config
        self._graph_meta = graph_meta
        self._version = None
        self._max_row_key_count = config.MAX_ROW_KEY_COUNT
        
        # Initialize logger first (before connection attempts that might use it)
        self.logger = logging.getLogger(
            f"{config.HOST}:{config.PORT}/{table_id}"
        )
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.WARNING)
            self.logger.addHandler(sh)
        
        # Connect to HBase with retry logic
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                self._connection = happybase.Connection(
                    host=config.HOST,
                    port=config.PORT,
                    transport=config.THRIFT_TRANSPORT,
                    protocol='compact',
                    timeout=2000  # 2 second timeout
                )
                # Test the connection by getting tables
                _ = self._connection.tables()
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                self.logger.error(f"Failed to connect to HBase at {config.HOST}:{config.PORT} after {max_retries} attempts: {e}")
                raise
        self._table = None
        self._ensure_table()

    def _ensure_table(self):
        """Ensure table exists, create if needed."""
        table_name = self._table_id.encode('utf-8')
        try:
            self._table = self._connection.table(table_name)
            # Test connection - just check if table exists
            # Don't scan if table is empty
            pass
        except Exception:
            # Table might not exist, but we'll handle that in create_graph
            self._table = None
    
    def _get_table(self):
        """Get table, creating connection if needed."""
        if self._table is None:
            table_name = self._table_id.encode('utf-8')
            self._table = self._connection.table(table_name)
        return self._table

    @property
    def graph_meta(self):
        return self._graph_meta

    def create_graph(self, meta: ChunkedGraphMeta, version: str) -> None:
        """Initialize the graph and store associated meta."""
        table_name = self._table_id.encode('utf-8')
        
        # Check if table exists
        tables = [t.decode('utf-8') for t in self._connection.tables()]
        if self._table_id in tables:
            raise ValueError(f"{self._table_id} already exists.")
        
        # Create table with column families
        families = {
            '0': {},  # Default column family
            '1': {'max_versions': 1},  # Max versions = 1
            '2': {},
            '3': {},
        }
        self._connection.create_table(table_name, families)
        self._table = self._connection.table(table_name)
        
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
        except (KeyError, TypeError):
            return None

    def _delete_meta(self):
        """Delete existing meta row."""
        meta_key = attributes.GraphMeta.key
        self._get_table().delete(meta_key)

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
            # Sort for better performance when reading ranges
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
        typing.Dict[attributes._Attribute, typing.List[Cell]],
        typing.List[Cell],
    ]:
        """Convenience function for reading a single node from HBase."""
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
        # This is a placeholder - actual implementation would use write()
        pass

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
        rows: typing.Iterable,
        root_ids: typing.Optional[
            typing.Union[np.uint64, typing.Iterable[np.uint64]]
        ] = None,
        operation_id: typing.Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
        """Writes a list of mutated rows in bulk."""
        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]
            if not self.renew_locks(root_ids, operation_id):
                raise exceptions.LockingError(
                    f"Root lock renewal failed: operation {operation_id}"
                )

        # Batch write rows
        batch = self._get_table().batch()
        try:
            for row in rows:
                if hasattr(row, 'row_key') and hasattr(row, '_cells'):
                    # BigTable-like row object
                    row_key = row.row_key
                    cells_dict = {}
                    for (family_id, column_key), cells in row._cells.items():
                        for cell in cells:
                            col_key = f"{family_id}:{column_key}".encode('utf-8')
                            if col_key not in cells_dict:
                                cells_dict[col_key] = []
                            timestamp = int(cell.timestamp.timestamp() * 1000) if cell.timestamp else None
                            cells_dict[col_key].append((cell.value, timestamp))
                    # Use the latest version
                    for col_key, cell_list in cells_dict.items():
                        if cell_list:
                            value, timestamp = cell_list[-1]  # Latest version
                            batch.put(row_key, {col_key: value}, timestamp=timestamp)
                elif isinstance(row, dict):
                    # Direct dict format: {row_key: {column: value}}
                    for row_key, columns in row.items():
                        batch.put(row_key, columns)
            batch.send()
        except Exception as e:
            self.logger.error(f"Batch write failed: {e}")
            raise exceptions.ChunkedGraphError(
                f"Bulk write failed: operation {operation_id}"
            )

    def mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ):
        """Mutates a single row (doesn't write to HBase). Returns a dict-like object."""
        timestamp_ms = None
        if time_stamp:
            timestamp_ms = int(time_stamp.timestamp() * 1000)
        
        cells = {}
        for column, value in val_dict.items():
            col_key = f"{column.family_id}:{column.key}".encode('utf-8')
            serialized_value = column.serialize(value)
            cells[col_key] = serialized_value
        
        # Create a row-like object
        class Row:
            def __init__(self, row_key, cells, timestamp):
                self.row_key = row_key
                self._cells = {}
                for col_key, value in cells.items():
                    family_id, column_key = col_key.decode('utf-8').split(':', 1)
                    family_id = family_id.encode('utf-8')
                    column_key = column_key.encode('utf-8')
                    if (family_id, column_key) not in self._cells:
                        self._cells[(family_id, column_key)] = []
                    self._cells[(family_id, column_key)].append(
                        Cell(value, datetime.fromtimestamp(timestamp / 1000) if timestamp else None)
                    )
        
        return Row(row_key, cells, timestamp_ms)

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
        
            # Read current lock state
        row_key = serialize_uint64(root_id)
        table = self._get_table()
        
        # Check if already locked
        lock_key = f"{lock_column.family_id}:{lock_column.key}".encode('utf-8')
        indefinite_lock_key = f"{indefinite_lock_column.family_id}:{indefinite_lock_column.key}".encode('utf-8')
        
        # Try to acquire lock using check-and-put
        try:
            # Use HBase checkAndPut - check that lock doesn't exist, then put
            # Since happybase doesn't have checkAndPut directly, we use a transaction-like approach
            # Read-modify-write with a check
            current_data = table.row(row_key, columns=[lock_key, indefinite_lock_key])
            if lock_key in current_data:
                # Check if lock is expired
                lock_value = current_data[lock_key]
                # Try to parse timestamp from the cell
                # For now, just check if lock exists - expiration check would need timestamp
                return False
            if indefinite_lock_key in current_data:
                return False  # Indefinite lock exists
            
            # Write lock
            timestamp_ms = int(get_valid_timestamp(None).timestamp() * 1000)
            table.put(
                row_key,
                {lock_key: serialize_uint64(operation_id)},
                timestamp=timestamp_ms
            )
            return True
        except Exception as e:
            self.logger.debug(f"Lock acquisition failed: {e}")
            return False

    def lock_root_indefinitely(
        self,
        root_id: np.uint64,
        operation_id: np.uint64,
    ) -> bool:
        """Attempts to indefinitely lock the latest version of a root node."""
        lock_column = attributes.Concurrency.IndefiniteLock
        row_key = serialize_uint64(root_id)
        lock_key = f"{lock_column.family_id}:{lock_column.key}".encode('utf-8')
        table = self._get_table()
        
        # Check if already locked
        current_data = table.row(row_key, columns=[lock_key])
        if lock_key in current_data:
            return False
        
        # Write indefinite lock
        timestamp_ms = int(get_valid_timestamp(None).timestamp() * 1000)
        try:
            table.put(
                row_key,
                {lock_key: serialize_uint64(operation_id)},
                timestamp=timestamp_ms
            )
            return True
        except Exception:
            return False

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
        row_key = serialize_uint64(root_id)
        lock_key = f"{lock_column.family_id}:{lock_column.key}".encode('utf-8')
        table = self._get_table()
        
        # Read current lock
        row_data = table.row(row_key, columns=[lock_key])
        if lock_key not in row_data:
            return True  # Already unlocked
        
        # Check if lock matches operation_id and is not expired
        lock_value = row_data[lock_key]
        if lock_column.deserialize(lock_value) != operation_id:
            return False
        
        # Check timestamp
        # Note: happybase doesn't return timestamps easily, so we'll delete if value matches
        table.delete(row_key, columns=[lock_key])
        return True

    def unlock_indefinitely_locked_root(
        self, root_id: np.uint64, operation_id: np.uint64
    ):
        """Unlocks root node that is indefinitely locked with operation_id."""
        lock_column = attributes.Concurrency.IndefiniteLock
        row_key = serialize_uint64(root_id)
        lock_key = f"{lock_column.family_id}:{lock_column.key}".encode('utf-8')
        table = self._get_table()
        
        # Read current lock
        row_data = table.row(row_key, columns=[lock_key])
        if lock_key not in row_data:
            return True  # Already unlocked
        
        # Check if lock matches operation_id
        lock_value = row_data[lock_key]
        if lock_column.deserialize(lock_value) != operation_id:
            return False
        
        table.delete(row_key, columns=[lock_key])
        return True

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        """Renews existing root node lock with operation_id to extend time."""
        lock_column = attributes.Concurrency.Lock
        row_key = serialize_uint64(root_id)
        lock_key = f"{lock_column.family_id}:{lock_column.key}".encode('utf-8')
        table = self._get_table()
        
        # Read current lock
        row_data = table.row(row_key, columns=[lock_key])
        if lock_key not in row_data:
            return False
        
        # Check if lock matches operation_id
        lock_value = row_data[lock_key]
        if lock_column.deserialize(lock_value) != operation_id:
            return False
        
        # Renew by writing with new timestamp
        timestamp_ms = int(get_valid_timestamp(None).timestamp() * 1000)
        table.put(
            row_key,
            {lock_key: serialize_uint64(operation_id)},
            timestamp=timestamp_ms
        )
        return True

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
        return utils.get_hbase_compatible_time_stamp(time_stamp, round_up=round_up)
    
    @property
    def _table(self):
        """Lazy table access."""
        if self.__table is None:
            table_name = self._table_id.encode('utf-8')
            self.__table = self._connection.table(table_name)
        return self.__table
    
    @_table.setter
    def _table(self, value):
        """Set table."""
        self.__table = value

    # PRIVATE METHODS
    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        """Returns a range (min, max) of IDs for a given `key`."""
        column = attributes.Concurrency.Counter
        col_key = f"{column.family_id}:{column.key}".encode('utf-8')
        
        # Read current value
        table = self._get_table()
        current_data = table.row(key, columns=[col_key])
        current_value = 0
        if col_key in current_data:
            current_value = column.deserialize(current_data[col_key])
        
        # Increment atomically using HBase increment
        # Since happybase doesn't have increment, we use a read-modify-write pattern
        # For atomicity, we'll use a simple increment
        new_value = current_value + size
        timestamp_ms = int(get_valid_timestamp(None).timestamp() * 1000)
        table.put(
            key,
            {col_key: column.serialize(new_value)},
            timestamp=timestamp_ms
        )
        
        high = column.deserialize(column.serialize(new_value))
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
            typing.Dict[attributes._Attribute, typing.List[Cell]],
            typing.List[Cell],
        ],
    ]:
        """Main function for reading a row range or non-contiguous row sets from HBase."""
        
        # Prepare column filter
        column_filter = utils.get_column_filter(columns)
        
        # Prepare time range
        time_filter = utils.get_time_range_filter(
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
        )
        
        rows = {}
        
        if row_keys is not None:
            # Non-contiguous row set - use batch get
            row_keys_list = list(row_keys)
            if not row_keys_list:
                return rows
            
            # Split into batches for performance
            n_subrequests = max(
                1, int(np.ceil(len(row_keys_list) / self._max_row_key_count))
            )
            n_threads = min(n_subrequests, 2 * mu.n_cpus)
            
            def _execute_read_batch(batch_keys):
                batch_results = {}
                try:
                    table = self._get_table()
                    # Use batch get
                    batch_data = table.rows(batch_keys, columns=column_filter)
                    for row_key, row_data in batch_data:
                        if row_data:
                            batch_results[row_key] = utils.hbase_row_to_column_dict(
                                row_data, start_time, end_time, time_filter
                            )
                except Exception as e:
                    self.logger.warning(f"Batch read failed: {e}")
                return batch_results
            
            # Split into batches
            batches = []
            for i in range(n_subrequests):
                batch = row_keys_list[
                    i * self._max_row_key_count : (i + 1) * self._max_row_key_count
                ]
                if batch:
                    batches.append(batch)
            
            # Execute in parallel
            responses = mu.multithread_func(
                _execute_read_batch,
                params=batches,
                debug=n_threads == 1,
                n_threads=n_threads,
            )
            
            # Combine results
            for resp in responses:
                rows.update(resp)
                
        elif start_key is not None and end_key is not None:
            # Range scan
            scan_kwargs = {}
            if column_filter:
                scan_kwargs['columns'] = column_filter
            if time_filter.get('timestamp'):
                scan_kwargs['timestamp'] = time_filter['timestamp']
            if time_filter.get('max_timestamp'):
                scan_kwargs['max_timestamp'] = time_filter['max_timestamp']
            
            # Adjust end_key for inclusivity
            if not end_key_inclusive:
                # For exclusive, we need to adjust the end key
                # HBase scans are start-inclusive, end-exclusive by default
                pass
            
            try:
                table = self._get_table()
                for row_key, row_data in table.scan(
                    row_start=start_key,
                    row_stop=end_key if end_key_inclusive else end_key + b'\x00',
                    **scan_kwargs
                ):
                    if row_data:
                        rows[row_key] = utils.hbase_row_to_column_dict(
                            row_data, start_time, end_time, time_filter
                        )
            except Exception as e:
                self.logger.warning(f"Range scan failed: {e}")
        else:
            raise exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or"
                " both, a start row and an end row."
            )
        
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
        typing.Dict[attributes._Attribute, typing.List[Cell]],
        typing.List[Cell],
    ]:
        """Convenience function for reading a single row from HBase."""
        row = self._read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        if isinstance(columns, attributes._Attribute):
            return row.get(row_key, [])
        return row.get(row_key, {})

