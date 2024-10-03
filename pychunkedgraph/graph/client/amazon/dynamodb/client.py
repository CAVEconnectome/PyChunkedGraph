import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Union, Optional, List, Any, Tuple, Sequence

import boto3
import botocore
import numpy as np
from boto3.dynamodb.types import TypeSerializer, Binary, TypeDeserializer
from botocore.exceptions import ClientError
from multiwrapper import multiprocessing_utils as mu

from . import AmazonDynamoDbConfig
from . import utils
from .ddb_table import Table
from .ddb_translator import DdbTranslator, to_column_name, to_lock_timestamp_column_name
from .item_compressor import ItemCompressor
from .row_set import RowSet
from .timestamped_cell import TimeStampedCell
from .utils import (
    DynamoDbFilter, append, get_current_time_microseconds, to_microseconds, remove_and_merge_duplicates,
)
from ...base import ClientWithIDGen
from ...base import OperationLogger
from .... import attributes
from .... import exceptions
from ....meta import ChunkedGraphMeta
from ....utils import basetypes
from ....utils.serializers import pad_node_id, serialize_key, serialize_uint64, deserialize_uint64

MAX_BATCH_READ_ITEMS = 100  # Max items to fetch using GetBatchItem operation
MAX_BATCH_WRITE_ITEMS = 25  # Max items to write using BatchWriteItem operation
MAX_QUERY_ITEMS = 1000  # Maximum items to fetch in one query


class Client(ClientWithIDGen, OperationLogger):
    def __init__(
        self,
        table_id: str = None,
        config: AmazonDynamoDbConfig = AmazonDynamoDbConfig(),
        graph_meta: ChunkedGraphMeta = None,
    ):
        self._table_name = (
            ".".join([config.TABLE_PREFIX, table_id])
            if config.TABLE_PREFIX
            else table_id
        )
        
        self._max_batch_read_page_size = MAX_BATCH_READ_ITEMS
        self._max_batch_write_page_size = MAX_BATCH_WRITE_ITEMS
        self._max_query_page_size = MAX_QUERY_ITEMS
        
        self._graph_meta = graph_meta
        self._version = None
        
        boto3_conf_ = botocore.config.Config(
            retries={"max_attempts": 10, "mode": "standard"}
        )
        kwargs = {}
        if config.REGION:
            kwargs["region_name"] = config.REGION
        if config.END_POINT:
            kwargs["endpoint_url"] = config.END_POINT
        self._main_db = boto3.client("dynamodb", config=boto3_conf_, **kwargs)
        
        self._ddb_serializer = TypeSerializer()
        
        self._ddb_translator = DdbTranslator()
        
        # Storing items in DynamoDB table by compressing all columns into one column named "v"
        # Certain columns which are either used in conditional checks (such as lock columns) or used for metadata
        # are not compressed and stored as is at the top level
        # The list below denotes such columns which should be excluded from compressing into "v"
        self._uncompressed_columns = [
            to_column_name(attributes.Concurrency.Lock),
            to_lock_timestamp_column_name(attributes.Concurrency.Lock),
            to_column_name(attributes.Hierarchy.NewParent),
            to_column_name(attributes.Concurrency.Counter),
            to_column_name(attributes.Concurrency.IndefiniteLock),
            to_lock_timestamp_column_name(attributes.Concurrency.IndefiniteLock),
            attributes.GraphVersion.Version.key,
            attributes.GraphMeta.Meta.key,
            attributes.OperationLogs.key,
        ]
        self._ddb_item_compressor = ItemCompressor(
            pk_name='key',
            sk_name='sk',
            exclude_keys=self._uncompressed_columns
        )
        
        # The "self._table" below is only used by the test code for inspecting the items written to the DB
        # and is not used by the actual code. The actual code uses the underlying "_ddb_table" instead.
        self._table = Table(
            self._main_db,
            self._table_name,
            translator=self._ddb_translator,
            compressor=self._ddb_item_compressor,
            boto3_conf=boto3_conf_,
            **kwargs
        )
        self._ddb_table = self._table.ddb_table
        self._ddb_deserializer = TypeDeserializer()
        
        # TODO: Remove _no_of_reads and _no_of_writes variables. These are added for debugging purposes only.
        self._no_of_reads = 0
        self._no_of_writes = 0
    
    """Initialize the graph and store associated meta."""
    
    def create_graph(self, meta: ChunkedGraphMeta, version: str) -> None:
        """Initialize the graph and store associated meta."""
        existing_version = self.read_graph_version()
        if not existing_version:
            self.add_graph_version(version)
        
        self.update_graph_meta(meta)
    
    """Add a version to the graph."""
    
    def add_graph_version(self, version):
        assert self.read_graph_version() is None, "Graph has already been versioned."
        self._version = version
        row = self.mutate_row(
            attributes.GraphVersion.key,
            {attributes.GraphVersion.Version: version},
        )
        self.write([row])
    
    """Read stored graph version."""
    
    def read_graph_version(self):
        row = self._read_byte_row(attributes.GraphVersion.key)
        cells = row.get(attributes.GraphVersion.Version, [])
        self._version = None
        if len(cells) > 0:
            self._version = cells[0].value
        return self._version
    
    """Update stored graph meta."""
    
    def update_graph_meta(
        self, meta: ChunkedGraphMeta, overwrite: Optional[bool] = False
    ):
        do_write = True
        
        if not overwrite:
            existing_meta = self.read_graph_meta()
            do_write = not existing_meta
        
        if do_write:
            self._graph_meta = meta
            row = self.mutate_row(
                attributes.GraphMeta.key,
                {attributes.GraphMeta.Meta: meta},
            )
            self.write([row])
    
    """Read stored graph meta."""
    
    def read_graph_meta(self):
        logging.debug("read_graph_meta")
        row = self._read_byte_row(attributes.GraphMeta.key)
        cells = row.get(attributes.GraphMeta.Meta, [])
        self._graph_meta = None
        if len(cells) > 0:
            self._graph_meta = cells[0].value
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
        logging.debug(
            f"read_nodes: {start_id}, {end_id}, {node_ids}, {properties}, {start_time}, {end_time}, {end_time_inclusive}"
        )
        
        if node_ids is not None and len(node_ids) > 0:
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
    
    """Read a single node and its properties."""
    
    def read_node(
        self,
        node_id: np.uint64,
        properties: Optional[
            Union[Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
    ) -> Union[
        Dict[attributes._Attribute, List[TimeStampedCell]],
        List[TimeStampedCell],
    ]:
        """Convenience function for reading a single node from Amazon DynamoDB.
        Arguments:
            node_id {np.uint64} -- the NodeID of the row to be read.
        Keyword Arguments:
            columns {Optional[Union[Iterable[attributes._Attribute], attributes._Attribute]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single
                column (not iterable), the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether or not `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})
        Returns:
            Union[Dict[attributes._Attribute, List[TimeStampedCell]],
                  List[TimeStampedCell]] --
                Returns a mapping of columns to a List of cells (one cell per timestamp). Each cell
                has a `value` property, which returns the deserialized field, and a `timestamp`
                property, which returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the List of cells is returned
                directly.
        """
        return self._read_byte_row(
            row_key=serialize_uint64(node_id, fake_edges=fake_edges),
            columns=properties,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
    
    """Writes/updates nodes (IDs along with properties)."""
    
    def write_nodes(self, nodes):
        logging.debug(f"write_nodes: {nodes}")
        raise NotImplementedError("write_nodes - Not yet implemented")
    
    # Helpers
    def write(
        self,
        rows: Iterable[Dict[str, Union[bytes, Dict[str, Iterable[TimeStampedCell]]]]],
        root_ids: Optional[
            Union[np.uint64, Iterable[np.uint64]]
        ] = None,
        operation_id: Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 25,
    ):
        """Writes a list of mutated rows in bulk
        WARNING: If <rows> contains the same row (same row_key) and column
        key two times only the last one is effectively written (even when the mutations were applied to
        different columns) --> no versioning!
        :param rows: list
            list of mutated rows
        :param root_ids: list of uint64
        :param operation_id: uint64 or None
            operation_id (or other unique id) that *was* used to lock the root
            the bulk write is only executed if the root is still locked with
            the same id.
        :param slow_retry: bool
        :param block_size: int
        """
        logging.debug(f"write {rows} {root_ids} {operation_id} {slow_retry} {block_size}")
        
        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]
            if not self.renew_locks(root_ids, operation_id):
                raise exceptions.LockingError(
                    f"Root lock renewal failed: operation {operation_id}"
                )
        
        # TODO: Implement retries with backoff and handle partial batch failures
        
        batch_size = min(self._max_batch_write_page_size, block_size)
        
        # There may be multiple rows with the same row key but with different columns
        # Merge such rows to avoid duplicates and write multiple columns when writing the row to DDB
        deduplicated_rows = remove_and_merge_duplicates(rows)
        
        for i in range(0, len(deduplicated_rows), batch_size):
            with self._ddb_table.batch_writer() as batch:
                self._no_of_writes += 1
                rows_in_this_batch = deduplicated_rows[i: i + batch_size]
                for row in rows_in_this_batch:
                    ddb_item = self._ddb_translator.row_to_ddb_item(row)
                    ddb_item = self._ddb_item_compressor.compress(ddb_item)
                    batch.put_item(Item=ddb_item)
    
    def mutate_row(
        self,
        row_key: bytes,
        val_dict: Dict[attributes._Attribute, Any],
        time_stamp: Optional[datetime] = None,
    ) -> Dict[str, Union[bytes, Dict[str, Iterable[TimeStampedCell]]]]:
        """Mutates a single row (doesn't write to DynamoDB)."""
        pk, sk = self._ddb_translator.to_pk_sk(row_key)
        self._no_of_reads += 1
        ret = self._ddb_table.get_item(Key={"key": pk, "sk": sk})
        item = ret.get('Item')
        row = {"key": row_key}
        if item is not None:
            item = self._ddb_item_compressor.decompress(item)
            b_real_key, row_from_db = self._ddb_translator.ddb_item_to_row(item)
            row.update(row_from_db)
        
        cells = self._ddb_translator.attribs_to_cells(attribs=val_dict, time_stamp=time_stamp)
        row.update(cells)
        
        return row
    
    def lock_root(
        self,
        root_id: np.uint64,
        operation_id: np.uint64,
    ) -> bool:
        """Locks root node with operation_id to prevent race conditions."""
        logging.debug(f"lock_root: {root_id}, {operation_id}")
        time_cutoff = self._get_lock_expiry_time_cutoff()
        
        pk, sk = self._ddb_translator.to_pk_sk(serialize_uint64(root_id))
        
        lock_column = attributes.Concurrency.Lock
        indefinite_lock_column = attributes.Concurrency.IndefiniteLock
        new_parents_column = attributes.Hierarchy.NewParent
        
        lock_column_name_in_ddb = to_column_name(lock_column)
        lock_timestamp_column_name_in_ddb = to_lock_timestamp_column_name(lock_column)
        
        indefinite_lock_column_name_in_ddb = to_column_name(indefinite_lock_column)
        
        new_parents_column_name_in_ddb = to_column_name(new_parents_column)
        
        # Add the given operation_id in the lock column ONLY IF the lock column is not already set or
        # if the lock column is set but the lock is expired
        # and if there is NO new parent (i.e., the new_parents column is not set).
        try:
            self._no_of_writes += 1
            self._ddb_table.update_item(
                Key={"key": pk, "sk": sk},
                UpdateExpression="SET #c = :c, #lock_timestamp = :current_time",
                ConditionExpression=f"(attribute_not_exists(#c) OR #lock_timestamp < :time_cutoff)"
                                    f" AND attribute_not_exists(#c_indefinite_lock)"
                                    f" AND attribute_not_exists(#new_parents)",
                ExpressionAttributeNames={
                    "#c": lock_column_name_in_ddb,
                    "#lock_timestamp": lock_timestamp_column_name_in_ddb,
                    "#c_indefinite_lock": indefinite_lock_column_name_in_ddb,
                    "#new_parents": new_parents_column_name_in_ddb,
                },
                ExpressionAttributeValues={
                    ':c': serialize_uint64(operation_id),
                    ':time_cutoff': time_cutoff,
                    ':current_time': get_current_time_microseconds(),
                }
            )
            
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logging.debug(f"lock_root: {root_id}, {operation_id} failed")
                return False
            else:
                raise e
    
    def lock_roots(
        self,
        root_ids: Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: Dict,
        max_tries: int = 1,
        waittime_s: float = 0.5,
    ) -> Tuple[bool, Iterable]:
        """Attempts to lock multiple nodes with same operation id"""
        i_try = 0
        while i_try < max_tries:
            lock_acquired = False
            # Collect latest root ids
            new_root_ids: List[np.uint64] = []
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
            logging.debug(f"Try {i_try}")
        return False, root_ids
    
    def lock_root_indefinitely(
        self,
        root_id: np.uint64,
        operation_id: np.uint64,
    ) -> bool:
        """Attempts to indefinitely lock the latest version of a root node."""
        logging.debug(f"lock_root_indefinitely: {root_id}, {operation_id}")
        
        pk, sk = self._ddb_translator.to_pk_sk(serialize_uint64(root_id))
        
        lock_column = attributes.Concurrency.IndefiniteLock
        lock_column_name_in_ddb = to_column_name(lock_column)
        lock_timestamp_column_name_in_ddb = to_lock_timestamp_column_name(lock_column)
        
        new_parents_column_name_in_ddb = to_column_name(attributes.Hierarchy.NewParent)
        
        # Add the given operation_id in the indefinite lock column ONLY IF the indefinite column is not already set
        # and if there is NO new parent (i.e., the new_parents column is not set).
        try:
            self._no_of_writes += 1
            self._ddb_table.update_item(
                Key={"key": pk, "sk": sk},
                UpdateExpression="SET #c = :c, #lock_timestamp = :current_time",
                ConditionExpression=f"attribute_not_exists(#c)"
                                    f" AND attribute_not_exists(#new_parents)",
                ExpressionAttributeNames={
                    "#c": lock_column_name_in_ddb,
                    "#lock_timestamp": lock_timestamp_column_name_in_ddb,
                    "#new_parents": new_parents_column_name_in_ddb,
                },
                ExpressionAttributeValues={
                    ':c': serialize_uint64(operation_id),
                    ':current_time': get_current_time_microseconds(),
                }
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logging.debug(f"lock_root: {root_id}, {operation_id} failed")
                return False
            else:
                raise e
    
    def lock_roots_indefinitely(
        self,
        root_ids: Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: Dict,
    ) -> Tuple[bool, Iterable]:
        """
        Attempts to indefinitely lock multiple nodes with same operation id to prevent structural damage to graph.
        This scenario is rare and needs asynchronous fix or inspection to unlock.
        """
        lock_acquired = False
        # Collect latest root ids
        new_root_ids: List[np.uint64] = []
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
            logging.debug(f"operation {operation_id} root_id {_id}")
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
    
    def unlock_root(self, root_id, operation_id):
        """Unlocks root node that is locked with operation_id."""
        logging.debug(f"unlock_root: {root_id}, {operation_id}")
        time_cutoff = self._get_lock_expiry_time_cutoff()
        
        pk, sk = self._ddb_translator.to_pk_sk(serialize_uint64(root_id))
        
        lock_column = attributes.Concurrency.Lock
        lock_column_name_in_ddb = to_column_name(lock_column)
        lock_timestamp_column_name_in_ddb = to_lock_timestamp_column_name(lock_column)
        
        # Delete (remove) the lock column ONLY IF the given operation_id is still the active lock holder and
        # the lock has not expired
        try:
            self._no_of_writes += 1
            self._ddb_table.update_item(
                Key={"key": pk, "sk": sk},
                UpdateExpression="REMOVE #c",
                ConditionExpression=f"(#lock_timestamp > :time_cutoff)"  # Ensure not expired
                                    f" AND #c = :c",  # Ensure operation_id is the active lock holder
                ExpressionAttributeNames={
                    "#c": lock_column_name_in_ddb,
                    "#lock_timestamp": lock_timestamp_column_name_in_ddb,
                },
                ExpressionAttributeValues={
                    ':c': serialize_uint64(operation_id),
                    ':time_cutoff': time_cutoff,
                }
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logging.debug(f"unlock_root: {root_id}, {operation_id} failed")
                return False
            else:
                raise e
    
    def unlock_indefinitely_locked_root(
        self, root_id: np.uint64, operation_id: np.uint64
    ):
        """Unlocks root node that is indefinitely locked with operation_id."""
        logging.debug(f"unlock_indefinitely_locked_root: {root_id}, {operation_id}")
        
        pk, sk = self._ddb_translator.to_pk_sk(serialize_uint64(root_id))
        
        lock_column = attributes.Concurrency.IndefiniteLock
        
        lock_column_name_in_ddb = to_column_name(lock_column)
        
        # Delete (remove) the lock column ONLY IF the given operation_id is still the active lock holder
        try:
            self._no_of_writes += 1
            self._ddb_table.update_item(
                Key={"key": pk, "sk": sk},
                UpdateExpression="REMOVE #c",
                ConditionExpression=f"#c = :c",  # Ensure operation_id is the active lock holder
                ExpressionAttributeNames={
                    "#c": lock_column_name_in_ddb,
                },
                ExpressionAttributeValues={
                    ':c': serialize_uint64(operation_id),
                }
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logging.debug(f"unlock_indefinitely_locked_root: {root_id}, {operation_id} failed")
                return False
            else:
                raise e
    
    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        """Renews existing root node lock with operation_id to extend time."""
        
        logging.debug(f"renew_lock: {root_id}, {operation_id}")
        
        pk, sk = self._ddb_translator.to_pk_sk(serialize_uint64(root_id))
        lock_column = attributes.Concurrency.Lock
        new_parents_column = attributes.Hierarchy.NewParent
        
        lock_column_name_in_ddb = to_column_name(lock_column)
        lock_timestamp_column_name_in_ddb = to_lock_timestamp_column_name(lock_column)
        
        new_parents_column_name_in_ddb = to_column_name(new_parents_column)
        
        # Update the given operation_id in the lock column and update the lock_timestamp
        # ONLY IF the given operation_id is still the current lock holder and if
        # there is NO new parent (i.e., the new_parents column is not set).
        # TODO: Do we also need to check that the lock has not expired before renewing it?
        #   Currently, the BigTable implementation does not check for expiry during renewals
        #   (See "renew_lock" method in "pychunkedgraph/graph/client/bigtable/client.py" for reference)
        #
        try:
            self._no_of_writes += 1
            self._ddb_table.update_item(
                Key={"key": pk, "sk": sk},
                UpdateExpression="SET #c = :c, #lock_timestamp = :current_time",
                ConditionExpression=f"#c = :c"  # Ensure operation_id is the active lock holder
                                    f" AND attribute_not_exists(#new_parents)",  # Ensure no new parents
                ExpressionAttributeNames={
                    "#c": lock_column_name_in_ddb,
                    "#lock_timestamp": lock_timestamp_column_name_in_ddb,
                    "#new_parents": new_parents_column_name_in_ddb,
                },
                ExpressionAttributeValues={
                    ':c': serialize_uint64(operation_id),
                    ':current_time': get_current_time_microseconds(),
                }
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logging.debug(f"renew_lock: {root_id}, {operation_id} failed")
                return False
            else:
                raise e
    
    """Renews existing node locks with operation_id for extended time."""
    
    def renew_locks(self, root_ids: Iterable[np.uint64], operation_id: np.uint64) -> bool:
        """Renews existing root node locks with operation_id to extend time."""
        for root_id in root_ids:
            if not self.renew_lock(root_id, operation_id):
                logging.warning(f"renew_lock failed - {root_id}")
                return False
        return True
    
    """Reads timestamp from lock row to get a consistent timestamp."""
    
    def get_lock_timestamp(
        self, root_id: np.uint64, operation_id: np.uint64
    ) -> Union[datetime, None]:
        logging.debug(f"get_lock_timestamp: {root_id}, {operation_id}")
        
        pk, sk = self._ddb_translator.to_pk_sk(serialize_uint64(root_id))
        
        lock_column = attributes.Concurrency.Lock
        
        lock_column_name = to_column_name(lock_column)
        lock_timestamp_column_name = to_lock_timestamp_column_name(lock_column)
        self._no_of_reads += 1
        res = self._ddb_table.get_item(
            Key={"key": pk, "sk": sk},
            ProjectionExpression='#c, #lock_timestamp',
            ConsistentRead=True,
            ExpressionAttributeNames={
                "#c": lock_column_name,
                "#lock_timestamp": lock_timestamp_column_name,
            },
        )
        item = res.get('Item', None)
        
        if item is None:
            logging.warning(f"No lock found for {root_id}")
            return None
        if operation_id != item.get(lock_column_name, None):
            logging.warning(f"{root_id} not locked with {operation_id}")
            return None
        
        return item.get(lock_timestamp_column_name, None)
    
    """Minimum of multiple lock timestamps."""
    
    def get_consolidated_lock_timestamp(
        self,
        root_ids: Sequence[np.uint64],
        operation_ids: Sequence[np.uint64],
    ) -> Union[datetime, None]:
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
    
    """Datetime time stamp compatible with client's services."""
    
    def get_compatible_timestamp(self, time_stamp):
        logging.debug(f"get_compatible_timestamp: {time_stamp}")
        raise NotImplementedError("get_compatible_timestamp - Not yet implemented")
    
    """Generate a range of unique IDs in the chunk."""
    
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
    
    """Generate a unique ID in the chunk."""
    
    def create_node_id(
        self, chunk_id: np.uint64, root_chunk=False
    ) -> basetypes.NODE_ID:
        """Generate a unique node ID in the chunk."""
        return self.create_node_ids(chunk_id, 1, root_chunk=root_chunk)[0]
    
    """Gets the current maximum node ID in the chunk."""
    
    def get_max_node_id(self, chunk_id, root_chunk=False):
        """Gets the current maximum segment ID in the chunk."""
        if root_chunk:
            n_counters = np.uint64(2 ** 8)
            max_value = 0
            for counter in range(n_counters):
                row_key = serialize_key(f"i{pad_node_id(chunk_id)}_{counter}")
                row = self._read_byte_row(
                    row_key,
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
    
    """Generate a unique operation ID."""
    
    def create_operation_id(self):
        """Generate a unique operation ID."""
        return self._get_ids_range(attributes.OperationLogs.key, 1)[1]
    
    """Gets the current maximum operation ID."""
    
    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""
        column = attributes.Concurrency.Counter
        row = self._read_byte_row(attributes.OperationLogs.key, columns=column)
        return row[0].value if row else column.basetype(0)
    
    def read_log_entry(self, operation_id: int) -> None:
        """Read log entry for a given operation ID."""
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
    
    """Read log entries for given operation IDs."""
    
    def read_log_entries(
        self,
        operation_ids: Optional[Iterable] = None,
        user_id: Optional[str] = None,
        properties: Optional[Iterable[attributes._Attribute]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
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
    
    def _read_byte_row(
        self,
        row_key: bytes,
        columns: Optional[
            Union[Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Union[
        Dict[attributes._Attribute, List[TimeStampedCell]],
        List[TimeStampedCell],
    ]:
        """Convenience function for reading a single row from Amazon DynamoDB using its `bytes` keys.

        Arguments:
            row_key {bytes} -- The row to be read.

        Keyword Arguments:
            columns {Optional[Union[Iterable[attributes._Attribute], attributes._Attribute]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single column (not iterable),
                the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})

        Returns:
            Union[Dict[attributes._Attribute, List[TimeStampedCell]],
                  List[TimeStampedCell]] --
                Returns a mapping of columns to a List of cells (one cell per timestamp). Each cell
                has a `value` property, which returns the deserialized field, and a `timestamp`
                property, which returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the List of cells is returned
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
    
    def _read_byte_rows(
        self,
        start_key: Optional[bytes] = None,
        end_key: Optional[bytes] = None,
        end_key_inclusive: bool = False,
        row_keys: Optional[Iterable[bytes]] = None,
        columns: Optional[
            Union[Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        end_time_inclusive: bool = False,
        user_id: Optional[str] = None,
    ) -> Dict[
        bytes,
        Union[
            Dict[attributes._Attribute, List[TimeStampedCell]],
            List[TimeStampedCell],
        ],
    ]:
        """Main function for reading a row range or non-contiguous row sets from Amazon DynamoDB using
        `bytes` keys.

        Keyword Arguments:
            start_key {Optional[bytes]} -- The first row to be read, ignored if `row_keys` is set.
                If None, no lower boundary is used. (default: {None})
            end_key {Optional[bytes]} -- The end of the row range, ignored if `row_keys` is set.
                If None, no upper boundary is used. (default: {None})
            end_key_inclusive {bool} -- Whether `end_key` itself should be included in the
                request, ignored if `row_keys` is set or `end_key` is None. (default: {False})
            row_keys {Optional[Iterable[bytes]]} -- An `Iterable` containing possibly
                non-contiguous row keys. Takes precedence over `start_key` and `end_key`.
                (default: {None})
            columns {Optional[Union[Iterable[attributes._Attribute], attributes._Attribute]]} --
                Optional filtering by columns to speed up the query. If `columns` is a single column (not iterable),
                the column key will be omitted from the result.
                (default: {None})
            start_time {Optional[datetime]} -- Ignore cells with timestamp before
                `start_time`. If None, no lower bound. (default: {None})
            end_time {Optional[datetime]} -- Ignore cells with timestamp after `end_time`.
                If None, no upper bound. (default: {None})
            end_time_inclusive {bool} -- Whether `end_time` itself should be included in the
                request, ignored if `end_time` is None. (default: {False})
            user_id {Optional[str]} -- Only return cells with userID equal to this

        Returns:
            Dict[bytes, Union[Dict[attributes._Attribute, List[TimeStampedCell]],
                              List[TimeStampedCell]]] --
                Returns a dictionary of `byte` rows as keys. Their value will be a mapping of
                columns to a List of cells (one cell per timestamp). Each cell has a `value`
                property, which returns the deserialized field, and a `timestamp` property, which
                returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the List of cells will be
                attached to the row dictionary directly (skipping the column dictionary).
        """
        
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
        
        rows = self._read(row_set=row_set, row_filter=filter_)
        
        # Deserialize cells
        for row_key, column_dict in rows.items():
            for column, cell_entries in column_dict.items():
                for cell_entry in cell_entries:
                    if isinstance(column, attributes._Attribute):
                        if isinstance(cell_entry.value, Binary):
                            cell_entry.value = column.deserialize(bytes(cell_entry.value))
            
            # If no column array was requested, reattach single column's values directly to the row
            if isinstance(columns, attributes._Attribute):
                rows[row_key] = column_dict[columns]
        
        return rows
    
    def _read(self, row_set: RowSet, row_filter: DynamoDbFilter = None) -> dict:
        """Core function to read rows from DynamoDB.
        :param row_set: Set of related to the rows to be read
        :param row_filter: An instance of DynamoDbFilter to filter which rows/columns to read
        :return: Dict
        """
        from pychunkedgraph.logging.log_db import TimeIt
        
        n_subrequests = max(
            1, int(np.ceil(len(row_set.row_keys) / self._max_batch_read_page_size))
        )
        n_threads = min(n_subrequests, 2 * mu.n_cpus)
        
        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[i * self._max_batch_read_page_size: (i + 1) * self._max_batch_read_page_size]
            row_sets.append(r)
        
        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges
        
        with TimeIt(
            "chunked_reads",
            f"{self._table_name}_ddb_profile",
            operation_id=-1,
            n_rows=len(row_set.row_keys),
            n_requests=n_subrequests,
            n_threads=n_threads,
        ):
            responses = mu.multithread_func(
                self._execute_read_thread,
                params=((r, row_filter) for r in row_sets),
                debug=n_threads == 1,
                n_threads=n_threads,
            )
            
            combined_response = {}
            for resp in responses:
                combined_response.update(resp)
            return combined_response
    
    def _execute_read_thread(self, args: Tuple[RowSet, DynamoDbFilter]):
        """Function to be executed in parallel."""
        row_set, row_filter = args
        if not row_set.row_keys and not row_set.row_ranges:
            return {}
        
        row_keys = np.unique(row_set.row_keys)
        
        rows = {}
        item_keys_to_get = []
        attr_names = {'#key': 'key'}
        kwargs = {
        }
        
        def __append_to_projection_expression(
            dict_obj: dict,
            attribs_to_get
        ):
            existing_expr = dict_obj.get("ProjectionExpression", "")
            attribs_expr = ",".join(attribs_to_get)
            if existing_expr and attribs_expr:
                dict_obj["ProjectionExpression"] = f"{existing_expr},{attribs_expr}"
            elif attribs_expr:
                dict_obj["ProjectionExpression"] = attribs_expr
        
        # User ID filter
        if row_filter.user_id_filter and row_filter.user_id_filter.user_id:
            # Project #uid and v both attribs - if the item is compressed then the uid will be part of the "v" column
            # else it will be part of the #uid column (i.e., the attributes.OperationLogs.UserID column)
            __append_to_projection_expression(kwargs, ["#key", "sk", "#ver", "#uid", "v"])
            user_id_attr = attributes.OperationLogs.UserID
            attr_names["#uid"] = to_column_name(user_id_attr)
            attr_names["#ver"] = "@"
            kwargs["ExpressionAttributeNames"] = attr_names
        
        # Column filter
        if row_filter.column_filter:
            ddb_columns = [
                f"#C{index}" for index in range(len(row_filter.column_filter))
            ]
            # Project the specified columns along with "v" column
            # if the item is compressed then the specified columns will be part of the "v" column else
            # they will be part of the specified columns
            ddb_columns.extend(["#key", "sk", "#ver", "v"])
            __append_to_projection_expression(kwargs, ddb_columns)
            
            for index, attr in enumerate(row_filter.column_filter):
                attr_names[f"#C{index}"] = f"{attr.family_id}.{attr.key.decode()}"
            
            attr_names["#ver"] = "@"
            kwargs["ExpressionAttributeNames"] = attr_names
        
        # TODO: "new" data for existing key is appended to the map, this needs to be revisited since it can
        #  potentially exceed the limit for the item size (400KB).
        #  Currently it is as is in the BigTable implementation
        for key in row_keys:
            pk, sk = self._ddb_translator.to_pk_sk(key)
            item_keys_to_get.append({
                # "batch_get_item" is not available on the boto3 DynamoDB resource abstraction (i.e., "self._ddb_table")
                # so we are forced to use low-level boto3 client (i.e.,  "self._main_db")
                # The low-level boto3 client does not handle serialization/deserialization automatically, so have to
                # do it manually using the "self._ddb_serializer" here
                'key': self._ddb_serializer.serialize(pk),
                'sk': self._ddb_serializer.serialize(sk),
            })
        
        if len(item_keys_to_get) > 0:
            # TODO: Handle partial batch retrieval failures
            params = {
                self._table_name: {
                    'Keys': item_keys_to_get,
                    **kwargs,
                },
            }
            
            self._no_of_reads += 1
            ret = self._main_db.batch_get_item(RequestItems=params)
            
            items = ret.get("Responses", {}).get(self._table_name, [])
            
            # each item comes with 'key', 'sk', [column_family] and '@' columns
            for index, item in enumerate(items):
                # The item is not deserialized automatically when using the low-level boto3 client
                # (i.e.,  "self._main_db"), so deserialize first
                item = self._deserialize(item)
                item = self._ddb_item_compressor.decompress(item)
                b_real_key, row = self._ddb_translator.ddb_item_to_row(
                    item={
                        'key': item_keys_to_get[index]['key'],
                        'sk': item_keys_to_get[index]['sk'],
                        **item,
                    },
                )
                rows[b_real_key] = row
        
        if len(row_set.row_ranges) > 0:
            expression_attrib_names = kwargs.get('ExpressionAttributeNames', {})
            expression_attrib_names['#key'] = 'key'
            kwargs['ExpressionAttributeNames'] = expression_attrib_names
            
            for row_range in row_set.row_ranges:
                pk, start_sk, end_sk = self._ddb_translator.to_sk_range(
                    row_range.start_key,
                    row_range.end_key,
                    row_range.start_inclusive,
                    row_range.end_inclusive,
                )
                
                attr_vals = {
                    ":key": pk,
                    ":st_sk": start_sk,
                    ":end_sk": end_sk,
                }
                
                query_kwargs = {
                    "Limit": self._max_query_page_size,
                    "KeyConditionExpression": "#key = :key AND sk BETWEEN :st_sk AND :end_sk",
                    "ExpressionAttributeValues": attr_vals,
                    **kwargs,
                }
                self._no_of_reads += 1
                ret = self._ddb_table.query(**query_kwargs)
                items = ret.get("Items", [])
                
                for item in items:
                    item = self._ddb_item_compressor.decompress(item)
                    b_real_key, row = self._ddb_translator.ddb_item_to_row(item)
                    rows[b_real_key] = row
        
        filtered_rows = self._apply_filters(rows, row_filter)
        
        return filtered_rows
    
    def _apply_filters(
        self,
        rows: Dict[str, Dict[attributes._Attribute, Iterable[TimeStampedCell]]],
        row_filter: DynamoDbFilter
    ):
        # the start_datetime and the end_datetime below are "datetime" instances (and NOT int timestamp)
        start_datetime = row_filter.time_filter.start if row_filter.time_filter else None
        end_datetime = row_filter.time_filter.end if row_filter.time_filter else None
        user_id = row_filter.user_id_filter.user_id if row_filter.user_id_filter else None
        
        columns_to_filter = None
        if row_filter.column_filter:
            columns_to_filter = [to_column_name(attr) for index, attr in enumerate(row_filter.column_filter)]
        
        def time_filter_fn(row_to_filter: Dict[attributes._Attribute, Iterable[TimeStampedCell]]):
            filtered_row = {}
            for attr, cells in row_to_filter.items():
                for cell in cells:
                    is_after_start_time = (not start_datetime) or (start_datetime <= cell.timestamp)
                    is_before_end_time = (not end_datetime) or (cell.timestamp <= end_datetime)
                    if is_after_start_time and is_before_end_time:
                        append(filtered_row, attr, cell)
            return filtered_row
        
        def user_id_filter_fn(row_to_filter: Dict[attributes._Attribute, Iterable[TimeStampedCell]]):
            if user_id == row_to_filter.get(attributes.OperationLogs.UserID, None):
                return row_to_filter
            return None
        
        def column_filter_fn(row_to_filter: Dict[attributes._Attribute, Iterable[TimeStampedCell]]):
            filtered_row = {}
            for attr, cells in row_to_filter.items():
                if to_column_name(attr) in columns_to_filter:
                    filtered_row[attr] = cells
            return filtered_row
        
        filtered_rows = {}
        for b_real_key, row in rows.items():
            filtered_row = row
            if start_datetime or end_datetime:
                filtered_row = time_filter_fn(filtered_row)
            
            if user_id:
                filtered_row = user_id_filter_fn(filtered_row)
            
            if columns_to_filter:
                filtered_row = column_filter_fn(filtered_row)
            
            if filtered_row:
                filtered_rows[b_real_key] = filtered_row
        
        return filtered_rows
    
    def _get_ids_range(self, key: bytes, size: int) -> Tuple:
        """Returns a range (min, max) of IDs for a given `key`."""
        column = attributes.Concurrency.Counter
        
        pk, sk = self._ddb_translator.to_pk_sk(key)
        
        column_name_in_ddb = to_column_name(column)
        
        time_microseconds = get_current_time_microseconds()
        
        def serialize_counter(x):
            return np.array([x], dtype=np.dtype('int64').newbyteorder('B')).tobytes()
        
        existing_counter = 0
        
        self._no_of_reads += 1
        res = self._ddb_table.get_item(
            Key={"key": pk, "sk": sk},
            ProjectionExpression='#c',
            
            # Need strongly consistent read here since we are
            # using the existing counter from the item and incrementing it
            ConsistentRead=True,
            
            ExpressionAttributeNames={
                "#c": column_name_in_ddb,
            },
        )
        existing_item = res.get('Item')
        if existing_item:
            existing_counter_column = existing_item.get(column_name_in_ddb, None)
            if existing_counter_column:
                existing_counter = column.deserialize(bytes(existing_counter_column[0][1]))
        
        counter = existing_counter + size
        
        self._no_of_writes += 1
        self._ddb_table.update_item(
            Key={"key": pk, "sk": sk},
            UpdateExpression="SET #c = :c",
            ExpressionAttributeNames={
                "#c": column_name_in_ddb,
            },
            ExpressionAttributeValues={
                ':c': [[
                    time_microseconds,
                    serialize_counter(counter),
                ]],
            }
        )
        high = counter
        
        return high + np.uint64(1) - size, high
    
    def _get_root_segment_ids_range(
        self, chunk_id: basetypes.CHUNK_ID, size: int = 1, counter: int = None
    ) -> np.ndarray:
        """Return unique segment ID for the root chunk."""
        n_counters = np.uint64(2 ** 8)
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
    
    def _get_lock_expiry_time_cutoff(self):
        """
        Returns the cutoff time for the lock expiry since the epoch in microseconds.
        The lock expiry time_cutoff is the current time minus the lock expiry time.
        
        For example,
        If the lock expiry is set to 1 minute, then the time_cutoff is the current time minus 1 minute.
        
        :return:
        """
        lock_expiry = self._graph_meta.graph_config.ROOT_LOCK_EXPIRY
        time_cutoff = datetime.now(timezone.utc) - lock_expiry
        # Change the resolution of the time_cutoff to milliseconds
        time_cutoff -= timedelta(microseconds=time_cutoff.microsecond % 1000)
        return to_microseconds(time_cutoff)
    
    def _deserialize(self, item: Dict):
        return {k: self._ddb_deserializer.deserialize(v) for k, v in item.items()}
