import logging
from datetime import datetime
from typing import Dict, Iterable, Union, Optional, List, Any, Tuple

import boto3
import botocore
import numpy as np
from multiwrapper import multiprocessing_utils as mu
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer, Binary

from . import AmazonDynamoDbConfig
from . import utils
from .ddb_helper import DdbHelper
from .ddb_table import Table
from .row_set import RowSet
from .timestamped_cell import TimeStampedCell
from .utils import (
    DynamoDbFilter,
)
from ...base import ClientWithIDGen
from ...base import OperationLogger
from .... import attributes
from .... import exceptions
from ....meta import ChunkedGraphMeta
from ....utils import basetypes
from ....utils.serializers import pad_node_id, serialize_key, serialize_uint64, deserialize_uint64

DEFAULT_ROW_PAGE_SIZE = 100  # Max items to fetch using GetBatchItem operation
DEFAULT_MAX_ITEMS_LIMIT = 1000  # Maximum items to fetch in one query


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
        self._row_page_size = DEFAULT_ROW_PAGE_SIZE
        self._max_items = DEFAULT_MAX_ITEMS_LIMIT
        self._ddb_serializer = TypeSerializer()
        self._ddb_deserializer = TypeDeserializer()
        # TODO: refactor column families to match graph-creation procedures
        # TODO: generalize bigtable GC property for columnfamilies into something like
        #       [KEEP_LAST_ITEM, KEEP_ALL_ITEMS]
        self._column_families = {"0": {}, "1": {}, "2": {}, "3": {}}
        
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
        
        dynamodb = boto3.resource('dynamodb', config=boto3_conf_, **kwargs)
        self._ddb_table = dynamodb.Table(self._table_name)
        
        self._table = Table(self._main_db, self._table_name, boto3_conf_, **kwargs)
        
        self._ddb_helper = DdbHelper()
    
    """Initialize the graph and store associated meta."""
    
    def create_graph(self, meta: ChunkedGraphMeta, version: str) -> None:
        """Initialize the graph and store associated meta."""
        # TODO: revisit table creation here. Is it needed for anything but tests?
        # even for tests, it's better to create in testsuite fixture/resource factory
        try:
            row = self._read_byte_row(attributes.GraphMeta.key)
            if row:
                raise ValueError(f"{self._table_name} table already exists.")
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                raise e
        self._main_db.create_table(
            TableName=self._table_name,
            KeySchema=[
                {"AttributeName": "key", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "key", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
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
        try:
            row = self._read_byte_row(attributes.GraphVersion.key)
            self._version = row[attributes.GraphVersion.Version][0].value
            return self._version
        except KeyError:
            return None
    
    """Update stored graph meta."""
    
    def update_graph_meta(
            self, meta: ChunkedGraphMeta, overwrite: Optional[bool] = False
    ):
        if overwrite:
            self._delete_meta()
        self._graph_meta = meta
        row = self.mutate_row(
            attributes.GraphMeta.key,
            {attributes.GraphMeta.Meta: meta},
        )
        self.write([row])
    
    """Read stored graph meta."""
    
    def read_graph_meta(self):
        logging.warning("read_graph_meta")
        row = self._read_byte_row(attributes.GraphMeta.key)
        logging.debug(f"ROW: {row}")
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
        logging.warning(
            f"read_nodes: {start_id}, {end_id}, {node_ids}, {properties}, {start_time}, {end_time}, {end_time_inclusive}"
        )
        if node_ids is not None and len(node_ids) > self._max_items:
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
        logging.warning(f"write_nodes: {nodes}")
        raise NotImplementedError("Not yet implemented")
    
    # Helpers
    def write(
            self,
            rows: Iterable[dict[str, Union[bytes, dict[str, Iterable[TimeStampedCell]]]]],
            root_ids: Optional[
                Union[np.uint64, Iterable[np.uint64]]
            ] = None,
            operation_id: Optional[np.uint64] = None,
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
        logging.warning(f"write {rows} {root_ids} {operation_id} {slow_retry} {block_size}")
        
        # TODO: Implement locking and retries with backoff
        
        for i in range(0, len(rows), block_size):
            with self._ddb_table.batch_writer() as batch:
                rows_in_this_batch = rows[i: i + block_size]
                for row in rows_in_this_batch:
                    ddb_item = self._ddb_helper.row_to_ddb_item(row)
                    batch.put_item(Item=ddb_item)
    
    def mutate_row(
            self,
            row_key: bytes,
            val_dict: Dict[attributes._Attribute, Any],
            time_stamp: Optional[datetime] = None,
    ) -> dict[str, Union[bytes, dict[str, Iterable[TimeStampedCell]]]]:
        """Mutates a single row (doesn't write to DynamoDB)."""
        pk, sk = self._ddb_helper.to_pk_sk(row_key)
        ret = self._ddb_table.get_item(Key={"key": pk, "sk": sk})
        item = ret.get('Item')
        
        row = {"key": row_key}
        if item is not None:
            b_real_key, row_from_db = self._ddb_helper.ddb_item_to_row(item)
            row.update(row_from_db)
        
        cells = self._ddb_helper.attribs_to_cells(attribs=val_dict, time_stamp=time_stamp)
        row.update(cells)
        
        return row
    
    """Locks root node with operation_id to prevent race conditions."""
    
    def lock_root(self, node_id, operation_id):
        logging.warning(f"lock_root: {node_id}, {operation_id}")
    
    """Locks root nodes to prevent race conditions."""
    
    def lock_roots(self, node_ids, operation_id):
        logging.warning(f"lock_roots: {node_ids}, {operation_id}")
    
    """Locks root node with operation_id to prevent race conditions."""
    
    def lock_root_indefinitely(self, node_id, operation_id):
        logging.warning(f"lock_root_indefinitely: {node_id}, {operation_id}")
    
    """
    Locks root nodes indefinitely to prevent structural damage to graph.
    This scenario is rare and needs asynchronous fix or inspection to unlock.
    """
    
    def lock_roots_indefinitely(self, node_ids, operation_id):
        logging.warning(f"lock_roots_indefinitely: {node_ids}, {operation_id}")
    
    """Unlocks root node that is locked with operation_id."""
    
    def unlock_root(self, node_id, operation_id):
        logging.warning(f"unlock_root: {node_id}, {operation_id}")
    
    """Unlocks root node that is indefinitely locked with operation_id."""
    
    def unlock_indefinitely_locked_root(self, node_id, operation_id):
        logging.warning(f"unlock_indefinitely_locked_root: {node_id}, {operation_id}")
    
    """Renews existing node lock with operation_id for extended time."""
    
    def renew_lock(self, node_id, operation_id):
        logging.warning(f"renew_lock: {node_id}, {operation_id}")
    
    """Renews existing node locks with operation_id for extended time."""
    
    def renew_locks(self, node_ids, operation_id):
        logging.warning(f"renew_locks: {node_ids}, {operation_id}")
    
    """Reads timestamp from lock row to get a consistent timestamp."""
    
    def get_lock_timestamp(self, node_ids, operation_id):
        logging.warning(f"get_lock_timestamp: {node_ids}, {operation_id}")
    
    """Minimum of multiple lock timestamps."""
    
    def get_consolidated_lock_timestamp(self, root_ids, operation_ids):
        logging.warning(f"get_consolidated_lock_timestamp: {root_ids}, {operation_ids}")
    
    """Datetime time stamp compatible with client's services."""
    
    def get_compatible_timestamp(self, time_stamp):
        logging.warning(f"get_compatible_timestamp: {time_stamp}")
    
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
    
    def create_node_id(self, chunk_id):
        logging.warning(f"create_node_id: {chunk_id}")
    
    """Gets the current maximum node ID in the chunk."""
    
    def get_max_node_id(self, chunk_id, root_chunk=False):
        """Gets the current maximum segment ID in the chunk."""
        print(f" --- get_max_node_id chunk_id, root_chunk = {chunk_id}, {root_chunk}")
        if root_chunk:
            n_counters = np.uint64(2 ** 8)
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
        print(f" --- get_max_node_id reading row with key = {serialize_uint64(chunk_id, counter=True)}")
        row = self._read_byte_row(
            serialize_uint64(chunk_id, counter=True), columns=column
        )
        print(f" --- get_max_node_id read counter = {row}")
        return chunk_id | basetypes.SEGMENT_ID.type(row[0].value if row else 0)
    
    """Generate a unique operation ID."""
    
    def create_operation_id(self):
        logging.warning(f"create_operation_id")
    
    """Gets the current maximum operation ID."""
    
    def get_max_operation_id(self):
        logging.warning(f"get_max_operation_id")
    
    """Read log entry for a given operation ID."""
    
    def read_log_entry(self, operation_id: int) -> None:
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
    
    def read_log_entries(self, operation_ids) -> None:
        logging.warning(f"read_log_entries: {operation_ids}")
    
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
            end_key_inclusive {bool} -- Whether or not `end_key` itself should be included in the
                request, ignored if `row_keys` is set or `end_key` is None. (default: {False})
            row_keys {Optional[Iterable[bytes]]} -- An `Iterable` containing possibly
                non-contiguous row keys. Takes precedence over `start_key` and `end_key`.
                (default: {None})
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
            logging.debug(f"KEYS: {row_set.row_keys}")
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
    
    # TODO: run multi-key requests concurrently (do we need concurrency if batch read is used?)
    # TODO: use batch-read if possible
    # TODO: use pagination (some rows may have too many cells to be fetched at once, but haven't seen them)
    def _read(self, row_set: RowSet, row_filter: DynamoDbFilter = None) -> dict:
        """Core function to read rows from DynamoDB.
        :param row_set: Set of related to the rows to be read
        :param row_filter: An instance of DynamoDbFilter to filter which rows/columns to read
        :return: typing.Dict
        """
        n_subrequests = max(
            1, int(np.ceil(len(row_set.row_keys) / self._row_page_size))
        )
        n_threads = min(n_subrequests, 2 * mu.n_cpus)
        
        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[i * self._row_page_size: (i + 1) * self._row_page_size]
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
    
    def _execute_read_thread(self, args: Tuple[Table, RowSet, DynamoDbFilter]):
        """Function to be executed in parallel."""
        print(f"\n\n --- _execute_read_thread START")
        table, row_set, row_filter = args
        if not row_set.row_keys and not row_set.row_ranges:
            return {}
        
        row_keys = row_set.row_keys
        
        rows = {}
        item_keys_to_get = []
        attr_names = {'#key': 'key'}
        kwargs = {
            'ProjectionExpression': '#key, sk',
            'ExpressionAttributeNames': attr_names
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
            __append_to_projection_expression(kwargs, ["#uid"])
            user_id_attr = attributes.OperationLogs.UserID
            attr_names["#uid"] = f"{user_id_attr.family_id}.{user_id_attr.key.decode()}"
            kwargs["ExpressionAttributeNames"] = attr_names
        
        # Column filter
        if row_filter.column_filter:
            ddb_columns = [
                f"#C{index}" for index in range(len(row_filter.column_filter))
            ]
            ddb_columns.extend(["#ver"])
            __append_to_projection_expression(kwargs, ddb_columns)
            
            for index, attr in enumerate(row_filter.column_filter):
                attr_names[f"#C{index}"] = f"{attr.family_id}.{attr.key.decode()}"
            attr_names["#ver"] = "@"
            
            kwargs["ExpressionAttributeNames"] = attr_names
        
        # there are 2 key entry types:
        #  * one or more "exact" keys
        #     ** If the request is for just one exact key, DDB can be called with GetItem
        #     ** If the request contains multiple exact keys, up to 100, DDB can be called with BatchGetItem
        #     ** If the request contains more than 100 exact keys, BatchGetItem can be called in multi-threading
        #  * one or more "range" keys (from..to)
        #     ** if the request has just one range, DDB can be called with QueryItem
        #     ** if the request has more than one range, multi-threading should be used
        # TODO: "new" data for existing key is appended to the the map... that's not so good because
        #       may exceed the limit for the row size eventually. Currently it is as is in the BigTable
        for key in row_keys:
            pk, sk = self._ddb_helper.to_pk_sk(key)
            logging.debug(f"QUERYING FOR: {key}, pk: {pk}, sk: {sk}")
            item_keys_to_get.append({
                # "batch_get_item" is not available on the boto3 DynamoDB resource abstraction (i.e., "self._ddb_table")
                # so we are forced to use low-level boto3 client (i.e.,  "self._main_db")
                # The low-level boto3 client does not handle serialization/deserialization automatically, so have to
                # do it manually using the "self._ddb_serializer" here
                'key': self._ddb_serializer.serialize(pk),
                'sk': self._ddb_serializer.serialize(sk),
            })
        
        if len(item_keys_to_get) > 0:
            # TODO: Handle partial batch retrival failures
            params = {
                self._table_name: {
                    'Keys': item_keys_to_get,
                    **kwargs,
                },
            }
            print(f" --- _execute_read_thread batch_get_item = {params}")
            ret = self._main_db.batch_get_item(RequestItems=params)
            
            items = ret.get("Responses", {}).get(self._table_name, [])
            
            print(f" --- _execute_read_thread batch_get_item returned = {items}")
            
            # each item comes with 'key', 'sk', [column_family] and '@' columns
            for index, item in enumerate(items):
                b_real_key, row = self._ddb_helper.raw_ddb_item_to_row(
                    {
                        'key': item_keys_to_get[index]['key'],
                        'sk': item_keys_to_get[index]['sk'],
                        **item,
                    }
                )
                rows[b_real_key] = row
        
        if len(row_set.row_ranges) > 0:
            for row_range in row_set.row_ranges:
                pk, start_sk, end_sk = self._ddb_helper.to_sk_range(
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
                    "Limit": self._max_items,
                    "KeyConditionExpression": "#key = :key AND sk BETWEEN :st_sk AND :end_sk",
                    "ExpressionAttributeValues": attr_vals,
                    **kwargs,
                }
                
                print(f" --- _execute_read_thread _ddb_table.query = {query_kwargs}")
                ret = self._ddb_table.query(**query_kwargs)
                items = ret.get("Items", [])
                
                print(f" --- _execute_read_thread _ddb_table.query items = {items}")
                
                # each item comes with 'key', 'sk', [column_family] and '@' columns
                for item in items:
                    b_real_key, row = self._ddb_helper.ddb_item_to_row(item)
                    rows[b_real_key] = row
        
        print(f" --- _execute_read_thread retrieved rows = {rows}")
        filtered_rows = self._apply_filters(rows, row_filter)
        print(f" --- _execute_read_thread filtered rows = {rows}")
        print(f" --- _execute_read_thread END\n\n")
        return filtered_rows
    
    def _apply_filters(
            self,
            rows: dict[str, dict[attributes._Attribute, Iterable[TimeStampedCell]]],
            row_filter: DynamoDbFilter
    ):
        # the start_datetime and the end_datetime below are "datetime" instances (and NOT int timestamp)
        start_datetime = row_filter.time_filter.start if row_filter.time_filter else None
        end_datetime = row_filter.time_filter.end if row_filter.time_filter else None
        user_id = row_filter.user_id_filter.user_id if row_filter.user_id_filter else None
        
        def time_filter_fn(row_to_filter: dict[attributes._Attribute, Iterable[TimeStampedCell]]):
            filtered_row = {}
            for attr, cells in row_to_filter.items():
                filtered_cells = []
                for cell in cells:
                    if start_datetime <= cell.to_datetime() <= end_datetime:
                        filtered_cells.append(cell)
                filtered_row[attr] = filtered_cells
            return filtered_row
        
        def user_id_filter_fn(row_to_filter: dict[attributes._Attribute, Iterable[TimeStampedCell]]):
            if user_id == row_to_filter.get(attributes.OperationLogs.UserID, None):
                return row_to_filter
            return None
        
        filtered_rows = {}
        for b_real_key, row in rows.items():
            filtered_row = row
            if start_datetime and end_datetime:
                filtered_row = time_filter_fn(filtered_row)
            if user_id:
                filtered_row = user_id_filter_fn(filtered_row)
            
            if filtered_row:
                filtered_rows[b_real_key] = filtered_row
        
        return filtered_rows
    
    def _get_ids_range(self, key: bytes, size: int) -> Tuple:
        """Returns a range (min, max) of IDs for a given `key`."""
        print(f" --- _get_ids_range key, size = {key},  {size}")
        
        column = attributes.Concurrency.Counter
        
        pk, sk = self._ddb_helper.to_pk_sk(key)
        
        column_name_in_ddb = f"{column.family_id}.{column.key.decode()}"
        
        # ret = self._main_db.put_item(
        #     TableName=self._table_name,
        #     Item={
        #         "key": self._ddb_serializer.serialize(pk),
        #         "sk": self._ddb_serializer.serialize(sk),
        #         column_name_in_ddb: self._ddb_serializer.serialize(size),
        #     }
        # )
        
        time_microseconds = TimeStampedCell.get_current_time_microseconds()
        size_bytes = basetypes.COUNTER.type(size).tobytes()
        # self._ddb_table.put_item(
        #     Item={
        #         "key": pk,
        #         "sk": sk,
        #
        #         # Each attribute column in DDB is stored as an array of "cells"
        #         # Each "cell" contains a timestamp and the value of the attribute
        #         # at the given timestamp
        #         column_name_in_ddb: [[
        #             int(time_microseconds),
        #             size_bytes,
        #         ]]
        #     }
        # )
        
        self._ddb_table.update_item(
            Key={"key": pk, "sk": sk},
            UpdateExpression="SET #c = :c",
            ExpressionAttributeNames={
                "#c": column_name_in_ddb,
            },
            ExpressionAttributeValues={
                ':c': [[
                    int(time_microseconds),
                    size_bytes,
                ]],
            }
        )
        high = size
        print(f" --- _get_ids_range returning = {high + np.uint64(1) - size}, {high}")
        
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
