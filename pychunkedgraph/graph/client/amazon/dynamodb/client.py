import math
import typing
import logging
from datetime import datetime, timezone
import numpy as np

import botocore
import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer, Binary

from . import AmazonDynamoDbConfig
from ...base import ClientWithIDGen
from ...base import OperationLogger
from ....meta import ChunkedGraphMeta
from .... import attributes
from .... import exceptions
from ....utils.serializers import pad_node_id
from ....utils.serializers import serialize_uint64
from ....utils.serializers import deserialize_uint64

from . import utils
from .utils import (
    DynamoDbTimeRangeFilter,
    DynamoDbColumnFilter,
    DynamoDbUserIdFilter,
    DynamoDbFilter,
)


class TimeStampedCell:
    def __init__(self, value: typing.Any, timestamp: int):
        self.value = value
        self.timestamp = timestamp

    def __repr__(self):
        return f"<Cell value={repr(self.value)} timestamp={datetime.fromtimestamp(self.timestamp / 1000000, tz=timezone.utc).isoformat(sep=' ')}>"


DEFAULT_ROW_PAGE_SIZE = 100


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
        self._ddb_serializer = TypeSerializer()
        self._ddb_deserializer = TypeDeserializer()
        # TODO: refactor column families to match graph-creation procedures
        # TODO: generalize bigtable GC property for columnfamilies into something like
        #       [KEEP_LAST_ITEM, KEEP_ALL_ITEMS]
        self._column_families = {"0": {}, "1": {}, "2": {}, "3": {}}

        self._graph_meta = graph_meta
        self._version = None

        boto3_conf_ = botocore.config.Config(
            region_name=config.REGION, retries={"max_attempts": 10, "mode": "standard"}
        )
        self._main_db = boto3.client("dynamodb", config=boto3_conf_)

    """Initialize the graph and store associated meta."""

    def create_graph(self) -> None:
        logging.warn(f"create_graph")

    """Add a version to the graph."""

    def add_graph_version(self, version):
        logging.warn(f"add_graph_version: {version}")

    """Read stored graph version."""

    def read_graph_version(self):
        logging.warn(f"read_graph_version")

    """Update stored graph meta."""

    def update_graph_meta(self, meta):
        logging.warn(f"update_graph_meta: {meta}")

    """Read stored graph meta."""

    def read_graph_meta(self):
        logging.debug("read_graph_meta")
        row = self._read_byte_row(attributes.GraphMeta.key)
        logging.debug(f"ROW: {row}")
        self._graph_meta = row[attributes.GraphMeta.Meta][0].value

        self.pk_key_shift = (
            self._graph_meta.graph_config.LAYER_ID_BITS
            + self._graph_meta.graph_config.SPATIAL_BITS
        )
        self.sk_key_mask = (1 << self.pk_key_shift) - 1
        pk_digits = math.ceil(math.log10(pow(2, 64 - self.pk_key_shift)))
        self.pk_int_format = f"0{pk_digits + 1}"

        return self._graph_meta

    """
    Read nodes and their properties.
    Accepts a range of node IDs or specific node IDs.
    """

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
        logging.warn(
            f"read_nodes: {start_id}, {end_id}, {node_ids}, {properties}, {start_time}, {end_time}, {end_time_inclusive}"
        )

    """Read a single node and its properties."""

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
        typing.Dict[attributes._Attribute, typing.List[TimeStampedCell]],
        typing.List[TimeStampedCell],
    ]:
        """Convenience function for reading a single node from Amazon DynamoDB.
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
            typing.Union[typing.Dict[attributes._Attribute, typing.List[TimeStampedCell]],
                  typing.List[TimeStampedCell]] --
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

    """Writes/updates nodes (IDs along with properties)."""

    def write_nodes(self, nodes):
        logging.warn(f"write_nodes: {nodes}")

    """Locks root node with operation_id to prevent race conditions."""

    def lock_root(self, node_id, operation_id):
        logging.warn(f"lock_root: {node_id}, {operation_id}")

    """Locks root nodes to prevent race conditions."""

    def lock_roots(self, node_ids, operation_id):
        logging.warn(f"lock_roots: {node_ids}, {operation_id}")

    """Locks root node with operation_id to prevent race conditions."""

    def lock_root_indefinitely(self, node_id, operation_id):
        logging.warn(f"lock_root_indefinitely: {node_id}, {operation_id}")

    """
    Locks root nodes indefinitely to prevent structural damage to graph.
    This scenario is rare and needs asynchronous fix or inspection to unlock.
    """

    def lock_roots_indefinitely(self, node_ids, operation_id):
        logging.warn(f"lock_roots_indefinitely: {node_ids}, {operation_id}")

    """Unlocks root node that is locked with operation_id."""

    def unlock_root(self, node_id, operation_id):
        logging.warn(f"unlock_root: {node_id}, {operation_id}")

    """Unlocks root node that is indefinitely locked with operation_id."""

    def unlock_indefinitely_locked_root(self, node_id, operation_id):
        logging.warn(f"unlock_indefinitely_locked_root: {node_id}, {operation_id}")

    """Renews existing node lock with operation_id for extended time."""

    def renew_lock(self, node_id, operation_id):
        logging.warn(f"renew_lock: {node_id}, {operation_id}")

    """Renews existing node locks with operation_id for extended time."""

    def renew_locks(self, node_ids, operation_id):
        logging.warn(f"renew_locks: {node_ids}, {operation_id}")

    """Reads timestamp from lock row to get a consistent timestamp."""

    def get_lock_timestamp(self, node_ids, operation_id):
        logging.warn(f"get_lock_timestamp: {node_ids}, {operation_id}")

    """Minimum of multiple lock timestamps."""

    def get_consolidated_lock_timestamp(self, root_ids, operation_ids):
        logging.warn(f"get_consolidated_lock_timestamp: {root_ids}, {operation_ids}")

    """Datetime time stamp compatible with client's services."""

    def get_compatible_timestamp(self, time_stamp):
        logging.warn(f"get_compatible_timestamp: {time_stamp}")

    """Generate a range of unique IDs in the chunk."""

    def create_node_ids(self, chunk_id):
        logging.warn(f"create_node_ids: {chunk_id}")

    """Generate a unique ID in the chunk."""

    def create_node_id(self, chunk_id):
        logging.warn(f"create_node_id: {chunk_id}")

    """Gets the current maximum node ID in the chunk."""

    def get_max_node_id(self, chunk_id):
        logging.warn(f"get_max_node_id: {chunk_id}")

    """Generate a unique operation ID."""

    def create_operation_id(self):
        logging.warn(f"create_operation_id")

    """Gets the current maximum operation ID."""

    def get_max_operation_id(self):
        logging.warn(f"get_max_operation_id")

    """Read log entry for a given operation ID."""

    def read_log_entry(self, operation_id: int) -> None:
        logging.warn(f"read_log_entry: {operation_id}")

    """Read log entries for given operation IDs."""

    def read_log_entries(self, operation_ids) -> None:
        logging.warn(f"read_log_entries: {operation_ids}")

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
        typing.Dict[attributes._Attribute, typing.List[TimeStampedCell]],
        typing.List[TimeStampedCell],
    ]:
        """Convenience function for reading a single row from Amazon DynamoDB using its `bytes` keys.

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
            typing.Union[typing.Dict[attributes._Attribute, typing.List[TimeStampedCell]],
                  typing.List[TimeStampedCell]] --
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
            typing.Dict[attributes._Attribute, typing.List[TimeStampedCell]],
            typing.List[TimeStampedCell],
        ],
    ]:
        """Main function for reading a row range or non-contiguous row sets from Amazon DynamoDB using
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
            typing.Dict[bytes, typing.Union[typing.Dict[attributes._Attribute, typing.List[TimeStampedCell]],
                              typing.List[TimeStampedCell]]] --
                Returns a dictionary of `byte` rows as keys. Their value will be a mapping of
                columns to a typing.List of cells (one cell per timestamp). Each cell has a `value`
                property, which returns the deserialized field, and a `timestamp` property, which
                returns the timestamp as `datetime` object.
                If only a single `attributes._Attribute` was requested, the typing.List of cells will be
                attached to the row dictionary directly (skipping the column dictionary).
        """

        key_set = {}
        if row_keys is not None:
            key_set["ROW_KEYS"] = list(row_keys)
            logging.debug(f"KEYS: {row_keys}")
        else:
            raise exceptions.PreconditionError("IMPLEMENTME")

        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
            user_id=user_id,
        )

        rows = self._read(key_set=key_set, row_filter=filter_)

        # Deserialize cells
        for row_key, column_dict in rows.items():
            for column, cell_entries in column_dict.items():
                for cell_entry in cell_entries:
                    cell_entry.value = column.deserialize(cell_entry.value)
            # If no column array was requested, reattach single column's values directly to the row
            if isinstance(columns, attributes._Attribute):
                rows[row_key] = cell_entries
        return rows

    # TODO: run multi-key requests concurrently (do we need cuncurrency if batch read is used?)
    # TODO: use batch-read if possible
    # TODO: use pagination (some rows may have too many cells to be fetched at once, but haven't seen them)
    def _read(self, key_set=dict[str, dict], row_filter: DynamoDbFilter = None) -> dict:
        rows = {}

        attr_names = {"#key": "key"}

        # TODO: refactor key_set into named tuple
        # TODO: consider multithreading:
        # there are 2 key entry types:
        #  * one or more "exact" keys
        #     ** If the request is for just one exact key, DDB can be called with GetItem
        #     ** If the request contains multiple exact keys, up to 100, DDB can be called with BatchGetItem
        #     ** If the request contains more than 100 exact keys, BatchGetItem can be called in multi-threading
        #  * one or more "range" keys (from..to)
        #     ** if the request has just one range, DDB can be called with QueryItem
        #     ** if the request has more than one range, multi-threading should be used
        # TODO: "new" data for existing key is appended to the the map... that's not so good because
        #       may exceed the limit for the row size eventually. Currenty it is as is in the BigTable
        for key in key_set["ROW_KEYS"]:
            skey = key.decode()
            if skey[0].isdigit():
                ikey = int(skey)
                pk = f"{(ikey >> self.pk_key_shift):{self.pk_int_format}}"
                sk = ikey & self.sk_key_mask
            elif skey[0] in ["f", "i"]:
                ikey = int(skey[1:])
                pk = f"{(ikey >> self.pk_key_shift):{self.pk_int_format}}"
                sk = ikey & self.sk_key_mask
            else:
                pk = skey
                sk = 0

            logging.debug(f"QUERYING FOR: {key}, pk: {pk}, sk: {sk}")
            attr_vals = {
                ":key": self._ddb_serializer.serialize(pk),
                ":sk": self._ddb_serializer.serialize(sk),
            }
            kwargs = {}
            # TODO: implement filters:
            #       user_id
            #       time
            #       columns
            if row_filter.column_filter:
                ddb_columns = [
                    f"#C{index}" for index in range(len(row_filter.column_filter))
                ]
                ddb_columns.extend(["#key", "sk", "#ver"])
                kwargs["ProjectionExpression"] = ",".join(ddb_columns)
                for index, attr in enumerate(row_filter.column_filter):
                    attr_names[f"#C{index}"] = f"{attr.family_id}.{attr.key.decode()}"
                attr_names[f"#ver"] = "@"

            ret = self._main_db.query(
                TableName=self._table_name,
                Limit=self._row_page_size,
                KeyConditionExpression="#key = :key AND sk = :sk",
                ExpressionAttributeNames=attr_names,
                ExpressionAttributeValues=attr_vals,
                **kwargs,
            )
            items = ret.get("Items", [])

            # each item comes with 'key', 'sk', [column_faimily] and '@' columns
            for item in items:
                row = {}
                pk = None
                sk = 0
                item_keys = list(item.keys())
                item_keys.sort()
                # ddb_clm is one of 'key', 'sk', [column_faimily.column_qualifier], '@'
                for ddb_clm in item_keys:
                    ddb_row_value = item[ddb_clm]
                    row_value = self._ddb_deserializer.deserialize(ddb_row_value)
                    if ddb_clm == "@":
                        # for '@' row_value is int
                        # TODO: store row version for optimistic locking (subject TBD)
                        ver = row_value
                    elif ddb_clm == "key":
                        # for key row_value is string
                        pk = row_value
                    elif ddb_clm == "sk":
                        # for sk row_value is int
                        sk = int(row_value)
                    else:
                        # ddb_clm here is column_family.column_qualifier
                        column_family, qualifier = ddb_clm.split(".")
                        attr = attributes.from_key(
                            column_family, qualifier.encode()
                        )
                        if attr not in row:
                            row[attr] = []
                        for timestamp, column_value in row_value:
                            # find _Attribute for ddb_clm:qualifier and put cell & timestamp pair associated with it
                            row[attr].append(
                                TimeStampedCell(
                                    column_value.value
                                    if isinstance(column_value, Binary)
                                    else column_value,
                                    int(timestamp),
                                )
                            )

                if pk[0].isdigit():
                    ikey = (int(pk) << self.pk_key_shift) | sk
                    real_key = pad_node_id(ikey)
                elif pk[0] in ["i", "f"]:
                    ikey = (int(pk[1:]) << self.pk_key_shift) | sk
                    real_key = pad_node_id(ikey)
                else:
                    real_key = pk

                b_real_key = real_key.encode()
                rows[b_real_key] = row

        return rows
