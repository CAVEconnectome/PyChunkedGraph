from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Iterable

import numpy as np

from ..meta import ChunkedGraphMeta


# TODO design api
# 1. create / overwrite
# 2. a counter to generate unique ids (IDs api?)
# 3. store metadata
# 4. read/write rows api


class Client(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def create_graph(self, graph_meta: ChunkedGraphMeta) -> None:
        """Create graph and store associated meta"""

    def read_byte_rows(
        self,
        start_key: Optional[bytes] = None,
        end_key: Optional[bytes] = None,
        row_keys: Optional[Iterable[bytes]] = None,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Dict[
        bytes,
        Union[
            Dict[column_keys._Column, List[bigtable.row_data.Cell]],
            List[bigtable.row_data.Cell],
        ],
    ]:
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
            end_inclusive=end_time_inclusive,
        )

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
            )
        else:
            raise cg_exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or"
                " both, a start row and an end row."
            )

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
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Union[
        Dict[column_keys._Column, List[bigtable.row_data.Cell]],
        List[bigtable.row_data.Cell],
    ]:
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
        row = self.read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        if isinstance(columns, column_keys._Column):
            return row.get(row_key, [])
        else:
            return row.get(row_key, {})

    @abstractmethod
    def read_nodes(
        self,
        start_id: Optional[np.uint64] = None,
        end_id: Optional[np.uint64] = None,
        node_ids: Optional[Iterable[np.uint64]] = None,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Dict[np.uint64,Union[Dict,List]]:
        pass

    def read_node_id_row(
        self,
        node_id: np.uint64,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Union[
        Dict[column_keys._Column, List[bigtable.row_data.Cell]],
        List[bigtable.row_data.Cell],
    ]:
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
        return self.read_byte_row(
            row_key=serializers.serialize_uint64(node_id),
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )

