from typing import Dict
from typing import Union
from typing import Iterable
from typing import Optional
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import numpy as np
from google.cloud.bigtable.row_data import PartialRowData
from google.cloud.bigtable.row_filters import RowFilter
from google.cloud.bigtable.row_filters import PassAllFilter
from google.cloud.bigtable.row_filters import BlockAllFilter
from google.cloud.bigtable.row_filters import TimestampRange
from google.cloud.bigtable.row_filters import RowFilterChain
from google.cloud.bigtable.row_filters import RowFilterUnion
from google.cloud.bigtable.row_filters import ValueRangeFilter
from google.cloud.bigtable.row_filters import CellsRowLimitFilter
from google.cloud.bigtable.row_filters import ColumnRangeFilter
from google.cloud.bigtable.row_filters import TimestampRangeFilter
from google.cloud.bigtable.row_filters import ConditionalRowFilter
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter

from ... import attributes


def partial_row_data_to_column_dict(
    partial_row_data: PartialRowData,
) -> Dict[attributes._Attribute, PartialRowData]:
    new_column_dict = {}
    for family_id, column_dict in partial_row_data._cells.items():
        for column_key, column_values in column_dict.items():
            column = attributes.from_key(family_id, column_key)
            new_column_dict[column] = column_values
    return new_column_dict


def get_google_compatible_time_stamp(
    time_stamp: datetime, round_up: bool = False
) -> datetime:
    """
    Makes a datetime time stamp compatible with googles' services.
    Google restricts the accuracy of time stamps to milliseconds. Hence, the
    microseconds are cut of. By default, time stamps are rounded to the lower
    number.
    """
    micro_s_gap = timedelta(microseconds=time_stamp.microsecond % 1000)
    if micro_s_gap == 0:
        return time_stamp
    if round_up:
        time_stamp += timedelta(microseconds=1000) - micro_s_gap
    else:
        time_stamp -= micro_s_gap
    return time_stamp


def _get_column_filter(
    columns: Union[Iterable[attributes._Attribute], attributes._Attribute] = None
) -> RowFilter:
    """Generates a RowFilter that accepts the specified columns"""
    if isinstance(columns, attributes._Attribute):
        return ColumnRangeFilter(
            columns.family_id, start_column=columns.key, end_column=columns.key
        )
    elif len(columns) == 1:
        return ColumnRangeFilter(
            columns[0].family_id, start_column=columns[0].key, end_column=columns[0].key
        )
    return RowFilterUnion(
        [
            ColumnRangeFilter(col.family_id, start_column=col.key, end_column=col.key)
            for col in columns
        ]
    )


def _get_user_filter(user_id: str):
    """generates a ColumnRegEx Filter which filters user ids

    Args:
        user_id (str): userID to select for
    """

    condition = RowFilterChain(
        [
            ColumnQualifierRegexFilter(attributes.OperationLogs.UserID.key),
            ValueRangeFilter(str.encode(user_id), str.encode(user_id)),
            CellsRowLimitFilter(1),
        ]
    )

    conditional_filter = ConditionalRowFilter(
        base_filter=condition,
        true_filter=PassAllFilter(True),
        false_filter=BlockAllFilter(True),
    )
    return conditional_filter


def _get_time_range_filter(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = True,
) -> RowFilter:
    """Generates a TimeStampRangeFilter which is inclusive for start and (optionally) end.

    :param start:
    :param end:
    :return:
    """
    # Comply to resolution of BigTables TimeRange
    if start_time is not None:
        start_time = get_google_compatible_time_stamp(start_time, round_up=False)
    if end_time is not None:
        end_time = get_google_compatible_time_stamp(end_time, round_up=end_inclusive)
    return TimestampRangeFilter(TimestampRange(start=start_time, end=end_time))


def get_time_range_and_column_filter(
    columns: Optional[
        Union[Iterable[attributes._Attribute], attributes._Attribute]
    ] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = False,
    user_id: Optional[str] = None,
) -> RowFilter:
    time_filter = _get_time_range_filter(
        start_time=start_time, end_time=end_time, end_inclusive=end_inclusive
    )
    filters = [time_filter]
    if columns is not None:
        if len(columns) == 0:
            raise ValueError(
                f"Empty column filter {columns} is ambiguous. Pass `None` if no column filter should be applied."
            )
        column_filter = _get_column_filter(columns)
        filters = [column_filter, time_filter]
    if user_id is not None:
        user_filter = _get_user_filter(user_id=user_id)
        filters.append(user_filter)
    if len(filters) > 1:
        return RowFilterChain(filters)
    return filters[0]


def get_root_lock_filter(
    lock_column, lock_expiry, indefinite_lock_column
) -> ConditionalRowFilter:
    time_cutoff = datetime.now(timezone.utc) - lock_expiry
    # Comply to resolution of BigTables TimeRange
    time_cutoff -= timedelta(microseconds=time_cutoff.microsecond % 1000)
    time_filter = TimestampRangeFilter(TimestampRange(start=time_cutoff))

    # Build a column filter which tests if a lock was set (== lock column
    # exists) and if it is still valid (timestamp younger than
    # LOCK_EXPIRED_TIME_DELTA) and if there is no new parent (== new_parents
    # exists)
    lock_key_filter = ColumnRangeFilter(
        column_family_id=lock_column.family_id,
        start_column=lock_column.key,
        end_column=lock_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    indefinite_lock_key_filter = ColumnRangeFilter(
        column_family_id=indefinite_lock_column.family_id,
        start_column=indefinite_lock_column.key,
        end_column=indefinite_lock_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    new_parents_column = attributes.Hierarchy.NewParent
    new_parents_key_filter = ColumnRangeFilter(
        column_family_id=new_parents_column.family_id,
        start_column=new_parents_column.key,
        end_column=new_parents_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    temporal_lock_filter = RowFilterChain([time_filter, lock_key_filter])
    return ConditionalRowFilter(
        base_filter=RowFilterUnion([indefinite_lock_key_filter, temporal_lock_filter]),
        true_filter=PassAllFilter(True),
        false_filter=new_parents_key_filter,
    )


def get_indefinite_root_lock_filter(lock_column) -> ConditionalRowFilter:
    lock_key_filter = ColumnRangeFilter(
        column_family_id=lock_column.family_id,
        start_column=lock_column.key,
        end_column=lock_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    new_parents_column = attributes.Hierarchy.NewParent
    new_parents_key_filter = ColumnRangeFilter(
        column_family_id=new_parents_column.family_id,
        start_column=new_parents_column.key,
        end_column=new_parents_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    return ConditionalRowFilter(
        base_filter=lock_key_filter,
        true_filter=PassAllFilter(True),
        false_filter=new_parents_key_filter,
    )


def get_renew_lock_filter(
    lock_column: attributes._Attribute, operation_id: np.uint64
) -> ConditionalRowFilter:
    new_parents_column = attributes.Hierarchy.NewParent
    operation_id_b = lock_column.serialize(operation_id)

    # Build a column filter which tests if a lock was set (== lock column
    # exists) and if the given operation_id is still the active lock holder
    # and there is no new parent (== new_parents column exists). The latter
    # is not necessary but we include it as a backup to prevent things
    # from going really bad.

    column_key_filter = ColumnRangeFilter(
        column_family_id=lock_column.family_id,
        start_column=lock_column.key,
        end_column=lock_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    value_filter = ValueRangeFilter(
        start_value=operation_id_b,
        end_value=operation_id_b,
        inclusive_start=True,
        inclusive_end=True,
    )

    new_parents_key_filter = ColumnRangeFilter(
        column_family_id=new_parents_column.family_id,
        start_column=new_parents_column.key,
        end_column=new_parents_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    return ConditionalRowFilter(
        base_filter=RowFilterChain([column_key_filter, value_filter]),
        true_filter=new_parents_key_filter,
        false_filter=PassAllFilter(True),
    )


def get_unlock_root_filter(lock_column, lock_expiry, operation_id) -> RowFilterChain:
    time_cutoff = datetime.now(timezone.utc) - lock_expiry
    # Comply to resolution of BigTables TimeRange
    time_cutoff -= timedelta(microseconds=time_cutoff.microsecond % 1000)
    time_filter = TimestampRangeFilter(TimestampRange(start=time_cutoff))

    # Build a column filter which tests if a lock was set (== lock column
    # exists) and if it is still valid (timestamp younger than
    # LOCK_EXPIRED_TIME_DELTA) and if the given operation_id is still
    # the active lock holder
    column_key_filter = ColumnRangeFilter(
        column_family_id=lock_column.family_id,
        start_column=lock_column.key,
        end_column=lock_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    value_filter = ValueRangeFilter(
        start_value=lock_column.serialize(operation_id),
        end_value=lock_column.serialize(operation_id),
        inclusive_start=True,
        inclusive_end=True,
    )

    # Chain these filters together
    return RowFilterChain([time_filter, column_key_filter, value_filter])


def get_indefinite_unlock_root_filter(lock_column, operation_id) -> RowFilterChain:
    column_key_filter = ColumnRangeFilter(
        column_family_id=lock_column.family_id,
        start_column=lock_column.key,
        end_column=lock_column.key,
        inclusive_start=True,
        inclusive_end=True,
    )

    value_filter = ValueRangeFilter(
        start_value=lock_column.serialize(operation_id),
        end_value=lock_column.serialize(operation_id),
        inclusive_start=True,
        inclusive_end=True,
    )

    # Chain these filters together
    return RowFilterChain([column_key_filter, value_filter])
