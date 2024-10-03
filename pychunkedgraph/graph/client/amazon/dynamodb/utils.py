from typing import (
    Union,
    Iterable,
    Optional,
)
from datetime import datetime, timedelta, timezone
from collections import namedtuple

from .... import attributes

DynamoDbTimeRangeFilter = namedtuple(
    "DynamoDbTimeRangeFilter", ("start", "end"), defaults=(None, None)
)

DynamoDbColumnFilter = namedtuple(
    "DynamoDbColumnFilter", ("family_id", "key"), defaults=(None, None)
)

DynamoDbUserIdFilter = namedtuple("DynamoDbUserIdFilter", ("user_id"), defaults=(None))

DynamoDbFilter = namedtuple(
    "DynamoDbFilter",
    ("time_filter", "column_filter", "user_id_filter"),
    defaults=(None, None, None),
)


def get_filter_time_stamp(time_stamp: datetime, round_up: bool = False) -> datetime:
    """
    Makes a datetime time stamp with the accuracy of milliseconds. Hence, the
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
    return time_stamp.replace(tzinfo=timezone.utc)


def _get_time_range_filter(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = True,
) -> DynamoDbTimeRangeFilter:
    """Generates a TimeStampRangeFilter which is inclusive for start and (optionally) end.

    :param start:
    :param end:
    :return:
    """
    if start_time is not None:
        start_time = get_filter_time_stamp(start_time, round_up=False)
    if end_time is not None:
        end_time = get_filter_time_stamp(end_time, round_up=end_inclusive)
    return DynamoDbTimeRangeFilter(start=start_time, end=end_time)


def _get_column_filter(
    columns: Union[Iterable[attributes._Attribute], attributes._Attribute] = None
) -> Union[DynamoDbColumnFilter, Iterable[DynamoDbColumnFilter]]:
    """Generates a RowFilter that accepts the specified columns"""
    if isinstance(columns, attributes._Attribute):
        return [DynamoDbColumnFilter(columns.family_id, key=columns.key)]
    return [DynamoDbColumnFilter(col.family_id, key=col.key) for col in columns]


# TODO: revisit how OperationLogs works and how filer by user_id would be implemented
def _get_user_filter(user_id: str):
    """generates a ColumnRegEx Filter which filters user ids

    Args:
        user_id (str): userID to select for
    """
    return DynamoDbUserIdFilter(user_id)


def get_time_range_and_column_filter(
    columns: Optional[
        Union[Iterable[attributes._Attribute], attributes._Attribute]
    ] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = False,
    user_id: Optional[str] = None,
) -> DynamoDbFilter:
    time_filter = (
        _get_time_range_filter(
            start_time=start_time, end_time=end_time, end_inclusive=end_inclusive
        )
        if start_time or end_time
        else None
    )
    column_filter = _get_column_filter(columns) if columns is not None else None
    user_filter = _get_user_filter(user_id=user_id) if user_id is not None else None
    return DynamoDbFilter(time_filter, column_filter, user_filter)


def append(d, attr, val):
    """
    utility function to append a value to the given array attribute of the given dictionary
    :param d:
    :param attr:
    :param val:
    :return: None
    """
    
    if attr not in d:
        d[attr] = []
    d[attr].append(val)


def to_microseconds(t: datetime):
    return int(t.timestamp() * 1e6)


def get_current_time_microseconds():
    return to_microseconds(datetime.now(timezone.utc))


def from_microseconds(microseconds):
    return datetime.fromtimestamp(microseconds / 1e6, tz=timezone.utc)


def remove_and_merge_duplicates(list_of_dicts: list[dict], key_to_compare="key") -> list[dict]:
    """
    A utility method to remove duplicates from a list containing dictionaries based on the specified key.
    If such duplicates are found then merge the duplicated dicts into a single dict.
    
    For example,
      original_list = [
          {"key": 123, "a": "a"},
          {"key": 345, "b": "b"},
          {"key": 567, "c": "c"},
          {"key": 123, "d": "d", 10: 100},
      ]
    print(remove_and_merge_duplicates(original_list)):
      [
          {"key": 123, "a": "a", "d": "d", 10: 100},
          {"key": 345, "b": "b"},
          {'key': 567, 'c': 'c'},
      ]
      
    :param list_of_dicts:
    :param key_to_compare:
    
    :return:
    """
    # Create a new list to store the unique elements
    unique_dicts = {}
    merged_list = []
    
    for d in list_of_dicts:
        if key_to_compare in d:
            value_to_compare = d[key_to_compare]
            if value_to_compare in unique_dicts:
                # Merge the duplicate with the existing one
                unique_dicts[value_to_compare].update(d)
            else:
                unique_dicts[value_to_compare] = d
                merged_list.append(d)
        else:
            # If the key is not present in the dict, then it is a new element.
            # Simply append it to the list
            merged_list.append(d)
    return merged_list
