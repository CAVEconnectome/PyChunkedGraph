from typing import Dict
from typing import Union
from typing import Iterable
from typing import Optional
from datetime import datetime
from datetime import timedelta

import numpy as np

from ... import attributes


class Cell:
    """Cell-like object to match BigTable cell interface."""
    def __init__(self, value, timestamp=None):
        self.value = value
        self.timestamp = timestamp


def hbase_row_to_column_dict(
    row_data: Dict,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    time_filter: Optional[Dict] = None,
) -> Dict[attributes._Attribute, list]:
    """Convert HBase row data to column dictionary format compatible with BigTable format."""
    new_column_dict = {}
    
    # happybase returns data as {b'family:qualifier': value}
    for col_key_bytes, cell_data in row_data.items():
        col_key = col_key_bytes.decode('utf-8')
        if ':' not in col_key:
            continue
        family_id, column_key = col_key.split(':', 1)
        family_id = family_id.encode('utf-8')
        column_key = column_key.encode('utf-8')
        
        try:
            column = attributes.from_key(family_id, column_key)
        except (KeyError, ValueError):
            # Skip unknown columns
            continue
        
        # happybase returns single value (latest version) by default
        # For multiple versions, we'd need to use include_timestamp=True
        # For now, create a single cell
        cell = Cell(
            value=cell_data,
            timestamp=None  # We'll need to get timestamp separately if needed
        )
        new_column_dict[column] = [cell]
    
    return new_column_dict


def get_hbase_compatible_time_stamp(
    time_stamp: datetime, round_up: bool = False
) -> datetime:
    """
    Makes a datetime time stamp compatible with HBase.
    Returns datetime (HBase uses milliseconds internally but we work with datetime).
    """
    if time_stamp is None:
        return None
    # HBase uses milliseconds, but we'll work with datetime objects
    # Round to milliseconds
    micro_s_gap = timedelta(microseconds=time_stamp.microsecond % 1000)
    if micro_s_gap == 0:
        return time_stamp
    if round_up:
        time_stamp += timedelta(microseconds=1000) - micro_s_gap
    else:
        time_stamp -= micro_s_gap
    return time_stamp


def get_time_range_filter(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = True,
) -> Dict:
    """Generates HBase filter parameters for time range."""
    filter_params = {}
    if start_time is not None:
        filter_params['timestamp'] = int(get_hbase_compatible_time_stamp(start_time, round_up=False).timestamp() * 1000)
    if end_time is not None:
        filter_params['max_timestamp'] = int(get_hbase_compatible_time_stamp(end_time, round_up=end_inclusive).timestamp() * 1000)
    return filter_params


def get_column_filter(
    columns: Union[Iterable[attributes._Attribute], attributes._Attribute] = None
) -> Optional[list]:
    """Generates column filter for HBase scan/get operations.
    Returns list of (family, qualifier) tuples or None.
    """
    if columns is None:
        return None
    
    if isinstance(columns, attributes._Attribute):
        return [f"{columns.family_id.decode('utf-8')}:{columns.key.decode('utf-8')}".encode('utf-8')]
    
    if len(columns) == 0:
        return None
    
    return [f"{col.family_id.decode('utf-8')}:{col.key.decode('utf-8')}".encode('utf-8') for col in columns]

