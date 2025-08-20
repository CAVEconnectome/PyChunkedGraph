from typing import Dict, Iterable, Union, Optional, List, Any, Tuple


class RowRange:
    def __init__(self, start_key: bytes = None, end_key: bytes = None, start_inclusive: bool = True,
                 end_inclusive: bool = False):
        self._start_key = start_key
        self._end_key = end_key
        self._start_inclusive = start_inclusive
        self._end_inclusive = end_inclusive
    
    @property
    def start_key(self):
        return self._start_key
    
    @property
    def end_key(self):
        return self._end_key
    
    @property
    def start_inclusive(self):
        return self._start_inclusive
    
    @property
    def end_inclusive(self):
        return self._end_inclusive


class RowSet:
    def __init__(self, row_keys: Iterable[bytes] = None, row_ranges: Iterable[RowRange] = None):
        if row_ranges is None:
            row_ranges = []
        if row_keys is None:
            row_keys = []
        self._row_keys = row_keys
        self._row_ranges = row_ranges
    
    @property
    def row_keys(self):
        return self._row_keys
    
    @row_keys.setter
    def row_keys(self, value):
        self._row_keys = value
    
    @property
    def row_ranges(self):
        return self._row_ranges
    
    @row_ranges.setter
    def row_ranges(self, value):
        self._row_ranges = value
    
    def add_row_range_from_keys(
            self,
            start_key=None,
            end_key=None,
            start_inclusive=True,
            end_inclusive=False
    ):
        self._row_ranges.append(
            RowRange(
                start_key,
                end_key,
                start_inclusive,
                end_inclusive
            )
        )
