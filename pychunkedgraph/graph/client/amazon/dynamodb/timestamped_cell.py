import typing

from .utils import from_microseconds


class TimeStampedCell:
    def __init__(self, value: typing.Any, timestamp: int):
        self.value = value
        self.timestamp_int = timestamp
        self.timestamp = from_microseconds(timestamp)
    
    def __repr__(self):
        return f"<Cell value={repr(self.value)} timestamp={self.timestamp.isoformat(sep=' ')}>"
