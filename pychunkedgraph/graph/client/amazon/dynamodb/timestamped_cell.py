import typing
import time
from datetime import datetime, timezone


class TimeStampedCell:
    def __init__(self, value: typing.Any, timestamp: int):
        self.value = value
        self.timestamp = timestamp
    
    def __repr__(self):
        return f"<Cell value={repr(self.value)} timestamp={self.to_datetime().isoformat(sep=' ')}>"
    
    def to_datetime(self):
        return datetime.fromtimestamp(self.timestamp / 1000000, tz=timezone.utc)
    
    @staticmethod
    def get_current_time_microseconds():
        time_seconds = time.time()
        time_microseconds = time_seconds * 1000 * 1000
        return time_microseconds
