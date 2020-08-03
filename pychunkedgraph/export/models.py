from dataclasses import dataclass
from datetime import datetime


@dataclass
class OperationLogBase:
    """
    Base class for log format.
    """

    id: str
    user: str
    timestamp: datetime
    status: int
    roots: list
    source_coords: list
    sink_coords: list
    old_roots: list
    old_roots_ts: list
    exception: str
    operation_ts: datetime

    def __init__(self, **kwargs):
        self.id = kwargs.get("id")
        self.user = kwargs.get("user")
        self.timestamp = kwargs.get("timestamp")
        self.status = kwargs.get("operation_status", 0)
        self.roots = kwargs.get("roots")
        self.source_coords = kwargs.get("source_coords")
        self.sink_coords = kwargs.get("sink_coords")
        self.old_roots = kwargs.get("old_roots")
        self.old_roots_ts = kwargs.get("old_roots_ts")
        self.exception = kwargs.get("operation_exception")
        # this was added recently
        # for older logs assume log_timestamp = operation_timestamp
        self.operation_ts = kwargs.get("operation_ts", self.timestamp)


class OperationLog:
    """
    Wrapper class for log format.
    """

    def __new__(cls, **kwargs):
        if "added_edges" in kwargs:
            return MergeLog(**kwargs)
        return SplitLog(**kwargs)


@dataclass
class MergeLog(OperationLogBase):
    """Log class for merge operation."""

    added_edges: list

    def __init__(self, **kwargs):
        added_edges = kwargs.pop("added_edges")
        super().__init__(**kwargs)
        self.added_edges = added_edges


@dataclass
class SplitLog(OperationLogBase):
    """Log class for split operation."""

    source_ids: list
    sink_ids: list
    bb_offset: list

    def __init__(self, **kwargs):
        source_ids = kwargs.pop("source_ids")
        sink_ids = kwargs.pop("sink_ids")
        bb_offset = kwargs.pop("bb_offset")
        super().__init__(**kwargs)
        self.source_ids = source_ids
        self.sink_ids = sink_ids
        self.bb_offset = bb_offset

