from collections import namedtuple

_operation_log_fields = ("NAMESPACE", "EXCLUDE_FROM_INDICES")
_operation_log_defaults = (
    "pychunkedgraph_operation_logs",
    (
        "added_edges",
        "source_coords",
        "sink_coords",
        "exception",
        "source_ids",
        "sink_ids",
        "bb_offset",
    ),
)
OperationLogsConfig = namedtuple(
    "OperationLogsConfig", _operation_log_fields, defaults=_operation_log_defaults,
)
