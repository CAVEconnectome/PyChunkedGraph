from collections import namedtuple


_export_info_fields = ("KIND", "LOGS_COUNT", "LAST_EXPORT_TS", "EXCLUDE_FROM_INDICES")
_export_info_defaults = (
    "export_info",
    "logs_count",
    "last_export_ts",
    ("logs_count", "last_export_ts",),
)
ExportInfo = namedtuple(
    "ExportInfo", _export_info_fields, defaults=_export_info_defaults,
)


_operation_log_fields = ("NAMESPACE", "EXPORT", "EXCLUDE_FROM_INDICES")
_operation_log_defaults = (
    "pychunkedgraph_operation_logs",
    ExportInfo(),
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
