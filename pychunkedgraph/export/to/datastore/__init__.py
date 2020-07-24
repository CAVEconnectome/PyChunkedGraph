from typing import Dict
from typing import Optional
from typing import Iterable
from datetime import datetime

from google.cloud import datastore

from ....graph import ChunkedGraph


def _create_col_for_each_root(parsed_logs: Iterable) -> Iterable[Dict]:
    """
    Creates a new column for each old and new roots of an operation.
    This makes querying easier. For eg, a split operation yields 2 new roots:
    old_roots = [123] -> old_root1_col = 123
    new_roots = [124,125] -> new_root1_col = 124, new_root2_col = 125
    """
    from dataclasses import asdict

    result = []
    for log in parsed_logs:
        log_d = asdict(log)
        roots = log_d.pop("roots")
        old_roots = log_d.pop("old_roots")
        old_roots_ts = log_d.pop("old_roots_ts")

        for i, root in enumerate(roots):
            log_d[f"root_{i+1}"] = str(root)

        if not (old_roots and old_roots_ts):
            result.append(log_d)
            continue

        for i, root_info in enumerate(zip(old_roots, old_roots_ts)):
            log_d[f"old_root_{i+1}"] = str(root_info[0])
            log_d[f"old_root_{i+1}_ts"] = root_info[1]
        result.append(log_d)
    return result


def _nested_lists_to_string(parsed_logs: Iterable[Dict]) -> Iterable[Dict]:
    result = []
    for log in parsed_logs:
        for k, v in log.items():
            if isinstance(v, list):
                log[k] = str(v).replace(" ", "")
        result.append(log)
    return result


def export_operation_logs(cg: ChunkedGraph, timestamp: datetime = None) -> None:
    """
    Exports operation logs from the given timestamp.
    If timestamp is None, logs since the last export are fetched.
        Timestamp of last export is stored in an entity, along with other information.
        Kind for this entity is `export_info` and key is `cg.graph_id`.
    """
    from .config import OperationLogsConfig
    from ... import operation_logs

    client = datastore.Client()
    config = OperationLogsConfig()

    export_key = client.key(config.EXPORT.KIND, cg.graph_id, namespace=config.NAMESPACE)
    if not timestamp:
        export_info = client.get(export_key)
        try:
            timestamp = export_info.get(config.EXPORT.LAST_EXPORT_TS)
        except AttributeError:
            timestamp = None

    logs = operation_logs.get_parsed_logs(cg, start_time=timestamp)
    if not logs:
        # nothing to do
        return
    # next export starts from this timestamp
    # unless overridden by passing custom timestamp
    new_timestamp = datetime.now()
    logs = operation_logs.get_logs_with_previous_roots(cg, logs)
    logs = _create_col_for_each_root(logs)
    logs = _nested_lists_to_string(logs)

    batch = client.batch()

    with batch:
        for log in logs:
            task_key = client.key(cg.graph_id, log["id"], namespace=config.NAMESPACE)
            del log["id"]
            operation_log = datastore.Entity(
                key=task_key, exclude_from_indexes=config.EXCLUDE_FROM_INDICES
            )
            operation_log.update(log)
            batch.put(operation_log)
        export_log = datastore.Entity(key=export_key)
        export_log[config.EXPORT.LAST_EXPORT_TS] = new_timestamp
        export_log[config.EXPORT.LOGS_COUNT] = len(logs)
        batch.put(export_log)

