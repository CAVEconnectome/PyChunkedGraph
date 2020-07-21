from typing import Dict
from typing import Optional
from typing import Iterable
from datetime import datetime

from google.cloud import datastore


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
            log_d[f"root_{i+1}"] = int(root)

        if not (old_roots and old_roots_ts):
            result.append(log_d)
            continue

        for i, root_info in enumerate(zip(old_roots, old_roots_ts)):
            log_d[f"old_root_{i+1}"] = int(root_info[0])
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


def export_operation_logs(graph_id: str, logs: Iterable) -> None:
    from .config import OperationLogsConfig

    config = OperationLogsConfig()
    logs = _create_col_for_each_root(logs)
    logs = _nested_lists_to_string(logs)

    client = datastore.Client()
    batch = client.batch()

    with batch:
        for log in logs:
            task_key = client.key(graph_id, log["id"], namespace=config.NAMESPACE)
            del log["id"]
            operation_log = datastore.Entity(
                key=task_key, exclude_from_indexes=config.EXCLUDE_FROM_INDICES
            )
            operation_log.update(log)
            batch.put(operation_log)
