from typing import Dict
from typing import Optional
from typing import Iterable
from datetime import datetime

from google.cloud import datastore

from .config import OperationLogsConfig
from ...models import OperationLog
from ....graph import ChunkedGraph
from ....utils.general import chunked


def _create_col_for_each_root(parsed_logs: Iterable[OperationLog]) -> Iterable[Dict]:
    """
    Creates a new column for each old and new roots of an operation.
    This makes querying easier. For eg, a split operation yields 2 new roots:
    old_roots = [123] -> old_root1_col = 123
    new_roots = [124,125] -> new_root1_col = 124, new_root2_col = 125
    """
    from dataclasses import asdict

    count = 0

    result = []
    for log in parsed_logs:
        if log.status == 4:
            count += 1
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
    print(f"failed {count}")
    # if count > int(len(result) * 0.2):
    #     raise ValueError(
    #         f"Something's wrong I can feel it, failed count {count}/{len(result)}"
    #     )
    return result


def _nested_lists_to_string(parsed_logs: Iterable[Dict]) -> Iterable[Dict]:
    result = []
    for log in parsed_logs:
        for k, v in log.items():
            if isinstance(v, list):
                log[k] = str(v).replace(" ", "")
        result.append(log)
    return result


def delete_entities(namespace: str, kind: str) -> None:
    """
    Deletes all entities of the given kind in given namespace.
    Use this only when you need to "clean up".
    """
    from google.cloud import datastore
    from datetime import timedelta

    client = datastore.Client()

    query = client.query(kind=kind, namespace=namespace)
    keys = []
    for result in query.fetch():
        keys.append(result.key)

    for chunk in chunked(keys, 500):
        client.delete_multi(chunk)


def _get_last_timestamp(
    client: datastore.Client,
    last_export_key: datastore.Key,
    operation_logs_config: OperationLogsConfig,
):
    from datetime import timedelta

    new_timestamp = None
    export_info = client.get(last_export_key)
    try:
        timestamp_ = export_info.get(operation_logs_config.EXPORT.LAST_EXPORT_TS)
        # set window to hour of the previous timestamp
        timestamp_ = datetime(*list(timestamp_.utctimetuple()[:4]))
        new_timestamp = timestamp_ + timedelta(hours=1)
    except AttributeError:
        timestamp_ = None
    # to make sure no logs are missed for next export
    # (new logs that could be generated during the process)
    # use timestamp before getting logs
    if not new_timestamp:
        # when export is run for the first time, there will be no new timestamp
        new_timestamp = datetime.now()
    return timestamp_, new_timestamp


def export_operation_logs(
    cg: ChunkedGraph, timestamp: datetime = None, namespace: str = None,
) -> None:
    """
    Exports operation logs from the given timestamp.
    If timestamp is None, logs since the last export are fetched.
        Timestamp of last export is stored in an entity, along with other information.
        Kind for this entity is `export_info` and key is `cg.graph_id`.
    """
    from .config import OperationLogsConfig
    from ... import operation_logs

    client = datastore.Client()
    namespace_ = "pychunkedgraph_operation_logs"
    if namespace:
        namespace_ = namespace
    config = OperationLogsConfig(NAMESPACE=namespace_)

    # next export starts from this timestamp
    # unless overridden by passing custom timestamp
    new_timestamp = None
    last_export_key = client.key(
        config.EXPORT.KIND, cg.graph_id, namespace=config.NAMESPACE
    )

    timestamp_, new_timestamp = _get_last_timestamp(client, last_export_key, config)
    if not timestamp:
        timestamp = timestamp_

    logs = operation_logs.get_parsed_logs(cg, start_time=timestamp)
    logs = operation_logs.get_logs_with_previous_roots(cg, logs)
    logs = _create_col_for_each_root(logs)
    logs = _nested_lists_to_string(logs)

    # datastore limits 500 entities per request
    print(f"total logs {len(logs)}")
    count = 0
    for chunk in chunked(logs, 500):
        entities = []
        for log in chunk:
            kind = cg.graph_id
            if log["status"] == 4:
                kind = f"{cg.graph_id}_failed"
            op_log = datastore.Entity(
                key=client.key(kind, log.pop("id"), namespace=config.NAMESPACE),
                exclude_from_indexes=config.EXCLUDE_FROM_INDICES,
            )
            op_log.update(log)
            entities.append(op_log)
        client.put_multi(entities)
        count += len(entities)
        print(f"exported {count}")
    _update_stats(cg.graph_id, client, config, last_export_key, count, new_timestamp)


def _update_stats(
    graph_id: str,
    client: datastore.Client,
    config: OperationLogsConfig,
    last_export_key: datastore.Key,
    logs_count: int,
    new_timestamp: datetime,
):
    export_log = datastore.Entity(
        key=last_export_key, exclude_from_indexes=config.EXPORT.EXCLUDE_FROM_INDICES
    )
    export_log[config.EXPORT.LAST_EXPORT_TS] = new_timestamp
    export_log[config.EXPORT.LOGS_COUNT] = logs_count

    this_export_key = client.key(
        config.EXPORT.KIND,
        f"{graph_id}_{int(new_timestamp.timestamp())}",
        namespace=config.NAMESPACE,
    )
    this_export_log = datastore.Entity(
        key=this_export_key, exclude_from_indexes=config.EXPORT.EXCLUDE_FROM_INDICES
    )
    this_export_log[config.EXPORT.LAST_EXPORT_TS] = new_timestamp
    this_export_log[config.EXPORT.LOGS_COUNT] = logs_count

    client.put_multi([export_log, this_export_log])

