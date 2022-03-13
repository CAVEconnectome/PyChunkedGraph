from typing import Dict
from typing import Optional
from typing import Iterable
from datetime import datetime

from google.cloud import datastore

from .config import OperationLogsConfig
from ...models import OperationLog
from .... import pcg_logger
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
    pcg_logger.log(pcg_logger.level, f"failed {count}")
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
    client = datastore.Client()

    query = client.query(kind=kind, namespace=namespace)
    keys = []
    for result in query.fetch():
        keys.append(result.key)

    for chunk in chunked(keys, 500):
        client.delete_multi(chunk)
    pcg_logger.log(pcg_logger.level, f"deleted {len(keys)} entities")


def _get_last_timestamp(
    client: datastore.Client,
    last_export_key: datastore.Key,
    operation_logs_config: OperationLogsConfig,
):
    from datetime import timedelta

    export_info = client.get(last_export_key)
    try:
        start_ts = export_info.get(operation_logs_config.EXPORT.LAST_EXPORT_TS)
        start_ts = datetime(*list(start_ts.utctimetuple()[:4]))
        start_ts -= timedelta(hours=1)
    except AttributeError:
        start_ts = None
    export_ts = datetime.now()
    # operation status changes while operation is running
    # export is at least an hour behind to ensure all logs have a finalized status.
    end_ts = datetime(*list(export_ts.utctimetuple()[:4]))
    end_ts -= timedelta(hours=1)
    if start_ts == end_ts:
        end_ts += timedelta(seconds=1)
    return start_ts, end_ts, export_ts


def _write_removed_edges(path: str, removed_edges: Iterable) -> None:
    """
    Store removed edges of an operation in a bucket.
    For some split operations there can be a large number of removed edges
    that can't be written to datastore due to size limitations.
    """
    if not len(removed_edges):
        return
    from json import dumps
    from cloudfiles import CloudFiles

    cf = CloudFiles(path)
    files_to_write = []
    for info in removed_edges:
        op_id, edges = info
        if not len(edges):
            continue
        files_to_write.append(
            {"content": dumps(edges), "path": f"{op_id}.gz", "compress": "gzip"}
        )
    cf.puts(files_to_write)


def export_operation_logs(
    cg: ChunkedGraph,
    start_ts: datetime = None,
    end_ts: datetime = None,
    namespace: str = None,
) -> int:
    """
    Main function to export logs to Datastore.
    Returns number of operations that failed to while persisting new IDs.
    """
    from os import environ
    from .config import DEFAULT_NS
    from .config import OperationLogsConfig
    from ... import operation_logs

    try:
        client = datastore.Client().from_service_account_json(
            environ["OPERATION_LOGS_DATASTORE_CREDENTIALS"]
        )
    except KeyError:
        # use GOOGLE_APPLICATION_CREDENTIALS
        # this is usually set to "/root/.cloudvolume/secrets/<some_secret>.json"
        client = datastore.Client()

    namespace_ = DEFAULT_NS
    if namespace:
        namespace_ = namespace
    config = OperationLogsConfig(NAMESPACE=namespace_)

    last_export_key = client.key(
        config.EXPORT.KIND, cg.graph_id, namespace=config.NAMESPACE
    )
    start_ts_, end_ts_, export_ts = _get_last_timestamp(client, last_export_key, config)
    if not start_ts:
        start_ts = start_ts_
    if not end_ts:
        end_ts = end_ts_

    pcg_logger.log(pcg_logger.level, f"getting logs from chunkedgraph {cg.graph_id}")
    pcg_logger.log(pcg_logger.level, f"start: {start_ts} end: {end_ts}")
    logs = operation_logs.get_parsed_logs(cg, start_time=start_ts, end_time=end_ts)
    logs = operation_logs.get_logs_with_previous_roots(cg, logs)
    logs = _create_col_for_each_root(logs)
    logs = _nested_lists_to_string(logs)
    pcg_logger.log(pcg_logger.level, f"total logs {len(logs)}")

    count = 0
    failed_count = 0
    # datastore limits 500 entities per request
    for chunk in chunked(logs, 500):
        entities = []
        removed_edges = []
        for log in chunk:
            kind = cg.graph_id
            if log["status"] == 4:
                kind = f"{cg.graph_id}_failed"
                failed_count += 1
            op_id = log.pop("id")
            op_log = datastore.Entity(
                key=client.key(kind, op_id, namespace=config.NAMESPACE),
                exclude_from_indexes=config.EXCLUDE_FROM_INDICES,
            )
            removed_edges.append((op_id, log.pop("removed_edges", [])))
            op_log.update(log)
            entities.append(op_log)
        client.put_multi(entities)
        count += len(entities)
        _write_removed_edges(f"{cg.meta.data_source.EDGES}/removed", removed_edges)
    _update_stats(cg.graph_id, client, config, last_export_key, count, export_ts)
    return failed_count


def _update_stats(
    graph_id: str,
    client: datastore.Client,
    config: OperationLogsConfig,
    last_export_key: datastore.Key,
    logs_count: int,
    export_ts: datetime,
):
    export_log = datastore.Entity(
        key=last_export_key, exclude_from_indexes=config.EXPORT.EXCLUDE_FROM_INDICES
    )
    export_log[config.EXPORT.LAST_EXPORT_TS] = export_ts
    export_log[config.EXPORT.LOGS_COUNT] = logs_count

    this_export_key = client.key(
        config.EXPORT.KIND,
        f"{graph_id}_{int(export_ts.timestamp())}",
        namespace=config.NAMESPACE,
    )
    this_export_log = datastore.Entity(
        key=this_export_key, exclude_from_indexes=config.EXPORT.EXCLUDE_FROM_INDICES
    )
    this_export_log[config.EXPORT.LAST_EXPORT_TS] = export_ts
    this_export_log[config.EXPORT.LOGS_COUNT] = logs_count

    client.put_multi([export_log, this_export_log])
    pcg_logger.log(pcg_logger.level, f"export time {export_ts}, count {logs_count}")
