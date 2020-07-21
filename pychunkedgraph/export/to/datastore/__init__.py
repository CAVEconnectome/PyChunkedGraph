from typing import Optional
from typing import Iterable
from datetime import datetime
from collections import namedtuple

from google.cloud import datastore

from ...operation_logs import get_parsed_logs
from ....graph import ChunkedGraph


_operation_log_fields = ("NAMESPACE", "PARENT_KEY")
_operation_log_defaults = ("pychunkedgraph_logs", "OperationLogs")
OperationLogsConfig = namedtuple(
    "OperationLogsConfig", _operation_log_fields, defaults=_operation_log_defaults,
)


def export_operation_logs(cg: ChunkedGraph) -> None:
    client = datastore.Client()
    log = get_parsed_logs(cg)[0]

    print(log)

    config = OperationLogsConfig()
    parent_key = client.key(config.PARENT_KEY, cg.graph_id, namespace=config.NAMESPACE)

    # The Cloud Datastore key for the new entity
    task_key = client.key(log["type"], int(log["id"]), parent=parent_key)
    del log["id"]
    del log["type"]

    # Prepares the new entity
    operation_log = datastore.Entity(key=task_key)
    operation_log.update(log)

    # Saves the entity
    client.put(operation_log)
