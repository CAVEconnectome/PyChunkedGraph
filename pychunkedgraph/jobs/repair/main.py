"""
Re run failed edit operations.
These jobs get data (failed operations) from Google Datastore.
"""

from ... import pcg_logger
from ...graph import ChunkedGraph
from ...export.models import OperationLog


def _repair_operation(cg: ChunkedGraph, log: OperationLog):
    from datetime import timedelta
    from ...graph.operation import GraphEditOperation

    operation = GraphEditOperation.from_operation_id(
        cg, log.id, multicut_as_split=False, privileged_mode=True
    )
    ts = log["timestamp"]
    result = operation.execute(
        operation_id=log.id,
        parent_ts=ts - timedelta(seconds=0.1),
        override_ts=ts + timedelta(microseconds=(ts.microsecond % 1000) + 10),
    )

    old_roots = operation._update_root_ids()
    pcg_logger.log(pcg_logger.level, f"old {old_roots} new {result.new_root_ids}")
    for root_ in old_roots:
        cg.client.unlock_indefinitely_locked_root(root_, result.operation_id)


def _repair_failed_operations(graph_id: str = None, datastore_ns: str = None):
    from os import environ
    from google.cloud import datastore

    # if not graph_id:
    #     graph_id = environ["GRAPH_IDS"]
    # if not datastore_ns:
    #     datastore_ns = environ.get("DATASTORE_NS")

    graph_id = "minnie3_v1"
    datastore_ns = "pcg_test"

    try:
        client = datastore.Client().from_service_account_json(
            environ["OPERATION_LOGS_DATASTORE_CREDENTIALS"]
        )
    except KeyError:
        pcg_logger.log(
            pcg_logger.level,
            "Datastore credentials not provided."
            f"Using {environ['GOOGLE_APPLICATION_CREDENTIALS']}",
        )
        client = datastore.Client()

    query = client.query(kind=f"{graph_id}_failed", namespace=datastore_ns)
    cg = ChunkedGraph(graph_id=graph_id)
    for log in query.fetch():
        pcg_logger.log(pcg_logger.level, f"Re-trying operation ID {log.id}")
        _repair_operation(cg, log)
        client.delete(log.key)


def repair_operations():
    _repair_failed_operations()


if __name__ == "__main__":
    repair_operations()
