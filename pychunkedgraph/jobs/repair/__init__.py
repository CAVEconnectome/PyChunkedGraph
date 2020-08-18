"""
Re run failed edit operations.

These jobs get data (failed operations) from Google Datastore.
"""


def _read_failed_logs(graph_id: str = None, datastore_ns: str = None):
    from os import environ
    from datetime import datetime
    from datetime import timedelta
    from google.cloud import datastore
    from pychunkedgraph.graph import ChunkedGraph
    from pychunkedgraph.graph.operation import GraphEditOperation

    if not graph_id:
        graph_id = environ["GRAPH_ID"]
    if not datastore_ns:
        datastore_ns = environ.get("DATASTORE_NS")

    try:
        client = datastore.Client().from_service_account_json(
            environ["OPERATION_LOGS_DATASTORE_CREDENTIALS"]
        )
    except KeyError:
        print("Datastore credentials not provided.")
        print(f"Using {environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        # use GOOGLE_APPLICATION_CREDENTIALS
        # this is usually "/root/.cloudvolume/secrets/<some_secret>.json"
        client = datastore.Client()

    query = client.query(kind=f"{graph_id}_failed", namespace=datastore_ns)
    cg = ChunkedGraph(graph_id=graph_id)
    for log in query.fetch():
        operation = GraphEditOperation.from_operation_id(
            cg, log.id, multicut_as_split=False
        )
        ts = log["timestamp"]

        print(f"Re-trying operation ID {log.id}")
        operation.execute(
            operation_id=log.id,
            override_ts=ts + timedelta(microseconds=(ts.microsecond % 1000) + 10),
            last_successful_ts=ts - timedelta(seconds=1),
        )
        client.delete(log.key)
        break


def repair_operations():
    _read_failed_logs()


if __name__ == "__main__":
    repair_operations()
