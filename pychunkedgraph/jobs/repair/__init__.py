"""
Re run failed edit operations.

These jobs currently get data (failed operations) from Google Datastore.
"""


def _read_failed_logs(graph_id: str = None, datastore_ns: str = None):
    from os import environ
    from google.cloud import datastore
    from pychunkedgraph.graph import ChunkedGraph
    from pychunkedgraph.export.models import OperationLog
    from pychunkedgraph.export.to.datastore import export_operation_logs

    if not graph_id:
        graph_id = environ["GRAPH_ID"]
    if not datastore_ns:
        datastore_ns = environ.get("DATASTORE_NS")

    try:
        client = datastore.Client().from_service_account_json(
            environ["OPERATION_LOGS_DATASTORE_CREDENTIALS"]
        )
    except KeyError:
        # use GOOGLE_APPLICATION_CREDENTIALS
        # this is usually set to "/root/.cloudvolume/secrets/<some_secret>.json"
        client = datastore.Client()
    query = client.query(kind=f"{graph_id}_failed", namespace=datastore_ns)

    failed_operations = []
    for log in query.fetch():
        failed_operations.append(OperationLog(**log))
        print(OperationLog(**log))
        print()

    print(f"failed count {len(failed_operations)}")
    # cg = ChunkedGraph(graph_id=graph_id)

    # ret = cg.add_edges(
    #     user_id=user_id,
    #     atomic_edges=np.array(atomic_edge, dtype=np.uint64),
    #     source_coords=coords[:1],
    #     sink_coords=coords[1:],
    # )

    # ret = cg.remove_edges(
    #     user_id=user_id,
    #     source_ids=data_dict["sources"]["id"],
    #     sink_ids=data_dict["sinks"]["id"],
    #     source_coords=data_dict["sources"]["coord"],
    #     sink_coords=data_dict["sinks"]["coord"],
    #     mincut=True,
    # )


def repair_operations():
    _read_failed_logs()


if __name__ == "__main__":
    repair_operations()
