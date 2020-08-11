"""
Export data to Google Datastore.
"""


def export_operation_logs(graph_id: str = None, datastore_ns: str = None):
    """
    Export job for processing edit logs and storing them in Datastore.
    
    If `graph_id` is passed, env var `GRAPH_ID` is ignored.
    One of `graph_id` or `GRAPH_ID` is required.
    
    Same applies for `datastore_ns` (namespace), but it is optional.
    Default ns: "pychunkedgraph_operation_logs"
    """
    from os import environ
    from pychunkedgraph.graph import ChunkedGraph
    from pychunkedgraph.export.to.datastore import export_operation_logs

    if not graph_id:
        graph_id = environ["GRAPH_ID"]
    if not datastore_ns:
        datastore_ns = environ.get("DATASTORE_NS")

    cg = ChunkedGraph(graph_id=graph_id)
    export_operation_logs(cg, namespace=datastore_ns)


if __name__ == "__main__":
    export_operation_logs()
