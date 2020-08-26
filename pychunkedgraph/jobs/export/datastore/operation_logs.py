"""
Export operation log data to Datastore.

Accepts a comma separated list of graph IDs
as command line argument or env variable `GRAPH_IDS`
"""
from typing import List
from typing import Optional
from os import environ

from ....graph import ChunkedGraph


def _get_chunkedgraphs(graph_ids: Optional[str] = None) -> List[ChunkedGraph]:
    """
    `graph_ids` comma separated list of graph IDs
    If None, load from env variable `GRAPH_IDS`
    """
    if not graph_ids:
        try:
            graph_ids = environ["GRAPH_IDS"]
        except KeyError:
            raise KeyError("Environment variable `GRAPH_IDS` is required.")

    chunkedgraphs = []
    for id_ in [id_.strip() for id_ in graph_ids.split(",")]:
        chunkedgraphs.append(ChunkedGraph(graph_id=id_))
    return chunkedgraphs


def run_export(chunkedgraphs: List[ChunkedGraph], datastore_ns: str = None) -> None:
    """
    Export job for processing edit logs and storing them in Datastore.
    Default namespace: pychunkedgraph.export.to.datastore.config.DEFAULT_NS

    Sends an email alert when there are failed writes.
    These cases must be inspected manually and are expected to be rare.
    """
    from ...alert import send_email
    from ....export.to.datastore import export_operation_logs

    if not datastore_ns:
        datastore_ns = environ.get("DATASTORE_NS")

    for cg in chunkedgraphs:
        print(f"Start log export job for {cg.graph_id}")
        failed = export_operation_logs(cg, namespace=datastore_ns)
        if failed:
            print(f"Failed writes {failed}")
        alert_emails = environ["EMAIL_LIST_FAILED_WRITES"]
        send_email(
            alert_emails.split(","),
            "Failed Writes",
            f"TEST: There are {failed} failed writes for {cg.graph_id}.",
        )


if __name__ == "__main__":
    from sys import argv

    assert len(argv) <= 3

    graph_ids = None
    try:
        graph_ids = argv[1]
    except IndexError:
        print("`graph_ids` not provided, using env variable `GRAPH_IDS`")

    datastore_ns = None
    try:
        datastore_ns = argv[2]
    except IndexError:
        print("`datastore_namespace` not provided, using env variable `DATASTORE_NS`.")
        print(
            "Default `pychunkedgraph.export.to.datastore.config.DEFAULT_NS`\
             if env variable `DATASTORE_NS` is not provided."
        )

    run_export(_get_chunkedgraphs(graph_ids=graph_ids), datastore_ns=datastore_ns)
