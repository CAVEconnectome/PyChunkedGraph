"""One-shot ingest setup — create the Bigtable table and write graph meta.

Run once per graph, inside the image, before any layer Jobs:
    python -m pychunkedgraph.pipeline.ingest.setup <graph_id> [--raw]

Reads the dataset yaml from its mounted location (no path passed); this is the
only step that touches the yaml. Folds the agglomeration source into
``meta.custom_data["agg"] = {"path": str, "raw": bool}`` so workers read
everything from Bigtable at run time — no yaml, no Redis. Errors out if the
table already exists (the operator runs setup explicitly once).
"""

import argparse
from os import environ

import yaml

from ...graph import ChunkedGraph
from ...graph.client import BackendClientInfo
from ...graph.client.bigtable import BigTableConfig
from ...graph.meta import ChunkedGraphMeta, DataSource, GraphConfig

# Predetermined mount path of the dataset yaml (the chart mounts the dataset
# ConfigMap here); overridable for local/testing.
DATASET_PATH = environ.get("PCG_DATASET", "/app/datasets/dataset.yml")


def setup(graph_id: str, raw: bool = False, dataset_path: str = DATASET_PATH) -> None:
    with open(dataset_path) as stream:
        config = yaml.safe_load(stream)
    client_config = BigTableConfig(**config["backend_client"]["CONFIG"])
    client_info = BackendClientInfo(config["backend_client"]["TYPE"], client_config)
    graph_config = GraphConfig(ID=str(graph_id), OVERWRITE=False, **config["graph_config"])
    data_source = DataSource(**config["data_source"])
    agg = {"path": config.get("ingest_config", {}).get("AGGLOMERATION"), "raw": raw}
    meta = ChunkedGraphMeta(graph_config, data_source, custom_data={"agg": agg})
    cg = ChunkedGraph(meta=meta, client_info=client_info)
    cg.create()


def main() -> None:
    parser = argparse.ArgumentParser(prog="pychunkedgraph.pipeline.ingest.setup")
    parser.add_argument("graph_id")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="raw agglomeration input; the L2 workers convert it to processed data",
    )
    args = parser.parse_args()
    setup(args.graph_id, raw=args.raw)


if __name__ == "__main__":
    main()
