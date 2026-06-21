"""One-shot ingest setup — create the graph table and write graph meta.

    python -m pychunkedgraph.pipeline.ingest.setup <graph_id> [--raw]

Reads the dataset yaml (mounted at ``PCG_DATASET``) and folds the agglomeration
source into ``meta.custom_data["agg"]`` so workers read it from the graph store.
"""

import argparse
from os import environ

import yaml
from kvdbclient import BigTableConfig

from ...graph import BackendClientInfo, ChunkedGraph
from ...graph.meta import ChunkedGraphMeta, DataSource, GraphConfig
from cave_pipeline.distribution import run_and_exit

# Predetermined mount path of the dataset yaml (the chart mounts the dataset
# ConfigMap here); overridable for local/testing.
DATASET_PATH = environ.get("PCG_DATASET", "/app/datasets/dataset.yml")


def setup(
    graph_id: str,
    raw: bool = False,
    exist_ok: bool = False,
    dataset_path: str = DATASET_PATH,
) -> None:
    with open(dataset_path) as stream:
        config = yaml.safe_load(stream)
    client_config = BigTableConfig(**config["backend_client"]["CONFIG"])
    client_info = BackendClientInfo(
        config["backend_client"].get("TYPE", "bigtable"), client_config
    )
    graph_config = GraphConfig(
        ID=str(graph_id), OVERWRITE=False, **config["graph_config"]
    )
    data_source = DataSource(**config["data_source"])
    agg = {"path": config.get("ingest_config", {}).get("AGGLOMERATION"), "raw": raw}
    meta = ChunkedGraphMeta(graph_config, data_source, custom_data={"agg": agg})
    cg = ChunkedGraph(meta=meta, client_info=client_info)
    try:
        cg.create()
    except ValueError:  # create() raises only when the table already exists
        if not exist_ok:
            raise
        print(f"graph '{graph_id}' already exists; skipping create")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pychunkedgraph.pipeline.ingest.setup")
    parser.add_argument("graph_id")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="raw agglomeration input; the L2 workers convert it to processed data",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="succeed (skip create) if the graph table already exists, for resumes",
    )
    args = parser.parse_args()
    setup(args.graph_id, raw=args.raw, exist_ok=args.exist_ok)


if __name__ == "__main__":
    run_and_exit(main)
