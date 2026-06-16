"""Container entrypoint for one-shot mesh metadata setup.

Run once per graph after root-layer ingest::

    python -m pychunkedgraph.pipeline.meshing.setup <graph_id>

reads ``mesh_config:`` from the mounted dataset yaml (``PCG_DATASET``) and
delegates to :func:`pychunkedgraph.meshing.setup.setup_mesh_meta`, the single
source of truth for the mesh meta a graph needs before serving fragments.
"""

import argparse
from os import environ

import yaml

from cave_pipeline.distribution import run_and_exit

from ...graph.chunkedgraph import ChunkedGraph
from ...meshing.meta import MeshConfig
from ...meshing.setup import setup_mesh_meta

# Predetermined mount path of the dataset yaml (the chart mounts the dataset
# ConfigMap here); overridable for local/testing.
DATASET_PATH = environ.get("PCG_DATASET", "/app/datasets/dataset.yml")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pychunkedgraph.pipeline.meshing.setup")
    parser.add_argument("graph_id")
    args = parser.parse_args()
    with open(DATASET_PATH) as stream:
        config = yaml.safe_load(stream)
    if "mesh_config" not in config:
        raise SystemExit(
            f"{DATASET_PATH} has no `mesh_config:` block — required for mesh meta setup."
        )
    mesh_cfg = MeshConfig.from_dict(config["mesh_config"])
    cg = ChunkedGraph(graph_id=args.graph_id)
    result = setup_mesh_meta(cg, mesh_cfg)
    print(f"mesh meta written for {args.graph_id}: {result}")


if __name__ == "__main__":
    run_and_exit(main)
