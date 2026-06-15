"""One-shot migration setup (pcgv3): prep an existing pcgv2 table for upgrade.

    python -m pychunkedgraph.pipeline.migrate.setup <graph_id>

Stamps the table version, adds the column family the upgrade writes need, and caches
earliest_ts in meta for the upgrade workers.
"""

import argparse

from pychunkedgraph import __version__

from ...graph import ChunkedGraph
from cave_pipeline.distribution import run_and_exit


def setup(graph_id: str) -> None:
    cg = ChunkedGraph(graph_id=graph_id)
    cg.client.add_table_version(__version__, overwrite=True)
    try:
        cg.client.create_column_family("4")
    except Exception:  # already present
        pass
    cg.meta.custom_data["earliest_ts"] = cg.get_earliest_timestamp().isoformat()
    cg.update_meta(cg.meta, overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pychunkedgraph.pipeline.migrate.setup")
    parser.add_argument("graph_id")
    args = parser.parse_args()
    setup(args.graph_id)


if __name__ == "__main__":
    run_and_exit(main)
