"""Migrate per-chunk processor: upgrade a chunk in place.

Idempotent (overwrites), so no per-chunk lock. ``--clean`` runs the cleanup pass
(fix corrupt nodes); otherwise the main upgrade. Plugged into ``pipeline.worker``.
"""

import argparse
import logging

from cave_pipeline.distribution.harness import run

from .. import cg_factory, layer_bounds
from . import dispatch

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="pychunkedgraph.pipeline.migrate")
    parser.add_argument(
        "--clean", action="store_true", help="cleanup pass: fix corrupt nodes only"
    )
    clean = parser.parse_args().clean

    def make_processor(cg, layer, env):
        def process_one(coord):
            try:
                dispatch.process_chunk(
                    cg, layer, coord, clean=clean, n_processes=env["n_processes"]
                )
                return "ok"
            except Exception:
                logger.exception(f"migrate failure on chunk {layer}_{tuple(coord)}")
                return "transient"

        return process_one

    return run(make_processor, context_factory=cg_factory, bounds_fn=layer_bounds)


if __name__ == "__main__":
    raise SystemExit(main())
