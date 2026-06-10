"""Mesh per-chunk processor: marching cubes at L2, sharded stitching above.

Mirrors ``meshing_sqs.MeshTask.execute``. Idempotent (overwrites shards), so it
needs no per-chunk lock. Plugged into the generic ``pipeline.worker`` harness. ``mip``
comes from the mesh meta written by setup; a per-chunk failure counts transient so
the batch retries it.
"""

import logging
import os

from ...meshing import meshgen
from ..worker import run

logger = logging.getLogger(__name__)


def make_processor(cg, layer, env):
    """Build the mesh per-chunk processor for this batch."""
    mesh_meta = cg.meta.custom_data.get("mesh") or {}
    mip = int(mesh_meta.get("mip", 0))
    cache = os.environ.get("PCG_MESH_CACHE", "1") != "0"

    def process_one(coord):
        chunk_id = int(cg.get_chunk_id(layer=layer, x=coord[0], y=coord[1], z=coord[2]))
        try:
            if layer == 2:
                meshgen.chunk_initial_mesh_task(
                    cg.graph_id, chunk_id, None, mip=mip, sharded=True, cache=cache
                )
            else:
                meshgen.chunk_initial_sharded_stitching_task(
                    cg.graph_id, chunk_id, mip, cache=cache
                )
            return "ok"
        except Exception:
            logger.exception(f"mesh failure on chunk {layer}_{tuple(coord)}")
            return "transient"

    return process_one


def main() -> int:
    return run(make_processor)


if __name__ == "__main__":
    raise SystemExit(main())
