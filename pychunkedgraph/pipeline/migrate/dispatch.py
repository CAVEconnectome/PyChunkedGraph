"""Upgrade one chunk for the pcgv2 -> pcgv3 migration — the branch-specific builder.

Idempotent, so no per-chunk lock. The upgrade bodies keep module-global ``CHILDREN``
and ``CX_EDGES``; clear them per chunk so a pod's batch does not accumulate state.
"""

from typing import Sequence

from ...ingest.upgrade import atomic_layer, parent_layer


def process_chunk(
    cg, layer: int, coord: Sequence[int], clean: bool = False, n_threads: int = 1
) -> None:
    """Upgrade a single chunk: L2 atomic IDs, or L>2 parent IDs."""
    coord = list(map(int, coord))
    if layer == 2:
        atomic_layer.CHILDREN.clear()
        atomic_layer.update_chunk(cg, coord, clean=clean)
    else:
        parent_layer.CHILDREN.clear()
        parent_layer.CX_EDGES.clear()
        parent_layer.update_chunk(cg, coord, layer, clean=clean)
