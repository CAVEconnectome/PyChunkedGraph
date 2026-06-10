"""Branch-aware shim: run one chunk's ingest body.

The ONLY branch-specific piece of this package. `main` builds L2 atomic chunks
via `add_atomic_edges` and parents via `add_layer`. The pcgv3 variant (OCDBT +
`create_parent_chunk` modes) is added here when the package is ported; the rest
of the package is identical across branches.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from ...ingest.common import get_atomic_chunk_data
from ...ingest.ran_agglomeration import get_active_edges
from ...ingest.create.atomic_layer import add_atomic_edges
from ...ingest.create.abstract_layers import add_layer


@dataclass
class IngestConfig:
    """Minimal atomic-chunk source config: agglomeration path + raw-input flags."""

    AGGLOMERATION: str = None
    USE_RAW_EDGES: bool = False
    USE_RAW_COMPONENTS: bool = False


def process_chunk(
    cg, layer: int, coord: Sequence[int], config: IngestConfig, n_threads: int = 1
) -> None:
    """Ingest a single chunk: L2 atomic edges, or L>2 parent agglomeration."""
    coord = np.array(list(coord), dtype=int)
    if layer == 2:
        _add_atomic_chunk(cg, coord, config)
    else:
        add_layer(cg, layer, coord, n_threads=n_threads)


def _add_atomic_chunk(cg, coord: np.ndarray, config: IngestConfig) -> None:
    # get_atomic_chunk_data only needs .config/.cg/.cg_meta — no Redis-backed manager.
    manager = SimpleNamespace(config=config, cg=cg, cg_meta=cg.meta)
    chunk_edges_all, mapping = get_atomic_chunk_data(manager, coord)
    chunk_edges_active, isolated_ids = get_active_edges(chunk_edges_all, mapping)
    add_atomic_edges(cg, coord, chunk_edges_active, isolated=isolated_ids)
