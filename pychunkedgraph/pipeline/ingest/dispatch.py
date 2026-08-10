"""Run one chunk's ingest body — the branch-specific builder.

pcgv3 builds L2 atomic chunks via ``add_atomic_chunk`` and parents via ``add_parent_chunk``.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from ...graph.edges import EDGE_TYPES
from ...ingest.create.atomic_layer import add_atomic_chunk
from ...ingest.create.parent_layer import add_parent_chunk
from ...ingest.ran_agglomeration import (
    get_active_edges,
    read_raw_agglomeration_data,
    read_raw_edge_data,
)
from ...io.components import get_chunk_components
from ...io.edges import get_chunk_edges


@dataclass
class IngestConfig:
    """Minimal atomic-chunk source config: agglomeration path + raw-input flags."""

    AGGLOMERATION: str = None
    USE_RAW_EDGES: bool = False
    USE_RAW_COMPONENTS: bool = False


def process_chunk(
    cg, layer: int, coord: Sequence[int], config: IngestConfig, n_processes: int = 1
) -> None:
    """Ingest a single chunk: L2 atomic edges, or L>2 parent agglomeration."""
    coord = np.array(list(coord), dtype=int)
    if layer == 2:
        _add_atomic_chunk(cg, coord, config)
    else:
        add_parent_chunk(cg, layer, coord, n_processes=n_processes)


def _add_atomic_chunk(cg, coord: np.ndarray, config: IngestConfig) -> None:
    # read_raw_* read .config / .cg_meta off the manager.
    manager = SimpleNamespace(config=config, cg=cg, cg_meta=cg.meta)
    chunk_edges_all = (
        read_raw_edge_data(manager, coord)
        if config.USE_RAW_EDGES
        else get_chunk_edges(cg.meta.data_source.EDGES, [coord])
    )
    _check_edges_direction(chunk_edges_all, cg, coord)
    mapping = (
        read_raw_agglomeration_data(manager, coord)
        if config.USE_RAW_COMPONENTS
        else get_chunk_components(cg.meta.data_source.COMPONENTS, coord)
    )
    chunk_edges_active, isolated_ids = get_active_edges(chunk_edges_all, mapping)
    add_atomic_chunk(cg, coord, chunk_edges_active, isolated_ids)


def _check_edges_direction(chunk_edges: dict, cg, coord: Sequence[int]) -> None:
    """Cross/between edges must have nodes1 inside the chunk (mirrors ingest.cluster)."""
    x, y, z = coord
    chunk_id = cg.get_chunk_id(layer=1, x=x, y=y, z=z)
    for edge_type in [EDGE_TYPES.between_chunk, EDGE_TYPES.cross_chunk]:
        edges = chunk_edges[edge_type]
        chunk_ids = cg.get_chunk_ids_from_node_ids(edges.node_ids1)
        assert np.all(chunk_ids == chunk_id), "all IDs must belong to same chunk"
