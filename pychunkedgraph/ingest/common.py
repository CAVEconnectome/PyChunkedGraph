from typing import Dict
from typing import Tuple
from typing import Sequence

from .manager import IngestionManager
from .ran_agglomeration import read_raw_edge_data
from .ran_agglomeration import read_raw_agglomeration_data
from ..graph import ChunkedGraph
from ..io.edges import get_chunk_edges
from ..io.components import get_chunk_components


def get_atomic_chunk_data(
    imanager: IngestionManager, coord: Sequence[int]
) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        read_raw_edge_data(imanager, coord)
        if imanager.config.USE_RAW_EDGES
        else get_chunk_edges(imanager.cg_meta.data_source.EDGES, [coord])
    )

    _check_edges_direction(chunk_edges, imanager.cg, coord)

    mapping = (
        read_raw_agglomeration_data(imanager, coord)
        if imanager.config.USE_RAW_COMPONENTS
        else get_chunk_components(imanager.cg_meta.data_source.COMPONENTS, coord)
    )
    return chunk_edges, mapping


def _check_edges_direction(
    chunk_edges: dict, cg: ChunkedGraph, coord: Sequence[int]
) -> None:
    """
    For between and cross chunk edges:
    Checks and flips edges such that nodes1 are always within a chunk and nodes2 outside the chunk.
    Where nodes1 = edges[:,0] and nodes2 = edges[:,1].
    """
    import numpy as np
    from ..graph.edges import Edges
    from ..graph.edges import EDGE_TYPES

    x, y, z = coord
    chunk_id = cg.get_chunk_id(layer=1, x=x, y=y, z=z)
    for edge_type in [EDGE_TYPES.between_chunk, EDGE_TYPES.cross_chunk]:
        edges = chunk_edges[edge_type]
        e1 = edges.node_ids1
        e2 = edges.node_ids2

        e2_chunk_ids = cg.get_chunk_ids_from_node_ids(e2)
        mask = e2_chunk_ids == chunk_id
        e1[mask], e2[mask] = e2[mask], e1[mask]

        e1_chunk_ids = cg.get_chunk_ids_from_node_ids(e1)
        mask = e1_chunk_ids == chunk_id
        assert np.all(mask), "all IDs must belong to same chunk"
