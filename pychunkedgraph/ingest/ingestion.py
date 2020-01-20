from typing import Dict
from typing import Tuple

import numpy as np

from .types import ChunkTask
from .manager import IngestionManager
from .backward_compat import get_chunk_data as get_chunk_data_old_format
from .ran_agglomeration import read_raw_edge_data
from .ran_agglomeration import read_raw_agglomeration_data
from ..io.edges import get_chunk_edges
from ..io.components import get_chunk_components


def create_atomic_chunk_helper(
    im_info: Dict, task: ChunkTask,
):
    """Helper to queue atomic chunk task."""
    imanager = IngestionManager(**im_info)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, task.coords)
    ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)

    retry = 5
    while retry:
        try:
            imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)
            retry = 0
        except:
            retry -= 1
    return task


def create_parent_chunk_helper(
    im_info: Dict, task: ChunkTask,
):
    """Helper to queue parent chunk task."""
    imanager = IngestionManager(**im_info)
    retry = 5
    while retry:
        try:
            imanager.cg.add_layer(task.layer, task.children_coords)
            retry = 0
        except:
            retry -= 1
    return task


def _get_atomic_chunk_data(
    imanager: IngestionManager, coord: np.ndarray
) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        read_raw_edge_data(imanager, coord)
        if imanager.cg_meta.data_source.use_raw_edges
        else get_chunk_edges(imanager.cg_meta.data_source.edges, [coord])
    )
    mapping = (
        read_raw_agglomeration_data(imanager, coord)
        if imanager.cg_meta.data_source.use_raw_components
        else get_chunk_components(imanager.cg_meta.data_source.components, coord)
    )
    return chunk_edges, mapping

