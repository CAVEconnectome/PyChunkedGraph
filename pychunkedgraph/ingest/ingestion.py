import time
from typing import Dict
from typing import Tuple
from multiprocessing import RLock

import numpy as np

from .types import ChunkTask
from .manager import IngestionManager
from .backward_compat import get_chunk_data as get_chunk_data_old_format
from .ran_agglomeration import read_raw_edge_data
from .ran_agglomeration import read_raw_agglomeration_data
from .ran_agglomeration import get_active_edges
from ..io.edges import get_chunk_edges
from ..io.edges import put_chunk_edges
from ..io.components import get_chunk_components
from ..utils.general import chunked
from ..backend.chunks.hierarchy import get_children_coords

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def get_parent_task(
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: RLock,
    imanager: IngestionManager,
    layer: int,
    coords: np.ndarray,
):
    parent_layer = layer + 1
    parent_coords = np.array(coords, int) // imanager.cg_meta.graph_config.fanout
    if parent_layer > imanager.cg_meta.layer_count:
        return ChunkTask(imanager.cg_meta, parent_coords, parent_layer)

    parent_chunk_str = chunk_id_str(parent_layer, parent_coords)

    with parent_children_count_d_lock:
        if not parent_chunk_str in parent_children_count_d_shared:
            children_count = len(
                get_children_coords(imanager.cg_meta, parent_layer, parent_coords)
            )
            # set initial number of child chunks
            parent_children_count_d_shared[parent_chunk_str] = children_count

        # decrement child count by 1
        parent_children_count_d_shared[parent_chunk_str] -= 1

        # if zero, all dependents complete -> start parent
        if parent_children_count_d_shared[parent_chunk_str] == 0:
            parent_children_count_d_shared.pop(parent_chunk_str, None)
            return ChunkTask(imanager.cg_meta, parent_coords, parent_layer)


def create_atomic_chunk_helper(
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: RLock,
    im_info: dict,
    coords: np.ndarray,
):
    """Helper to queue atomic chunk task."""
    imanager = IngestionManager(**im_info)
    coords = np.array(list(coords), dtype=np.int)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, coords)

    ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)
    imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)
    return get_parent_task(
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        imanager,
        2,
        coords,
    )


def create_parent_chunk_helper(args):
    """Helper to queue parent chunk task."""
    (
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        im_info,
        layer,
        chunk_coords,
    ) = args
    imanager = IngestionManager(**im_info)
    chunk_coords = np.array(list(chunk_coords), dtype=np.int)

    children = get_children_coords(imanager.cg_meta, layer, chunk_coords)
    imanager.cg.add_layer(layer, children)
    return get_parent_task(
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        imanager,
        layer,
        chunk_coords,
    )


def _get_atomic_chunk_data(imanager: IngestionManager, coord) -> Tuple[Dict, Dict]:
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

