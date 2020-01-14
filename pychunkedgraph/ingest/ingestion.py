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
    task: ChunkTask,
):
    parent = task.parent()
    if parent.layer > parent.cg_meta.layer_count:
        return parent

    with parent_children_count_d_lock:
        if not parent.id in parent_children_count_d_shared:
            children_count = len(
                get_children_coords(parent.cg_meta, parent.layer, parent.coords)
            )
            # set initial number of child chunks
            parent_children_count_d_shared[parent.id] = children_count

        # decrement child count by 1
        parent_children_count_d_shared[parent.id] -= 1
        # if zero, all dependents complete -> return parent
        if parent_children_count_d_shared[parent.id] == 0:
            parent_children_count_d_shared.pop(parent.id, None)
            return parent


def create_atomic_chunk_helper(
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: RLock,
    im_info: Dict,
    coords: np.ndarray,
):
    """Helper to queue atomic chunk task."""
    imanager = IngestionManager(**im_info)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, coords)
    ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)

    success = False
    while not success:
        try:
            imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)
            success = True
        except:
            pass
    return get_parent_task(
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        imanager,
        2,
        coords,
    )


def create_parent_chunk_helper(
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: RLock,
    im_info: Dict,
    layer: int,
    chunk_coords: np.ndarray,
):
    """Helper to queue parent chunk task."""
    imanager = IngestionManager(**im_info)
    children = get_children_coords(imanager.cg_meta, layer, chunk_coords)

    success = False
    while not success:
        try:
            imanager.cg.add_layer(layer, children)
            success = True
        except:
            pass
    return get_parent_task(
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        imanager,
        layer,
        chunk_coords,
    )


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

