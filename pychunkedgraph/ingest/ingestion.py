"""
Ingest / create chunkedgraph on a single machine / instance
"""

import time
import multiprocessing as mp
from itertools import product
from typing import List
from typing import Sequence
from typing import Dict
from typing import Tuple

import numpy as np
from multiwrapper import multiprocessing_utils as mu

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


def start_ingest(imanager: IngestionManager):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    with mp.Manager() as manager:
        parent_children_count_d_shared = manager.dict()
        parent_children_count_d_lock = manager.RLock()  # pylint: disable=no-member
        jobs = chunked(chunk_coords, len(chunk_coords) // mp.cpu_count())
        multi_args = []
        for job in jobs:
            multi_args.append(
                (
                    parent_children_count_d_shared,
                    parent_children_count_d_lock,
                    imanager.get_serialized_info(),
                    job,
                )
            )
        mu.multiprocess_func(
            _create_atomic_chunks_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )


def _post_task_completion(
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: mp.Lock,
    imanager: IngestionManager,
    layer: int,
    coords: np.ndarray,
):
    parent_layer = layer + 1
    if parent_layer > imanager.chunkedgraph_meta.layer_count:
        return

    parent_coords = (
        np.array(coords, int) // imanager.chunkedgraph_meta.graph_config.fanout
    )
    parent_chunk_str = chunk_id_str(parent_layer, parent_coords)

    with parent_children_count_d_lock:
        if not parent_chunk_str in parent_children_count_d_shared:
            children_count = len(
                get_children_coords(
                    imanager.chunkedgraph_meta, parent_layer, parent_coords
                )
            )
            # set initial number of child chunks
            parent_children_count_d_shared[parent_chunk_str] = children_count

        # decrement child count by 1
        parent_children_count_d_shared[parent_chunk_str] -= 1

        # if zero, all dependents complete -> start parent
        if parent_children_count_d_shared[parent_chunk_str] == 0:
            parent_children_count_d_shared.pop(parent_chunk_str, None)
            children = get_children_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            )
            imanager.cg.add_layer(parent_layer, children)
            _post_task_completion(
                parent_children_count_d_shared,
                parent_children_count_d_lock,
                imanager,
                parent_layer,
                parent_coords,
            )


def _create_atomic_chunks_helper(args):
    """ helper to start atomic tasks """
    (
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        im_info,
        chunk_coords,
    ) = args
    imanager = IngestionManager(**im_info)
    chunk_coords = np.array(list(chunk_coords), dtype=np.int)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, chunk_coords)

    ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)
    imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)
    _post_task_completion(
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        imanager,
        2,
        chunk_coords,
    )


def _get_atomic_chunk_data(imanager: IngestionManager, coord) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        read_raw_edge_data(imanager, coord)
        if imanager.chunkedgraph_meta.data_source.use_raw_edges
        else get_chunk_edges(imanager.chunkedgraph_meta.data_source.edges, [coord])
    )
    mapping = (
        read_raw_agglomeration_data(imanager, coord)
        if imanager.chunkedgraph_meta.data_source.use_raw_components
        else get_chunk_components(
            imanager.chunkedgraph_meta.data_source.components, coord
        )
    )
    return chunk_edges, mapping

