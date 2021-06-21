"""
Ingest / create chunkedgraph on a single machine / instance
"""

import time
import math
import multiprocessing as mp
from typing import Dict
from itertools import product

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from .manager import IngestionManager
from ..utils.general import chunked
from ..graph.chunks.hierarchy import get_children_chunk_coords

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def start_ingest(imanager: IngestionManager):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    with mp.Manager() as manager:
        parent_children_count_d_shared = manager.dict()
        task_size = int(math.ceil(len(chunk_coords) / mp.cpu_count() / 10))
        jobs = chunked(chunk_coords, task_size)
        multi_args = []
        for job in jobs:
            multi_args.append(
                (parent_children_count_d_shared, imanager.serialized(), job)
            )
        mu.multiprocess_func(
            _create_atomic_chunks_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )


def _create_atomic_chunks_helper(args):
    """ helper to start atomic tasks """
    parent_children_count_d_shared, im_info, chunk_coords = args
    imanager = IngestionManager(**im_info)
    for chunk_coord in chunk_coords:
        chunk_coord = np.array(list(chunk_coord), dtype=int)
        chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, chunk_coord)

        ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)
        imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)
        _post_task_completion(parent_children_count_d_shared, imanager, 2, chunk_coord)


def _post_task_completion(
    parent_children_count_d_shared: Dict,
    imanager: IngestionManager,
    layer: int,
    coords: np.ndarray,
):
    parent_layer = layer + 1
    if parent_layer > imanager.chunkedgraph_meta.layer_count:
        return

    parent_coords = (
        np.array(coords, int) // imanager.chunkedgraph_meta.graph_config.FANOUT
    )
    parent_chunk_str = chunk_id_str(parent_layer, parent_coords)

    if not parent_chunk_str in parent_children_count_d_shared:
        children_count = len(
            get_children_chunk_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            )
        )
        # set initial number of child chunks
        parent_children_count_d_shared[parent_chunk_str] = children_count
    # decrement child count by 1
    parent_children_count_d_shared[parent_chunk_str] -= 1

    # if zero, all dependents complete -> start parent
    if parent_children_count_d_shared[parent_chunk_str] == 0:
        children = get_children_chunk_coords(
            imanager.chunkedgraph_meta, parent_layer, parent_coords
        )
        imanager.cg.add_layer(parent_layer, children)
        del parent_children_count_d_shared[parent_chunk_str]
        _post_task_completion(
            parent_children_count_d_shared, imanager, parent_layer, parent_coords
        )
