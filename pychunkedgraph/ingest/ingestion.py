"""
Ingest / create chunkedgraph
"""

import time
from itertools import product
from typing import Dict
from typing import Tuple

import numpy as np

from .manager import IngestionManager
from .backward_compat import get_chunk_data as get_chunk_data_old_format
from .ran_agglomeration import read_raw_edge_data
from .ran_agglomeration import read_raw_agglomeration_data
from .ran_agglomeration import get_active_edges
from ..io.edges import get_chunk_edges
from ..io.edges import put_chunk_edges
from ..io.components import get_chunk_components
from ..backend.chunks.hierarchy import get_children_coords

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def enqueue_atomic_tasks(imanager: IngestionManager):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    for chunk_coord in chunk_coords:
        atomic_queue = imanager.get_task_queue(imanager.config.atomic_q_name)
        # for optimal use of redis memory wait if queue limit is reached
        if len(atomic_queue) > imanager.config.atomic_q_limit:
            print(f"Sleeping {imanager.config.atomic_q_interval}s...")
            time.sleep(imanager.config.atomic_q_interval)
        atomic_queue.enqueue(
            _create_atomic_chunk,
            job_id=chunk_id_str(2, chunk_coord),
            job_timeout="59m",
            result_ttl=0,
            args=(imanager.get_serialized_info(), chunk_coord),
        )


def _create_atomic_chunk(im_info, coord):
    """ Creates single atomic chunk"""
    imanager = IngestionManager(**im_info)
    coord = np.array(list(coord), dtype=np.int)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, coord)

    ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)
    imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)

    _post_task_completion(imanager, 2, coord)


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


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.hdel(f"{layer}q", chunk_str)
    imanager.redis.hset(f"{layer}c", chunk_str, "")

    parent_layer = layer + 1
    if parent_layer > imanager.chunkedgraph_meta.layer_count:
        return

    parent_coords = (
        np.array(coords, int) // imanager.chunkedgraph_meta.graph_config.fanout
    )
    parent_chunk_str = "_".join(map(str, parent_coords))
    if not imanager.redis.hget(parent_layer, parent_chunk_str):
        children_count = len(
            get_children_coords(imanager.chunkedgraph_meta, parent_layer, parent_coords)
        )
        imanager.redis.hset(parent_layer, parent_chunk_str, children_count)
    imanager.redis.hincrby(parent_layer, parent_chunk_str, -1)
    children_left = int(
        imanager.redis.hget(parent_layer, parent_chunk_str).decode("utf-8")
    )

    if children_left == 0:
        imanager.cg.add_layer(
            layer,
            get_children_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            ),
        )
        _post_task_completion(imanager, layer, parent_coords)

        imanager.redis.hdel(parent_layer, parent_chunk_str)
        imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")
