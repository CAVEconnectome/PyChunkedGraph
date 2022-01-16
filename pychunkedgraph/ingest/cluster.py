"""
Ingest / create chunkedgraph with workers.
"""

import time
from itertools import product
from typing import List
from typing import Dict
from typing import Tuple
from typing import Sequence

import numpy as np

from .utils import chunk_id_str
from .manager import IngestionManager
from .common import get_atomic_chunk_data
from .ran_agglomeration import get_active_edges
from .initial.atomic_layer import add_atomic_edges
from .initial.abstract_layers import add_layer
from ..utils.redis import keys as r_keys
from ..graph.meta import ChunkedGraphMeta
from ..graph.chunks.hierarchy import get_children_chunk_coords


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.hdel(f"{layer}q", chunk_str)
    imanager.redis.hset(f"{layer}c", chunk_str, "")
    return  # TODO remove this once ingest process is robust

    parent_layer = layer + 1
    if parent_layer > imanager.chunkedgraph_meta.layer_count:
        return

    parent_coords = (
        np.array(coords, int) // imanager.chunkedgraph_meta.graph_config.FANOUT
    )
    parent_chunk_str = "_".join(map(str, parent_coords))
    if not imanager.redis.hget(parent_layer, parent_chunk_str):
        children_count = len(
            get_children_chunk_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            )
        )
        imanager.redis.hset(parent_layer, parent_chunk_str, children_count)
    imanager.redis.hincrby(parent_layer, parent_chunk_str, -1)
    children_left = int(
        imanager.redis.hget(parent_layer, parent_chunk_str).decode("utf-8")
    )

    if children_left == 0:
        parents_queue = imanager.get_task_queue(imanager.config.CLUSTER.PARENTS_Q_NAME)
        parents_queue.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(parent_layer, parent_coords),
            job_timeout=f"{parent_layer*parent_layer}m",
            result_ttl=0,
            args=(imanager.serialized(pickled=True), parent_layer, parent_coords,),
        )
        imanager.redis.hdel(parent_layer, parent_chunk_str)
        imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


def create_parent_chunk(
    im_info: str, layer: int, parent_coords: Sequence[int],
) -> None:
    imanager = IngestionManager.from_pickle(im_info)
    add_layer(
        imanager.cg,
        layer,
        parent_coords,
        get_children_chunk_coords(imanager.chunkedgraph_meta, layer, parent_coords),
    )
    _post_task_completion(imanager, layer, parent_coords)


def enqueue_atomic_tasks(imanager: IngestionManager):
    imanager.redis.flushdb()
    chunk_coords = _get_test_chunks(imanager.cg.meta)

    if not imanager.config.TEST_RUN:
        atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
        chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
        np.random.shuffle(chunk_coords)

    for chunk_coord in chunk_coords:
        atomic_queue = imanager.get_task_queue(imanager.config.CLUSTER.ATOMIC_Q_NAME)
        # for optimal use of redis memory wait if queue limit is reached
        if len(atomic_queue) > imanager.config.CLUSTER.ATOMIC_Q_LIMIT:
            print(f"Sleeping {imanager.config.CLUSTER.ATOMIC_Q_INTERVAL}s...")
            time.sleep(imanager.config.CLUSTER.ATOMIC_Q_INTERVAL)
        atomic_queue.enqueue(
            _create_atomic_chunk,
            job_id=chunk_id_str(2, chunk_coord),
            job_timeout="4m",
            result_ttl=0,
            args=(imanager.serialized(pickled=True), chunk_coord),
        )


def _create_atomic_chunk(im_info: str, coord: Sequence[int]):
    """ Creates single atomic chunk """
    imanager = IngestionManager.from_pickle(im_info)
    coord = np.array(list(coord), dtype=int)
    chunk_edges_all, mapping = get_atomic_chunk_data(imanager, coord)
    chunk_edges_active, isolated_ids = get_active_edges(
        imanager, coord, chunk_edges_all, mapping
    )
    # if not imanager.config.build_graph:
    #     # to keep track of jobs when only creating edges and components per chunk
    #     imanager.redis.hset(r_keys.ATOMIC_HASH_FINISHED, chunk_id_str(2, coord), "")
    #     return
    add_atomic_edges(imanager.cg, coord, chunk_edges_active, isolated=isolated_ids)
    for k, v in chunk_edges_active.items():
        print(k, len(v))
    raise RuntimeError("me die")
    _post_task_completion(imanager, 2, coord)


def _get_test_chunks(meta: ChunkedGraphMeta):
    """
    Returns chunks that lie at the center of the dataset
    """
    f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(-1, 3)
    x, y, z = np.array(meta.layer_chunk_bounds[2]) // 2
    return f((x, x + 1), (y, y + 1), (z, z + 1))
