"""
Ingest / create chunkedgraph with workers.
"""

import time
from itertools import product
from typing import Sequence

import numpy as np

from .utils import chunk_id_str
from .manager import IngestionManager
from .common import get_atomic_chunk_data
from .ran_agglomeration import get_active_edges
from .initial.atomic_layer import add_atomic_edges
from .initial.abstract_layers import add_layer
from ..graph.meta import ChunkedGraphMeta
from ..graph.chunks.hierarchy import get_children_chunk_coords
from ..utils.redis import keys as r_keys
from ..utils.redis import get_redis_connection


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # mark chunk as completed - "c"
    imanager.redis.hset(f"{layer}c", chunk_str, "")

    parent_layer = layer + 1
    if parent_layer > imanager.cg_meta.layer_count:
        return

    parent_coords = np.array(coords, int) // imanager.cg_meta.graph_config.FANOUT
    parent_chunk_str = "_".join(map(str, parent_coords))
    if not imanager.redis.hget(parent_layer, parent_chunk_str):
        # add children chunk count to redis cache
        # tracked by another worker to enqueue parent chunk
        children_count = len(
            get_children_chunk_coords(imanager.cg_meta, parent_layer, parent_coords)
        )
        imanager.redis.hset(parent_layer, parent_chunk_str, children_count)


def enqueue_parent_task():
    PARENT_TASK_TRACKER = "parent_task_tracker"
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    layer = chunk_info[0]
    coords = chunk_info[1:]
    queue = imanager.get_task_queue(PARENT_TASK_TRACKER)
    queue.enqueue(
        create_parent_chunk,
        job_id=chunk_id_str(layer, coords),
        job_timeout=f"{int(layer * layer)}m",
        result_ttl=0,
        args=(
            imanager.serialized(pickled=True),
            layer,
            coords,
        ),
    )


def create_parent_chunk(
    parent_layer: int,
    parent_coords: Sequence[int],
) -> None:
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    add_layer(
        imanager.cg,
        parent_layer,
        parent_coords,
        get_children_chunk_coords(
            imanager.cg_meta,
            parent_layer,
            parent_coords,
        ),
    )
    _post_task_completion(imanager, parent_layer, parent_coords)


def enqueue_atomic_tasks(imanager: IngestionManager):
    chunk_coords = _get_test_chunks(imanager.cg.meta)

    if not imanager.config.TEST_RUN:
        atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
        chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
        np.random.shuffle(chunk_coords)

    for chunk_coord in chunk_coords:
        atomic_queue = imanager.get_task_queue(imanager.config.CLUSTER.ATOMIC_Q_NAME)
        # buffer for optimal use of redis memory
        if len(atomic_queue) > imanager.config.CLUSTER.ATOMIC_Q_LIMIT:
            print(f"Sleeping {imanager.config.CLUSTER.ATOMIC_Q_INTERVAL}s...")
            time.sleep(imanager.config.CLUSTER.ATOMIC_Q_INTERVAL)
        atomic_queue.enqueue(
            _create_atomic_chunk,
            job_id=chunk_id_str(2, chunk_coord),
            job_timeout="3m",
            result_ttl=0,
            args=(imanager.serialized(pickled=True), chunk_coord),
        )


def _create_atomic_chunk(coord: Sequence[int]):
    """Creates single atomic chunk"""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    coord = np.array(list(coord), dtype=int)
    chunk_edges_all, mapping = get_atomic_chunk_data(imanager, coord)
    chunk_edges_active, isolated_ids = get_active_edges(
        imanager, coord, chunk_edges_all, mapping
    )
    add_atomic_edges(imanager.cg, coord, chunk_edges_active, isolated=isolated_ids)
    if imanager.config.TEST_RUN:
        # print for debugging
        for k, v in chunk_edges_all.items():
            print(k, len(v))
        for k, v in chunk_edges_active.items():
            print(k, len(v))
    _post_task_completion(imanager, 2, coord)


def _get_test_chunks(meta: ChunkedGraphMeta):
    """Chunks at center of the dataset most likely not to be empty, for testing."""
    f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(-1, 3)
    x, y, z = np.array(meta.layer_chunk_bounds[2]) // 2
    return f((x, x + 1), (y, y + 1), (z, z + 1))
