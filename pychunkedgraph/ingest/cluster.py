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
from .types import ChunkTask
from ..backend import ChunkedGraphMeta
from ..backend.chunks.hierarchy import get_children_coords


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.hdel(f"{layer}q", chunk_str)
    imanager.redis.hset(f"{layer}c", chunk_str, "")
    return


def create_parent_chunk(
    im_info: str,
    layer: int,
    parent_coords: Sequence[int],
) -> None:
    from .ingestion import create_parent_chunk_helper

    imanager = IngestionManager.from_pickle(im_info)
    create_parent_chunk_helper(
        ChunkTask(imanager.cg_meta, parent_coords, layer), imanager
    )
    _post_task_completion(imanager, layer, parent_coords)


def enqueue_atomic_tasks(imanager: IngestionManager):
    imanager.redis.flushdb()
    chunk_coords = _get_test_chunks(imanager.cg.meta)

    if not imanager.config.TEST_RUN:
        atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
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
            job_timeout="6m",
            result_ttl=0,
            args=(imanager.get_serialized_info(pickled=True), chunk_coord),
        )


def _create_atomic_chunk(im_info: str, coord: Sequence[int]):
    """ Creates single atomic chunk """
    from .ingestion import create_atomic_chunk_helper

    imanager = IngestionManager.from_pickle(im_info)
    coord = np.array(list(coord), dtype=int)
    create_atomic_chunk_helper(ChunkTask(imanager.cg_meta, coord), imanager)
    _post_task_completion(imanager, 2, coord)


def _get_test_chunks(meta: ChunkedGraphMeta):
    """
    Returns chunks that lie at the center of the dataset
    """
    f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(-1, 3)
    x, y, z = np.array(meta.layer_chunk_bounds[2]) // 2
    return f((x, x + 1), (y, y + 1), (z, z + 1))
