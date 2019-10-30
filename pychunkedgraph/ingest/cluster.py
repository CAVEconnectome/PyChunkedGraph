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

from .manager import IngestionManager
from .common import get_atomic_chunk_data
from .utils import chunk_id_str
from .ran_agglomeration import get_active_edges
from .initialization.atomic_layer import add_atomic_edges
from .initialization.abstract_layers import add_layer
from ..backend import ChunkedGraphMeta
from ..utils.redis import keys as r_keys
from ..io.edges import get_chunk_edges
from ..io.components import get_chunk_components
from ..backend.edges import Edges
from ..backend.chunks.hierarchy import get_children_coords


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
        parents_queue = imanager.get_task_queue(imanager.config.parents_q_name)
        parents_queue.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(parent_layer, parent_coords),
            job_timeout=f"{int(1.5 * parent_layer)}m",
            result_ttl=0,
            args=(
                imanager.serialized(),
                parent_layer,
                parent_coords,
                get_children_coords(
                    imanager.chunkedgraph_meta, parent_layer, parent_coords
                ),
            ),
        )
        imanager.redis.hdel(parent_layer, parent_chunk_str)
        imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


def create_parent_chunk(
    im_info: str, layer: int, parent_coords: Sequence[int], child_chunk_coords: List
) -> None:
    imanager = IngestionManager(**im_info)
    add_layer(imanager.cg, layer, parent_coords, child_chunk_coords)
    _post_task_completion(imanager, layer, parent_coords)


def enqueue_atomic_tasks(imanager: IngestionManager):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    # test chunks

    chunk_coords = [
        [42, 24, 10],
        [42, 24, 11],
        [42, 25, 10],
        [42, 25, 11],
        [43, 24, 10],
        [43, 24, 11],
        [43, 25, 10],
        [43, 25, 11],
    ]

    # chunk_coords = [
    #     [400, 100, 14],
    #     [400, 100, 15],
    #     [400, 101, 14],
    #     [400, 101, 15],
    #     [401, 100, 14],
    #     [401, 100, 15],
    #     [401, 101, 14],
    #     [401, 101, 15],
    # ]

    # chunk_coords = [
    #     [270, 100, 4],
    #     [270, 100, 5],
    #     [270, 101, 4],
    #     [270, 101, 5],
    #     [271, 100, 4],
    #     [271, 100, 5],
    #     [271, 101, 4],
    #     [271, 101, 5],
    # ]

    # chunk_coords = [
    #     [100, 102, 8],
    #     [100, 102, 9],
    #     [100, 103, 8],
    #     [100, 103, 9],
    #     [101, 102, 8],
    #     [101, 102, 9],
    #     [101, 103, 8],
    #     [101, 103, 9],
    # ]

    for chunk_coord in chunk_coords:
        atomic_queue = imanager.get_task_queue(imanager.config.atomic_q_name)
        # for optimal use of redis memory wait if queue limit is reached
        if len(atomic_queue) > imanager.config.atomic_q_limit:
            print(f"Sleeping {imanager.config.atomic_q_interval}s...")
            time.sleep(imanager.config.atomic_q_interval)
        atomic_queue.enqueue(
            _create_atomic_chunk,
            job_id=chunk_id_str(2, chunk_coord),
            job_timeout="1m",
            result_ttl=0,
            args=(imanager.serialized(), chunk_coord),
        )


def _create_atomic_chunk(im_info: str, coord: Sequence[int]):
    """ Creates single atomic chunk """
    imanager = IngestionManager(**im_info)
    coord = np.array(list(coord), dtype=np.int)
    chunk_edges_all, mapping = get_atomic_chunk_data(imanager, coord)
    chunk_edges_active, isolated_ids = get_active_edges(
        imanager, coord, chunk_edges_all, mapping
    )
    if not imanager.config.build_graph:
        # to keep track of jobs when only creating edges and components per chunk
        imanager.redis.hset(r_keys.ATOMIC_HASH_FINISHED, chunk_id_str(2, coord), "")
        return
    add_atomic_edges(imanager.cg, coord, chunk_edges_active, isolated=isolated_ids)
    _post_task_completion(imanager, 2, coord)
