# pylint: disable=invalid-name, missing-function-docstring, import-outside-toplevel

"""
Ingest / create chunkedgraph with workers in a cluster.
"""

import logging
from os import environ
from time import sleep
from typing import Callable, Iterable, Sequence

import numpy as np
from rq import Queue as RQueue


from .utils import chunk_id_str
from .utils import randomize_grid_points
from .manager import IngestionManager
from .common import get_atomic_chunk_data
from .ran_agglomeration import get_active_edges
from .create.atomic_layer import add_atomic_edges
from .create.parent_layer import add_layer
from ..graph.edges import Edges, put_edges
from ..graph.meta import ChunkedGraphMeta
from ..graph.chunks.hierarchy import get_children_chunk_coords
from ..graph.utils.basetypes import NODE_ID
from ..utils.redis import keys as r_keys
from ..utils.redis import get_redis_connection


def _post_task_completion(
    imanager: IngestionManager,
    layer: int,
    coords: np.ndarray,
):
    chunk_str = "_".join(map(str, coords))
    # mark chunk as completed - "c"
    imanager.redis.sadd(f"{layer}c", chunk_str)


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


def create_atomic_chunk(coords: Sequence[int]):
    """Creates single atomic chunk"""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    coords = np.array(list(coords), dtype=int)

    chunk_edges_all, mapping = get_atomic_chunk_data(imanager, coords)
    chunk_edges_active, isolated_ids = get_active_edges(chunk_edges_all, mapping)
    add_atomic_edges(imanager.cg, coords, chunk_edges_active, isolated=isolated_ids)

    for k, v in chunk_edges_all.items():
        logging.debug(f"{k}: {len(v)}")
    for k, v in chunk_edges_active.items():
        logging.debug(f"active_{k}: {len(v)}")
    _post_task_completion(imanager, 2, coords)


def convert_to_ocdbt(coords: Sequence[int]):
    """
    Convert edges stored per chunk to ajacency list in the tensorstore ocdbt kv store.
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    coords = np.array(list(coords), dtype=int)
    chunk_edges_all, mapping = get_atomic_chunk_data(imanager, coords)

    node_ids1 = []
    node_ids2 = []
    affinities = []
    areas = []
    for edges in chunk_edges_all.values():
        node_ids1.extend(edges.node_ids1)
        node_ids2.extend(edges.node_ids2)
        affinities.extend(edges.affinities)
        areas.extend(edges.areas)

    edges = Edges(node_ids1, node_ids2, affinities=affinities, areas=areas)
    nodes = np.concatenate(
        [edges.node_ids1, edges.node_ids2, np.fromiter(mapping.keys(), dtype=NODE_ID)]
    )
    nodes = np.unique(nodes)

    chunk_id = imanager.cg.get_chunk_id(layer=1, x=coords[0], y=coords[1], z=coords[2])
    chunk_ids = imanager.cg.get_chunk_ids_from_node_ids(nodes)

    host = imanager.redis.get("OCDBT_COORDINATOR_HOST").decode()
    port = imanager.redis.get("OCDBT_COORDINATOR_PORT").decode()
    environ["OCDBT_COORDINATOR_HOST"] = host
    environ["OCDBT_COORDINATOR_PORT"] = port
    logging.info(f"OCDBT Coordinator address {host}:{port}")

    put_edges(
        f"{imanager.cg.meta.data_source.EDGES}/ocdbt",
        nodes[chunk_ids == chunk_id],
        edges,
    )
    _post_task_completion(imanager, 2, coords)


def _get_test_chunks(meta: ChunkedGraphMeta):
    """Chunks at the center most likely not to be empty"""
    parent_coords = np.array(meta.layer_chunk_bounds[3]) // 2
    return get_children_chunk_coords(meta, 3, parent_coords)


def _queue_tasks(imanager: IngestionManager, chunk_fn: Callable, coords: Iterable):
    queue_name = f"{imanager.config.CLUSTER.ATOMIC_Q_NAME}"
    q = imanager.get_task_queue(queue_name)
    job_datas = []
    batch_size = int(environ.get("L2JOB_BATCH_SIZE", 1000))
    for chunk_coord in coords:
        # buffer for optimal use of redis memory
        if len(q) > int(environ.get("QUEUE_SIZE", 100000)):
            interval = int(environ.get("QUEUE_INTERVAL", 300))
            logging.info(f"Queue full; sleeping {interval}s...")
            sleep(interval)

        x, y, z = chunk_coord
        chunk_str = f"{x}_{y}_{z}"
        if imanager.redis.sismember("2c", chunk_str):
            continue
        job_datas.append(
            RQueue.prepare_data(
                chunk_fn,
                args=(chunk_coord,),
                timeout=environ.get("L2JOB_TIMEOUT", "3m"),
                result_ttl=0,
                job_id=chunk_id_str(2, chunk_coord),
            )
        )
        if len(job_datas) % batch_size == 0:
            q.enqueue_many(job_datas)
            job_datas = []
    q.enqueue_many(job_datas)


def enqueue_l2_tasks(imanager: IngestionManager, chunk_fn: Callable):
    """
    `chunk_fn`: function to process a given layer 2 chunk.
    """
    chunk_coords = _get_test_chunks(imanager.cg.meta)
    chunk_count = len(chunk_coords)
    if not imanager.config.TEST_RUN:
        atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
        chunk_coords = randomize_grid_points(*atomic_chunk_bounds)
        chunk_count = imanager.cg_meta.layer_chunk_counts[0]
    logging.info(f"Chunk count: {chunk_count}, queuing...")
    _queue_tasks(imanager, chunk_fn, chunk_coords)
