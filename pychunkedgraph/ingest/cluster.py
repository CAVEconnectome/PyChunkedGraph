# pylint: disable=invalid-name, missing-function-docstring, import-outside-toplevel

"""
Ingest / create chunkedgraph with workers on a cluster.
"""

import logging
from os import environ
from time import sleep
from typing import Callable, Dict, Iterable, Tuple, Sequence

import numpy as np
from rq import Queue as RQueue


from .utils import chunk_id_str, get_chunks_not_done, randomize_grid_points
from .manager import IngestionManager
from .ran_agglomeration import (
    get_active_edges,
    read_raw_edge_data,
    read_raw_agglomeration_data,
)
from .create.atomic_layer import add_atomic_chunk
from .create.parent_layer import add_parent_chunk
from .upgrade.atomic_layer import update_chunk as update_atomic_chunk
from .upgrade.parent_layer import update_chunk as update_parent_chunk
from ..graph.edges import EDGE_TYPES, Edges, put_edges
from ..graph import ChunkedGraph, ChunkedGraphMeta
from ..graph.chunks.hierarchy import get_children_chunk_coords
from ..graph.utils.basetypes import NODE_ID
from ..io.edges import get_chunk_edges
from ..io.components import get_chunk_components
from ..utils.redis import keys as r_keys, get_redis_connection
from ..utils.general import chunked


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
    add_parent_chunk(
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


def upgrade_parent_chunk(
    parent_layer: int,
    parent_coords: Sequence[int],
) -> None:
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    update_parent_chunk(imanager.cg, parent_coords, layer=parent_layer)
    _post_task_completion(imanager, parent_layer, parent_coords)


def _get_atomic_chunk_data(
    imanager: IngestionManager, coord: Sequence[int]
) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        read_raw_edge_data(imanager, coord)
        if imanager.config.USE_RAW_EDGES
        else get_chunk_edges(imanager.cg_meta.data_source.EDGES, [coord])
    )

    _check_edges_direction(chunk_edges, imanager.cg, coord)

    mapping = (
        read_raw_agglomeration_data(imanager, coord)
        if imanager.config.USE_RAW_COMPONENTS
        else get_chunk_components(imanager.cg_meta.data_source.COMPONENTS, coord)
    )
    return chunk_edges, mapping


def _check_edges_direction(
    chunk_edges: dict, cg: ChunkedGraph, coord: Sequence[int]
) -> None:
    """
    For between and cross chunk edges:
    Checks and flips edges such that nodes1 are always within a chunk and nodes2 outside the chunk.
    Where nodes1 = edges[:,0] and nodes2 = edges[:,1].
    """
    x, y, z = coord
    chunk_id = cg.get_chunk_id(layer=1, x=x, y=y, z=z)
    for edge_type in [EDGE_TYPES.between_chunk, EDGE_TYPES.cross_chunk]:
        edges = chunk_edges[edge_type]
        chunk_ids = cg.get_chunk_ids_from_node_ids(edges.node_ids1)
        mask = chunk_ids == chunk_id
        assert np.all(mask), "all IDs must belong to same chunk"


def create_atomic_chunk(coords: Sequence[int]):
    """Creates single atomic chunk"""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    coords = np.array(list(coords), dtype=int)

    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, coords)
    chunk_edges_active, isolated_ids = get_active_edges(chunk_edges_all, mapping)
    add_atomic_chunk(imanager.cg, coords, chunk_edges_active, isolated=isolated_ids)

    for k, v in chunk_edges_all.items():
        logging.debug(f"{k}: {len(v)}")
    for k, v in chunk_edges_active.items():
        logging.debug(f"active_{k}: {len(v)}")
    _post_task_completion(imanager, 2, coords)


def upgrade_atomic_chunk(coords: Sequence[int]):
    """Upgrades single atomic chunk"""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    coords = np.array(list(coords), dtype=int)
    update_atomic_chunk(imanager.cg, coords, layer=2)
    _post_task_completion(imanager, 2, coords)


def convert_to_ocdbt(coords: Sequence[int]):
    """
    Convert edges stored per chunk to ajacency list in the tensorstore ocdbt kv store.
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    coords = np.array(list(coords), dtype=int)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, coords)

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
    queue_name = "l2"
    q = imanager.get_task_queue(queue_name)
    batch_size = int(environ.get("JOB_BATCH_SIZE", 10000))
    batches = chunked(coords, batch_size)
    for batch in batches:
        _coords = get_chunks_not_done(imanager, 2, batch)
        # buffer for optimal use of redis memory
        if len(q) > int(environ.get("QUEUE_SIZE", 1000000)):
            interval = int(environ.get("QUEUE_INTERVAL", 300))
            logging.info(f"Queue full; sleeping {interval}s...")
            sleep(interval)

        job_datas = []
        for chunk_coord in _coords:
            job_datas.append(
                RQueue.prepare_data(
                    chunk_fn,
                    args=(chunk_coord,),
                    timeout=environ.get("L2JOB_TIMEOUT", "3m"),
                    result_ttl=0,
                    job_id=chunk_id_str(2, chunk_coord),
                )
            )
        q.enqueue_many(job_datas)
        logging.info(f"Queued {len(job_datas)} chunks.")


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
