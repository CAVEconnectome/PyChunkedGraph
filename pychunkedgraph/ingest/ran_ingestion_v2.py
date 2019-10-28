"""
Module for ingesting in chunkedgraph format with edges stored outside bigtable
"""

import time
import json
from collections import defaultdict
from collections import Counter
from itertools import product
from typing import Dict
from typing import Tuple
from typing import Sequence

import pandas as pd
import cloudvolume
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
import zstandard as zstd

from .ingestion_utils import postprocess_edge_data
from .ingestionmanager import IngestionManager
from .initialization.atomic_layer import add_atomic_edges
from .initialization.abstract_layers import add_layer
from ..backend import ChunkedGraphMeta
from ..utils.redis import keys as r_keys
from ..io.edges import get_chunk_edges
from ..io.edges import put_chunk_edges
from ..io.components import get_chunk_components
from ..io.components import put_chunk_components
from ..backend.utils import basetypes
from ..backend.chunkedgraph_utils import compute_chunk_id
from ..backend.definitions.edges import Edges
from ..backend.definitions.edges import (
    IN_CHUNK,
    BT_CHUNK,
    CX_CHUNK,
    TYPES as EDGE_TYPES,
)

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def _get_children_coords(
    cg_meta: ChunkedGraphMeta, layer: int, chunk_coords
) -> np.ndarray:
    chunk_coords = np.array(chunk_coords, dtype=int)
    children_layer = layer - 1
    layer_boundaries = cg_meta.layer_chunk_bounds[children_layer]
    children_coords = []

    for dcoord in product(*[range(cg_meta.graph_config.fanout)] * 3):
        dcoord = np.array(dcoord, dtype=int)
        child_coords = chunk_coords * cg_meta.graph_config.fanout + dcoord
        check_bounds = np.less(child_coords, layer_boundaries)
        if np.all(check_bounds):
            children_coords.append(child_coords)
    return children_coords


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
            _get_children_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            )
        )
        imanager.redis.hset(parent_layer, parent_chunk_str, children_count)
    imanager.redis.hincrby(parent_layer, parent_chunk_str, -1)
    children_left = int(
        imanager.redis.hget(parent_layer, parent_chunk_str).decode("utf-8")
    )

    if children_left == 0:
        parents_queue = imanager.get_task_queue(imanager.config.parents_q_name)
        parents_queue.enqueue(
            _create_parent_chunk,
            job_id=chunk_id_str(parent_layer, parent_coords),
            job_timeout="59m",
            result_ttl=0,
            args=(
                imanager.get_serialized_info(),
                parent_layer,
                parent_coords,
                _get_children_coords(
                    imanager.chunkedgraph_meta, parent_layer, parent_coords
                ),
            ),
        )
        imanager.redis.hdel(parent_layer, parent_chunk_str)
        imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


def _create_parent_chunk(im_info, layer, parent_coords, child_chunk_coords):
    imanager = IngestionManager(**im_info)
    add_layer(imanager.cg, layer, parent_coords, child_chunk_coords)
    _post_task_completion(imanager, layer, parent_coords)


def enqueue_atomic_tasks(
    imanager: IngestionManager, batch_size: int = 50000, interval: float = 300.0
):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    # test chunks
    # chunk_coords = [
    #     [26, 4, 10],
    #     [26, 4, 11],
    #     [26, 5, 10],
    #     [26, 5, 11],
    #     [27, 4, 10],
    #     [27, 4, 11],
    #     [27, 5, 10],
    #     [27, 5, 11],
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
            job_timeout="59m",
            result_ttl=0,
            args=(imanager.get_serialized_info(), chunk_coord),
        )


def _create_atomic_chunk(im_info, coord):
    """ Creates single atomic chunk"""
    imanager = IngestionManager(**im_info)
    coord = np.array(list(coord), dtype=np.int)
    chunk_edges_all, mapping = _get_chunk_data(imanager, coord)
    chunk_edges_active, isolated_ids = _get_active_edges(
        imanager, coord, chunk_edges_all, mapping
    )
    if not imanager.config.build_graph:
        imanager.redis.hset(r_keys.ATOMIC_HASH_FINISHED, chunk_id_str(2, coord), "")
        return
    add_atomic_edges(imanager.cg, coord, chunk_edges_active, isolated=isolated_ids)
    _post_task_completion(imanager, 2, coord)


def _get_chunk_data(imanager, coord) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        _read_raw_edge_data(imanager, coord)
        if imanager.chunkedgraph_meta.data_source.use_raw_edges
        else get_chunk_edges(imanager.chunkedgraph_meta.data_source.edges, [coord])
    )
    mapping = (
        _read_raw_agglomeration_data(imanager, coord)
        if imanager.chunkedgraph_meta.data_source.use_raw_components
        else get_chunk_components(
            imanager.chunkedgraph_meta.data_source.components, coord
        )
    )
    return chunk_edges, mapping