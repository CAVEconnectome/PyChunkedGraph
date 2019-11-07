import time
import math
import multiprocessing as mp
from collections import defaultdict
from typing import Optional
from typing import Sequence
from typing import List
from typing import Dict

import numpy as np
from multiwrapper.multiprocessing_utils import multiprocess_func

from ...utils.general import chunked
from .. import flatgraph_utils
from ..utils import basetypes
from ..utils import serializers
from ..utils import column_keys
from ..chunkedgraph import ChunkedGraph
from ..chunkedgraph_utils import get_valid_timestamp
from ..chunkedgraph_utils import filter_failed_node_ids
from ..chunks.atomic import get_touching_atomic_chunks
from ..chunks.atomic import get_bounding_atomic_chunks


def get_children_chunk_cross_edges(cg_instance, layer, chunk_coord) -> np.ndarray:
    """ cross edges that connect children chunks """
    atomic_chunks = get_touching_atomic_chunks(cg_instance.meta, layer, chunk_coord)
    if not len(atomic_chunks):
        return []

    print(f"start reading cross edges, atomic chunks {len(atomic_chunks)}")
    start = time.time()
    cg_info = cg_instance.get_serialized_info(credentials=False)
    with mp.Manager() as manager:
        edge_ids_shared = manager.list()
        edge_ids_shared.append(np.empty([0, 2], dtype=basetypes.NODE_ID))

        task_size = int(math.ceil(len(atomic_chunks) / mp.cpu_count()))
        chunked_l2chunk_list = chunked(atomic_chunks, task_size)
        multi_args = []
        for atomic_chunks in chunked_l2chunk_list:
            multi_args.append((edge_ids_shared, cg_info, atomic_chunks, layer - 1))

        print(
            f"prep: {time.time()-start} jobs: {len(multi_args)} task_size: {task_size}"
        )
        multiprocess_func(
            _get_children_chunk_cross_edges_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )

        cross_edges = np.concatenate(edge_ids_shared)
        print("reading cross edges complete")
        if cross_edges.size:
            return np.unique(cross_edges, axis=0)
        return cross_edges


def _get_children_chunk_cross_edges_helper(args) -> None:
    edge_ids_shared, cg_info, atomic_chunks, layer = args
    cg_instance = ChunkedGraph(**cg_info)

    cross_edges = [np.empty([0, 2], dtype=basetypes.NODE_ID)]
    for layer2_chunk in atomic_chunks:
        edges = _read_atomic_chunk_cross_edges(cg_instance, layer2_chunk, layer)
        cross_edges.append(edges)

    cross_edges = np.concatenate(cross_edges)
    cross_edges[:, 0] = cg_instance.get_roots(cross_edges[:, 0], stop_layer=layer)
    cross_edges[:, 1] = cg_instance.get_roots(cross_edges[:, 1], stop_layer=layer)
    if len(cross_edges):
        edge_ids_shared.append(np.unique(cross_edges, axis=0))


def _read_atomic_chunk_cross_edges(
    cg_instance, chunk_coord: Sequence[int], cross_edge_layer: int
) -> np.ndarray:
    cross_edge_col = column_keys.Connectivity.CrossChunkEdge[cross_edge_layer]
    range_read, l2ids = _read_atomic_chunk(cg_instance, chunk_coord, [cross_edge_layer])

    parent_neighboring_chunk_supervoxels_d = defaultdict(list)
    for l2id in l2ids:
        if not cross_edge_col in range_read[l2id]:
            continue
        edges = range_read[l2id][cross_edge_col][0].value
        parent_neighboring_chunk_supervoxels_d[l2id] = edges[:, 1]

    cross_edges = [np.empty([0, 2], dtype=basetypes.NODE_ID)]
    for l2id in parent_neighboring_chunk_supervoxels_d:
        nebor_svs = parent_neighboring_chunk_supervoxels_d[l2id]
        chunk_parent_ids = np.array([l2id] * len(nebor_svs), dtype=basetypes.NODE_ID)
        cross_edges.append(np.vstack([chunk_parent_ids, nebor_svs]).T)
    cross_edges = np.concatenate(cross_edges)
    return cross_edges


def get_chunk_nodes_cross_edge_layer(
    cg_instance, layer: int, chunk_coord: Sequence[int]
) -> Dict:
    """
    gets nodes in a chunk that are part of cross chunk edges
    return_type dict {node_id: layer}
    the lowest layer (>= current layer) at which a node_id is part of a cross edge
    """
    atomic_chunks = get_bounding_atomic_chunks(cg_instance.meta, layer, chunk_coord)
    if not len(atomic_chunks):
        return {}

    print(f"atomic_chunks: {len(atomic_chunks)}")
    start = time.time()
    cg_info = cg_instance.get_serialized_info(credentials=False)
    manager = mp.Manager()  # needs to be closed?
    node_layer_t_shared_list = manager.list()
    task_size = int(math.ceil(len(atomic_chunks) / mp.cpu_count()))
    chunked_l2chunk_list = chunked(atomic_chunks, task_size)
    multi_args = []
    for i, atomic_chunks in enumerate(chunked_l2chunk_list):
        multi_args.append((node_layer_t_shared_list, cg_info, atomic_chunks, layer, i))

    print(f"prep: {time.time()-start} jobs: {len(multi_args)} task_size: {task_size}")
    start = time.time()
    multiprocess_func(
        _get_chunk_nodes_cross_edge_layer_helper,
        multi_args,
        n_threads=min(len(multi_args), mp.cpu_count()),
    )
    print(f"node_layer_d mp 1 complete, {time.time()-start}")

    start = time.time()
    node_layer_d_shared = manager.dict()
    _find_min_layer(node_layer_d_shared, node_layer_t_shared_list)
    print(f"_find_min_layer, {time.time()-start}")
    return node_layer_d_shared


def _get_chunk_nodes_cross_edge_layer_helper(args):
    node_layer_t_shared_list, cg_info, atomic_chunks, layer, job_id = args
    cg_instance = ChunkedGraph(**cg_info)

    start = time.time()
    atomic_node_layer_d = {}
    for atomic_chunk in atomic_chunks:
        chunk_node_layer_d = _read_atomic_chunk_cross_edge_nodes(
            cg_instance, atomic_chunk, range(layer, cg_instance.n_layers + 1)
        )
        atomic_node_layer_d.update(chunk_node_layer_d)

    l2ids = np.fromiter(atomic_node_layer_d.keys(), dtype=basetypes.NODE_ID)
    parents = cg_instance.get_roots(l2ids, stop_layer=layer - 1)
    layers = np.fromiter(atomic_node_layer_d.values(), dtype=np.int)

    node_layer_d = defaultdict(lambda: cg_instance.n_layers)
    for i, parent in enumerate(parents):
        node_layer_d[parent] = min(node_layer_d[parent], layers[i])

    node_layer_t_shared_list.append(
        (list(node_layer_d.keys()), list(node_layer_d.values()))
    )
    time_spent = time.time() - start
    print(
        f"{job_id}: chunks {len(atomic_chunks)}, nodes {len(node_layer_d)}, l2ids {len(l2ids)}, time {time_spent}"
    )


def _read_atomic_chunk_cross_edge_nodes(cg_instance, chunk_coord, cross_edge_layers):
    node_layer_d = {}
    range_read, l2ids = _read_atomic_chunk(cg_instance, chunk_coord, cross_edge_layers)
    for l2id in l2ids:
        node_layer_d[l2id] = cg_instance.n_layers
        for layer in cross_edge_layers:
            if column_keys.Connectivity.CrossChunkEdge[layer] in range_read[l2id]:
                node_layer_d[l2id] = layer
                break
    return node_layer_d


def _find_min_layer(node_layer_d_shared, node_layer_t_shared_list):
    print(f"node_layer_t_shared_list {len(node_layer_t_shared_list)}")
    node_ids = []
    layers = []
    for node_layer_t in node_layer_t_shared_list:
        node_ids.append(node_layer_t[0])
        layers.append(node_layer_t[1])

    node_ids = np.concatenate(node_ids)
    layers = np.concatenate(layers)
    print(f"_find_min_layer node_ids: {len(node_ids)}")
    print(f"_find_min_layer node_ids unique: {len(np.unique(node_ids))}")

    return
    for i, node_id in enumerate(node_ids):
        layer = node_layer_d_shared.get(node_id, layers[i])
        node_layer_d_shared[node_id] = min(layer, layers[i])
    print(f"_find_min_layer node_ids: {len(node_layer_d_shared)}")


def _read_atomic_chunk(cg_instance, chunk_coord, layers):
    """ utility function to read atomic chunk data """
    x, y, z = chunk_coord
    child_col = column_keys.Hierarchy.Child
    columns = [child_col] + [column_keys.Connectivity.CrossChunkEdge[l] for l in layers]
    range_read = cg_instance.range_read_chunk(2, x, y, z, columns=columns)

    row_ids = []
    max_children_ids = []
    for row_id, row_data in range_read.items():
        row_ids.append(row_id)
        max_children_ids.append(np.max(row_data[child_col][0].value))

    row_ids = np.array(row_ids, dtype=basetypes.NODE_ID)
    segment_ids = np.array([cg_instance.get_segment_id(r_id) for r_id in row_ids])
    l2ids = filter_failed_node_ids(row_ids, segment_ids, max_children_ids)
    return range_read, l2ids
