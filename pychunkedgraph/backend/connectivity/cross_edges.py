import time
import multiprocessing as mp
from collections import defaultdict
from typing import Optional
from typing import Sequence
from typing import List
from typing import Dict

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from ...utils.general import chunked
from ...backend import flatgraph_utils
from ...backend.utils import basetypes
from ...backend.utils import serializers
from ...backend.utils import column_keys
from ...backend.chunkedgraph import ChunkedGraph
from ...backend.chunkedgraph_utils import get_valid_timestamp
from ...backend.chunkedgraph_utils import filter_failed_node_ids
from ...backend.chunks.atomic import get_touching_atomic_chunks
from ...backend.chunks.atomic import get_bounding_atomic_chunks


def get_children_chunk_cross_edges(cg_instance, layer_id, chunk_coord) -> np.ndarray:
    layer2_chunks = get_touching_atomic_chunks(
        cg_instance.meta, layer_id, chunk_coord, include_both=False
    )
    if not len(layer2_chunks):
        return []

    cg_info = cg_instance.get_serialized_info(credentials=False)
    with mp.Manager() as manager:
        edge_ids_shared = manager.list()
        edge_ids_shared.append(np.empty([0, 2], dtype=basetypes.NODE_ID))

        chunked_l2chunk_list = chunked(
            layer2_chunks, len(layer2_chunks) // mp.cpu_count()
        )
        multi_args = []
        for layer2_chunks in chunked_l2chunk_list:
            multi_args.append((edge_ids_shared, cg_info, layer2_chunks, layer_id - 1))

        mu.multiprocess_func(
            _get_children_chunk_cross_edges_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )

        cross_edges = np.concatenate(edge_ids_shared)
        if cross_edges.size:
            return np.unique(cross_edges, axis=0)
        return cross_edges


def _get_children_chunk_cross_edges_helper(args):
    edge_ids_shared, cg_info, layer2_chunks, cross_edge_layer = args
    cg_instance = ChunkedGraph(**cg_info)

    # start = time.time()
    cross_edges = [np.empty([0, 2], dtype=basetypes.NODE_ID)]
    for layer2_chunk in layer2_chunks:
        edges = _read_atomic_chunk_cross_edges(
            cg_instance, layer2_chunk, cross_edge_layer
        )
        cross_edges.append(edges)
    cross_edges = np.concatenate(cross_edges)
    # print(f"reading raw edges {time.time()-start}s")

    # start = time.time()
    parents_1 = cg_instance.get_roots(cross_edges[:, 0], stop_layer=cross_edge_layer)
    # print(f"getting parents1 {time.time()-start}s")

    # start = time.time()
    parents_2 = cg_instance.get_roots(cross_edges[:, 1], stop_layer=cross_edge_layer)
    # print(f"getting parents2 {time.time()-start}s")

    cross_edges[:, 0] = parents_1
    cross_edges[:, 1] = parents_2
    if len(cross_edges):
        cross_edges = np.unique(cross_edges, axis=0)
        edge_ids_shared.append(cross_edges)


def _read_atomic_chunk_cross_edges(
    cg_instance, chunk_coord, cross_edge_layer
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


def get_chunk_nodes_cross_edge_layer(cg_instance, layer_id, chunk_coord) -> Dict:
    """
    gets nodes in a chunk that are part of cross chunk edges
    return_type dict {node_id: layer}
    the lowest layer (>= current layer) at which a node_id is part of a cross edge
    """
    layer2_chunks = get_bounding_atomic_chunks(cg_instance.meta, layer_id, chunk_coord)
    node_layer_d = defaultdict(lambda: cg_instance.n_layers)
    if not len(layer2_chunks):
        return node_layer_d

    cg_info = cg_instance.get_serialized_info(credentials=False)
    with mp.Manager() as manager:
        node_layer_tuples_shared = manager.list()
        chunked_l2chunk_list = chunked(
            layer2_chunks, len(layer2_chunks) // mp.cpu_count()
        )
        multi_args = []
        for layer2_chunks in chunked_l2chunk_list:
            multi_args.append(
                (node_layer_tuples_shared, cg_info, layer2_chunks, layer_id)
            )

        mu.multiprocess_func(
            _get_chunk_nodes_cross_edge_layer_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )

        node_ids = []
        layers = []
        for tup in node_layer_tuples_shared:
            node_ids.append(tup[0])
            layers.append(tup[1])

        node_ids = np.concatenate(node_ids)
        layers = np.concatenate(layers)

        for i, node_id in enumerate(node_ids):
            node_layer_d[node_id] = min(node_layer_d[node_id], layers[i])
        return {**node_layer_d}


def _get_chunk_nodes_cross_edge_layer_helper(args):
    node_layer_tuples_shared, cg_info, layer2_chunks, layer_id = args
    cg_instance = ChunkedGraph(**cg_info)

    # start = time.time()
    node_layer_d = {}
    for layer2_chunk in layer2_chunks:
        chunk_node_layer_d = _read_atomic_chunk_cross_edge_nodes(
            cg_instance, layer2_chunk, range(layer_id, cg_instance.n_layers + 1)
        )
        node_layer_d.update(chunk_node_layer_d)
    # print(f"reading raw edges {time.time()-start}s")

    # start = time.time()
    l2ids = np.fromiter(node_layer_d.keys(), dtype=basetypes.NODE_ID)
    parents = cg_instance.get_roots(l2ids, stop_layer=layer_id - 1)
    layers = np.fromiter(node_layer_d.values(), dtype=np.int)
    # print(f"getting parents {time.time()-start}s")

    node_layer_tuples_shared.append((parents, layers))


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
    return (range_read, l2ids)
