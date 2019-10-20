import time
import multiprocessing as mp
from collections import defaultdict
from typing import Optional
from typing import Sequence
from typing import List

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


def get_children_chunk_cross_edges(cg_instance, layer_id, chunk_coord) -> List:
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
            _get_cross_edges_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )

        cross_edges = np.concatenate(edge_ids_shared)
        if len(cross_edges):
            cross_edges = np.unique(cross_edges, axis=0)
        return list(cross_edges)


def _get_cross_edges_helper(args):
    edge_ids_shared, cg_info, layer2_chunks, cross_edge_layer = args
    cg_instance = ChunkedGraph(**cg_info)

    start = time.time()
    cross_edges = [np.empty([0, 2], dtype=basetypes.NODE_ID)]
    for layer2_chunk in layer2_chunks:
        edges = _read_atomic_chunk_cross_edges(
            cg_instance, layer2_chunk, cross_edge_layer
        )
        cross_edges.append(edges)
    cross_edges = np.concatenate(cross_edges)
    print(f"reading raw edges {time.time()-start}s")

    start = time.time()
    parents_1 = cg_instance.get_roots(cross_edges[:, 0], stop_layer=cross_edge_layer)
    print(f"getting parents1 {time.time()-start}s")

    start = time.time()
    parents_2 = cg_instance.get_roots(cross_edges[:, 1], stop_layer=cross_edge_layer)
    print(f"getting parents2 {time.time()-start}s")

    cross_edges[:, 0] = parents_1
    cross_edges[:, 1] = parents_2
    if len(cross_edges):
        cross_edges = np.unique(cross_edges, axis=0)
        edge_ids_shared.append(cross_edges)


def _read_atomic_chunk_cross_edges(cg_instance, chunk_coord, cross_edge_layer):
    x, y, z = chunk_coord
    child_key = column_keys.Hierarchy.Child
    cross_edge_key = column_keys.Connectivity.CrossChunkEdge[cross_edge_layer]
    range_read = cg_instance.range_read_chunk(
        2, x, y, z, columns=[child_key, cross_edge_key]
    )

    row_ids = []
    max_children_ids = []
    for row_id, row_data in range_read.items():
        row_ids.append(row_id)
        max_children_ids.append(np.max(row_data[child_key][0].value))

    row_ids = np.array(row_ids, dtype=basetypes.NODE_ID)
    segment_ids = np.array([cg_instance.get_segment_id(r_id) for r_id in row_ids])
    l2ids = filter_failed_node_ids(row_ids, segment_ids, max_children_ids)
    return _extract_atomic_cross_edges(range_read, l2ids, cross_edge_key)


def _extract_atomic_cross_edges(range_read, l2ids, cross_edge_key):
    parent_neighboring_chunk_supervoxels_d = defaultdict(list)
    for l2id in l2ids:
        if not cross_edge_key in range_read[l2id]:
            continue
        edges = range_read[l2id][cross_edge_key][0].value
        parent_neighboring_chunk_supervoxels_d[l2id] = edges[:, 1]

    cross_edges = [np.empty([0, 2], dtype=basetypes.NODE_ID)]
    for l2id in parent_neighboring_chunk_supervoxels_d:
        nebor_svs = parent_neighboring_chunk_supervoxels_d[l2id]
        chunk_parent_ids = np.array([l2id] * len(nebor_svs), dtype=basetypes.NODE_ID)
        cross_edges.append(np.vstack([chunk_parent_ids, nebor_svs]).T)
    cross_edges = np.concatenate(cross_edges)
    return cross_edges
