import numpy as np
import itertools
import time

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend.utils import column_keys

from multiwrapper import multiprocessing_utils as mu


def count_nodes_and_edges(table_id, n_threads=1):
    cg = chunkedgraph.ChunkedGraph(table_id)

    bounds = np.array(cg.cv.bounds.to_list()).reshape(2, -1).T
    bounds -= bounds[:, 0:1]

    chunk_id_bounds = np.ceil((bounds / cg.chunk_size[:, None])).astype(np.int)

    chunk_coord_gen = itertools.product(*[range(*r) for r in chunk_id_bounds])
    chunk_coords = np.array(list(chunk_coord_gen), dtype=np.int)

    order = np.arange(len(chunk_coords))
    np.random.shuffle(order)

    n_blocks = np.min([len(order), n_threads * 3])
    blocks = np.array_split(order, n_blocks)

    cg_serialized_info = cg.get_serialized_info()
    if n_threads > 1:
        del cg_serialized_info["credentials"]

    multi_args = []
    for block in blocks:
        multi_args.append([cg_serialized_info, chunk_coords[block]])

    if n_threads == 1:
        results = mu.multiprocess_func(_count_nodes_and_edges,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_count_nodes_and_edges,
                                          multi_args, n_threads=n_threads)

    n_edges_per_chunk = []
    n_nodes_per_chunk = []
    for result in results:
        n_nodes_per_chunk.extend(result[0])
        n_edges_per_chunk.extend(result[1])

    return n_nodes_per_chunk, n_edges_per_chunk


def _count_nodes_and_edges(args):
    serialized_cg_info, chunk_coords = args

    time_start = time.time()

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    n_edges_per_chunk = []
    n_nodes_per_chunk = []
    for chunk_coord in chunk_coords:
        x, y, z = chunk_coord
        rr = cg.range_read_chunk(layer=1, x=x, y=y, z=z)

        n_nodes_per_chunk.append(len(rr))
        n_edges = 0

        for k in rr.keys():
            n_edges += len(rr[k][column_keys.Connectivity.Partner][0].value)

        n_edges_per_chunk.append(n_edges)

    print(f"{len(chunk_coords)} took {time.time() - time_start}s")
    return n_nodes_per_chunk, n_edges_per_chunk