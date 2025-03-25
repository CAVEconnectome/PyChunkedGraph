import numpy as np
import itertools
import time
import os
import h5py
import pandas as pd

from pychunkedgraph.backend import chunkedgraph, chunkedgraph_comp
from pychunkedgraph.backend.utils import column_keys

from multiwrapper import multiprocessing_utils as mu


HOME = os.path.expanduser("~")


def count_nodes_and_edges(table_id, n_threads=1):
    cg = chunkedgraph.ChunkedGraph(table_id)

    bounds = np.array(cg.cv.bounds.to_list()).reshape(2, -1).T
    bounds -= bounds[:, 0:1]

    chunk_id_bounds = np.ceil((bounds / cg.chunk_size[:, None])).astype(int)

    chunk_coord_gen = itertools.product(*[range(*r) for r in chunk_id_bounds])
    chunk_coords = np.array(list(chunk_coord_gen), dtype=int)

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


def count_and_download_nodes(table_id, save_dir=f"{HOME}/benchmarks/",
                             n_threads=1):
    cg = chunkedgraph.ChunkedGraph(table_id)

    bounds = np.array(cg.cv.bounds.to_list()).reshape(2, -1).T
    bounds -= bounds[:, 0:1]

    chunk_id_bounds = np.ceil((bounds / cg.chunk_size[:, None])).astype(int)

    chunk_coord_gen = itertools.product(*[range(*r) for r in chunk_id_bounds])
    chunk_coords = np.array(list(chunk_coord_gen), dtype=int)

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
        results = mu.multiprocess_func(_count_and_download_nodes,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_count_and_download_nodes,
                                          multi_args, n_threads=n_threads)

    n_nodes_per_l2_node = []
    n_l2_nodes_per_chunk = []
    n_l1_nodes_per_chunk = []
    rep_l1_nodes = []
    for result in results:
        n_nodes_per_l2_node.extend(result[0])
        n_l2_nodes_per_chunk.extend(result[1])
        n_l1_nodes_per_chunk.extend(result[2])
        rep_l1_nodes.extend(result[3])

    save_folder = f"{save_dir}/{table_id}/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with h5py.File(f"{save_folder}/l1_l2_stats.h5", "w") as f:
        f.create_dataset("n_nodes_per_l2_node", data=n_nodes_per_l2_node,
                         compression="gzip")
        f.create_dataset("n_l2_nodes_per_chunk", data=n_l2_nodes_per_chunk,
                         compression="gzip")
        f.create_dataset("n_l1_nodes_per_chunk", data=n_l1_nodes_per_chunk,
                         compression="gzip")
        f.create_dataset("rep_l1_nodes", data=rep_l1_nodes,
                         compression="gzip")


def _count_and_download_nodes(args):
    serialized_cg_info, chunk_coords = args

    time_start = time.time()

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    n_nodes_per_l2_node = []
    n_l2_nodes_per_chunk = []
    n_l1_nodes_per_chunk = []
    # l1_nodes = []
    rep_l1_nodes = []
    for chunk_coord in chunk_coords:
        x, y, z = chunk_coord
        rr = cg.range_read_chunk(layer=2, x=x, y=y, z=z,
                                 columns=[column_keys.Hierarchy.Child])

        n_l2_nodes_per_chunk.append(len(rr))
        n_l1_nodes = 0

        for k in rr.keys():
            children = rr[k][column_keys.Hierarchy.Child][0].value
            rep_l1_nodes.append(children[np.random.randint(0, len(children))])
            # l1_nodes.extend(children)

            n_nodes_per_l2_node.append(len(children))
            n_l1_nodes += len(children)

        n_l1_nodes_per_chunk.append(n_l1_nodes)

    print(f"{len(chunk_coords)} took {time.time() - time_start}s")
    return n_nodes_per_l2_node, n_l2_nodes_per_chunk, n_l1_nodes_per_chunk, rep_l1_nodes


def get_root_ids_and_sv_chunks(table_id, save_dir=f"{HOME}/benchmarks/",
                               n_threads=1):
    cg = chunkedgraph.ChunkedGraph(table_id)

    save_folder = f"{save_dir}/{table_id}/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(f"{save_folder}/root_ids.h5"):
        root_ids = chunkedgraph_comp.get_latest_roots(cg, n_threads=n_threads)

        with h5py.File(f"{save_folder}/root_ids.h5", "w") as f:
            f.create_dataset("root_ids", data=root_ids)
    else:
        with h5py.File(f"{save_folder}/root_ids.h5", "r") as f:
            root_ids = f["root_ids"].value

    cg_serialized_info = cg.get_serialized_info()
    if n_threads > 1:
        del cg_serialized_info["credentials"]

    order = np.arange(len(root_ids))
    np.random.shuffle(order)

    order = order

    n_blocks = np.min([len(order), n_threads * 3])
    blocks = np.array_split(order, n_blocks)

    multi_args = []
    for block in blocks:
        multi_args.append([cg_serialized_info, root_ids[block]])

    if n_threads == 1:
        results = mu.multiprocess_func(_get_root_ids_and_sv_chunks,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_get_root_ids_and_sv_chunks,
                                          multi_args, n_threads=n_threads)

    root_ids = []
    n_l1_nodes_per_root = []
    rep_l1_nodes = []
    rep_l1_chunk_ids = []
    for result in results:
        root_ids.extend(result[0])
        n_l1_nodes_per_root.extend(result[1])
        rep_l1_nodes.extend(result[2])
        rep_l1_chunk_ids.extend(result[3])

    save_folder = f"{save_dir}/{table_id}/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with h5py.File(f"{save_folder}/root_stats.h5", "w") as f:
        f.create_dataset("root_ids", data=root_ids,
                         compression="gzip")
        f.create_dataset("n_l1_nodes_per_root", data=n_l1_nodes_per_root,
                         compression="gzip")
        f.create_dataset("rep_l1_nodes", data=rep_l1_nodes,
                         compression="gzip")
        f.create_dataset("rep_l1_chunk_ids", data=rep_l1_chunk_ids,
                         compression="gzip")


def _get_root_ids_and_sv_chunks(args):
    serialized_cg_info, root_ids = args

    time_start = time.time()

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    n_l1_nodes_per_root = []
    rep_l1_nodes = []
    rep_l1_chunk_ids = []
    for root_id in root_ids:
        l1_ids = cg.get_subgraph_nodes(root_id)

        n_l1_nodes_per_root.append(len(l1_ids))
        rep_l1_node = l1_ids[np.random.randint(0, len(l1_ids))]
        rep_l1_nodes.append(rep_l1_node)
        rep_l1_chunk_ids.append(cg.get_chunk_coordinates(rep_l1_node))

    print(f"{len(root_ids)} took {time.time() - time_start}s")
    return root_ids, n_l1_nodes_per_root, rep_l1_nodes, rep_l1_chunk_ids


def get_merge_candidates(table_id, save_dir=f"{HOME}/benchmarks/",
                             n_threads=1):
    cg = chunkedgraph.ChunkedGraph(table_id)

    bounds = np.array(cg.cv.bounds.to_list()).reshape(2, -1).T
    bounds -= bounds[:, 0:1]

    chunk_id_bounds = np.ceil((bounds / cg.chunk_size[:, None])).astype(int)

    chunk_coord_gen = itertools.product(*[range(*r) for r in chunk_id_bounds])
    chunk_coords = np.array(list(chunk_coord_gen), dtype=int)

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
        results = mu.multiprocess_func(_get_merge_candidates,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_get_merge_candidates,
                                          multi_args, n_threads=n_threads)
    merge_edges = []
    merge_edge_weights = []
    for result in results:
        merge_edges.extend(result[0])
        merge_edge_weights.extend(result[1])

    save_folder = f"{save_dir}/{table_id}/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with h5py.File(f"{save_folder}/merge_edge_stats.h5", "w") as f:
        f.create_dataset("merge_edges", data=merge_edges,
                         compression="gzip")
        f.create_dataset("merge_edge_weights", data=merge_edge_weights,
                         compression="gzip")


def _get_merge_candidates(args):
    serialized_cg_info, chunk_coords = args

    time_start = time.time()

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    merge_edges = []
    merge_edge_weights = []
    for chunk_coord in chunk_coords:
        chunk_id = cg.get_chunk_id(layer=1, x=chunk_coord[0],
                                   y=chunk_coord[1], z=chunk_coord[2])

        rr = cg.range_read_chunk(chunk_id=chunk_id,
                                 columns=[column_keys.Connectivity.Partner,
                                          column_keys.Connectivity.Connected,
                                          column_keys.Hierarchy.Parent])

        ps = []
        edges = []
        for it in rr.items():
            e, _, _ = cg._retrieve_connectivity(it, connected_edges=False)
            edges.extend(e)
            ps.extend([it[1][column_keys.Hierarchy.Parent][0].value] * len(e))

        if len(edges) == 0:
            continue

        edges = np.sort(np.array(edges), axis=1)
        cols = {"sv1": edges[:, 0], "sv2": edges[:, 1], "parent": ps}

        df = pd.DataFrame(data=cols)
        dfg = df.groupby(["sv1", "sv2"]).aggregate(np.sum).reset_index()

        _, i, c = np.unique(dfg[["parent"]], return_counts=True,
                            return_index=True)

        merge_edges.extend(np.array(dfg.loc[i][["sv1", "sv2"]],
                                    dtype=np.uint64))
        merge_edge_weights.extend(c)


    print(f"{len(chunk_coords)} took {time.time() - time_start}s")

    return merge_edges, merge_edge_weights



def run_graph_measurements(table_id, save_dir=f"{HOME}/benchmarks/",
                           n_threads=1):
    get_root_ids_and_sv_chunks(table_id=table_id, save_dir=save_dir,
                               n_threads=n_threads)
    count_and_download_nodes(table_id=table_id, save_dir=save_dir,
                             n_threads=n_threads)
    get_merge_candidates(table_id=table_id, save_dir=save_dir,
                         n_threads=n_threads)

