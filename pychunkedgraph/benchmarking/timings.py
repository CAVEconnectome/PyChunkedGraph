import matplotlib as mpl

try:
    mpl.use('Agg')
except:
    pass

import numpy as np
import itertools
import time
import os
import h5py
from functools import lru_cache
import pickle as pkl
from matplotlib import pyplot as plt
import glob

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend.utils import column_keys

from multiwrapper import multiprocessing_utils as mu


HOME = os.path.expanduser("~")


@lru_cache(maxsize=None)
def load_l1_l2_stats(save_folder):
    with h5py.File(f"{save_folder}/l1_l2_stats.h5", "r") as f:
        rep_l1_nodes = f["rep_l1_nodes"].value
        n_nodes_per_l2_node = f["n_nodes_per_l2_node"].value

    return rep_l1_nodes, n_nodes_per_l2_node


@lru_cache(maxsize=None)
def load_root_stats(save_folder):
    with h5py.File(f"{save_folder}/root_stats.h5", "r") as f:
        root_ids = f["root_ids"].value
        n_l1_nodes_per_root = f["n_l1_nodes_per_root"].value
        rep_l1_chunk_ids = f["rep_l1_chunk_ids"].value

    return root_ids, n_l1_nodes_per_root, rep_l1_chunk_ids


@lru_cache(maxsize=None)
def load_merge_stats(save_folder):
    with h5py.File(f"{save_folder}/merge_edge_stats.h5", "r") as f:
        merge_edges = f["merge_edges"].value
        merge_edge_weights = f["merge_edge_weights"].value

    return merge_edges, merge_edge_weights


def plot_scaling(re_path, key=8):
    save_dir = f"{os.path.dirname(os.path.dirname(re_path))}/scaling/"

    paths = sorted(glob.glob(re_path))
    save_name = f"{os.path.basename(os.path.dirname(paths[0]))[:-3]}_{os.path.basename(paths[0]).split('.')[0]}_key{key}"

    sizes = []
    percentiles = []
    for i_path, path in enumerate(paths):
        with open(path, "rb") as f:
            percentiles.append(pkl.load(f)[key]["percentiles"])
            sizes.append(i_path)

    percentiles = np.array(percentiles) * 1000
    sizes = np.array(sizes) + 2

    plt.figure(figsize=(10, 8))

    plt.tick_params(length=8, width=1.5, labelsize=20)
    plt.axes().spines['bottom'].set_linewidth(1.5)
    plt.axes().spines['left'].set_linewidth(1.5)
    plt.axes().spines['right'].set_linewidth(1.5)
    plt.axes().spines['top'].set_linewidth(1.5)

    plt.plot(sizes, percentiles[:, 98], marker="o", linestyle="--", lw=2, c=".6", markersize=10, label="p99")
    plt.plot(sizes, percentiles[:, 94], marker="o", linestyle="--", lw=2, c=".3", markersize=10, label="p95")
    plt.plot(sizes, percentiles[:, 49], marker="o", linestyle="-", lw=2, c="k", markersize=10, label="median")
    plt.plot(sizes, percentiles[:, 4], marker="o", linestyle="-", lw=2, c=".3", markersize=10, label="p05")
    plt.plot(sizes, percentiles[:, 0], marker="o", linestyle="-", lw=2, c=".6", markersize=10, label="p01")

    plt.ylim(0, np.max(percentiles) * 1.05)
    plt.xlim(1, np.max(sizes) * 1.05)


    plt.xlabel("Number of layers", fontsize=22)
    plt.ylabel("Time (ms)", fontsize=22)

    plt.legend(frameon=False, fontsize=18, loc="upper left")

    plt.tight_layout()

    plt.savefig(f"{save_dir}/{save_name}.png", dpi=300)
    plt.close()



def plot_timings(path):
    save_dir = os.path.dirname(path)
    save_name = os.path.basename(path).split(".")[0]

    with open(path, "rb") as f:
        timings = pkl.load(f)

    loads = []
    percentiles = []
    for k in timings:
        percentiles.append(timings[k]["percentiles"])
        loads.append(timings[k]["requests_per_s"])

    percentiles = np.array(percentiles) * 1000
    loads = np.array(loads)

    plt.figure(figsize=(10, 8))

    plt.tick_params(length=8, width=1.5, labelsize=20)
    plt.axes().spines['bottom'].set_linewidth(1.5)
    plt.axes().spines['left'].set_linewidth(1.5)
    plt.axes().spines['right'].set_linewidth(1.5)
    plt.axes().spines['top'].set_linewidth(1.5)

    plt.plot(loads, percentiles[:, 98], marker="o", linestyle="--", lw=2, c=".6", markersize=10, label="p99")
    plt.plot(loads, percentiles[:, 94], marker="o", linestyle="--", lw=2, c=".3", markersize=10, label="p95")
    plt.plot(loads, percentiles[:, 49], marker="o", linestyle="-", lw=2, c="k", markersize=10, label="median")
    plt.plot(loads, percentiles[:, 4], marker="o", linestyle="-", lw=2, c=".3", markersize=10, label="p05")
    plt.plot(loads, percentiles[:, 0], marker="o", linestyle="-", lw=2, c=".6", markersize=10, label="p01")

    plt.ylim(0, np.max(percentiles) * 1.05)
    plt.xlim(0, np.max(loads) * 1.05)

    plt.xlabel("Load (requests/s)", fontsize=22)
    plt.ylabel("Time (ms)", fontsize=22)

    plt.legend(frameon=False, fontsize=18, loc="upper left")

    plt.tight_layout()

    plt.savefig(f"{save_dir}/{save_name}.png", dpi=300)
    plt.close()


def plot_all_timings(save_dir=f"{HOME}/benchmarks/"):
    paths = glob.glob(f"{save_dir}/*/*.pkl")

    for path in paths:
        print(path)
        plot_timings(path)


def benchmark_root_timings(table_id, save_dir=f"{HOME}/benchmarks/",
                           job_size=500):
    save_folder = f"{save_dir}/{table_id}/"

    n_thread_list = [1, 4, 8, 16, 24, 32, 40, 48, 64]
    results = {}

    for n_threads in n_thread_list:
        results[n_threads] = get_root_timings(table_id, save_dir, job_size,
                                              n_threads=n_threads)

        print(n_threads, results[n_threads])

    with open(f"{save_folder}/root_timings_js{job_size}.pkl", "wb") as f:
        pkl.dump(results, f)

    return results


def get_root_timings(table_id, save_dir=f"{HOME}/benchmarks/", job_size=500,
                     n_threads=1):
    save_folder = f"{save_dir}/{table_id}/"

    rep_l1_nodes, n_nodes_per_l2_node = load_l1_l2_stats(save_folder)

    probs = n_nodes_per_l2_node / np.sum(n_nodes_per_l2_node)

    if n_threads == 1:
        n_jobs = n_threads * 3
    else:
        n_jobs = n_threads * 3

    cg = chunkedgraph.ChunkedGraph(table_id)
    cg_serialized_info = cg.get_serialized_info()
    if n_threads > 0:
        del cg_serialized_info["credentials"]

    time_start = time.time()
    np.random.seed(int(time.time()))

    if len(rep_l1_nodes) < job_size * 64 * 3 * 10:
        replace = True
    else:
        replace = False

    blocks = np.random.choice(rep_l1_nodes, job_size * n_jobs, p=probs,
                              replace=replace).reshape(n_jobs, job_size)

    multi_args = []
    for block in blocks:
        multi_args.append([cg_serialized_info, block])
    print(f"Building jobs took {time.time()-time_start}s")

    time_start = time.time()
    if n_threads == 1:
        results = mu.multiprocess_func(_get_root_timings,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_get_root_timings,
                                          multi_args, n_threads=n_threads)
    dt = time.time() - time_start

    timings = []
    for result in results:
        timings.extend(result)

    percentiles = [np.percentile(timings, k) for k in range(1, 100, 1)]
    mean = np.mean(timings)
    std = np.std(timings)
    median = np.median(timings)

    result_dict = {"percentiles": percentiles,
                   "p01": percentiles[0],
                   "p05": percentiles[4],
                   "p95": percentiles[94],
                   "p99": percentiles[98],
                   "mean": mean,
                   "std": std,
                   "median": median,
                   "total_time_s": dt,
                   "job_size": job_size,
                   "n_jobs": n_jobs,
                   "n_threads": n_threads,
                   "replace": replace,
                   "requests_per_s": job_size * n_jobs / dt}

    return result_dict


def _get_root_timings(args):
    serialized_cg_info, l1_ids = args
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    timings = []
    for l1_id in l1_ids:

        time_start = time.time()
        root = cg.get_root(l1_id)
        dt = time.time() - time_start
        timings.append(dt)

    return timings


def benchmark_subgraph_timings(table_id, save_dir=f"{HOME}/benchmarks/",
                               job_size=500):
    save_folder = f"{save_dir}/{table_id}/"

    n_thread_list = [1, 4, 8, 16, 24, 32, 40, 48, 64]
    results = {}

    for n_threads in n_thread_list:
        results[n_threads] = get_subgraph_timings(table_id, save_dir, job_size,
                                                  n_threads=n_threads)

        print(n_threads, results[n_threads])

    with open(f"{save_folder}/subgraph_timings_js{job_size}.pkl", "wb") as f:
        pkl.dump(results, f)

    return results


def get_subgraph_timings(table_id, save_dir=f"{HOME}/benchmarks/", job_size=500,
                     n_threads=1):
    save_folder = f"{save_dir}/{table_id}/"

    root_ids, n_l1_nodes_per_root, rep_l1_chunk_ids = load_root_stats(save_folder)

    probs = n_l1_nodes_per_root / np.sum(n_l1_nodes_per_root)

    if n_threads == 1:
        n_jobs = n_threads * 3
    else:
        n_jobs = n_threads * 3

    cg = chunkedgraph.ChunkedGraph(table_id)
    cg_serialized_info = cg.get_serialized_info()
    if n_threads > 0:
        del cg_serialized_info["credentials"]

    time_start = time.time()
    order = np.arange(len(n_l1_nodes_per_root))

    np.random.seed(int(time.time()))

    if len(order) < job_size * 64 * 3 * 10:
        replace = True
    else:
        replace = False

    blocks = np.random.choice(order, job_size * n_jobs, p=probs,
                              replace=replace).reshape(n_jobs, job_size)

    multi_args = []
    for block in blocks:
        multi_args.append([cg_serialized_info, root_ids[block],
                           rep_l1_chunk_ids[block]])
    print(f"Building jobs took {time.time()-time_start}s")

    time_start = time.time()
    if n_threads == 1:
        results = mu.multiprocess_func(_get_subgraph_timings,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_get_subgraph_timings,
                                          multi_args, n_threads=n_threads)
    dt = time.time() - time_start

    timings = []
    for result in results:
        timings.extend(result)

    percentiles = [np.percentile(timings, k) for k in range(1, 100, 1)]
    mean = np.mean(timings)
    std = np.std(timings)
    median = np.median(timings)

    result_dict = {"percentiles": percentiles,
                   "p01": percentiles[0],
                   "p05": percentiles[4],
                   "p95": percentiles[94],
                   "p99": percentiles[98],
                   "mean": mean,
                   "std": std,
                   "median": median,
                   "total_time_s": dt,
                   "job_size": job_size,
                   "n_jobs": n_jobs,
                   "n_threads": n_threads,
                   "replace": replace,
                   "requests_per_s": job_size * n_jobs / dt}

    return result_dict


def _get_subgraph_timings(args):
    serialized_cg_info, root_ids, rep_l1_chunk_ids = args
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    timings = []
    for root_id, rep_l1_chunk_id in zip(root_ids, rep_l1_chunk_ids):
        bb = np.array([rep_l1_chunk_id, rep_l1_chunk_id + 1], dtype=int)

        time_start = time.time()
        sv_ids = cg.get_subgraph_nodes(root_id, bb, bb_is_coordinate=False)
        dt = time.time() - time_start
        timings.append(dt)

    return timings


def benchmark_merge_split_timings(table_id, save_dir=f"{HOME}/benchmarks/",
                                  job_size=250):
    save_folder = f"{save_dir}/{table_id}/"

    n_thread_list = [1]
    merge_results = {}
    split_results = {}

    for n_threads in n_thread_list:
        results = get_merge_split_timings(table_id, save_dir, job_size,
                                          n_threads=n_threads)
        merge_results[n_threads] = results[0]
        split_results[n_threads] = results[1]

        print(n_threads, merge_results[n_threads])
        print(n_threads, split_results[n_threads])

    with open(f"{save_folder}/merge_timings_js{job_size}.pkl", "wb") as f:
        pkl.dump(merge_results, f)

    with open(f"{save_folder}/split_timings_js{job_size}.pkl", "wb") as f:
        pkl.dump(split_results, f)

    return merge_results, split_results


def get_merge_split_timings(table_id, save_dir=f"{HOME}/benchmarks/", job_size=500,
                      n_threads=1):
    save_folder = f"{save_dir}/{table_id}/"

    merge_edges, merge_edge_weights = load_merge_stats(save_folder)

    probs = merge_edge_weights / np.sum(merge_edge_weights)

    if n_threads == 1:
        n_jobs = n_threads * 3
    else:
        n_jobs = n_threads * 3

    cg = chunkedgraph.ChunkedGraph(table_id)
    cg_serialized_info = cg.get_serialized_info()
    if n_threads > 0:
        del cg_serialized_info["credentials"]

    time_start = time.time()
    order = np.arange(len(merge_edges))

    np.random.seed(int(time.time()))

    replace = False

    blocks = np.random.choice(order, job_size * n_jobs, p=probs,
                              replace=replace).reshape(n_jobs, job_size)

    multi_args = []
    for block in blocks:
        multi_args.append([cg_serialized_info, merge_edges[block]])
    print(f"Building jobs took {time.time()-time_start}s")

    time_start = time.time()
    if n_threads == 1:
        results = mu.multiprocess_func(_get_merge_timings,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_get_merge_timings,
                                          multi_args, n_threads=n_threads)
    dt = time.time() - time_start

    timings = []
    for result in results:
        timings.extend(result[0])

    percentiles = [np.percentile(timings, k) for k in range(1, 100, 1)]
    mean = np.mean(timings)
    std = np.std(timings)
    median = np.median(timings)

    merge_results = {"percentiles": percentiles,
                     "p01": percentiles[0],
                     "p05": percentiles[4],
                     "p95": percentiles[94],
                     "p99": percentiles[98],
                     "mean": mean,
                     "std": std,
                     "median": median,
                     "total_time_s": dt,
                     "job_size": job_size,
                     "n_jobs": n_jobs,
                     "n_threads": n_threads,
                     "replace": replace,
                     "requests_per_s": job_size * n_jobs / dt}

    timings = []
    for result in results:
        timings.extend(result[1])

    percentiles = [np.percentile(timings, k) for k in range(1, 100, 1)]
    mean = np.mean(timings)
    std = np.std(timings)
    median = np.median(timings)

    split_results = {"percentiles": percentiles,
                     "p01": percentiles[0],
                     "p05": percentiles[4],
                     "p95": percentiles[94],
                     "p99": percentiles[98],
                     "mean": mean,
                     "std": std,
                     "median": median,
                     "total_time_s": dt,
                     "job_size": job_size,
                     "n_jobs": n_jobs,
                     "n_threads": n_threads,
                     "replace": replace,
                     "requests_per_s": job_size * n_jobs / dt}

    return merge_results, split_results


def _get_merge_timings(args):
    serialized_cg_info, merge_edges = args
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    merge_timings = []
    for merge_edge in merge_edges:
        time_start = time.time()
        root_ids = cg.add_edges(user_id="ChuckNorris",
                                atomic_edges=[merge_edge]).new_root_ids
        dt = time.time() - time_start
        merge_timings.append(dt)

    split_timings = []
    for merge_edge in merge_edges:
        time_start = time.time()
        root_ids = cg.remove_edges(user_id="ChuckNorris",
                                   atomic_edges=[merge_edge],
                                   mincut=False).new_root_ids

        dt = time.time() - time_start
        split_timings.append(dt)

    return merge_timings, split_timings



def run_timings(table_id, save_dir=f"{HOME}/benchmarks/", job_size=500):
    benchmark_root_timings(table_id=table_id, save_dir=save_dir,
                           job_size=job_size)
    benchmark_subgraph_timings(table_id=table_id, save_dir=save_dir,
                               job_size=job_size)

