"""
Proposed stitch v2: multiwave with in-memory cache retention between waves.

Cache is a module-level global. Parent process updates it between waves.
Workers inherit it via fork COW — no serialization needed for reads.
Workers return cache snapshots, parent merges for next wave.
"""

import json
import random
import time
from multiprocessing import Pool

from tqdm import tqdm

from pychunkedgraph.graph import ChunkedGraph

from .proposed import stitch
from .reader import CachedReader
from .tables import setup_env
from .utils import _convert_for_json
from .worker_utils import (
    pool_init,
    create_cg_worker,
    load_edges,
    default_n_workers,
    list_wave_files,
    list_all_waves,
)

_shared_cache = None  # (parents_dict, children_dict, acx_dict) — set by parent before Pool


def _worker_v2(args: tuple) -> tuple:
    graph_id, edge_path, idx = args
    setup_env()

    edges = load_edges(edge_path)
    cg = create_cg_worker(graph_id)

    reader = CachedReader(cg, preloaded=_shared_cache)

    t0 = time.time()
    result = stitch(cg, edges, verbose=False, reader=reader)
    t_stitch = time.time() - t0

    t0 = time.time()
    cg.client.write(result.entries)
    t_write = time.time() - t0

    ctx = result.ctx
    cache_snapshot = (
        reader._parents_local,
        reader._children_local,
        reader._acx_local,
        dict(ctx.parents_cache),
        dict(ctx.children_cache),
        dict(ctx.l2_atomic_cx),
    )

    file_result = {
        "idx": idx,
        "edge_file": edge_path,
        "n_edges": len(edges),
        "new_roots": [int(r) for r in result.new_roots],
        "elapsed": t_stitch + t_write,
        "t_stitch": t_stitch,
        "t_write": t_write,
        "perf": result.perf,
    }
    result.ctx = None
    return file_result, cache_snapshot


_inprocess_cg = None


def _run_single_inprocess(table_name: str, edge_path: str, idx: int) -> dict:
    """Run one file in-process with _shared_cache, update cache directly."""
    global _inprocess_cg, _shared_cache
    if _inprocess_cg is None or _inprocess_cg.graph_id != table_name:
        _inprocess_cg = ChunkedGraph(graph_id=table_name)
    cg = _inprocess_cg

    edges = load_edges(edge_path)
    reader = CachedReader(cg, preloaded=_shared_cache)

    t0 = time.time()
    result = stitch(cg, edges, verbose=False, reader=reader)
    t_stitch = time.time() - t0

    t0 = time.time()
    cg.client.write(result.entries)
    t_write = time.time() - t0

    if _shared_cache is None:
        _shared_cache = ({}, {}, {})
    parents, children, acx = _shared_cache
    parents.update(reader._parents_local)
    children.update(reader._children_local)
    acx.update(reader._acx_local)
    ctx = result.ctx
    parents.update(ctx.parents_cache)
    children.update(ctx.children_cache)
    acx.update(ctx.l2_atomic_cx)

    return {
        "idx": idx,
        "edge_file": edge_path,
        "n_edges": len(edges),
        "new_roots": [int(r) for r in result.new_roots],
        "elapsed": t_stitch + t_write,
        "t_stitch": t_stitch,
        "t_write": t_write,
        "perf": result.perf,
    }


def run_wave_v2(
    table_name: str,
    wave: int,
    shared_init: tuple,
    n_workers: int = None,
) -> tuple:
    """Run a wave. Returns (file_results, elapsed)."""
    edge_files = list_wave_files(wave)
    n = len(edge_files)
    args = [(table_name, f, i) for i, f in enumerate(edge_files)]
    random.shuffle(args)

    if n == 1:
        t0 = time.time()
        file_results = [_run_single_inprocess(*args[0])]
        elapsed = time.time() - t0
        return file_results, elapsed

    if n_workers is None:
        n_workers = default_n_workers(n)

    meta_bytes, cv_info = shared_init
    t0 = time.time()
    file_results = []
    with Pool(n_workers, initializer=pool_init, initargs=(meta_bytes, cv_info)) as pool:
        for file_result, cache_snapshot in tqdm(
            pool.imap_unordered(_worker_v2, args),
            total=n,
            desc=f"wave {wave}",
        ):
            file_results.append(file_result)
            _merge_snapshot(cache_snapshot)
    elapsed = time.time() - t0
    return file_results, elapsed


def _merge_snapshot(snapshot: tuple) -> None:
    """Merge a single worker's cache snapshot into _shared_cache."""
    global _shared_cache
    rp, rc, ra, cp, cc, ca = snapshot
    if _shared_cache is None:
        _shared_cache = ({}, {}, {})
    parents, children, acx = _shared_cache
    parents.update(rp)
    children.update(rc)
    acx.update(ra)
    parents.update(cp)
    children.update(cc)
    acx.update(ca)


def run_multiwave_v2(
    table_name: str, shared_init: tuple, n_workers: int = None, save_path: str = None
) -> tuple:
    """Run all waves with cache retention. Returns (all_file_results, total_wall)."""
    global _shared_cache
    _shared_cache = None

    waves = list_all_waves()
    all_file_results = []
    total_wall = 0

    for wave in waves:
        file_results, elapsed = run_wave_v2(table_name, wave, shared_init, n_workers)
        all_file_results.extend(file_results)
        total_wall += elapsed
        total_req = sum(n for r in file_results for _, n, _, _ in r.get("perf", {}).get("rpc_log", []))
        total_read = sum(n for r in file_results for _, _, n, _ in r.get("perf", {}).get("rpc_log", []))
        hit_pct = (total_req - total_read) / total_req * 100 if total_req else 0
        cache_size = f"{len(_shared_cache[0])} parents, {len(_shared_cache[1])} children, {len(_shared_cache[2])} acx" if _shared_cache else "empty"
        print(f"  wave {wave}: {elapsed:.1f}s, {total_read}/{total_req} reads ({hit_pct:.0f}% cached), cache: {cache_size}")

        if save_path:
            with open(save_path, "w") as f:
                json.dump(_convert_for_json({
                    "file_results": all_file_results,
                    "elapsed": total_wall,
                    "table_name": table_name,
                }), f)

    return all_file_results, total_wall
