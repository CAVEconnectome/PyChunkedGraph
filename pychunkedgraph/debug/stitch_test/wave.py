"""
Stitch test runner: unified entry points for single, wave, and multiwave experiments.
Uses multiprocessing to run edge files in parallel within each wave.
"""

import json
import os
import pickle
import random
import shutil
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

import numpy as np
from cloudfiles import CloudFile, CloudFiles
from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph, basetypes


from .current import run_current_stitch
from . import proposed as proposed_mod
from . import reader as reader_mod
from .proposed import stitch, run_proposed_stitch
from .tables import restore_test_table, setup_env, set_autoscaling, set_autoscaling_cpu, PREFIX, EDGES_SRC
from .utils import extract_structure, batched_extract_structure, batched_extract_and_compare, layer_counts_from_shards
from .compare import (
    LOGS_ROOT,
    generate_run_id,
    _save_run_result,
    _load_result,
    compare_stitch_results,
)
from .utils import _convert_for_json

_CG_INIT_RETRIES = 5
_CG_INIT_DELAY = 2

# set per-worker via Pool initializer
_worker_meta = None
_worker_cv_info = None


def _pool_init(meta_bytes, cv_info):
    """Pool initializer: deserialize meta + cv_info once per worker process."""
    global _worker_meta, _worker_cv_info
    _worker_meta = pickle.loads(meta_bytes)
    _worker_cv_info = cv_info


def _create_cg_worker(graph_id):
    """Create ChunkedGraph in worker using pre-loaded meta. No BigTable or GCS reads."""
    if _worker_meta is not None:
        cg = ChunkedGraph(meta=_worker_meta)
        if _worker_cv_info is not None:
            cg.meta._ws_cv = CloudVolume(
                cg.meta._data_source.WATERSHED, info=_worker_cv_info, progress=False
            )
        return cg
    return _create_cg(graph_id)


def _create_cg(graph_id):
    """Create ChunkedGraph with retry — meta read can transiently fail."""
    for attempt in range(_CG_INIT_RETRIES):
        cg = ChunkedGraph(graph_id=graph_id)
        if cg.meta is not None:
            return cg
        time.sleep(_CG_INIT_DELAY * (attempt + 1))
    raise RuntimeError(f"ChunkedGraph meta is None after {_CG_INIT_RETRIES} retries for {graph_id}")


def _prepare_shared_init(table_name):
    """Read meta + cv info once in parent process, return bytes for Pool initializer."""
    cg = _create_cg(table_name)
    cv_info = cg.meta.ws_cv.info
    meta_bytes = pickle.dumps(cg.meta)
    return meta_bytes, cv_info




def list_wave_files(wave: int = 0) -> list[str]:
    """List all edge files for a wave, sorted by subtask index."""
    cf = CloudFiles(EDGES_SRC)
    prefix = f"task_{wave}_"
    files = [f for f in cf.list() if f.startswith(prefix)]
    files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]))
    return [f"{EDGES_SRC}/{f}" for f in files]


def list_all_waves() -> list[int]:
    """List all wave indices available."""
    cf = CloudFiles(EDGES_SRC)
    waves = set()
    for f in cf.list():
        parts = f.replace(".edges", "").split("_")
        if len(parts) >= 3 and parts[0] == "task":
            waves.add(int(parts[1]))
    return sorted(waves)


def _load_edges(path: str) -> np.ndarray:
    return np.asarray(pickle.loads(CloudFile(path).get()), dtype=basetypes.NODE_ID)


def _default_n_workers(n_tasks):
    return min(n_tasks, 4 * os.cpu_count())




def _worker_current(args):
    graph_id, edge_path, idx = args
    setup_env()
    os.environ["PCG_PROFILER_ENABLED"] = "0"

    edges = _load_edges(edge_path)
    cg = _create_cg_worker(graph_id)

    t0 = time.time()
    result = cg.add_edges(
        user_id="test",
        atomic_edges=edges,
        stitch_mode=True,
        allow_same_segment_merge=True,
        do_sanity_check=False,
    )
    elapsed = time.time() - t0
    return {
        "idx": idx,
        "edge_file": edge_path,
        "n_edges": len(edges),
        "new_roots": [int(r) for r in result.new_root_ids],
        "elapsed": elapsed,
    }


def _worker_proposed(args):
    graph_id, edge_path, idx = args
    setup_env()

    edges = _load_edges(edge_path)
    cg = _create_cg_worker(graph_id)


    t0 = time.time()
    result = stitch(cg, edges, verbose=False)
    t_stitch = time.time() - t0

    t0 = time.time()
    cg.client.write(result.node_entries)
    cg.client.write(result.parent_entries)
    t_write = time.time() - t0

    elapsed = t_stitch + t_write
    return {
        "idx": idx,
        "edge_file": edge_path,
        "n_edges": len(edges),
        "new_roots": [int(r) for r in result.new_roots],
        "elapsed": elapsed,
        "t_stitch": t_stitch,
        "t_write": t_write,
        "perf": result.perf,
    }




def _run_wave(table_name, wave, worker_fn, n_workers=None):
    """Run all files in a wave in parallel. Returns list of per-file results."""
    edge_files = list_wave_files(wave)
    if n_workers is None:
        n_workers = _default_n_workers(len(edge_files))
    n = len(edge_files)

    meta_bytes, cv_info = _prepare_shared_init(table_name)
    args = [(table_name, f, i) for i, f in enumerate(edge_files)]
    random.shuffle(args)
    t0 = time.time()
    file_results = []
    with Pool(n_workers, initializer=_pool_init, initargs=(meta_bytes, cv_info)) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_fn, args),
            total=n,
            desc=f"wave {wave}",
        ):
            file_results.append(result)
    elapsed = time.time() - t0
    return file_results, elapsed




def _clear_log_dir(log_dir):
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


SINGLE_EDGE_FILE = "task_0_591.edges"


def run_current(experiment: str = "single", n_workers: int = None):
    """
    Run the current stitch path.
      "single"    — one file (task_0_0.edges)
      "wave"      — all files in wave 0 in parallel
      "multiwave" — all waves sequentially, files within each wave in parallel
    """
    setup_env()
    table_name = f"{PREFIX}hsmith_mec_current_{experiment}"
    log_dir = LOGS_ROOT / experiment / "current"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[current] experiment={experiment}" + (f", file={SINGLE_EDGE_FILE}" if experiment == "single" else ""))
    if experiment != "single":
        set_autoscaling(target_cpu=25, min_nodes=5)
    try:
        return _run_current_impl(experiment, table_name, log_dir, n_workers)
    finally:
        if experiment != "single":
            set_autoscaling(target_cpu=60, min_nodes=1)


def _run_current_impl(experiment, table_name, log_dir, n_workers):

    # check for resumable stitch (extraction crashed after stitch completed)
    stitch_results_path = log_dir / "stitch_results.json"
    if stitch_results_path.exists():
        print("resuming from saved stitch results...")
        with open(stitch_results_path) as f:
            saved = json.load(f)
        all_file_results = saved["file_results"]
        total_wall = saved["elapsed"]
    elif experiment == "single":
        _clear_log_dir(log_dir)
        restore_test_table(table_name)
        edge_file = f"{EDGES_SRC}/{SINGLE_EDGE_FILE}"
        edges = _load_edges(edge_file)
        result = run_current_stitch(table_name, edges, do_sanity_check=False)
        _save_run_result(log_dir, "current", result)
        print(f"current done: {result.elapsed:.1f}s")
        return result
    else:
        _clear_log_dir(log_dir)
        restore_test_table(table_name)
        waves = [0] if experiment == "wave" else list_all_waves()
        all_file_results = []
        total_wall = 0
        for wave in waves:
            file_results, wave_wall = _run_wave(table_name, wave, _worker_current, n_workers)
            all_file_results.extend(file_results)
            total_wall += wave_wall
        # save stitch results immediately so we can resume
        with open(stitch_results_path, "w") as f:
            json.dump(_convert_for_json({"file_results": all_file_results, "elapsed": total_wall}), f)

    all_roots = []
    for fr in all_file_results:
        all_roots.extend(fr["new_roots"])
    roots_arr = np.array(all_roots, dtype=basetypes.NODE_ID)

    print(f"extracting structure: {len(roots_arr)} roots...")
    t0 = time.time()
    struct_dir = batched_extract_structure(table_name, roots_arr, save_dir=log_dir)
    t_struct = time.time() - t0
    print(f"structure extracted: {t_struct:.1f}s")

    result = {
        "new_roots": [int(r) for r in roots_arr],
        "elapsed": total_wall,
        "table_name": table_name,
        "n_files": len(all_file_results),
        "n_waves": len(waves) if "waves" in dir() else 0,
        "layer_counts": layer_counts_from_shards(struct_dir),
        "file_results": all_file_results,
    }
    with open(log_dir / "wave_meta.json", "w") as f:
        json.dump(_convert_for_json(result), f, indent=2)

    total_stitch = sum(r["elapsed"] for r in all_file_results)
    print(f"\ncurrent {experiment}: {total_wall:.1f}s wall, {total_stitch:.1f}s total stitch, {len(roots_arr)} roots")
    return result


def run_proposed_and_compare(experiment: str = "single", n_workers: int = None, run_id: str = None):
    """
    Run the proposed stitch path and compare against the current baseline.
      "single"    — one file (task_0_0.edges)
      "wave"      — all files in wave 0 in parallel
      "multiwave" — all waves sequentially, files within each wave in parallel
    Pass run_id to resume a failed run (skips stitch, goes to extraction/comparison).
    Returns (match, result_current, result_proposed).
    """
    setup_env()
    if experiment != "single":
        set_autoscaling(target_cpu=25, min_nodes=5)
    try:
        return _run_proposed_impl(experiment, n_workers, run_id)
    finally:
        if experiment != "single":
            set_autoscaling(target_cpu=60, min_nodes=1)


def _run_proposed_impl(experiment, n_workers, run_id):
    resume = run_id is not None
    if not resume:
        run_id = generate_run_id()
    log_dir = LOGS_ROOT / experiment / run_id
    stitch_results_path = log_dir / "stitch_results.json"

    print(f"[proposed] experiment={experiment}, run_id={run_id}{' (resume)' if resume else ''}" + (f", file={SINGLE_EDGE_FILE}" if experiment == "single" else ""))
    print(f"logs: {log_dir}")

    current_log_dir = LOGS_ROOT / experiment / "current"
    result_current = _load_current_result(current_log_dir) if experiment == "single" else None

    if resume and stitch_results_path.exists():
        print("resuming from saved stitch results...")
        with open(stitch_results_path) as f:
            saved = json.load(f)
        table_name = saved["table_name"]
        if experiment == "single":
            result_proposed = saved
        else:
            all_file_results = saved["file_results"]
            total_wall = saved["elapsed"]
    else:
        if not resume:
            _clear_log_dir(log_dir)
        table_name = f"{PREFIX}hsmith_mec_{run_id}_proposed"
        restore_test_table(table_name)

        if experiment == "single":
            reader_mod.VERBOSE = True
            edge_file = f"{EDGES_SRC}/{SINGLE_EDGE_FILE}"
            edges = _load_edges(edge_file)
            result_proposed = run_proposed_stitch(table_name, edges)
            reader_mod.VERBOSE = False
            result_proposed.table_name = table_name
            _save_run_result(log_dir, "proposed", result_proposed)
            with open(stitch_results_path, "w") as f:
                json.dump(_convert_for_json(result_proposed.meta), f)
        else:
            waves = [0] if experiment == "wave" else list_all_waves()
            all_file_results = []
            total_wall = 0
            for wave in waves:
                file_results, wave_wall = _run_wave(table_name, wave, _worker_proposed, n_workers)
                all_file_results.extend(file_results)
                total_wall += wave_wall
            with open(stitch_results_path, "w") as f:
                json.dump(_convert_for_json({
                    "file_results": all_file_results,
                    "elapsed": total_wall,
                    "table_name": table_name,
                }), f)

    if experiment == "single":
        t_pro = result_proposed.elapsed
        t_cur = result_current.elapsed
        print(f"\ncurrent: {t_cur:.1f}s, proposed: {t_pro:.1f}s")
        match = compare_stitch_results(result_current, result_proposed)
    else:
        all_roots = []
        for fr in all_file_results:
            all_roots.extend(fr["new_roots"])
        roots_proposed = np.array(all_roots, dtype=basetypes.NODE_ID)

        total_stitch = sum(r["elapsed"] for r in all_file_results)
        total_edges = sum(r["n_edges"] for r in all_file_results)
        print(f"\nproposed {experiment}: {total_wall:.1f}s wall, {total_stitch:.1f}s total stitch, {len(roots_proposed)} roots, {total_edges} edges")

        with open(current_log_dir / "wave_meta.json") as f:
            current_meta = json.load(f)
        roots_current = np.array(current_meta["new_roots"], dtype=basetypes.NODE_ID)
        table_current = current_meta["table_name"]

        print(f"\ncomparing: {len(roots_current)} current vs {len(roots_proposed)} proposed roots")
        match = batched_extract_and_compare(
            table_current, roots_current,
            table_name, roots_proposed,
            save_dir=log_dir,
            current_extract_dir=current_log_dir,
        )

        result_proposed = {
            "new_roots": [int(r) for r in roots_proposed],
            "elapsed": total_wall,
            "table_name": table_name,
            "n_files": len(all_file_results),
            "n_waves": len(waves) if "waves" in dir() else 0,
            "file_results": all_file_results,
        }
        with open(log_dir / "wave_meta.json", "w") as f:
            json.dump(_convert_for_json(result_proposed), f, indent=2)

    summary = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "match": match,
        "time_current": result_current.elapsed if result_current else 0,
        "time_proposed": total_wall if experiment != "single" else result_proposed.elapsed,
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(_convert_for_json(summary), f, indent=2)

    print(f"\n{'MATCH' if match else 'MISMATCH'}")

    return match, result_current, result_proposed


def _load_current_result(log_dir):
    """Load current baseline result from saved files."""
    meta_path = log_dir / "current_meta.json"
    struct_path = log_dir / "current_structure.pkl.gz"
    if not meta_path.exists() or not struct_path.exists():
        raise FileNotFoundError(
            f"current baseline not found in {log_dir}. Run run_current('{log_dir.parent.name}') first."
        )
    return _load_result(log_dir, "current")
