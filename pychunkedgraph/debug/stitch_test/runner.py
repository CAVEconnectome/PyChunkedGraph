"""
Multiwave orchestration: baseline, proposed stitch, comparison.

Entry points: run_baseline(experiment), run_proposed(experiment), compare_run(run_id)
"""

import json
import logging
import os
import pickle
import random
import time
import uuid
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil
from cloudfiles import CloudFiles
from tqdm import tqdm

from pychunkedgraph.graph import ChunkedGraph

from .local_cg import LocalChunkedGraph
from .stitch import stitch
from .tables import restore_test_table, set_autoscaling
from .utils import _convert_for_json, batched_extract_and_compare

log = logging.getLogger(__name__)

_process = psutil.Process()


def _mem_str() -> str:
    return f"{_process.memory_info().rss / 1024**2:.0f}MB"


EDGES_SRC = "gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges"
LOGS_ROOT = Path(__file__).parent.parent.parent.parent / ".env" / "stitching" / "runs"
REFERENCE_PATH = LOGS_ROOT / "reference.json"
SINGLE_EDGE_FILE = "task_0_591.edges"

_shared_cache = None
_shared_inc = None
_reference = None
_edge_files_cache = None


def _load_edges(path: str) -> np.ndarray:
    cf = CloudFiles(EDGES_SRC)
    return pickle.loads(cf.get(path))


def _get_edge_files() -> list:
    global _edge_files_cache
    if _edge_files_cache is None:
        cf = CloudFiles(EDGES_SRC)
        _edge_files_cache = [
            f for f in cf.list()
            if f.startswith("task_") and f.endswith(".edges")
        ]
    return _edge_files_cache


def _list_wave_files(wave: int) -> list:
    prefix = f"task_{wave}_"
    return sorted(
        [f for f in _get_edge_files() if f.startswith(prefix)],
        key=lambda f: int(f.split("_")[2].split(".")[0]),
    )


def _list_all_waves() -> list:
    waves = set()
    for f in _get_edge_files():
        waves.add(int(f.split("_")[1]))
    return sorted(waves)


def _default_n_workers(n_tasks: int) -> int:
    return min(n_tasks, 3 * os.cpu_count())


def _load_reference() -> dict:
    global _reference
    if _reference is None:
        if REFERENCE_PATH.exists():
            with open(REFERENCE_PATH) as f:
                _reference = json.load(f)
        else:
            _reference = {}
    return _reference


def _check_roots(edge_file: str, roots_before: int, new_roots: int) -> None:
    ref = _load_reference()
    entry = ref.get(edge_file)
    if entry is None:
        return
    if roots_before != entry["roots_before"]:
        raise AssertionError(
            f"ROOTS BEFORE MISMATCH {edge_file}: got {roots_before}, expected {entry['roots_before']}"
        )
    if new_roots != entry["new_roots"]:
        raise AssertionError(
            f"NEW ROOTS MISMATCH {edge_file}: got {new_roots}, expected {entry['new_roots']}"
        )


def _worker_proposed(args: tuple) -> tuple:
    graph_id, edge_path, idx = args
    edges = _load_edges(edge_path)
    lcg = LocalChunkedGraph.create_worker(graph_id, preloaded=_shared_cache, incremental=_shared_inc)
    lcg.begin_stitch()

    roots_before = len(np.unique(lcg.get_roots(edges.ravel())))

    t0 = time.time()
    result = stitch(lcg, edges)
    t_stitch = time.time() - t0

    t0 = time.time()
    lcg.mutate_rows(result.rows)
    t_write = time.time() - t0

    _check_roots(edge_path, roots_before, len(result.new_roots))

    snapshot = lcg.wave_snapshot()

    file_result = {
        "idx": idx, "edge_file": edge_path, "n_edges": len(edges),
        "new_roots": [int(r) for r in result.new_roots],
        "roots_before": roots_before,
        "elapsed": t_stitch + t_write, "t_stitch": t_stitch, "t_write": t_write,
        "n_rows": len(result.rows),
        "perf": result.perf,
    }
    return file_result, snapshot


def _run_wave_pool(
    pool: Pool, table_name: str, wave: int, lcg: LocalChunkedGraph,
    edge_files: list = None,
) -> tuple:
    edge_files = edge_files or _list_wave_files(wave)
    args = [(table_name, f, i) for i, f in enumerate(edge_files)]
    random.shuffle(args)

    t0 = time.time()
    file_results = []
    snapshots = []

    for file_result, snapshot in tqdm(
        pool.imap_unordered(_worker_proposed, args),
        total=len(args), desc=f"wave {wave}",
    ):
        file_results.append(file_result)
        snapshots.append(snapshot)

    lcg.merge_wave_results(snapshots)
    elapsed = time.time() - t0
    return file_results, elapsed


def _run_wave_inprocess(
    table_name: str, wave: int, lcg: LocalChunkedGraph,
) -> tuple:
    edge_files = _list_wave_files(wave)
    assert len(edge_files) == 1

    edges = _load_edges(edge_files[0])
    lcg.begin_stitch()

    roots_before = len(np.unique(lcg.get_roots(edges.ravel())))

    t0 = time.time()
    result = stitch(lcg, edges)
    t_stitch = time.time() - t0

    t0_w = time.time()
    lcg.mutate_rows(result.rows)
    t_write = time.time() - t0_w

    _check_roots(edge_files[0], roots_before, len(result.new_roots))
    lcg.end_stitch()
    elapsed = time.time() - t0

    file_result = {
        "idx": 0, "edge_file": edge_files[0], "n_edges": len(edges),
        "new_roots": [int(r) for r in result.new_roots],
        "roots_before": roots_before,
        "elapsed": t_stitch + t_write, "t_stitch": t_stitch, "t_write": t_write,
        "n_rows": len(result.rows),
        "perf": result.perf,
    }
    return [file_result], elapsed


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _log_wave_stats(wave: int, results: list, elapsed: float, lcg: LocalChunkedGraph) -> None:
    total_req = sum(n for r in results for _, n, _, _ in r.get("perf", {}).get("rpc_log", []))
    total_read = sum(n for r in results for _, _, n, _ in r.get("perf", {}).get("rpc_log", []))
    hit_pct = (total_req - total_read) / total_req * 100 if total_req else 0
    n_roots = sum(len(r.get("new_roots", [])) for r in results)
    n_rows = sum(r.get("n_rows", 0) for r in results)
    t_stitch = sum(r.get("t_stitch", 0) for r in results)
    t_write = sum(r.get("t_write", 0) for r in results)
    print(
        f"  wave {wave}: {elapsed:.0f}s {len(results)}f {_fmt(n_roots)} roots "
        f"stitch={t_stitch:.0f}s write={t_write:.0f}s {_fmt(n_rows)} rows "
        f"{_fmt(total_read)}/{_fmt(total_req)} reads({hit_pct:.0f}%) "
        f"{_mem_str()} {lcg.stats()}"
    )


def _save_results(path: str, results: list, elapsed: float, table: str, wave_walls: dict = None) -> None:
    if not path:
        return
    data = {"file_results": results, "elapsed": elapsed, "table_name": table}
    if wave_walls:
        data["wave_walls"] = wave_walls
    with open(path, "w") as f:
        json.dump(_convert_for_json(data), f)


def run_proposed(
    experiment: str = "multiwave", n_workers: int = None,
    baseline_table: str = "stitch_redesign_test_cmp_prod_wave0",
    baseline_roots_file: str = "/tmp/wave0_prod_checkpoint.pkl",
    filename: str = None,
) -> tuple:
    global _shared_cache, _shared_inc

    run_id = uuid.uuid4().hex[:8]
    log_dir = LOGS_ROOT / experiment / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    table_name = f"stitch_redesign_test_hsmith_{run_id}"

    print(f"[proposed] run_id={run_id}")
    print(f"logs: {log_dir}")

    restore_test_table(table_name)
    set_autoscaling(target_cpu=25, min_nodes=5)

    if filename:
        waves = [0]
    elif experiment == "wave":
        waves = [0]
    else:
        waves = _list_all_waves()

    lcg = LocalChunkedGraph(table_name)
    init = lcg.prepare_pool_init()
    all_results = []
    wave_walls = {}
    total_wall = 0
    results_path = str(log_dir / "stitch_results.json")

    # Wave 0
    wave0_files = [filename] if filename else _list_wave_files(waves[0])
    nw0 = n_workers or _default_n_workers(len(wave0_files))
    _shared_cache = lcg.preloaded()
    _shared_inc = None

    with Pool(nw0, initializer=LocalChunkedGraph.pool_init, initargs=init) as pool:
        results, elapsed = _run_wave_pool(pool, table_name, waves[0], lcg, edge_files=wave0_files)
    all_results.extend(results)
    wave_walls[waves[0]] = elapsed
    total_wall += elapsed
    _log_wave_stats(waves[0], results, elapsed, lcg)
    _save_results(results_path, all_results, total_wall, table_name, wave_walls)

    # Waves 1+
    min_nodes_reset = False
    for wave in waves[1:]:
        files = _list_wave_files(wave)
        _shared_cache = lcg.preloaded()
        _shared_inc = lcg.incremental_state()

        if len(files) == 1:
            results, elapsed = _run_wave_inprocess(table_name, wave, lcg)
        else:
            nw = n_workers or _default_n_workers(len(files))
            with Pool(nw, initializer=LocalChunkedGraph.pool_init, initargs=init) as pool:
                results, elapsed = _run_wave_pool(pool, table_name, wave, lcg)

        all_results.extend(results)
        wave_walls[wave] = elapsed
        total_wall += elapsed
        _log_wave_stats(wave, results, elapsed, lcg)
        _save_results(results_path, all_results, total_wall, table_name, wave_walls)

        if not min_nodes_reset:
            set_autoscaling(min_nodes=1)
            min_nodes_reset = True

        errors = [r for r in results if r.get("error")]
        if errors:
            print(f"STOPPING: {len(errors)} errors in wave {wave}")
            for r in errors:
                print(f"  {r['edge_file']}: {r.get('error_msg')}")
            break

    n_roots = sum(len(r.get("new_roots", [])) for r in all_results)
    n_edges = sum(r["n_edges"] for r in all_results)
    total_stitch = sum(r.get("t_stitch", r["elapsed"]) for r in all_results)
    print(
        f"\nproposed {experiment}: {total_wall:.1f}s wall, "
        f"{total_stitch:.1f}s total stitch, {n_roots} roots, {n_edges} edges\n"
        f"memory: {_mem_str()}"
    )

    # Compare
    proposed_roots = [r for fr in all_results for r in fr["new_roots"]]

    if baseline_table and baseline_roots_file:
        _bl_table = baseline_table
        with open(baseline_roots_file, "rb") as f:
            import pickle as _pkl
            baseline_roots = _pkl.load(f)
        _bl_extract_dir = None
    else:
        baseline_dir = LOGS_ROOT / experiment / "baseline"
        _bl_table = f"stitch_redesign_test_hsmith_mec_baseline_{experiment}"
        _bl_extract_dir = baseline_dir
        if not (baseline_dir / "stitch_results.json").exists():
            print("no baseline to compare against")
            set_autoscaling(target_cpu=60, min_nodes=1)
            return None, None
        with open(baseline_dir / "stitch_results.json") as f:
            baseline_data = json.load(f)
        baseline_roots = [r for fr in baseline_data["file_results"] for r in fr.get("new_roots", [])]

    print(f"\ncomparing: {len(baseline_roots)} baseline vs {len(proposed_roots)} proposed roots")
    if len(baseline_roots) != len(proposed_roots):
        print(f"ROOT COUNT MISMATCH: {len(baseline_roots)} vs {len(proposed_roots)}")

    set_autoscaling(min_nodes=5)
    match = batched_extract_and_compare(
        _bl_table, np.array(baseline_roots, dtype=np.uint64),
        table_name, np.array(proposed_roots, dtype=np.uint64),
        save_dir=log_dir,
        baseline_extract_dir=_bl_extract_dir,
    )

    print(f"\n{'MATCH' if match else 'MISMATCH'}")
    set_autoscaling(target_cpu=60, min_nodes=1)
    return match, {"run_id": run_id, "table": table_name, "results": all_results}


def run_baseline(experiment: str = "multiwave", n_workers: int = None, filename: str = None) -> dict:
    run_id = uuid.uuid4().hex[:8]
    log_dir = LOGS_ROOT / experiment / "baseline"
    log_dir.mkdir(parents=True, exist_ok=True)
    table_name = f"stitch_redesign_test_hsmith_mec_baseline_{experiment}"

    print(f"[baseline] run_id={run_id}, table={table_name}")
    restore_test_table(table_name)

    lcg = LocalChunkedGraph(table_name)
    init = lcg.prepare_pool_init()

    def _worker_baseline(args):
        graph_id, edge_path, idx = args
        bl = LocalChunkedGraph.create_worker(graph_id)
        edges = _load_edges(edge_path)
        t0 = time.time()
        bl.cg.add_edges("stitch", edges, affinities=np.ones(len(edges)))
        elapsed = time.time() - t0
        return {"idx": idx, "edge_file": edge_path, "n_edges": len(edges), "new_roots": [], "elapsed": elapsed}

    if filename:
        waves = [0]
    elif experiment == "wave":
        waves = [0]
    else:
        waves = _list_all_waves()
    all_results = []
    total_wall = 0

    for wave in waves:
        files = [filename] if filename and wave == 0 else _list_wave_files(wave)
        nw = n_workers or _default_n_workers(len(files))
        args = [(table_name, f, i) for i, f in enumerate(files)]
        random.shuffle(args)
        t0 = time.time()
        with Pool(nw, initializer=LocalChunkedGraph.pool_init, initargs=init) as pool:
            results = list(tqdm(pool.imap_unordered(_worker_baseline, args), total=len(files), desc=f"wave {wave}"))
        elapsed = time.time() - t0
        total_wall += elapsed
        all_results.extend(results)
        print(f"  wave {wave}: {elapsed:.1f}s, {len(files)} files")
        _save_results(str(log_dir / "stitch_results.json"), all_results, total_wall, table_name)

    print(f"baseline {experiment}: {total_wall:.1f}s wall, {len(all_results)} files")
    return {"table_name": table_name, "results": all_results, "elapsed": total_wall}


def compare_run(run_id: str, experiment: str = "multiwave") -> bool:
    log_dir = LOGS_ROOT / experiment / run_id
    baseline_dir = LOGS_ROOT / experiment / "baseline"
    baseline_table = f"stitch_redesign_test_hsmith_mec_baseline_{experiment}"

    with open(log_dir / "stitch_results.json") as f:
        proposed_data = json.load(f)
    with open(baseline_dir / "stitch_results.json") as f:
        baseline_data = json.load(f)

    baseline_roots = [r for fr in baseline_data["file_results"] for r in fr.get("new_roots", [])]
    proposed_roots = [r for fr in proposed_data["file_results"] for r in fr.get("new_roots", [])]

    print(f"comparing: {len(baseline_roots)} baseline vs {len(proposed_roots)} proposed roots")
    table_name = proposed_data.get("table_name", f"stitch_redesign_test_hsmith_{run_id}")

    match = batched_extract_and_compare(
        baseline_table, np.array(baseline_roots, dtype=np.uint64),
        table_name, np.array(proposed_roots, dtype=np.uint64),
        save_dir=log_dir,
        baseline_extract_dir=baseline_dir,
    )
    print(f"\n{'MATCH' if match else 'MISMATCH'}")
    return match
