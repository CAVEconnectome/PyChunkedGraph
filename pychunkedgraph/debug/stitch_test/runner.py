"""
Multiwave orchestration: baseline (production add_edges), proposed (our stitch), comparison.

Usage:
    bl = BaselineRun("wave")
    bl.run()  # once, cached forever

    pr = ProposedRun("wave")
    pr.run(baseline=bl)  # runs stitch, compares against baseline
"""

import json
import logging
import os
import pickle
from contextlib import contextmanager
import random
import time
import uuid
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil
from cloudfiles import CloudFiles
from tqdm import tqdm

from pychunkedgraph.graph import ChunkedGraph, basetypes

from . import local_cg as _lcg_mod
from .stitch import stitch
from .tables import restore_test_table, set_autoscaling
from .utils import _convert_for_json, batched_extract_and_compare, batched_extract_structure

log = logging.getLogger(__name__)

_process = psutil.Process()

EDGES_SRC = "gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges"
LOGS_ROOT = Path(__file__).parent.parent.parent.parent / ".env" / "stitching" / "runs"
REFERENCE_PATH = LOGS_ROOT / "reference.json"
SINGLE_EDGE_FILE = "task_0_591.edges"

_shared_cache = None
_shared_inc = None
_reference = None
_edge_files_cache = None


# -- Utilities --

def _mem_str() -> str:
    return f"{_process.memory_info().rss / 1024**2:.0f}MB"


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


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


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


# -- Pool workers (module-level for pickling) --

def _worker_baseline(args):
    graph_id, edge_path, idx = args
    bl = _lcg_mod.LocalChunkedGraph.create_worker(graph_id)
    edges = _load_edges(edge_path)
    t0 = time.time()
    result = bl.cg.add_edges("stitch", edges, stitch_mode=True)
    elapsed = time.time() - t0
    new_roots = [int(r) for r in result.new_root_ids]
    return {"idx": idx, "edge_file": edge_path, "n_edges": len(edges), "new_roots": new_roots, "elapsed": elapsed}


def _worker_proposed(args: tuple) -> tuple:
    graph_id, edge_path, idx = args
    edges = _load_edges(edge_path)
    lcg = _lcg_mod.LocalChunkedGraph.create_worker(graph_id, preloaded=_shared_cache, incremental=_shared_inc)
    lcg.acquire_stitch_timestamp(edge_path)
    lcg.begin_stitch()

    roots_before = len(np.unique(lcg.get_roots(edges.ravel())))

    t0 = time.time()
    result = stitch(lcg, edges)
    t_stitch = time.time() - t0

    t0 = time.time()
    lcg.mutate_rows(result.rows)
    t_write = time.time() - t0

    lcg.release_stitch_timestamp(edge_path)
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


# -- Base class --

class StitchRun:

    SINGLE_FILE = "task_0_591.edges"

    def __init__(self, experiment: str, table_name: str, log_dir: Path) -> None:
        self.experiment = experiment
        self.table_name = table_name
        self.log_dir = log_dir
        self.extract_dir = log_dir / "extract"
        self.json_path = log_dir / "stitch_results.json"
        self.roots_path = log_dir / "roots.pkl"

    def get_waves(self) -> list[int]:
        return _list_all_waves() if self.experiment == "multiwave" else [0]

    def get_wave_files(self, wave: int) -> list[str]:
        if self.experiment == "single":
            return [self.SINGLE_FILE]
        return _list_wave_files(wave)

    @property
    def use_pool(self) -> bool:
        return self.experiment != "single"

    @staticmethod
    @contextmanager
    def _autoscale(min_nodes: int):
        """Scale up BigTable for heavy parallel reads, revert after. No-op if min_nodes=1."""
        if min_nodes > 1:
            set_autoscaling(target_cpu=25, min_nodes=min_nodes)
        try:
            yield
        finally:
            if min_nodes > 1:
                set_autoscaling(target_cpu=60, min_nodes=1)

    @property
    def has_roots(self) -> bool:
        return self.roots_path.exists()

    @property
    def has_extraction(self) -> bool:
        return any(self.extract_dir.rglob("shard_*.pkl.gz"))

    @property
    def is_complete(self) -> bool:
        return self.has_roots and self.has_extraction

    def load_roots(self) -> list:
        with open(self.roots_path, "rb") as f:
            return pickle.load(f)

    def save_roots(self, roots: list) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.roots_path, "wb") as f:
            pickle.dump(roots, f)

    def save_results(self, results: list, elapsed: float, **kw) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        data = {"file_results": results, "elapsed": elapsed, "table_name": self.table_name, **kw}
        with open(self.json_path, "w") as f:
            json.dump(_convert_for_json(data), f)

    def extract(self, force: bool = False) -> None:
        roots = np.array(self.load_roots(), dtype=basetypes.NODE_ID)
        batched_extract_structure(self.table_name, roots, self.extract_dir, force=force)

    def compare_with(self, other: "StitchRun") -> bool:
        return batched_extract_and_compare(
            self.table_name, np.array(self.load_roots(), dtype=basetypes.NODE_ID),
            other.table_name, np.array(other.load_roots(), dtype=basetypes.NODE_ID),
            save_dir=other.log_dir,
            baseline_extract_dir=self.extract_dir,
        )


# -- Baseline --

class BaselineRun(StitchRun):

    def __init__(self, experiment: str = "wave") -> None:
        table_name = f"stitch_redesign_test_hsmith_mec_baseline_{experiment}"
        log_dir = LOGS_ROOT / experiment / "baseline"
        super().__init__(experiment, table_name, log_dir)

    def run(self, n_workers: int = None, force: bool = False, bt_min_nodes: int = 5) -> list:
        if self.is_complete and not force:
            roots = self.load_roots()
            print(f"baseline cached: {len(roots)} roots at {self.roots_path}")
            return roots

        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[baseline] table={self.table_name}")
        restore_test_table(self.table_name)

        lcg = _lcg_mod.LocalChunkedGraph(self.table_name)
        init = lcg.prepare_pool_init()

        all_results = []
        total_wall = 0

        for wi, wave in enumerate(self.get_waves()):
            files = self.get_wave_files(wave)
            args = [(self.table_name, f, i) for i, f in enumerate(files)]
            t0 = time.time()
            if self.use_pool:
                nw = n_workers or _default_n_workers(len(files))
                random.shuffle(args)
                scale = bt_min_nodes if wi == 0 else 1
                with self._autoscale(scale):
                    with Pool(nw, initializer=_lcg_mod.LocalChunkedGraph.pool_init, initargs=init) as pool:
                        results = list(tqdm(pool.imap_unordered(_worker_baseline, args), total=len(files), desc=f"wave {wave}"))
            else:
                results = [self._run_baseline_inprocess(a) for a in args]
            elapsed = time.time() - t0
            total_wall += elapsed
            all_results.extend(results)
            print(f"  wave {wave}: {elapsed:.1f}s, {len(files)} files")
            self.save_results(all_results, total_wall)

        all_roots = [r for fr in all_results for r in fr["new_roots"]]
        print(f"baseline: {total_wall:.1f}s, {len(all_results)} files, {len(all_roots)} roots")

        self.save_roots(all_roots)
        print(f"extracting baseline structure...")
        with self._autoscale(bt_min_nodes):
            self.extract(force=True)
        print(f"baseline cached: {len(all_roots)} roots")

        return all_roots

    @staticmethod
    def _run_baseline_inprocess(args: tuple) -> dict:
        graph_id, edge_path, idx = args
        cg = ChunkedGraph(graph_id=graph_id)
        edges = _load_edges(edge_path)
        t0 = time.time()
        result = cg.add_edges("stitch", edges, stitch_mode=True)
        elapsed = time.time() - t0
        new_roots = [int(r) for r in result.new_root_ids]
        return {"idx": idx, "edge_file": edge_path, "n_edges": len(edges), "new_roots": new_roots, "elapsed": elapsed}


# -- Proposed --

class ProposedRun(StitchRun):

    def __init__(self, experiment: str = "wave", run_id: str = None) -> None:
        run_id = run_id or uuid.uuid4().hex[:8]
        table_name = f"stitch_redesign_test_hsmith_{run_id}"
        log_dir = LOGS_ROOT / experiment / run_id
        super().__init__(experiment, table_name, log_dir)
        self.run_id = run_id

    def run(
        self, baseline: BaselineRun, n_workers: int = None, filename: str = None,
        bt_min_nodes: int = 5,
    ) -> bool:
        global _shared_cache, _shared_inc

        if not baseline.is_complete:
            print("no baseline — run BaselineRun().run() first")
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[proposed] run_id={self.run_id}")
        print(f"logs: {self.log_dir}")

        restore_test_table(self.table_name)

        waves = self.get_waves()
        lcg = _lcg_mod.LocalChunkedGraph(self.table_name)
        init = lcg.prepare_pool_init()
        all_results = []
        wave_walls = {}
        total_wall = 0

        wave0_files = [filename] if filename else self.get_wave_files(waves[0])
        print(f"mode={self.experiment}, waves={waves}, files={len(wave0_files)}")
        _shared_cache = lcg.preloaded()
        _shared_inc = None

        if self.use_pool:
            nw0 = n_workers or _default_n_workers(len(wave0_files))
            with self._autoscale(bt_min_nodes):
                with Pool(nw0, initializer=_lcg_mod.LocalChunkedGraph.pool_init, initargs=init) as pool:
                    results, elapsed = self._run_wave_pool(pool, waves[0], lcg, edge_files=wave0_files)
        else:
            results, elapsed = self._run_wave_inprocess(waves[0], lcg, edge_files=wave0_files)
        all_results.extend(results)
        wave_walls[waves[0]] = elapsed
        total_wall += elapsed
        self._log_wave_stats(waves[0], results, elapsed, lcg)
        self.save_results(all_results, total_wall, wave_walls=wave_walls)

        for wave in waves[1:]:
            files = self.get_wave_files(wave)
            _shared_cache = lcg.preloaded()
            _shared_inc = lcg.incremental_state()

            if len(files) == 1:
                results, elapsed = self._run_wave_inprocess(wave, lcg)
            else:
                nw = n_workers or _default_n_workers(len(files))
                with Pool(nw, initializer=_lcg_mod.LocalChunkedGraph.pool_init, initargs=init) as pool:
                    results, elapsed = self._run_wave_pool(pool, wave, lcg)

            all_results.extend(results)
            wave_walls[wave] = elapsed
            total_wall += elapsed
            self._log_wave_stats(wave, results, elapsed, lcg)
            self.save_results(all_results, total_wall, wave_walls=wave_walls)

            errors = [r for r in results if r.get("error")]
            if errors:
                print(f"STOPPING: {len(errors)} errors in wave {wave}")
                for r in errors:
                    print(f"  {r['edge_file']}: {r.get('error_msg')}")
                break

        all_roots = [r for fr in all_results for r in fr["new_roots"]]
        n_edges = sum(r["n_edges"] for r in all_results)
        total_stitch = sum(r.get("t_stitch", r["elapsed"]) for r in all_results)
        print(
            f"\nproposed: {total_wall:.1f}s wall, "
            f"{total_stitch:.1f}s total stitch, {len(all_roots)} roots, {n_edges} edges\n"
            f"memory: {_mem_str()}"
        )

        self.save_roots(all_roots)

        baseline_roots = baseline.load_roots()
        print(f"\ncomparing: {len(baseline_roots)} baseline vs {len(all_roots)} proposed roots")
        if len(baseline_roots) != len(all_roots):
            print(f"ROOT COUNT MISMATCH: {len(baseline_roots)} vs {len(all_roots)}")

        with self._autoscale(bt_min_nodes):
            match = baseline.compare_with(self)
        print(f"\n{'MATCH' if match else 'MISMATCH'}")
        return match

    def _run_wave_pool(
        self, pool: Pool, wave: int, lcg: _lcg_mod.LocalChunkedGraph,
        edge_files: list = None,
    ) -> tuple:
        edge_files = edge_files or _list_wave_files(wave)
        args = [(self.table_name, f, i) for i, f in enumerate(edge_files)]
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

    def _run_wave_inprocess(self, wave: int, lcg: _lcg_mod.LocalChunkedGraph, edge_files: list = None) -> tuple:
        edge_files = edge_files or _list_wave_files(wave)
        assert len(edge_files) == 1

        edges = _load_edges(edge_files[0])
        lcg.acquire_stitch_timestamp(edge_files[0])
        lcg.begin_stitch()

        roots_before = len(np.unique(lcg.get_roots(edges.ravel())))

        t0 = time.time()
        result = stitch(lcg, edges)
        t_stitch = time.time() - t0

        t0_w = time.time()
        lcg.mutate_rows(result.rows)
        t_write = time.time() - t0_w

        lcg.release_stitch_timestamp(edge_files[0])
        _check_roots(edge_files[0], roots_before, len(result.new_roots))
        cache_stats = lcg.stats()
        lcg.end_stitch()
        elapsed = time.time() - t0

        file_result = {
            "idx": 0, "edge_file": edge_files[0], "n_edges": len(edges),
            "new_roots": [int(r) for r in result.new_roots],
            "roots_before": roots_before,
            "elapsed": t_stitch + t_write, "t_stitch": t_stitch, "t_write": t_write,
            "n_rows": len(result.rows),
            "perf": result.perf,
            "cache_stats": cache_stats,
        }
        return [file_result], elapsed

    @staticmethod
    def _log_wave_stats(wave: int, results: list, elapsed: float, lcg: _lcg_mod.LocalChunkedGraph) -> None:
        total_req = sum(e.n_requested for r in results for e in r.get("perf", {}).get("rpc_log", []))
        total_read = sum(e.n_read for r in results for e in r.get("perf", {}).get("rpc_log", []))
        hit_pct = (total_req - total_read) / total_req * 100 if total_req else 0
        n_roots = sum(len(r.get("new_roots", [])) for r in results)
        n_rows = sum(r.get("n_rows", 0) for r in results)
        t_stitch = sum(r.get("t_stitch", 0) for r in results)
        t_write = sum(r.get("t_write", 0) for r in results)
        print(
            f"  wave {wave}: {elapsed:.0f}s {len(results)}f {_fmt(n_roots)} roots "
            f"stitch={t_stitch:.0f}s write={t_write:.0f}s {_fmt(n_rows)} rows "
            f"{_fmt(total_read)}/{_fmt(total_req)} reads({hit_pct:.0f}%) "
            f"{_mem_str()} {results[0].get('cache_stats') or lcg.stats()}"
        )


# -- Backward-compatible entry points --

def run_baseline(experiment: str = "wave", bt_min_nodes: int = 5, **kw) -> dict:
    bl = BaselineRun(experiment)
    roots = bl.run(bt_min_nodes=bt_min_nodes, **kw)
    return {"table_name": bl.table_name, "results": [], "elapsed": 0, "roots": roots}


def run_proposed(experiment: str = "wave", bt_min_nodes: int = 5, sanity_check: bool = False, **kw) -> tuple:
    _lcg_mod.SANITY_CHECK = sanity_check
    bl = BaselineRun(experiment)
    pr = ProposedRun(experiment)
    match = pr.run(baseline=bl, bt_min_nodes=bt_min_nodes, **kw)
    return match, {"run_id": pr.run_id, "table": pr.table_name}


def compare_run(run_id: str, experiment: str = "wave") -> bool:
    bl = BaselineRun(experiment)
    pr = ProposedRun(experiment, run_id=run_id)
    return bl.compare_with(pr)
