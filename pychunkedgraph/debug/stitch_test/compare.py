import json
import pickle
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cloudfiles import CloudFile

from pychunkedgraph.graph import basetypes

from .baseline import run_baseline_stitch
from .proposed import run_proposed_stitch
from .tables import restore_test_table, setup_env, PREFIX, EDGES_SRC, _get_instance
from .utils import compare_structures, _convert_for_json, _save_structure_file, _load_structure_file
from .stitch_types import RunResult

LOGS_ROOT = Path("/home/akhilesh/opt/zetta_utils/.env/pcg/.env/stitching/runs")


def generate_run_id() -> str:
    return secrets.token_hex(4)


# ─────────────────────────────────────────────────────────────────────
# Top-level API
# ─────────────────────────────────────────────────────────────────────


def run_baseline_single(experiment: str = "single", edge_file: str = None):
    """
    Run the baseline stitch path once for an experiment type.
    If the table + saved results already exist, skips and prints "reusing".
    """
    setup_env()
    if edge_file is None:
        edge_file = f"{EDGES_SRC}/task_0_0.edges"

    table_name = f"{PREFIX}hsmith_mec_baseline_{experiment}"
    log_dir = LOGS_ROOT / experiment / "baseline"
    log_dir.mkdir(parents=True, exist_ok=True)
    structure_path = log_dir / "baseline_structure.pkl.gz"

    instance = _get_instance()
    if instance.table(table_name).exists() and structure_path.exists():
        print(f"reusing {table_name}")
        return

    print(f"restoring and running baseline path for '{experiment}'")
    restore_test_table(table_name)
    edges = pickle.loads(CloudFile(edge_file).get())
    edges = np.asarray(edges, dtype=basetypes.NODE_ID)
    result = run_baseline_stitch(table_name, edges, do_sanity_check=False)
    _save_run_result(log_dir, "baseline", result)
    print(f"baseline {experiment} done: {result['elapsed']:.1f}s")


def run_proposed_and_compare(experiment: str = "single", edge_file: str = None):
    """
    Run the proposed stitch path and compare against the baseline.
    Returns (match, result_baseline, result_proposed).
    """
    setup_env()
    if edge_file is None:
        edge_file = f"{EDGES_SRC}/task_0_0.edges"

    run_id = generate_run_id()
    log_dir = LOGS_ROOT / experiment / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    table_proposed = f"{PREFIX}hsmith_mec_{run_id}_proposed"

    print(f"run_id: {run_id}")
    print(f"logs: {log_dir}")

    baseline_log_dir = LOGS_ROOT / experiment / "baseline"
    result_baseline = _load_result(baseline_log_dir, "baseline")

    restore_test_table(table_proposed)
    edges = pickle.loads(CloudFile(edge_file).get())
    edges = np.asarray(edges, dtype=basetypes.NODE_ID)
    result_proposed = run_proposed_stitch(table_proposed, edges)
    _save_run_result(log_dir, "proposed", result_proposed)

    print(f"\nbaseline: {result_baseline['elapsed']:.1f}s, proposed: {result_proposed['elapsed']:.1f}s")
    match = compare_stitch_results(result_baseline, result_proposed)

    summary = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "edge_file": edge_file,
        "match": match,
        "time_baseline": result_baseline["elapsed"],
        "time_proposed": result_proposed["elapsed"],
        "proposed_perf": result_proposed.get("perf", {}),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(_convert_for_json(summary), f, indent=2)

    print(f"\n{'MATCH' if match else 'MISMATCH'}")
    return match, result_baseline, result_proposed


# ─────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────


def compare_stitch_results(result_a: RunResult, result_b: RunResult) -> bool:
    ids_match = _compare_new_ids_per_layer(result_a, result_b)
    struct_match = compare_structures(result_a.structure, result_b.structure)
    return ids_match and struct_match


def _compare_new_ids_per_layer(result_a: RunResult, result_b: RunResult):
    lc_a = {int(k): v for k, v in result_a.layer_counts.items()}
    lc_b = {int(k): v for k, v in result_b.layer_counts.items()}
    all_layers = sorted(set(lc_a.keys()) | set(lc_b.keys()))
    match = True
    for layer in all_layers:
        if lc_a.get(layer, 0) != lc_b.get(layer, 0):
            print(f"  NEW IDS MISMATCH layer {layer}: {lc_a.get(layer,0)} vs {lc_b.get(layer,0)}")
            match = False
    if match:
        print(f"  NEW IDS MATCH: {sum(lc_a.values())} across {len(all_layers)} layers")
    return match


# ─────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────


def _save_run_result(log_dir, name, result: RunResult):
    _save_structure_file(str(log_dir / f"{name}_structure.json"), result.structure)
    with open(log_dir / f"{name}_meta.json", "w") as f:
        json.dump(_convert_for_json(result.meta), f, indent=2)


def _load_result(log_dir, name) -> RunResult:
    with open(log_dir / f"{name}_meta.json") as f:
        meta = json.load(f)
    structure = _load_structure_file(str(log_dir / f"{name}_structure.json"))
    return RunResult(structure=structure, **{k: v for k, v in meta.items() if k in RunResult.__dataclass_fields__})
