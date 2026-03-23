import json
import pickle
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cloudfiles import CloudFile

from pychunkedgraph.graph import basetypes

from .current import run_current_stitch
from .proposed import run_proposed_stitch
from .tables import restore_test_table, setup_env, PREFIX, EDGES_SRC, _get_instance
from .utils import _compare_components, _compare_cross_edges, _convert_for_json

LOGS_ROOT = Path("/home/akhilesh/opt/zetta_utils/.env/pcg/.env/stitching/runs")


def generate_run_id() -> str:
    return secrets.token_hex(4)


# ─────────────────────────────────────────────────────────────────────
# Top-level API
# ─────────────────────────────────────────────────────────────────────


def run_current_baseline(experiment: str = "single", edge_file: str = None):
    """
    Run the current stitch path once for an experiment type.
    If the table + saved results already exist, skips and prints "reusing".
    """
    setup_env()
    if edge_file is None:
        edge_file = f"{EDGES_SRC}/task_0_0.edges"

    table_name = f"{PREFIX}hsmith_mec_current_{experiment}"
    log_dir = LOGS_ROOT / experiment / "current"
    log_dir.mkdir(parents=True, exist_ok=True)
    structure_path = log_dir / "current_structure.json"

    instance = _get_instance()
    if instance.table(table_name).exists() and structure_path.exists():
        print(f"reusing {table_name}")
        return

    print(f"restoring and running current path for '{experiment}'")
    restore_test_table(table_name)
    edges = pickle.loads(CloudFile(edge_file).get())
    edges = np.asarray(edges, dtype=basetypes.NODE_ID)
    result = run_current_stitch(table_name, edges, do_sanity_check=False)
    _save_run_result(log_dir, "current", result)
    print(f"current {experiment} done: {result['elapsed']:.1f}s")


def run_proposed_and_compare(experiment: str = "single", edge_file: str = None):
    """
    Run the proposed stitch path and compare against the current baseline.
    Returns (match, result_current, result_proposed).
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

    current_log_dir = LOGS_ROOT / experiment / "current"
    result_current = _load_result(current_log_dir, "current")

    restore_test_table(table_proposed)
    edges = pickle.loads(CloudFile(edge_file).get())
    edges = np.asarray(edges, dtype=basetypes.NODE_ID)
    result_proposed = run_proposed_stitch(table_proposed, edges)
    _save_run_result(log_dir, "proposed", result_proposed)

    print(f"\ncurrent: {result_current['elapsed']:.1f}s, proposed: {result_proposed['elapsed']:.1f}s")
    match = compare_stitch_results(result_current, result_proposed)

    summary = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "edge_file": edge_file,
        "match": match,
        "time_current": result_current["elapsed"],
        "time_proposed": result_proposed["elapsed"],
        "proposed_perf": result_proposed.get("perf", {}),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(_convert_for_json(summary), f, indent=2)

    print(f"\n{'MATCH' if match else 'MISMATCH'}")
    return match, result_current, result_proposed


# ─────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────


def compare_stitch_results(result_a: dict, result_b: dict) -> bool:
    ids_match = _compare_new_ids_per_layer(result_a, result_b)
    comp_match = _compare_components(result_a["structure"], result_b["structure"])
    cx_match = _compare_cross_edges(result_a["structure"], result_b["structure"])
    return ids_match and comp_match and cx_match


def _compare_new_ids_per_layer(result_a, result_b):
    lc_a = {int(k): v for k, v in result_a.get("layer_counts", {}).items()}
    lc_b = {int(k): v for k, v in result_b.get("layer_counts", {}).items()}
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


def _save_structure(log_dir, name, structure):
    serializable = {}
    comps = structure.get("components", {})
    serializable["components"] = {
        str(layer): [sorted(c) for c in ccs] for layer, ccs in comps.items()
    }
    cx = structure.get("cross_edges", {})
    serializable["cross_edges"] = {
        str(layer): [[sorted(src), sorted(dst)] for src, dst in pairs]
        for layer, pairs in cx.items()
    }
    with open(log_dir / f"{name}_structure.json", "w") as f:
        json.dump(_convert_for_json(serializable), f, indent=2)


def _save_run_result(log_dir, name, result):
    _save_structure(log_dir, name, result["structure"])
    meta = {k: v for k, v in result.items() if k != "structure"}
    with open(log_dir / f"{name}_meta.json", "w") as f:
        json.dump(_convert_for_json(meta), f, indent=2)


def _load_structure(path):
    with open(path) as f:
        data = json.load(f)
    return {
        "components": {
            int(layer): [frozenset(c) for c in ccs]
            for layer, ccs in data.get("components", {}).items()
        },
        "cross_edges": {
            int(layer): [(frozenset(src), frozenset(dst)) for src, dst in pairs]
            for layer, pairs in data.get("cross_edges", {}).items()
        },
    }


def _load_result(log_dir, name):
    with open(log_dir / f"{name}_meta.json") as f:
        result = json.load(f)
    result["structure"] = _load_structure(log_dir / f"{name}_structure.json")
    return result
