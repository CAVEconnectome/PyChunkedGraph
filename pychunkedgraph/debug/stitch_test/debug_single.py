"""Debug: single-file comparison and wave 0 comparison."""

import importlib
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from cloudfiles import CloudFiles
from google.cloud.bigtable.backup import Backup
from tqdm import tqdm

os.environ.setdefault("BIGTABLE_PROJECT", "zetta-proofreading")
os.environ.setdefault("BIGTABLE_INSTANCE", "pychunkedgraph")

from . import local_cg as _lcg_mod
from . import resolver as _res_mod
from . import row_cache as _rc_mod
from . import stitch as _stitch_mod
from . import tree as _tree_mod
from . import wave_cache as _wc_mod
from .tables import BACKUP_ID, CLUSTER_ID, EDGES_SRC, _get_instance
from .utils import batch_get_l2children

from pychunkedgraph.graph import ChunkedGraph, basetypes, types
from pychunkedgraph.graph.utils import flatgraph

_ALL_MODS = [_rc_mod, _wc_mod, _tree_mod, _res_mod, _lcg_mod, _stitch_mod]
_VERSION = 34


def _get_components(cg, root_ids, layer_count: int) -> dict:
    """Walk hierarchy, collect full SV set per node at each layer. Batched reads."""
    node_svs = {}
    nodes_by_layer = defaultdict(set)
    root_arr = np.array([np.uint64(r) for r in root_ids], dtype=basetypes.NODE_ID)
    l2_map = batch_get_l2children(cg, root_arr)
    all_l2s = set()
    for l2s in l2_map.values():
        all_l2s.update(l2s)
    if all_l2s:
        l2_arr = np.array(list(all_l2s), dtype=basetypes.NODE_ID)
        ch_d = cg.get_children(l2_arr)
        for l2 in l2_arr:
            svs = ch_d.get(l2, ch_d.get(int(l2), []))
            if len(svs) > 0:
                node_svs[int(l2)] = frozenset(int(s) for s in svs)
                nodes_by_layer[2].add(int(l2))
    for rid in root_ids:
        nodes_by_layer[layer_count].add(int(rid))
    for layer in range(layer_count, 1, -1):
        layer_nodes = list(nodes_by_layer[layer])
        if not layer_nodes:
            continue
        batch_arr = np.array(layer_nodes, dtype=basetypes.NODE_ID)
        ch_d = cg.get_children(batch_arr)
        for nid in layer_nodes:
            children = ch_d.get(nid, ch_d.get(np.uint64(nid), []))
            all_svs = set()
            for child in children:
                child_int = int(child)
                if layer - 1 > 2:
                    nodes_by_layer[layer - 1].add(child_int)
                child_svs = node_svs.get(child_int, frozenset())
                all_svs.update(child_svs)
            if all_svs:
                node_svs[int(nid)] = frozenset(all_svs)
    per_layer = {}
    for layer in range(2, layer_count + 1):
        components = set()
        for nid in nodes_by_layer[layer]:
            svs = node_svs.get(int(nid))
            if svs:
                components.add(svs)
        per_layer[layer] = components
    return per_layer


def _print_component_comparison(comp_prod: dict, comp_ours: dict, layer_count: int) -> None:
    print(f"\n--- Component comparison (full SV sets) ---")
    for layer in range(2, layer_count + 1):
        p = comp_prod.get(layer, set())
        o = comp_ours.get(layer, set())
        only_p = p - o
        only_o = o - p
        tag = "MATCH" if not only_p and not only_o else "MISMATCH"
        print(f"  L{layer}: prod={len(p)} ours={len(o)} +prod={len(only_p)} +ours={len(only_o)} {tag}")


def load_edges(filename: str) -> np.ndarray:
    cf = CloudFiles(EDGES_SRC)
    return pickle.loads(cf.get(filename))


def _setup_table(filename: str) -> tuple:
    for m in _ALL_MODS:
        importlib.reload(m)
    print(f"[debug_single v{_VERSION}]")
    table_name = f"stitch_redesign_test_debug_{filename.split('.')[0].split('_')[-1]}"
    instance = _get_instance()
    t = instance.table(table_name)
    if t.exists():
        t.delete()
    Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID).restore(table_name).result()
    return table_name, instance


def _cleanup_table(table_name: str, instance) -> None:
    instance.table(table_name).delete()
    print("deleted")


def debug_file(filename: str = "task_0_579.edges") -> dict:
    table_name, instance = _setup_table(filename)
    try:
        return _run(filename, table_name)
    finally:
        _cleanup_table(table_name, instance)


def debug_compare(filename: str = None, rerun_baseline: bool = False) -> dict:
    """Compare production vs ours. Single file or full wave 0 if filename=None."""
    for m in _ALL_MODS:
        importlib.reload(m)
    print(f"[debug_single v{_VERSION}]")
    instance = _get_instance()

    if filename is None:
        return _run_wave0_compare(instance, rerun_baseline=rerun_baseline)

    suffix = filename.split(".")[0].split("_")[-1]
    tbl_prod = f"stitch_redesign_test_cmp_prod_{suffix}"
    tbl_ours = f"stitch_redesign_test_cmp_ours_{suffix}"
    for t in [tbl_prod, tbl_ours]:
        tab = instance.table(t)
        if tab.exists():
            tab.delete()
        Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID).restore(t).result()
    print("restored both tables")

    try:
        return _run_compare(filename, tbl_prod, tbl_ours)
    finally:
        for t in [tbl_prod, tbl_ours]:
            instance.table(t).delete()
        print("deleted both")


def _wave0_worker_prod(args):
    graph_id, edge_path = args
    cg = ChunkedGraph(graph_id=graph_id)
    edges = load_edges(edge_path)
    t0 = time.time()
    result = cg.add_edges("stitch", edges, stitch_mode=True)
    elapsed = time.time() - t0
    return [int(r) for r in result.new_root_ids], elapsed


def _wave0_worker_ours(args):
    graph_id, edge_path = args
    lcg = _lcg_mod.LocalChunkedGraph.create_worker(graph_id)
    edges = load_edges(edge_path)
    lcg.begin_stitch()
    t0 = time.time()
    result = _stitch_mod.stitch(lcg, edges)
    t_stitch = time.time() - t0
    t0 = time.time()
    lcg.mutate_rows(result.rows)
    t_write = time.time() - t0
    return [int(r) for r in result.new_roots], t_stitch, t_write


_WAVE0_CHECKPOINT = "/tmp/wave0_prod_checkpoint.pkl"
_WAVE0_TBL_PROD = "stitch_redesign_test_cmp_prod_wave0"
_WAVE0_TBL_OURS = "stitch_redesign_test_cmp_ours_wave0"


def _run_wave0_compare(instance, rerun_baseline: bool = False) -> dict:
    """Run all wave 0 files with Pool, compare components.
    Baseline (prod table + checkpoint) kept across runs. Only re-run if missing or forced."""
    cf = CloudFiles(EDGES_SRC)
    wave0_files = sorted([f for f in cf.list() if f.startswith("task_0_")])
    n_workers = min(len(wave0_files), 3 * os.cpu_count())
    print(f"wave 0: {len(wave0_files)} files, {n_workers} workers")

    tbl_prod = _WAVE0_TBL_PROD
    tbl_ours = _WAVE0_TBL_OURS
    prod_tab = instance.table(tbl_prod)
    need_baseline = rerun_baseline or not (os.path.exists(_WAVE0_CHECKPOINT) and prod_tab.exists())

    if need_baseline:
        if prod_tab.exists():
            prod_tab.delete()
        Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID).restore(tbl_prod).result()
        print(f"restored {tbl_prod}")
    else:
        with open(_WAVE0_CHECKPOINT, "rb") as f:
            prod_roots = pickle.load(f)
        print(f"baseline cached: {len(prod_roots)} roots")

    # Always restore ours table fresh
    ours_tab = instance.table(tbl_ours)
    if ours_tab.exists():
        ours_tab.delete()
    Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID).restore(tbl_ours).result()
    print(f"restored {tbl_ours}")

    lcg_init = _lcg_mod.LocalChunkedGraph(tbl_prod)
    pool_init_args = lcg_init.prepare_pool_init()

    if need_baseline:
        prod_roots = []
        t0 = time.time()
        with Pool(n_workers, initializer=_lcg_mod.LocalChunkedGraph.pool_init, initargs=pool_init_args) as pool:
            for roots, _ in tqdm(
                pool.imap_unordered(_wave0_worker_prod, [(tbl_prod, f) for f in wave0_files]),
                total=len(wave0_files), desc="production",
            ):
                prod_roots.extend(roots)
        print(f"production: {len(prod_roots)} roots in {time.time() - t0:.1f}s")
        with open(_WAVE0_CHECKPOINT, "wb") as f:
            pickle.dump(prod_roots, f)

    # Ours
    ours_roots = []
    ours_stitch_total = 0
    ours_write_total = 0
    t0 = time.time()
    with Pool(n_workers, initializer=_lcg_mod.LocalChunkedGraph.pool_init, initargs=pool_init_args) as pool:
        for roots, t_s, t_w in tqdm(
            pool.imap_unordered(_wave0_worker_ours, [(tbl_ours, f) for f in wave0_files]),
            total=len(wave0_files), desc="ours",
        ):
            ours_roots.extend(roots)
            ours_stitch_total += t_s
            ours_write_total += t_w
    ours_elapsed = time.time() - t0
    print(f"ours: {len(ours_roots)} roots in {ours_elapsed:.1f}s (stitch {ours_stitch_total:.1f}s + write {ours_write_total:.1f}s)")
    print(f"\nroots: prod={len(prod_roots)} ours={len(ours_roots)}")
    if len(prod_roots) != len(ours_roots):
        print(f"ROOT COUNT MISMATCH: {len(prod_roots)} vs {len(ours_roots)} — skipping component comparison")
        instance.table(tbl_ours).delete()
        return {"prod_roots": len(prod_roots), "ours_roots": len(ours_roots), "match": False}

    # Compare components
    layer_count = lcg_init.meta.layer_count
    comp_prod = _get_components(ChunkedGraph(graph_id=tbl_prod), prod_roots, layer_count)
    comp_ours = _get_components(ChunkedGraph(graph_id=tbl_ours), ours_roots, layer_count)
    _print_component_comparison(comp_prod, comp_ours, layer_count)

    instance.table(tbl_ours).delete()
    print(f"deleted {tbl_ours}")

    return {"prod_roots": len(prod_roots), "ours_roots": len(ours_roots)}


def _run_compare(filename: str, tbl_prod: str, tbl_ours: str) -> dict:
    edges = load_edges(filename)
    print(f"\n=== {filename}: {len(edges)} edges ===")

    # Production: add_edges
    cg_prod = ChunkedGraph(graph_id=tbl_prod)
    t0 = time.time()
    prod_result = cg_prod.add_edges("stitch", edges, stitch_mode=True)
    prod_elapsed = time.time() - t0
    new_roots_prod = prod_result.new_root_ids
    print(f"production roots: {len(new_roots_prod)} ({prod_elapsed:.1f}s)")

    # Ours: stitch
    lcg = _lcg_mod.LocalChunkedGraph(tbl_ours)
    lcg.begin_stitch()
    t0 = time.time()
    result = _stitch_mod.stitch(lcg, edges)
    ours_stitch_elapsed = time.time() - t0
    c = lcg._cache
    t0 = time.time()
    lcg.mutate_rows(result.rows)
    ours_write_elapsed = time.time() - t0
    print(f"our roots: {len(result.new_roots)} (stitch {ours_stitch_elapsed:.1f}s + write {ours_write_elapsed:.1f}s)")

    # --- Capture stitch internal state ---
    diag = {}
    diag["counterpart_ids"] = set(c.counterpart_ids)
    diag["new_node_ids"] = set(c.new_node_ids)
    diag["sibling_ids"] = set(c.sibling_ids)
    diag["dirty_siblings"] = set(c.dirty_siblings)
    diag["new_ids_d"] = {l: list(ids) for l, ids in c.new_ids_d.items()}
    diag["new_to_old"] = {k: set(v) for k, v in c.new_to_old.items()}
    diag["siblings_d"] = {int(l): list(ids) for l, ids in c.siblings_d.items()}
    diag["perf"] = result.perf
    diag["old_to_new"] = dict(c.old_to_new)
    diag["rows_written"] = len(result.rows)

    # Node map that _update_counterpart_cx would build
    full_node_map = {}
    for nid in c.new_node_ids:
        for old_id in c.new_to_old.get(int(nid), set()):
            full_node_map[old_id] = int(nid)
    diag["full_node_map_size"] = len(full_node_map)

    # Per-layer: what counterparts were found, what CX they have
    layer_count = lcg.meta.layer_count
    layer_diag = {}
    for layer in range(2, layer_count):
        new_nodes = c.new_ids_d.get(layer, [])
        if not new_nodes:
            continue
        ld = {"new_nodes": len(new_nodes)}

        # What CX do new nodes have?
        cx_layers_present = defaultdict(int)
        for nid in new_nodes:
            for lyr in c.cx.get(int(nid), {}):
                cx_layers_present[lyr] += 1
        ld["cx_layers"] = dict(cx_layers_present)

        # What unresolved_acx layers do new nodes have?
        acx_layers_present = defaultdict(int)
        for nid in new_nodes:
            for lyr in c.unresolved_acx.get(int(nid), {}):
                acx_layers_present[lyr] += 1
        ld["acx_layers"] = dict(acx_layers_present)

        # Counterparts from resolved CX only (old approach)
        in_scope = set(c.new_node_ids) | c.sibling_ids
        cp_from_cx = set()
        for nid in new_nodes:
            for edges in c.cx.get(int(nid), {}).values():
                if len(edges) > 0:
                    cp_from_cx.update(int(x) for x in edges[:, 1])
        cp_from_cx -= in_scope
        ld["counterparts_from_cx"] = len(cp_from_cx)

        # Counterparts from unresolved_acx at ALL layers (new approach)
        cp_from_acx = set()
        get_layer = lambda nid: lcg.get_chunk_layer(np.uint64(nid))
        child_to_parent = {}  # empty — check if this is the problem
        for nid in new_nodes:
            raw = c.unresolved_acx.get(int(nid), {})
            for lyr in range(layer, layer_count):
                acx_edges = raw.get(lyr, types.empty_2d)
                for sv in acx_edges[:, 1]:
                    resolved = _res_mod.resolve_sv_to_layer(
                        int(sv), lyr, c, child_to_parent, get_layer)
                    cp_from_acx.add(resolved)
        cp_from_acx -= in_scope
        ld["counterparts_from_acx_empty_c2p"] = len(cp_from_acx)

        # Node map for this layer (old approach)
        old_nm = {}
        for nid in new_nodes:
            for old_id in c.new_to_old.get(int(nid), set()):
                old_nm[old_id] = int(nid)
        ld["node_map_old"] = len(old_nm)

        # Node map cumulative (new approach)
        cum_nm = {}
        for nid in (*c.new_node_ids, *new_nodes):
            for old_id in c.new_to_old.get(int(nid), set()):
                cum_nm[old_id] = int(nid)
        ld["node_map_cumulative"] = len(cum_nm)

        layer_diag[layer] = ld
    diag["per_layer"] = layer_diag

    # CX written for counterparts in build_rows
    cp_cx_written = {}
    for nid in c.counterpart_ids:
        cx = c.cx.get(nid, {})
        cp_cx_written[nid] = {lyr: len(edges) for lyr, edges in cx.items()}
    diag["counterpart_cx_written"] = cp_cx_written

    # --- Component comparison: full SV sets per parent at each layer ---
    comp_prod = _get_components(cg_prod, new_roots_prod, layer_count)
    comp_ours = _get_components(lcg.cg, [np.uint64(r) for r in result.new_roots], layer_count)
    _print_component_comparison(comp_prod, comp_ours, layer_count)
    diag["comp_prod_counts"] = {l: len(c) for l, c in comp_prod.items()}
    diag["comp_ours_counts"] = {l: len(c) for l, c in comp_ours.items()}

    # Save diagnostics
    diag_path = f"/tmp/stitch_diag_{filename.split('.')[0]}.pkl"
    with open(diag_path, "wb") as f:
        pickle.dump(diag, f)
    print(f"\nDiagnostics saved to {diag_path}")

    return diag


def debug_cx_comparison(filename: str = "task_0_579.edges") -> dict:
    table_name, instance = _setup_table(filename)
    try:
        return _run_cx_comparison(filename, table_name)
    finally:
        _cleanup_table(table_name, instance)


def _run(filename: str, table_name: str) -> dict:
    lcg = _lcg_mod.LocalChunkedGraph(table_name)
    edges = load_edges(filename)
    lcg.begin_stitch()
    result = _stitch_mod.stitch(lcg, edges)
    c = lcg._cache

    print(f"=== v4 ROOTS: {len(result.new_roots)} ===\n")

    for layer in range(2, 8):
        new = len(c.new_ids_d.get(layer, []))
        sibs = len(c.siblings_d.get(layer, []))
        n2o = sum(1 for k in c.new_to_old if (int(k) >> 56) & 0xFF == layer)
        o2n = sum(1 for k in c.old_to_new if (int(k) >> 56) & 0xFF == layer)
        pwl = sum(1 for p in c.new_ids_d.get(layer, []) if c.new_to_old.get(int(p)))
        print(f"L{layer}: new={new} sibs={sibs} n2o={n2o} o2n={o2n} lineage={pwl}/{new}")

    print(f"\nBigTable reads: {len(lcg._read_row_keys)}")
    return {"roots": len(result.new_roots), "perf": result.perf}


def _run_cx_comparison(filename: str, table_name: str) -> dict:
    lcg = _lcg_mod.LocalChunkedGraph(table_name)
    edges = load_edges(filename)
    lcg.begin_stitch()
    result = _stitch_mod.stitch(lcg, edges)
    c = lcg._cache
    layer_count = lcg.meta.layer_count

    print(f"=== v{_VERSION} ROOTS: {len(result.new_roots)} ===\n")

    # Build canonical ID: node → min(SVs in subtree)
    # L2: min(children SVs). L3+: min across all L2 descendants.
    canonical = {}

    # L2 nodes: children from RowCache
    all_l2 = set(int(x) for x in c.new_ids_d.get(2, []))
    all_l2.update(c.sibling_ids)
    for l2 in all_l2:
        ch = c.children.get(l2)
        if ch is not None and len(ch) > 0:
            canonical[l2] = int(np.min(ch))

    # L3+ nodes: descend to L2 children, use their canonical
    for layer in range(3, layer_count + 1):
        for nid in list(c.new_ids_d.get(layer, [])) + list(c.siblings_d.get(layer, [])):
            nid_int = int(nid)
            ch = c.children.get(nid_int)
            if ch is None:
                ch = []
            min_sv = None
            for child in ch:
                child_can = canonical.get(int(child))
                if child_can is not None:
                    min_sv = min(min_sv, child_can) if min_sv is not None else child_can
            if min_sv is not None:
                canonical[nid_int] = min_sv

    def canonicalize_edges(cx_edges):
        result = set()
        for e in cx_edges:
            src, tgt = int(e[0]), int(e[1])
            cs = canonical.get(src)
            ct = canonical.get(tgt)
            if cs is not None and ct is not None and cs != ct:
                result.add((cs, ct))
        return result

    # --- Per-layer CX comparison using canonical IDs ---
    print("--- CX comparison (canonical SV-based) ---")
    get_layer = lambda nid: lcg.get_chunk_layer(np.uint64(nid))
    new_set = set(c.new_node_ids)

    for layer in range(2, layer_count):
        all_nodes = list(c.new_ids_d.get(layer, [])) + list(c.siblings_d.get(layer, []))
        if not all_nodes:
            continue

        # Our resolved CX
        our_cx = _res_mod.resolve_cx_at_layer(all_nodes, layer, c, {}, get_layer)
        our_can = canonicalize_edges(our_cx)

        # BigTable CX for siblings + our resolved CX for new nodes
        prod_parts = []
        sib_nodes = [n for n in all_nodes if n not in new_set]
        new_nodes = [n for n in all_nodes if n in new_set]

        for nid in new_nodes:
            cx = c.cx.get(int(nid), {}).get(layer)
            if cx is not None and len(cx) > 0:
                prod_parts.append(cx)

        if sib_nodes:
            sib_arr = np.array(sib_nodes, dtype=basetypes.NODE_ID)
            bt = lcg.cg.get_cross_chunk_edges(sib_arr)
            for sid in sib_arr:
                cx = bt[sid].get(layer, types.empty_2d)
                if len(cx) > 0:
                    prod_parts.append(cx)

        if prod_parts:
            prod_cx = np.concatenate(prod_parts).astype(basetypes.NODE_ID)
        else:
            prod_cx = types.empty_2d

        # Canonicalize BigTable edges: need canonical for BT node IDs too
        # BT targets might be old node IDs not in our canonical map — read their SVs
        bt_nodes = set()
        if len(prod_cx) > 0:
            for e in prod_cx:
                for nid in [int(e[0]), int(e[1])]:
                    if nid not in canonical:
                        bt_nodes.add(nid)

        if bt_nodes:
            bt_arr = np.array(list(bt_nodes), dtype=basetypes.NODE_ID)
            bt_l2_map = batch_get_l2children(lcg, bt_arr)
            for nid_key, l2_set in bt_l2_map.items():
                nid_int = int(nid_key)
                min_sv = None
                for l2 in l2_set:
                    l2_can = canonical.get(int(l2))
                    if l2_can is not None:
                        min_sv = min(min_sv, l2_can) if min_sv is not None else l2_can
                if min_sv is not None:
                    canonical[nid_int] = min_sv

        prod_can = canonicalize_edges(prod_cx)

        only_prod = prod_can - our_can
        only_ours = our_can - prod_can
        tag = "MATCH" if not only_prod and not only_ours else "MISMATCH"
        print(f"  L{layer}: ours={len(our_can)} prod={len(prod_can)} +prod={len(only_prod)} +ours={len(only_ours)} {tag}")

        if only_prod:
            for e in list(only_prod)[:3]:
                print(f"    prod-only: sv({e[0]}) -> sv({e[1]})")
        if only_ours:
            for e in list(only_ours)[:3]:
                print(f"    ours-only: sv({e[0]}) -> sv({e[1]})")

    print(f"\nBigTable reads: {len(lcg._read_row_keys)}")
    return {"roots": len(result.new_roots)}
