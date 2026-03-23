from collections import defaultdict
import gzip
import json
from multiprocessing import Pool
import os
from pathlib import Path
import time

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from pychunkedgraph.graph import ChunkedGraph, basetypes
from .tables import setup_env


def batch_get_l2children(cg: ChunkedGraph, node_ids: np.ndarray) -> dict:
    """
    Get L2 descendants for each node_id, returned as {node_id: frozenset(l2_ids)}.
    Uses level-by-level batch get_children to minimize RPCs.
    """
    node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
    node_to_root = {int(n): int(n) for n in node_ids}
    root_l2 = defaultdict(set)
    current_level = node_ids.copy()

    while len(current_level) > 0:
        layers = cg.get_chunk_layers(current_level)

        l2_mask = layers <= 2
        for n in current_level[l2_mask]:
            root_l2[node_to_root[int(n)]].add(int(n))

        non_l2 = current_level[~l2_mask]
        if len(non_l2) == 0:
            break

        children_d = cg.get_children(non_l2)
        next_level = []
        for parent_id, children in children_d.items():
            root = node_to_root[int(parent_id)]
            for c in children:
                node_to_root[int(c)] = root
                next_level.append(c)
        current_level = np.array(next_level, dtype=basetypes.NODE_ID) if next_level else np.array([], dtype=basetypes.NODE_ID)

    return {n: frozenset(root_l2.get(int(n), set())) for n in node_ids}


def extract_structure(cg: ChunkedGraph, roots: np.ndarray) -> dict:
    """
    Extract full hierarchy structure from roots.
    Walks down level by level, collecting at each non-root layer:
    - components: {layer: [frozenset(l2_descendants), ...]}
    - cross_edges: {layer: [(frozenset(l2_src), frozenset(l2_dst)), ...]}
    """
    roots = np.asarray(roots, dtype=basetypes.NODE_ID)
    components = defaultdict(list)
    cross_edges = defaultdict(list)
    node_l2 = {}  # {node_id_int: frozenset(l2_ids)} — built bottom-up

    # pass 1: walk down from roots, record parent→children relationships
    parent_children = {}  # {node_int: [child_ints]}
    all_nodes_by_layer = defaultdict(list)  # {layer: [node_ids]}
    current_nodes = roots.copy()

    while len(current_nodes) > 0:
        layers = cg.get_chunk_layers(current_nodes)

        for node, nl in zip(current_nodes, layers):
            all_nodes_by_layer[int(nl)].append(int(node))

        l2_mask = layers <= 2
        for n in current_nodes[l2_mask]:
            node_l2[int(n)] = frozenset([int(n)])

        non_l2 = current_nodes[~l2_mask]
        if len(non_l2) == 0:
            break

        children_d = cg.get_children(non_l2)
        next_level = []
        for parent_id, children in children_d.items():
            parent_children[int(parent_id)] = [int(c) for c in children]
            for c in children:
                next_level.append(c)

        current_nodes = np.array(next_level, dtype=basetypes.NODE_ID) if next_level else np.array([], dtype=basetypes.NODE_ID)

    # pass 2: compute L2 descendants bottom-up
    for layer in sorted(all_nodes_by_layer.keys()):
        if layer <= 2:
            continue
        for node in all_nodes_by_layer[layer]:
            children = parent_children.get(node, [])
            l2_desc = set()
            for c in children:
                l2_desc.update(node_l2.get(c, set()))
            node_l2[node] = frozenset(l2_desc)

    # pass 2b: resolve L2 → SVs so components use SV IDs (stable across tables)
    all_l2 = set()
    for l2set in node_l2.values():
        all_l2.update(l2set)
    all_l2_arr = np.array(list(all_l2), dtype=basetypes.NODE_ID)
    l2_children = cg.get_children(all_l2_arr) if len(all_l2_arr) > 0 else {}
    l2_to_svs = {int(l2): frozenset(int(sv) for sv in l2_children.get(l2, [])) for l2 in all_l2_arr}

    node_svs = {}  # {node_int: frozenset(sv_ids)}
    for node_int, l2set in node_l2.items():
        svs = set()
        for l2 in l2set:
            svs.update(l2_to_svs.get(l2, set()))
        node_svs[node_int] = frozenset(svs)

    # pass 3: collect components and cross edges at each non-root layer
    # batch read cross edges for all non-root, non-L2 nodes
    all_non_root = []
    for layer, nodes in all_nodes_by_layer.items():
        if layer > 2 and layer < cg.meta.layer_count:
            all_non_root.extend(nodes)

    if all_non_root:
        non_root_arr = np.array(all_non_root, dtype=basetypes.NODE_ID)
        all_cx = cg.get_cross_chunk_edges(non_root_arr, raw_only=True)

        # collect all unique partners and batch-resolve their SV descendants
        all_partners = set()
        for node_id, cx_d in all_cx.items():
            for layer, edges in cx_d.items():
                if len(edges):
                    all_partners.update(np.unique(edges[:, 1]).tolist())

        unknown_partners = np.array(
            [p for p in all_partners if p not in node_svs],
            dtype=basetypes.NODE_ID,
        )
        if len(unknown_partners) > 0:
            partner_l2_map = batch_get_l2children(cg, unknown_partners)
            # resolve partner L2s to SVs
            partner_l2_all = set()
            for l2set in partner_l2_map.values():
                partner_l2_all.update(l2set)
            new_l2s = np.array([l2 for l2 in partner_l2_all if l2 not in l2_to_svs], dtype=basetypes.NODE_ID)
            if len(new_l2s) > 0:
                new_l2_children = cg.get_children(new_l2s)
                for l2 in new_l2s:
                    l2_to_svs[int(l2)] = frozenset(int(sv) for sv in new_l2_children.get(l2, []))

            for p, l2set in partner_l2_map.items():
                svs = set()
                for l2 in l2set:
                    svs.update(l2_to_svs.get(int(l2), set()))
                node_svs[int(p)] = frozenset(svs)

        # build components and cross edges using SV sets
        for node_int in all_non_root:
            nl = cg.get_chunk_layer(np.uint64(node_int))
            nl_int = int(nl)
            my_svs = node_svs.get(node_int, frozenset())
            if my_svs:
                components[nl_int].append(my_svs)

            cx_d = all_cx.get(np.uint64(node_int), {})
            for cx_layer, edges in cx_d.items():
                if len(edges) == 0:
                    continue
                partners = np.unique(edges[:, 1])
                for p in partners:
                    psvs = node_svs.get(int(p), frozenset())
                    if psvs:
                        cross_edges[int(cx_layer)].append((my_svs, psvs))

    # sort for deterministic comparison
    result_comps = {}
    for layer in components:
        result_comps[layer] = sorted(components[layer], key=lambda s: min(s))
    result_cx = {}
    for layer in cross_edges:
        result_cx[layer] = sorted(
            cross_edges[layer],
            key=lambda pair: (min(pair[0]), min(pair[1])),
        )

    return {"components": result_comps, "cross_edges": result_cx}


BATCH_SIZE = 500_000
MAX_RETRIES = 3


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=1, max=8))
def _extract_with_retry(graph_id, roots_list):
    setup_env()
    cg = ChunkedGraph(graph_id=graph_id)
    roots = np.array(roots_list, dtype=basetypes.NODE_ID)
    return extract_structure(cg, roots)


def _extract_and_save_worker(args):
    """Worker: extract structure for a shard of roots, save to disk.
    args: (graph_id, roots_list, save_path)
    """
    graph_id, roots_list, save_path = args
    save_path = Path(save_path)
    npz_path = Path(str(save_path).replace(".json", ".npz"))
    if npz_path.exists() or save_path.exists():
        return str(save_path)

    structure = _extract_with_retry(graph_id, roots_list)
    _save_structure_file(save_path, structure)
    return str(save_path)


def _shard_roots(roots, n_workers):
    """Split roots into shards, one per worker."""
    roots = np.asarray(roots, dtype=basetypes.NODE_ID)
    n = min(len(roots), n_workers)
    if n <= 0:
        return []
    return [chunk for chunk in np.array_split(roots, n) if len(chunk) > 0]


def batched_extract_structure(graph_id, roots, save_dir):
    """
    Extract structure in BATCH_SIZE batches.
    Each batch is sharded across cpu_count workers, each worker saves its own JSON.
    """
    roots = np.asarray(roots, dtype=basetypes.NODE_ID)
    n_batches = max(1, (len(roots) + BATCH_SIZE - 1) // BATCH_SIZE)
    batches = np.array_split(roots, n_batches)
    n_workers = os.cpu_count()

    for i, batch in enumerate(batches):
        shards = _shard_roots(batch, n_workers)
        shard_dir = save_dir / f"batch_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        args = [
            (graph_id, shard.tolist(), str(shard_dir / f"shard_{j}.json"))
            for j, shard in enumerate(shards)
        ]
        # skip shards already on disk
        pending = [a for a in args if not Path(a[2]).exists()]
        cached = len(args) - len(pending)

        print(f"  batch {i+1}/{n_batches}: {len(batch)} roots, {len(shards)} shards ({cached} cached)...", end="", flush=True)
        t0 = time.time()
        if pending:
            with Pool(len(pending)) as pool:
                pool.map(_extract_and_save_worker, pending)
        print(f" {time.time() - t0:.1f}s")

    return save_dir


def layer_counts_from_shards(save_dir):
    """Compute layer_counts by summing across all shard files without merging."""
    counts = defaultdict(int)
    for shard_path in sorted(save_dir.rglob("shard_*.*")):
        structure = _load_structure_file(shard_path)
        for layer, ccs in structure["components"].items():
            counts[layer] += len(ccs)
    return dict(counts)


def batched_extract_and_compare(graph_id_a, roots_a, graph_id_b, roots_b, save_dir):
    """
    Extract structure from both tables, then compare using SV-based components.
    Each side is extracted independently into its own subdirectory.
    Comparison is order-independent (uses sets, not sorted lists).
    """
    dir_a = save_dir / "current"
    dir_b = save_dir / "proposed"

    print("extracting current...")
    batched_extract_structure(graph_id_a, roots_a, save_dir=dir_a)
    print("extracting proposed...")
    batched_extract_structure(graph_id_b, roots_b, save_dir=dir_b)

    print("comparing...")
    t0 = time.time()
    comps_a = _collect_components_from_shards(dir_a)
    comps_b = _collect_components_from_shards(dir_b)
    cx_a = _collect_cross_edges_from_shards(dir_a)
    cx_b = _collect_cross_edges_from_shards(dir_b)

    match = True
    all_layers = sorted(set(comps_a.keys()) | set(comps_b.keys()))
    for layer in all_layers:
        sa = comps_a.get(layer, set())
        sb = comps_b.get(layer, set())
        if sa != sb:
            only_a = len(sa - sb)
            only_b = len(sb - sa)
            print(f"  COMPONENT MISMATCH layer {layer}: {len(sa)} vs {len(sb)}, only_a={only_a}, only_b={only_b}")
            match = False
        else:
            print(f"  components layer {layer}: {len(sa)} OK")

    all_cx_layers = sorted(set(cx_a.keys()) | set(cx_b.keys()))
    for layer in all_cx_layers:
        sa = cx_a.get(layer, set())
        sb = cx_b.get(layer, set())
        if sa != sb:
            only_a = len(sa - sb)
            only_b = len(sb - sa)
            print(f"  CX EDGE MISMATCH layer {layer}: {len(sa)} vs {len(sb)}, only_a={only_a}, only_b={only_b}")
            match = False
        else:
            print(f"  cx_edges layer {layer}: {len(sa)} OK")

    print(f"comparison: {'MATCH' if match else 'MISMATCH'} ({time.time() - t0:.1f}s)")
    return match


def _collect_components_from_shards(save_dir):
    """Load all shard files, return {layer: set(frozenset(svs))}."""
    result = defaultdict(set)
    for shard_path in sorted(save_dir.rglob("shard_*.*")):
        structure = _load_structure_file(shard_path)
        for layer, ccs in structure["components"].items():
            for cc in ccs:
                result[layer].add(cc)
    return dict(result)


def _collect_cross_edges_from_shards(save_dir):
    """Load all shard files, return {layer: set(frozenset(src_svs, dst_svs))}."""
    result = defaultdict(set)
    for shard_path in sorted(save_dir.rglob("shard_*.*")):
        structure = _load_structure_file(shard_path)
        for layer, pairs in structure["cross_edges"].items():
            for src, dst in pairs:
                result[layer].add(frozenset([src, dst]))
    return dict(result)


def _convert_for_json(obj):
    if isinstance(obj, dict):
        return {_convert_for_json(k): _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(_convert_for_json(x) for x in obj)
    return obj


def _compare_components(struct_a, struct_b):
    comps_a = struct_a.get("components", {})
    comps_b = struct_b.get("components", {})
    all_layers = sorted(set(comps_a.keys()) | set(comps_b.keys()))
    match = True
    for layer in all_layers:
        ccs_a = sorted(comps_a.get(layer, []), key=lambda s: min(s))
        ccs_b = sorted(comps_b.get(layer, []), key=lambda s: min(s))
        if len(ccs_a) != len(ccs_b):
            print(f"  COMPONENT MISMATCH layer {layer}: {len(ccs_a)} vs {len(ccs_b)}")
            match = False
            continue
        for i, (a, b) in enumerate(zip(ccs_a, ccs_b)):
            if a != b:
                print(f"  COMPONENT MISMATCH layer {layer} cc {i}: only in A={a-b}, only in B={b-a}")
                match = False
    if match:
        total = sum(len(v) for v in comps_a.values())
        print(f"  COMPONENTS MATCH: {total} across {len(all_layers)} layers")
    return match


def _compare_cross_edges(struct_a, struct_b):
    cx_a = struct_a.get("cross_edges", {})
    cx_b = struct_b.get("cross_edges", {})
    all_layers = sorted(set(cx_a.keys()) | set(cx_b.keys()))
    match = True
    for layer in all_layers:
        pairs_a = {frozenset([src, dst]) for src, dst in cx_a.get(layer, [])}
        pairs_b = {frozenset([src, dst]) for src, dst in cx_b.get(layer, [])}
        if pairs_a != pairs_b:
            print(f"  CROSS EDGE MISMATCH layer {layer}: {len(pairs_a-pairs_b)} only in A, {len(pairs_b-pairs_a)} only in B")
            match = False
        else:
            print(f"  CROSS EDGES MATCH layer {layer}: {len(pairs_a)} connections")
    return match


def _save_structure_file(path, structure):
    path = Path(str(path).replace(".json", ".npz"))
    layers = sorted(set(structure["components"].keys()) | set(structure["cross_edges"].keys()))
    arrays = {}
    for layer in layers:
        ccs = structure["components"].get(layer, [])
        if ccs:
            # store as flat array + offsets for variable-length component sets
            offsets = np.array([0] + [len(c) for c in ccs], dtype=np.int64)
            offsets = np.cumsum(offsets)
            flat = np.concatenate([np.array(sorted(c), dtype=np.uint64) for c in ccs])
            arrays[f"comp_{layer}_flat"] = flat
            arrays[f"comp_{layer}_offsets"] = offsets

        cx = structure["cross_edges"].get(layer, [])
        if cx:
            # store as Nx2 array of (min_src_sv, min_dst_sv) for each pair
            # plus flat arrays for full SV sets
            src_offsets = [0]
            dst_offsets = [0]
            src_flat = []
            dst_flat = []
            for src, dst in cx:
                s = sorted(src)
                d = sorted(dst)
                src_flat.extend(s)
                dst_flat.extend(d)
                src_offsets.append(len(src_flat))
                dst_offsets.append(len(dst_flat))
            arrays[f"cx_{layer}_src_flat"] = np.array(src_flat, dtype=np.uint64)
            arrays[f"cx_{layer}_src_offsets"] = np.array(src_offsets, dtype=np.int64)
            arrays[f"cx_{layer}_dst_flat"] = np.array(dst_flat, dtype=np.uint64)
            arrays[f"cx_{layer}_dst_offsets"] = np.array(dst_offsets, dtype=np.int64)

    np.savez_compressed(path, **arrays)


def _load_structure_file(path):
    path = Path(path)
    npz_path = Path(str(path).replace(".json", ".npz"))
    if npz_path.exists():
        return _load_npz_structure(npz_path)
    # fallback for old json/gz files
    gz_path = Path(str(path).removesuffix(".gz") + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rt") as f:
            data = json.load(f)
    else:
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


def _load_npz_structure(path):
    data = np.load(path)
    keys = list(data.keys())
    components = {}
    cross_edges = {}

    # find all layers from key names
    comp_layers = set()
    cx_layers = set()
    for k in keys:
        if k.startswith("comp_") and k.endswith("_flat"):
            comp_layers.add(int(k.split("_")[1]))
        if k.startswith("cx_") and k.endswith("_src_flat"):
            cx_layers.add(int(k.split("_")[1]))

    for layer in comp_layers:
        flat = data[f"comp_{layer}_flat"]
        offsets = data[f"comp_{layer}_offsets"]
        ccs = []
        for i in range(len(offsets) - 1):
            ccs.append(frozenset(flat[offsets[i]:offsets[i+1]].tolist()))
        components[layer] = ccs

    for layer in cx_layers:
        src_flat = data[f"cx_{layer}_src_flat"]
        src_offsets = data[f"cx_{layer}_src_offsets"]
        dst_flat = data[f"cx_{layer}_dst_flat"]
        dst_offsets = data[f"cx_{layer}_dst_offsets"]
        pairs = []
        for i in range(len(src_offsets) - 1):
            src = frozenset(src_flat[src_offsets[i]:src_offsets[i+1]].tolist())
            dst = frozenset(dst_flat[dst_offsets[i]:dst_offsets[i+1]].tolist())
            pairs.append((src, dst))
        cross_edges[layer] = pairs

    return {"components": components, "cross_edges": cross_edges}
