from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from pychunkedgraph.graph import ChunkedGraph, basetypes
from .tables import setup_env


def batch_get_l2children(cg_or_reader, node_ids: np.ndarray) -> dict:
    """
    Get L2 descendants for each node_id, returned as {node_id: frozenset(l2_ids)}.
    Uses level-by-level batch get_children to minimize RPCs.
    Accepts ChunkedGraph or CachedReader (for cache reuse).
    """
    node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
    cg = cg_or_reader.cg if hasattr(cg_or_reader, "cg") else cg_or_reader
    reader = cg_or_reader
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

        children_d = reader.get_children(non_l2)
        next_level = []
        for parent_id, children in children_d.items():
            root = node_to_root[int(parent_id)]
            for c in children:
                node_to_root[int(c)] = root
                next_level.append(c)
        current_level = (
            np.array(next_level, dtype=basetypes.NODE_ID)
            if next_level
            else np.array([], dtype=basetypes.NODE_ID)
        )

    return {n: frozenset(root_l2.get(int(n), set())) for n in node_ids}


def extract_structure(cg: ChunkedGraph, roots: np.ndarray) -> dict:
    """
    Extract per-node structure using canonical IDs (table-independent).

    Canonical ID for L2 = frozenset(svs) — its SV children (immutable L1 nodes).
    Canonical ID for layer L>2 = frozenset(canonical_ids of children at layer L-1).

    Cross edges are stored as sets of (canonical_self, canonical_partner) per cx_layer.

    Returns {layer: {canonical_id: {cx_layer: set((canonical_self, canonical_partner))}}}
    """
    roots = np.asarray(roots, dtype=basetypes.NODE_ID)

    # pass 1: walk down from roots, record parent→children
    parent_children = {}  # {node_int: [child_ints]}
    all_nodes_by_layer = defaultdict(list)
    current_nodes = roots.copy()

    while len(current_nodes) > 0:
        layers = cg.get_chunk_layers(current_nodes)
        for node, nl in zip(current_nodes, layers):
            all_nodes_by_layer[int(nl)].append(int(node))

        l2_mask = layers <= 2
        non_l2 = current_nodes[~l2_mask]
        if len(non_l2) == 0:
            break

        children_d = cg.get_children(non_l2)
        next_level = []
        for parent_id, children in children_d.items():
            parent_children[int(parent_id)] = [int(c) for c in children]
            for c in children:
                next_level.append(c)
        current_nodes = (
            np.array(next_level, dtype=basetypes.NODE_ID)
            if next_level
            else np.array([], dtype=basetypes.NODE_ID)
        )

    # pass 2: resolve L2 → SVs to build L2 canonical IDs
    all_l2 = [int(n) for n in all_nodes_by_layer.get(2, [])]
    all_l2_arr = np.array(all_l2, dtype=basetypes.NODE_ID)
    l2_children = cg.get_children(all_l2_arr) if len(all_l2_arr) > 0 else {}

    # canonical_id: node_int → frozenset
    # L2: frozenset of SVs
    canonical = {}
    for l2 in all_l2:
        canonical[l2] = frozenset(int(sv) for sv in l2_children.get(np.uint64(l2), []))

    # pass 2b: build canonical IDs bottom-up for layers 3+
    for layer in sorted(all_nodes_by_layer.keys()):
        if layer <= 2:
            continue
        for node in all_nodes_by_layer[layer]:
            children = parent_children.get(node, [])
            canonical[node] = frozenset(
                canonical[c] for c in children if c in canonical
            )

    # pass 3: read CrossChunkEdge for all non-root nodes (L2+)
    all_non_root = []
    for layer, node_list in all_nodes_by_layer.items():
        if layer >= 2 and layer < cg.meta.layer_count:
            all_non_root.extend(node_list)

    node_cx = {}  # {node_int: {cx_layer: set((canonical_self, canonical_partner))}}
    if all_non_root:
        non_root_arr = np.array(all_non_root, dtype=basetypes.NODE_ID)
        all_cx = cg.get_cross_chunk_edges(non_root_arr, raw_only=True)

        # collect all partner IDs we need to resolve
        all_partners = set()
        for node_id, cx_d in all_cx.items():
            for cx_layer, edges in cx_d.items():
                if len(edges) > 0:
                    all_partners.update(int(p) for p in np.unique(edges[:, 1]))

        # resolve partners not in our tree to their canonical IDs
        unknown = np.array(
            [p for p in all_partners if p not in canonical], dtype=basetypes.NODE_ID
        )
        if len(unknown) > 0:
            # partners are nodes in the same table — get their L2 descendants then SVs
            partner_l2_map = batch_get_l2children(cg, unknown)
            # get SVs for any new L2s
            new_l2s = set()
            for l2set in partner_l2_map.values():
                new_l2s.update(l2set)
            new_l2s -= set(canonical.keys())
            if new_l2s:
                new_l2_arr = np.array(list(new_l2s), dtype=basetypes.NODE_ID)
                new_l2_ch = cg.get_children(new_l2_arr)
                for l2 in new_l2s:
                    canonical[int(l2)] = frozenset(
                        int(sv) for sv in new_l2_ch.get(np.uint64(l2), [])
                    )

            # build canonical for partner nodes from their L2 descendants
            for p_int, l2set in partner_l2_map.items():
                canonical[int(p_int)] = frozenset(
                    canonical.get(int(l2), frozenset()) for l2 in l2set
                )

        # build cross edges using canonical IDs
        for node_id, cx_d in all_cx.items():
            node_int = int(node_id)
            self_canon = canonical.get(node_int)
            if self_canon is None:
                continue
            cx_edges = {}
            for cx_layer, edges in cx_d.items():
                if len(edges) == 0:
                    continue
                pairs = set()
                for row in edges:
                    partner_canon = canonical.get(int(row[1]))
                    if partner_canon is not None:
                        pair = (
                            min(self_canon, partner_canon),
                            max(self_canon, partner_canon),
                        )
                        pairs.add(pair)
                if pairs:
                    cx_edges[int(cx_layer)] = pairs
            node_cx[node_int] = cx_edges

    # pass 4: build output keyed by canonical ID
    nodes = {}
    for layer, node_list in all_nodes_by_layer.items():
        if layer < 2 or layer >= cg.meta.layer_count:
            continue
        layer_nodes = {}
        for node_int in node_list:
            canon = canonical.get(node_int)
            if canon is None or len(canon) == 0:
                continue
            layer_nodes[canon] = node_cx.get(node_int, {})
        nodes[layer] = layer_nodes

    return {"nodes": nodes}


BATCH_SIZE = 500_000
MAX_RETRIES = 3


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
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
    pkl_path = Path(str(save_path).replace(".json", ".pkl.gz"))
    if pkl_path.exists():
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

    # clear stale extraction shards
    for old_batch in save_dir.glob("batch_*"):
        shutil.rmtree(old_batch)

    for i, batch in enumerate(batches):
        roots_per_shard = 250
        n_shards = max(n_workers, (len(batch) + roots_per_shard - 1) // roots_per_shard)
        shards = _shard_roots(batch, n_shards)
        shard_dir = save_dir / f"batch_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        args = [
            (graph_id, shard.tolist(), str(shard_dir / f"shard_{j}.json"))
            for j, shard in enumerate(shards)
        ]

        print(f"  batch {i+1}/{n_batches}: {len(batch)} roots, {len(shards)} shards")
        if args:
            with Pool(min(len(args), 4 * os.cpu_count())) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(_extract_and_save_worker, args),
                        total=len(args),
                        desc=f"  extracting",
                    )
                )

    return save_dir


def _count_layers_worker(shard_path):
    """Worker: count nodes per layer from one shard."""
    structure = _load_structure_file(shard_path)
    return {layer: len(node_dict) for layer, node_dict in structure["nodes"].items()}


def layer_counts_from_shards(save_dir):
    """Compute layer_counts by reading only meta arrays from npz files."""
    shard_paths = sorted(save_dir.rglob("shard_*.pkl.gz"))
    if not shard_paths:
        return {}
    with Pool(min(len(shard_paths), os.cpu_count())) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    _count_layers_worker, [str(p) for p in shard_paths]
                ),
                total=len(shard_paths),
                desc="  counting layers",
            )
        )
    totals = defaultdict(int)
    for counts in results:
        for layer, n in counts.items():
            totals[layer] += n
    return dict(totals)


def batched_extract_and_compare(
    graph_id_a, roots_a, graph_id_b, roots_b, save_dir, baseline_extract_dir=None
):
    """
    Extract structure from both tables, then compare per-node at each layer.
    Each node is identified by its SV set, compared with its cross edges at all layers.
    If baseline_extract_dir is provided, skips baseline extraction and loads from there.
    """
    dir_a = Path(baseline_extract_dir) if baseline_extract_dir else save_dir / "baseline"
    dir_b = save_dir / "proposed"

    if not baseline_extract_dir:
        print("extracting baseline...")
        batched_extract_structure(graph_id_a, roots_a, save_dir=dir_a)
    print("extracting proposed...")
    batched_extract_structure(graph_id_b, roots_b, save_dir=dir_b)

    print("comparing per-node...")
    t0 = time.time()
    # load both sides in parallel using threads (IO-bound: reading files from disk)
    with ThreadPoolExecutor(max_workers=2) as tpe:
        fut_a = tpe.submit(_collect_nodes_from_shards, dir_a)
        fut_b = tpe.submit(_collect_nodes_from_shards, dir_b)
        nodes_a = fut_a.result()
        nodes_b = fut_b.result()

    match = True
    all_layers = sorted(set(nodes_a.keys()) | set(nodes_b.keys()))
    for layer in all_layers:
        na = nodes_a.get(layer, {})
        nb = nodes_b.get(layer, {})
        keys_a = set(na.keys())
        keys_b = set(nb.keys())

        # check node count
        if len(keys_a) != len(keys_b):
            print(f"  MISMATCH layer {layer}: {len(keys_a)} vs {len(keys_b)} nodes")
            match = False

        # check component membership (SV sets)
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        if only_a or only_b:
            sizes_a = sorted([len(s) for s in only_a], reverse=True)[:5]
            sizes_b = sorted([len(s) for s in only_b], reverse=True)[:5]
            print(
                f"  COMPONENT MISMATCH layer {layer}: {len(only_a)} only in A (sizes: {sizes_a}), {len(only_b)} only in B (sizes: {sizes_b})"
            )
            match = False
        else:
            # components match — check cx edge counts per node
            cx_mismatches = 0
            for svs in keys_a:
                if na[svs] != nb[svs]:
                    cx_mismatches += 1
            total_svs = sum(len(s) for s in keys_a)
            if cx_mismatches:
                print(
                    f"  CX COUNT MISMATCH layer {layer}: {cx_mismatches}/{len(keys_a)} nodes have different cx edge counts"
                )
                match = False
            else:
                print(
                    f"  layer {layer}: {len(keys_a)} nodes, {total_svs} total SVs, cx counts match — OK"
                )

    print(f"comparison: {'MATCH' if match else 'MISMATCH'} ({time.time() - t0:.1f}s)")
    return match


def _load_shard_worker(shard_path):
    """Worker: load one shard file and return its nodes dict."""
    return _load_structure_file(shard_path)["nodes"]


def _collect_nodes_from_shards(save_dir):
    """Load all shard files in parallel, return {layer: {frozenset(svs): {cx_layer: set((sv_a,sv_b))}}}."""
    shard_paths = sorted(str(p) for p in save_dir.rglob("shard_*.pkl.gz"))
    if not shard_paths:
        return {}
    with Pool(min(len(shard_paths), os.cpu_count())) as pool:
        all_nodes = pool.map(_load_shard_worker, shard_paths)
    result = defaultdict(dict)
    for nodes in all_nodes:
        for layer, node_dict in nodes.items():
            result[layer].update(node_dict)
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


def compare_structures(struct_a, struct_b):
    """Compare two structures by component SV sets at each layer. Returns True if match."""
    nodes_a = struct_a["nodes"]
    nodes_b = struct_b["nodes"]
    all_layers = sorted(set(nodes_a.keys()) | set(nodes_b.keys()))
    match = True
    for layer in all_layers:
        na = nodes_a.get(layer, {})
        nb = nodes_b.get(layer, {})
        keys_a = set(na.keys())
        keys_b = set(nb.keys())

        if len(keys_a) != len(keys_b):
            print(f"  MISMATCH layer {layer}: {len(keys_a)} vs {len(keys_b)} nodes")
            match = False

        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        if only_a or only_b:
            sizes_a = sorted([len(s) for s in only_a], reverse=True)[:5]
            sizes_b = sorted([len(s) for s in only_b], reverse=True)[:5]
            print(
                f"  COMPONENT MISMATCH layer {layer}: {len(only_a)} only in A (sizes: {sizes_a}), {len(only_b)} only in B (sizes: {sizes_b})"
            )
            match = False
        else:
            total_svs = sum(len(s) for s in keys_a)
            print(f"  layer {layer}: {len(keys_a)} nodes, {total_svs} total SVs — OK")
    return match


def _save_structure_file(path, structure):
    """Save structure as compressed pickle (canonical IDs are nested frozensets)."""
    path = Path(str(path).replace(".json", ".pkl.gz").replace(".npz", ".pkl.gz"))
    with gzip.open(path, "wb") as f:
        pickle.dump(structure, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_structure_file(path):
    """Load structure from compressed pickle."""
    path = Path(path)
    pkl_path = Path(str(path).replace(".json", ".pkl.gz").replace(".npz", ".pkl.gz"))
    with gzip.open(pkl_path, "rb") as f:
        return pickle.load(f)
