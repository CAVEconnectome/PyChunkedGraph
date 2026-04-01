"""
Extract sampled subgraph from real BigTable for offline e2e tests.
Produces test_data/e2e_fixture.pkl.

Algorithm:
1. Sample edges from wave files
2. Edge SVs → roots via get_roots
3. BFS from roots top-down: read ALL nodes with ALL data
4. Filter L2 children: edge L2s keep edge SVs, sibling L2s keep ACX source SVs
5. Filter ACX to edges with both SVs in graph
6. Add SV parent rows for all SVs in graph (including ACX targets)

Run:
    from pychunkedgraph.debug.stitch_test.sample_test_data import extract
    extract(force=True)
"""

import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
from tqdm import tqdm
from google.cloud.bigtable.backup import Backup

os.environ.setdefault("BIGTABLE_PROJECT", "zetta-proofreading")
os.environ.setdefault("BIGTABLE_INSTANCE", "pychunkedgraph")

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes
from .tables import BACKUP_ID, CLUSTER_ID, EDGES_SRC, _get_instance

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def _fixture_path(n_edges: int) -> str:
    return os.path.join(FIXTURE_DIR, f"e2e_fixture_{n_edges}e.pkl")
SAMPLE_EDGES_PER_FILE = 2
RNG_SEED = 42
READ_BATCH = 1000


def _list_wave_files() -> dict:
    cf = CloudFiles(EDGES_SRC)
    all_files = sorted(f for f in cf.list() if f.startswith("task_") and f.endswith(".edges"))
    waves = defaultdict(list)
    for f in all_files:
        waves[int(f.split("_")[1])].append(f)
    return dict(sorted(waves.items()))


def _sample_edges(args: tuple) -> np.ndarray:
    cf_url, filename, n, seed = args
    cf = CloudFiles(cf_url)
    edges = pickle.loads(cf.get(filename))
    if len(edges) <= n:
        return edges
    rng = np.random.default_rng(seed)
    return edges[rng.choice(len(edges), size=n, replace=False)]


def _read_node_batch_worker(args: tuple) -> dict:
    """Worker: create CG client, read batch of nodes."""
    table_name, node_ids_list = args
    cg = ChunkedGraph(graph_id=table_name)
    node_ids = np.array(node_ids_list, dtype=basetypes.NODE_ID)
    lc = cg.meta.layer_count
    props = (
        [attributes.Hierarchy.Parent, attributes.Hierarchy.Child]
        + [attributes.Connectivity.AtomicCrossChunkEdge[l] for l in range(2, lc)]
        + [attributes.Connectivity.CrossChunkEdge[l] for l in range(2, lc)]
    )
    raw = cg.client.read_nodes(node_ids=node_ids, properties=props)
    result = {}
    for nid in node_ids:
        data = raw.get(nid, {})
        node = {}
        p = data.get(attributes.Hierarchy.Parent, [])
        if p:
            node["parent"] = int(p[0].value)
        ch = data.get(attributes.Hierarchy.Child, [])
        if ch:
            node["children"] = ch[0].value.copy()
        acx = {}
        for layer in range(2, lc):
            cells = data.get(attributes.Connectivity.AtomicCrossChunkEdge[layer], [])
            if cells and len(cells[0].value) > 0:
                acx[layer] = cells[0].value.copy()
        if acx:
            node["acx"] = acx
        cx = {}
        for layer in range(2, lc):
            cells = data.get(attributes.Connectivity.CrossChunkEdge[layer], [])
            if cells and len(cells[0].value) > 0:
                cx[layer] = cells[0].value.copy()
        if cx:
            node["cx"] = cx
        result[int(nid)] = node
    return result


def _bfs_from_roots(cg: ChunkedGraph, root_ids: np.ndarray) -> dict:
    """BFS top-down from roots. Read ALL nodes at each level in parallel batches."""
    store = {}
    current = root_ids.copy()
    level = 0
    t0 = time.time()
    table_name = cg.meta.graph_config.ID
    n_workers = max(os.cpu_count(), 16)

    while len(current) > 0:
        level += 1
        batches = [
            current[i:i + READ_BATCH].tolist()
            for i in range(0, len(current), READ_BATCH)
        ]
        args = [(table_name, b) for b in batches]
        with get_context("fork").Pool(min(n_workers, len(args))) as pool:
            for batch_data in tqdm(pool.imap_unordered(_read_node_batch_worker, args), total=len(args), desc=f"  level {level}"):
                store.update(batch_data)
        elapsed = time.time() - t0
        layers = cg.get_chunk_layers(current)
        layer_counts = dict(zip(*np.unique(layers, return_counts=True)))
        print(f"  level {level}: {len(current)} nodes {layer_counts} ({elapsed:.1f}s)", flush=True)

        next_level = []
        for nid in current:
            data = store.get(int(nid), {})
            children = data.get("children")
            if children is not None:
                layer = (int(nid) >> 56) & 0xFF
                if layer > 2:
                    next_level.extend(int(c) for c in children if int(c) not in store)

        if not next_level:
            break
        current = np.array(list(set(next_level)), dtype=basetypes.NODE_ID)

    print(f"  BFS complete: {len(store)} nodes in {time.time() - t0:.1f}s")
    return store


def _build_fixture_store(
    real_data: dict, edge_svs: set, edge_l2_of_sv: dict,
) -> dict:
    """Build self-consistent fixture from complete BFS data.

    - L3+ nodes: copied as-is
    - L2 nodes: children filtered (edge SVs or ACX source SVs), CX preserved, ACX filtered
    - SVs: parent row for every SV in the graph (including ACX targets)
    """
    rng = np.random.default_rng(99)
    store = {}
    edge_l2s = set(edge_l2_of_sv.values())

    # Build SV→L2 reverse map from all L2 children in real data
    real_sv_to_l2 = {}
    for nid, data in real_data.items():
        if (nid >> 56) & 0xFF == 2 and "children" in data:
            for sv in data["children"]:
                real_sv_to_l2[int(sv)] = nid

    # L3+ nodes: as-is
    for nid, data in real_data.items():
        layer = (nid >> 56) & 0xFF
        if layer > 2:
            store[nid] = dict(data)

    # L2 nodes: filter children, preserve CX, filter ACX
    all_kept_svs = set()
    for nid, data in real_data.items():
        layer = (nid >> 56) & 0xFF
        if layer != 2:
            continue
        real_children = data.get("children")
        if real_children is None or len(real_children) == 0:
            store[nid] = {k: v for k, v in data.items() if k != "children"}
            continue

        real_ch_set = set(int(sv) for sv in real_children)

        if nid in edge_l2s:
            keep_svs = set(sv for sv, l2 in edge_l2_of_sv.items() if l2 == nid)
        else:
            keep_svs = set()
            if "acx" in data:
                for edges in data["acx"].values():
                    for e in edges:
                        src, tgt = int(e[0]), int(e[1])
                        if src in real_ch_set:
                            keep_svs.add(src)
                        if tgt in real_ch_set:
                            keep_svs.add(tgt)
            if not keep_svs:
                keep_svs.add(int(real_children[rng.choice(len(real_children))]))

        all_kept_svs.update(keep_svs)
        my_svs = np.array(sorted(keep_svs), dtype=basetypes.NODE_ID)

        node = {"children": my_svs}
        if "parent" in data:
            node["parent"] = data["parent"]
        if "cx" in data:
            node["cx"] = data["cx"]
        if "acx" in data:
            sv_set = set(int(s) for s in my_svs)
            filtered_acx = {}
            for lyr, edges in data["acx"].items():
                mask = np.array([int(e[0]) in sv_set for e in edges])
                if mask.any():
                    filtered_acx[lyr] = edges[mask]
            if filtered_acx:
                node["acx"] = filtered_acx
        store[nid] = node

    # ACX target SVs: add to their L2's children + parent rows
    acx_target_svs = set()
    for nid, data in store.items():
        if "acx" not in data:
            continue
        for edges in data["acx"].values():
            for e in edges:
                acx_target_svs.add(int(e[1]))

    for tgt_sv in acx_target_svs:
        l2 = real_sv_to_l2.get(tgt_sv)
        if l2 is None or l2 not in store:
            continue
        all_kept_svs.add(tgt_sv)
        existing = set(int(s) for s in store[l2].get("children", []))
        if tgt_sv not in existing:
            existing.add(tgt_sv)
            store[l2]["children"] = np.array(sorted(existing), dtype=basetypes.NODE_ID)

    # SV parent rows
    for sv_int in all_kept_svs:
        if sv_int not in store:
            l2 = real_sv_to_l2.get(sv_int)
            if l2 is not None:
                store[sv_int] = {"parent": l2}

    return store


def extract(
    edges_range: tuple = (1, 4),
    seed: int = RNG_SEED,
    force: bool = False,
) -> list:
    """Extract fixtures for each edge count in range. Restores table once."""
    start, end = edges_range
    counts = list(range(start, end))

    # Check which need extraction
    to_extract = []
    for n in counts:
        path = _fixture_path(n)
        if os.path.exists(path) and not force:
            print(f"Fixture exists: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
        else:
            to_extract.append(n)

    if not to_extract:
        return [_fixture_path(n) for n in counts]

    # Restore table once
    table_name = "stitch_redesign_test_e2e_sample"
    instance = _get_instance()
    tab = instance.table(table_name)
    if not tab.exists():
        Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID).restore(table_name).result()
        print(f"Restored {table_name}")
    else:
        print(f"Reusing {table_name}")

    cf = CloudFiles(EDGES_SRC)
    wave_files = _list_wave_files()
    print(f"Waves: {list(wave_files.keys())}, files: {sum(len(v) for v in wave_files.values())}")

    try:
        cg = ChunkedGraph(graph_id=table_name)
        meta_bytes = pickle.dumps(cg.meta)
        cv_info = CloudVolume(cg.meta.data_source.WATERSHED, mip=0).info

        for n in to_extract:
            print(f"\n--- Extracting {n} edges per file ---")
            rng = np.random.default_rng(seed)
            file_seeds = rng.integers(0, 2**31, size=sum(len(v) for v in wave_files.values()))
            idx = 0
            edges_per_wave = {}
            for wave, files in wave_files.items():
                wave_args = []
                for f in files:
                    wave_args.append((EDGES_SRC, f, n, int(file_seeds[idx])))
                    idx += 1
                edges_per_wave[wave] = wave_args

            all_sampled_svs = set()
            sampled_edges = {}
            for wave, wave_args in edges_per_wave.items():
                with ThreadPoolExecutor(max_workers=16) as pool:
                    results = list(pool.map(_sample_edges, wave_args))
                sampled_edges[wave] = results
                for e in results:
                    all_sampled_svs.update(e.ravel().tolist())
            print(f"Sampled {len(all_sampled_svs)} unique SVs")

            if not all_sampled_svs:
                print("No SVs sampled, skipping")
                continue

            sv_arr = np.array(list(all_sampled_svs), dtype=basetypes.NODE_ID)
            sv_parents = cg.get_parents(sv_arr)
            edge_l2_of_sv = {int(sv): int(p) for sv, p in zip(sv_arr, sv_parents)}
            roots = cg.get_roots(sv_arr)
            unique_roots = np.unique(roots)
            print(f"Edge SVs: {len(sv_arr)}, L2s: {len(np.unique(sv_parents))}, roots: {len(unique_roots)}")

            print("BFS from roots...")
            real_data = _bfs_from_roots(cg, unique_roots)

            node_store = _build_fixture_store(real_data, all_sampled_svs, edge_l2_of_sv)
            print(f"Fixture store: {len(node_store)} nodes")

            fixture_path = _fixture_path(n)
            fixture = {
                "edges_per_wave": {w: [e.tolist() for e in el] for w, el in sampled_edges.items()},
                "node_store": node_store,
                "meta_bytes": meta_bytes,
                "cv_info": cv_info,
                "sample_config": {"edges_per_file": n, "seed": seed},
            }
            with open(fixture_path, "wb") as f:
                pickle.dump(fixture, f)
            print(f"Fixture saved: {fixture_path} ({os.path.getsize(fixture_path) / 1e6:.1f} MB)")

    finally:
        instance.table(table_name).delete()
        print(f"Deleted {table_name}")

    return [_fixture_path(n) for n in counts]
