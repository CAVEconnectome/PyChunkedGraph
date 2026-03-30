"""
One-time script to extract a sampled subgraph from real BigTable + edge files.
Produces test_data/e2e_fixture.pkl for offline e2e tests.

The fixture is a SELF-CONSISTENT mini-graph:
- L2 children = only sampled SVs (not all real SVs)
- ACX filtered to edges where both SVs are sampled
- Higher-layer children = only nodes with sampled descendants
- CX filtered to edges between nodes in the sampled graph

Run:
    from pychunkedgraph.debug.stitch_test.sample_test_data import extract
    extract(force=True)  # force=True to regenerate
"""

import os
import pickle
from collections import defaultdict

import numpy as np
from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
from google.cloud.bigtable.backup import Backup

os.environ.setdefault("BIGTABLE_PROJECT", "zetta-proofreading")
os.environ.setdefault("BIGTABLE_INSTANCE", "pychunkedgraph")

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes
from .tables import BACKUP_ID, CLUSTER_ID, EDGES_SRC, _get_instance

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "test_data", "e2e_fixture.pkl")
SAMPLE_EDGES_PER_FILE = 5
RNG_SEED = 42


def _list_wave_files():
    cf = CloudFiles(EDGES_SRC)
    all_files = sorted([f for f in cf.list() if f.startswith("task_")])
    waves = defaultdict(list)
    for f in all_files:
        wave = int(f.split("_")[1])
        waves[wave].append(f)
    return dict(sorted(waves.items()))


def _sample_edges(cf, filename: str, rng: np.random.Generator, n: int) -> np.ndarray:
    edges = pickle.loads(cf.get(filename))
    if len(edges) <= n:
        return edges
    idx = rng.choice(len(edges), size=n, replace=False)
    return edges[idx]


def _read_node_data(cg: ChunkedGraph, node_ids: np.ndarray) -> dict:
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
        parent_cells = data.get(attributes.Hierarchy.Parent, [])
        if parent_cells:
            node["parent"] = int(parent_cells[0].value)
        child_cells = data.get(attributes.Hierarchy.Child, [])
        if child_cells:
            node["children"] = child_cells[0].value.copy()
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


def _walk_to_root(cg: ChunkedGraph, node_ids: np.ndarray, node_store: dict) -> None:
    """Walk parent chain from node_ids to root, reading and storing each level."""
    root_layer = cg.meta.layer_count
    current = node_ids
    visited = set(int(n) for n in node_ids)
    while len(current) > 0:
        layers = cg.get_chunk_layers(current)
        non_root = current[layers < root_layer]
        if len(non_root) == 0:
            break
        parents = cg.get_parents(non_root)
        new_parents = []
        for nid, parent in zip(non_root, parents):
            if int(parent) != 0 and int(parent) not in visited:
                new_parents.append(int(parent))
                visited.add(int(parent))
        if not new_parents:
            break
        parent_arr = np.array(new_parents, dtype=basetypes.NODE_ID)
        parent_data = _read_node_data(cg, parent_arr)
        node_store.update(parent_data)
        current = parent_arr


def _rewrite_graph(real_data: dict, sampled_svs: set, l2_of_sv: dict) -> dict:
    """Build self-consistent mini-graph.

    - L2 nodes with edge SVs: children = all edge SVs for this L2, ACX filtered
    - Sibling L2 nodes (no edge SVs): children = 2 sampled real SVs
    - L3+ nodes: as-is from real table
    - SVs: parent row for each SV in the graph
    """
    rng = np.random.default_rng(99)
    store = {}
    edge_l2s = set(l2_of_sv.values())
    all_svs_in_graph = set(sampled_svs)
    sv_to_l2 = dict(l2_of_sv)  # will add sibling SVs too

    # Edge L2 nodes: children = edge SVs, ACX filtered
    for l2_int in edge_l2s:
        real = real_data.get(l2_int, {})
        my_svs = np.array(
            sorted(sv for sv, l2 in l2_of_sv.items() if l2 == l2_int),
            dtype=basetypes.NODE_ID,
        )
        node = {"children": my_svs}
        if "parent" in real:
            node["parent"] = real["parent"]
        if "acx" in real:
            sv_set = set(int(s) for s in my_svs)
            filtered_acx = {}
            for layer, edges in real["acx"].items():
                mask = np.array([int(e[0]) in sv_set and int(e[1]) in sv_set for e in edges])
                if mask.any():
                    filtered_acx[layer] = edges[mask]
            if filtered_acx:
                node["acx"] = filtered_acx
        store[l2_int] = node

    # Sibling L2 nodes: sample 2 real SVs as children
    for nid_int, data in real_data.items():
        if nid_int in store:
            continue
        layer = (nid_int >> 56) & 0xFF
        if layer != 2:
            continue
        real_children = data.get("children")
        if real_children is None or len(real_children) == 0:
            store[nid_int] = {k: v for k, v in data.items() if k != "children"}
            continue
        n_pick = min(1, len(real_children))
        picked = rng.choice(len(real_children), size=n_pick, replace=False)
        my_svs = real_children[picked].astype(basetypes.NODE_ID)
        for sv in my_svs:
            all_svs_in_graph.add(int(sv))
            sv_to_l2[int(sv)] = nid_int
        node = {"children": my_svs}
        if "parent" in data:
            node["parent"] = data["parent"]
        if "acx" in data:
            sv_set = set(int(s) for s in my_svs)
            filtered_acx = {}
            for layer_k, edges in data["acx"].items():
                mask = np.array([int(e[0]) in sv_set for e in edges])
                if mask.any():
                    filtered_acx[layer_k] = edges[mask]
            if filtered_acx:
                node["acx"] = filtered_acx
        store[nid_int] = node

    # L3+ nodes: as-is
    for nid_int, data in real_data.items():
        if nid_int in store:
            continue
        store[nid_int] = dict(data)

    # SV rows: parent pointer for every SV in the graph
    for sv_int in all_svs_in_graph:
        if sv_int not in store:
            l2 = sv_to_l2.get(sv_int)
            if l2 is not None:
                store[sv_int] = {"parent": l2}

    return store


def extract(
    sample_edges_per_file: int = SAMPLE_EDGES_PER_FILE,
    seed: int = RNG_SEED,
    force: bool = False,
) -> str:
    if os.path.exists(FIXTURE_PATH) and not force:
        print(f"Fixture exists: {FIXTURE_PATH} ({os.path.getsize(FIXTURE_PATH) / 1e6:.1f} MB)")
        return FIXTURE_PATH

    rng = np.random.default_rng(seed)
    cf = CloudFiles(EDGES_SRC)
    wave_files = _list_wave_files()
    print(f"Waves: {list(wave_files.keys())}, files: {sum(len(v) for v in wave_files.values())}")

    # Sample edges
    edges_per_wave = {}
    all_sampled_svs = set()
    for wave, files in wave_files.items():
        wave_edges = []
        for f in files:
            sampled = _sample_edges(cf, f, rng, sample_edges_per_file)
            wave_edges.append(sampled)
            all_sampled_svs.update(sampled.ravel().tolist())
        edges_per_wave[wave] = wave_edges
    print(f"Sampled {len(all_sampled_svs)} unique SVs")

    # Restore table
    table_name = "stitch_redesign_test_e2e_sample"
    instance = _get_instance()
    tab = instance.table(table_name)
    if tab.exists():
        tab.delete()
    Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID).restore(table_name).result()
    print(f"Restored {table_name}")

    try:
        cg = ChunkedGraph(graph_id=table_name)
        meta_bytes = pickle.dumps(cg.meta)
        cv_info = CloudVolume(cg.meta.data_source.WATERSHED, mip=0).info

        # SVs → L2 parents
        sv_arr = np.array(list(all_sampled_svs), dtype=basetypes.NODE_ID)
        sv_parents = cg.get_parents(sv_arr)
        l2_of_sv = {int(sv): int(p) for sv, p in zip(sv_arr, sv_parents)}
        l2_ids = np.unique(sv_parents)
        print(f"L2 nodes: {len(l2_ids)}")

        # Read L2 data
        real_data = _read_node_data(cg, l2_ids)
        print(f"L2 nodes read: {len(real_data)}")

        # Walk full hierarchy to root
        print("Walking to root...")
        _walk_to_root(cg, l2_ids, real_data)

        # Read ALL children at each level (siblings) so hierarchy is complete
        rounds = 0
        while True:
            missing = set()
            for nid_int, data in list(real_data.items()):
                children = data.get("children")
                if children is not None:
                    for ch in children:
                        if int(ch) not in real_data:
                            missing.add(int(ch))
            if not missing:
                break
            miss_arr = np.array(list(missing), dtype=basetypes.NODE_ID)
            miss_data = _read_node_data(cg, miss_arr)
            real_data.update(miss_data)
            _walk_to_root(cg, miss_arr, real_data)
            rounds += 1
            print(f"  round {rounds}: {len(missing)} sibling nodes read")

        print(f"Total real data: {len(real_data)} nodes")

        # Build mini-graph
        node_store = _rewrite_graph(real_data, all_sampled_svs, l2_of_sv)
        print(f"Rewritten graph: {len(node_store)} nodes")

        # Save
        fixture = {
            "edges_per_wave": {w: [e.tolist() for e in el] for w, el in edges_per_wave.items()},
            "node_store": node_store,
            "meta_bytes": meta_bytes,
            "cv_info": cv_info,
            "sample_config": {"edges_per_file": sample_edges_per_file, "seed": seed},
        }
        with open(FIXTURE_PATH, "wb") as f:
            pickle.dump(fixture, f)
        print(f"Fixture saved: {FIXTURE_PATH} ({os.path.getsize(FIXTURE_PATH) / 1e6:.1f} MB)")

    finally:
        instance.table(table_name).delete()
        print(f"Deleted {table_name}")

    return FIXTURE_PATH
