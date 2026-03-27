"""
LocalChunkedGraph: composite cached ChunkedGraph.

Owns WaveCache. Delegates to stateless operation modules (hierarchy, topology).
External code calls lcg methods — never touches cache directly.
"""

import logging
import os
import pickle
import random
import time
from collections import defaultdict
import numpy as np
from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes, types
from pychunkedgraph.graph.chunks import hierarchy as chunk_hierarchy
from pychunkedgraph.graph.edges.utils import get_cross_chunk_edges_layer
from pychunkedgraph.graph.utils import flatgraph
from kvdbclient import serializers

from . import topology
from . import tree
from .id_allocator import batch_create as batch_create_ids
from .utils import batch_get_l2children, timed
from .wave_cache import WaveCache

log = logging.getLogger(__name__)


_worker_meta = None
_worker_cv_info = None


def _setup_env() -> None:
    os.environ.setdefault("BIGTABLE_PROJECT", "zetta-proofreading")
    os.environ.setdefault("BIGTABLE_INSTANCE", "pychunkedgraph")


class LocalChunkedGraph:

    def __init__(self, graph_id: str, preloaded: tuple = None, incremental: tuple = None, meta=None) -> None:
        _setup_env()
        self.cg = ChunkedGraph(graph_id=graph_id, meta=meta) if meta else ChunkedGraph(graph_id=graph_id)
        self._cache = WaveCache(preloaded, incremental=incremental)
        self.rpc_log: list[tuple] = []

    def prepare_pool_init(self) -> tuple:
        meta_bytes = pickle.dumps(self.cg.meta)
        cv = CloudVolume(self.cg.meta.data_source.WATERSHED, mip=0)
        return meta_bytes, cv.info

    @staticmethod
    def pool_init(meta_bytes: bytes, cv_info: dict) -> None:
        global _worker_meta, _worker_cv_info
        _worker_meta = pickle.loads(meta_bytes)
        _worker_cv_info = cv_info
        _setup_env()

    @staticmethod
    def create_worker(graph_id: str, preloaded: tuple = None, incremental: tuple = None) -> "LocalChunkedGraph":
        assert _worker_meta is not None, "pool_init not called"
        lcg = LocalChunkedGraph(graph_id, preloaded=preloaded, incremental=incremental, meta=_worker_meta)
        if _worker_cv_info is not None:
            lcg.cg.meta._ws_cv = CloudVolume(
                lcg.cg.meta.data_source.WATERSHED, info=_worker_cv_info, progress=False,
            )
        return lcg

    @property
    def meta(self):
        return self.cg.meta

    # -- Lifecycle --

    def begin_stitch(self) -> None:
        self._cache.begin_stitch()
        self.rpc_log = []

    def end_stitch(self) -> None:
        c = self._cache
        c.save_wave_state(
            old_to_new=c.old_to_new, new_node_ids=c.new_node_ids,
            sibling_ids=c.sibling_ids, raw_cx_edges=c.raw_cx_edges,
            children=c.children_d,
        )

    # -- CG passthrough --

    def get_chunk_layer(self, node_id) -> int:
        return self.cg.get_chunk_layer(node_id)

    def get_chunk_layers(self, node_ids: np.ndarray) -> np.ndarray:
        return self.cg.get_chunk_layers(node_ids)

    def get_chunk_id(self, node_id) -> int:
        return self.cg.get_chunk_id(node_id)

    def get_segment_id(self, node_id) -> int:
        return self.cg.get_segment_id(node_id)

    def get_roots(self, node_ids: np.ndarray) -> np.ndarray:
        return self.cg.get_roots(node_ids)

    def mutate_rows(self, rows: dict, time_stamp=None) -> None:
        entries = [self.cg.client.mutate_row(rk, vd, time_stamp=time_stamp) for rk, vd in rows.items()]
        self.cg.client.write(entries)

    # -- Cached reads --

    def get_parents(self, node_ids: np.ndarray) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(node_ids, "get_parents")
        return np.array([self._cache.parents[int(n)] for n in node_ids], dtype=basetypes.NODE_ID)

    def get_children(self, node_ids: np.ndarray) -> dict:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(node_ids, "get_children")
        return {n: self._cache.children[int(n)] for n in node_ids}

    def get_acx(self, l2_ids: np.ndarray) -> dict:
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(l2_ids, "get_acx")
        return {n: self._cache.acx[int(n)] for n in l2_ids}

    def read_l2(self, l2_ids: np.ndarray) -> tuple:
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(l2_ids, "read_l2")
        return (
            {n: self._cache.children[int(n)] for n in l2_ids},
            {n: self._cache.acx[int(n)] for n in l2_ids},
        )

    def bulk_read_parent_child(self, node_ids: np.ndarray) -> tuple:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(node_ids, "bulk_pc")
        parents = np.array([self._cache.parents.get(int(n), 0) for n in node_ids], dtype=basetypes.NODE_ID)
        return parents, {n: self._cache.children[int(n)] for n in node_ids}

    # -- Stitch phases (called by stitch.py orchestrator) --

    def read_upfront(self, atomic_edges: np.ndarray, perf: dict) -> None:
        c = self._cache

        with timed(perf, "read_sv_parents"):
            svs = np.unique(atomic_edges)
            sv_parents = self.get_parents(svs)
            c.sv_to_l2 = dict(zip(svs, sv_parents))
            edge_layers = get_cross_chunk_edges_layer(self.meta, atomic_edges)

        l2_edges_list = []
        l2_cx_edges = defaultdict(lambda: defaultdict(list))
        for i, edge in enumerate(atomic_edges):
            layer = edge_layers[i]
            sv0, sv1 = int(edge[0]), int(edge[1])
            p0, p1 = c.sv_to_l2[sv0], c.sv_to_l2[sv1]
            if layer == 1:
                l2_edges_list.append([p0, p1])
            else:
                l2_cx_edges[p0][layer].append([sv0, sv1])
                l2_cx_edges[p1][layer].append([sv1, sv0])
                l2_edges_list.append([p0, p0])
                l2_edges_list.append([p1, p1])

        c.l2ids = np.unique(np.array(l2_edges_list, dtype=basetypes.NODE_ID)) if l2_edges_list else np.array([], dtype=basetypes.NODE_ID)
        c.l2_edges = np.array(l2_edges_list, dtype=basetypes.NODE_ID) if l2_edges_list else types.empty_2d
        c.l2_cx_edges = l2_cx_edges

        with timed(perf, "read_children_and_acx"):
            children_d, atomic_cx = self.read_l2(c.l2ids)
        c.children_d = children_d
        c.atomic_cx_stitch = atomic_cx

        c.l2ids = tree.filter_orphaned(self, c.l2ids, children_d)

        with timed(perf, "read_partner_chains"):
            partner_svs = set()
            for layer_d in atomic_cx.values():
                for edges in layer_d.values():
                    if len(edges) > 0:
                        partner_svs.update(edges[:, 1])
            for layer_d in l2_cx_edges.values():
                for edge_list in layer_d.values():
                    if edge_list:
                        partner_svs.update(np.array(edge_list, dtype=basetypes.NODE_ID)[:, 1])

            known_svs = set()
            for l2id in c.l2ids:
                known_svs.update(children_d.get(l2id, []))

            partner_sv_to_l2 = tree.resolve_partner_sv_parents(self, partner_svs - known_svs)
            unique_partner_l2s = np.array(list(set(partner_sv_to_l2.values())), dtype=basetypes.NODE_ID)

            all_chain_l2s = np.concatenate([c.l2ids, unique_partner_l2s]) if len(unique_partner_l2s) > 0 else c.l2ids
            all_chains = tree.get_all_parents_filtered(self, np.unique(all_chain_l2s))

            l2id_set = set(int(x) for x in c.l2ids)
            c.old_hierarchy = {k: v for k, v in all_chains.items() if k in l2id_set}

            partner_chains = {}
            for sv_int, l2_int in partner_sv_to_l2.items():
                chain = all_chains.get(l2_int, all_chains.get(np.uint64(l2_int), {}))
                partner_chains[sv_int] = {2: l2_int, **{int(l): int(p) for l, p in chain.items()}}

        resolver = {}
        for sv_int, l2_int in c.sv_to_l2.items():
            resolver[sv_int] = {2: l2_int}
        for sv, chain in partner_chains.items():
            resolver[int(sv)] = {int(l): int(p) for l, p in chain.items()}
        c.resolver = resolver

    def merge_l2(self, perf: dict) -> list:
        c = self._cache
        if c.l2_edges.size == 0:
            return []

        with timed(perf, "merge_l2_graph"):
            graph, _, _, graph_ids = flatgraph.build_gt_graph(c.l2_edges, make_directed=True)
            components = flatgraph.connected_components(graph)

        with timed(perf, "merge_l2_alloc"):
            chunk_counts = defaultdict(int)
            for cc in components:
                chunk_counts[self.get_chunk_id(graph_ids[cc][0])] += 1
            chunks = list(chunk_counts.keys())
            random.shuffle(chunks)
            chunk_ids_map = batch_create_ids(self.cg, {ch: chunk_counts[ch] for ch in chunks})

        new_l2_ids = []
        old_to_new = {}
        raw_cx_edges = {}
        l2_atomic_cx = {}

        with timed(perf, "merge_l2_loop"):
            for cc in components:
                old_ids = graph_ids[cc]
                new_id = chunk_ids_map[self.get_chunk_id(old_ids[0])].pop()
                new_l2_ids.append(new_id)
                for old_id in old_ids:
                    old_to_new[int(old_id)] = int(new_id)

                merged_children = np.concatenate([c.children_d[l2id] for l2id in old_ids]).astype(basetypes.NODE_ID)
                c.children_cache[new_id] = merged_children
                tree.update_parents_cache(c, merged_children, new_id)

                merged = defaultdict(list)
                for old_l2 in old_ids:
                    for layer, acx in c.atomic_cx_stitch.get(old_l2, {}).items():
                        if len(acx) > 0:
                            merged[layer].append(acx)
                    for layer, edge_list in c.l2_cx_edges.get(int(old_l2), {}).items():
                        if edge_list:
                            merged[layer].append(np.array(edge_list, dtype=basetypes.NODE_ID))

                raw_acx = {}
                for layer, arrs in merged.items():
                    raw_acx[layer] = np.unique(np.concatenate(arrs).astype(basetypes.NODE_ID), axis=0)
                l2_atomic_cx[int(new_id)] = raw_acx
                raw_cx_edges[int(new_id)] = topology.acx_to_cx(raw_acx, new_id)

        with timed(perf, "merge_l2_resolver"):
            for new_id in new_l2_ids:
                new_int = int(new_id)
                sv_parent = {2: new_int}
                for sv in c.children_cache.get(new_id, []):
                    sv_int = int(sv)
                    if sv_int not in c.resolver:
                        c.resolver[sv_int] = sv_parent

        c.old_to_new = {**self._cache.accumulated_replacements, **old_to_new}
        c.raw_cx_edges = raw_cx_edges
        c.l2_atomic_cx = l2_atomic_cx
        c.new_ids_d[2] = list(new_l2_ids)
        return new_l2_ids

    def discover_siblings(self, perf: dict) -> None:
        c = self._cache
        all_old_parents = set()
        for l2id in c.l2ids:
            for parent in c.old_hierarchy.get(l2id, {}).values():
                all_old_parents.add(int(parent))

        with timed(perf, "siblings_get_l2"):
            old_parents_arr = np.array(list(all_old_parents), dtype=basetypes.NODE_ID)
            all_l2_in_subtree = set()
            if len(old_parents_arr) > 0:
                parent_l2_map = batch_get_l2children(self, old_parents_arr)
                for l2set in parent_l2_map.values():
                    all_l2_in_subtree.update(l2set)

            known_l2 = set(int(x) for x in c.l2ids)
            all_siblings = np.array(list(all_l2_in_subtree - known_l2), dtype=basetypes.NODE_ID)

        if len(all_siblings) == 0:
            c.siblings_d = defaultdict(list)
            return

        with timed(perf, "siblings_restore"):
            known, unknown = c.split_known_siblings(all_siblings)
            perf["siblings_known"] = len(known)
            perf["siblings_unknown"] = len(unknown)

            if len(known) > 0:
                tree.restore_known_siblings(self, c, known)

        with timed(perf, "siblings_reads"):
            if len(unknown) > 0:
                sib_ch, sib_acx = self.read_l2(unknown)
                unknown = tree.filter_orphaned(self, unknown, sib_ch)
            else:
                sib_ch, sib_acx = {}, {}

        with timed(perf, "siblings_partner_chains"):
            # Resolve partner chains for ALL siblings (unknown + known).
            # Must run BEFORE compute_dirty_siblings — dirty check needs resolver
            # entries for partner SVs to detect if their L2 identity was replaced.
            # atomic_cx_stitch has known siblings' ACX, sib_acx has unknown siblings'.
            # known_svs filters already-resolved SVs, so clean siblings add negligible cost.
            if len(c.atomic_cx_stitch) > 0 or len(sib_acx) > 0:
                known_svs = set(c.resolver.keys())
                for l2id in c.l2ids:
                    known_svs.update(c.children_d.get(l2id, []))
                for l2id in unknown:
                    known_svs.update(sib_ch.get(l2id, []))
                for l2id in known:
                    entry = c.get_sibling(int(l2id))
                    if entry is not None:
                        known_svs.update(entry.resolver_entries.keys())
                topology.collect_and_resolve_partners(self, c.atomic_cx_stitch, known_svs, c.resolver)
                if len(sib_acx) > 0:
                    topology.collect_and_resolve_partners(self, sib_acx, known_svs, c.resolver)

        with timed(perf, "siblings_setup"):
            for l2id in unknown:
                l2id_int = int(l2id)
                c.children_d[l2id] = sib_ch.get(l2id, np.array([], dtype=basetypes.NODE_ID))
                c.atomic_cx_stitch[l2id] = sib_acx.get(l2id, {})
                sv_parent = {2: l2id_int}
                for sv in sib_ch.get(l2id, []):
                    c.resolver[int(sv)] = sv_parent
                c.raw_cx_edges[l2id_int] = topology.acx_to_cx(sib_acx.get(l2id, {}), l2id)

            all_siblings = (
                np.concatenate([known, unknown]).astype(basetypes.NODE_ID)
                if len(known) > 0 and len(unknown) > 0
                else (known if len(known) > 0 else unknown)
            )
            c.siblings_d = defaultdict(list)
            c.siblings_d[2] = [int(x) for x in all_siblings]
            c.sibling_ids = set(int(x) for x in all_siblings)

    def compute_dirty_siblings(self) -> None:
        self._cache.compute_dirty_siblings()

    def build_hierarchy(self, perf: dict) -> tuple:
        c = self._cache
        child_to_parent = {}
        deferred_roots = []
        layer_perf = {}
        get_layer = lambda nid: self.get_chunk_layer(np.uint64(nid))
        clean_sibs = c.sibling_ids - c.dirty_siblings

        for layer in range(2, self.meta.layer_count):
            new_nodes = c.new_ids_d[layer]
            sib_nodes = c.siblings_d.get(layer, [])
            all_nodes = list(new_nodes) + list(sib_nodes)
            if not all_nodes:
                continue
            c.new_node_ids.update(new_nodes)
            lp = {}

            with timed(lp, "resolve_cx"):
                if layer == 2 and clean_sibs:
                    resolve_nodes = [n for n in all_nodes if int(n) not in clean_sibs]
                    cx_resolved = topology.resolve_cx_at_layer(resolve_nodes, layer, c, child_to_parent, get_layer)
                    cached_cx = []
                    for n in all_nodes:
                        if int(n) in clean_sibs:
                            entry = c.get_sibling(int(n))
                            if entry and layer in entry.written_cx:
                                cached_cx.append(entry.written_cx[layer])
                    if cached_cx:
                        all_cx = (
                            np.concatenate([cx_resolved] + cached_cx).astype(basetypes.NODE_ID)
                            if len(cx_resolved) > 0
                            else np.concatenate(cached_cx).astype(basetypes.NODE_ID)
                        )
                    else:
                        all_cx = cx_resolved
                else:
                    all_cx = topology.resolve_cx_at_layer(all_nodes, layer, c, child_to_parent, get_layer)
                    cx_resolved = all_cx

            with timed(lp, "store_cx"):
                if layer == 2 and clean_sibs:
                    topology.store_cx_from_resolved(c, cx_resolved, layer)
                else:
                    topology.store_cx_from_resolved(c, all_cx, layer)

            with timed(lp, "cc"):
                nodes_arr = np.array(all_nodes, dtype=basetypes.NODE_ID)
                self_edges = np.vstack([nodes_arr, nodes_arr]).T
                all_edges = np.concatenate([all_cx, self_edges]).astype(basetypes.NODE_ID)
                graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
                ccs = flatgraph.connected_components(graph)

            with timed(lp, "create_parents"):
                self._create_parents(layer, ccs, graph_ids, child_to_parent, deferred_roots)
            layer_perf[layer] = lp

        with timed(perf, "phase3_alloc_resolve"):
            self._allocate_roots(deferred_roots)
            roots = np.array(c.new_ids_d[self.meta.layer_count], dtype=basetypes.NODE_ID)
            c.new_node_ids.update(roots.tolist())
            topology.resolve_remaining_cx(c, self, child_to_parent)
        return roots, layer_perf

    def build_rows(self) -> dict:
        c = self._cache
        rows = {}

        for nid in c.new_node_ids:
            children = c.children_cache.get(nid)
            if children is None:
                continue
            rk = serializers.serialize_uint64(nid)
            vd = rows.setdefault(rk, {})
            vd[attributes.Hierarchy.Child] = children
            for layer, cx in c.cx_cache.get(nid, {}).items():
                vd[attributes.Connectivity.CrossChunkEdge[layer]] = cx
            for layer, acx in c.l2_atomic_cx.get(int(nid), {}).items():
                vd[attributes.Connectivity.AtomicCrossChunkEdge[layer]] = acx
            for child in children:
                crk = serializers.serialize_uint64(child)
                rows.setdefault(crk, {})[attributes.Hierarchy.Parent] = nid

        for nid in c.sibling_ids:
            if nid not in c.cx_cache:
                continue
            rk = serializers.serialize_uint64(np.uint64(nid))
            vd = rows.setdefault(rk, {})
            for layer, cx_edges in c.cx_cache[nid].items():
                vd[attributes.Connectivity.CrossChunkEdge[layer]] = cx_edges

        return rows

    # -- Pool support --

    def preloaded(self) -> tuple:
        return self._cache.preloaded()

    def incremental_state(self) -> tuple:
        return self._cache.incremental_state()

    def wave_snapshot(self) -> tuple:
        c = self._cache
        c.flush_created()
        return (
            c.local_snapshot(),
            c.inc_snapshot_from(
                old_to_new=c.old_to_new, new_node_ids=c.new_node_ids,
                sibling_ids=c.sibling_ids, raw_cx_edges=c.raw_cx_edges,
                children=c.children_d,
            ),
        )

    def merge_wave_results(self, snapshots: list[tuple]) -> None:
        for local_snap, inc_snap in snapshots:
            self._cache.merge_reader(local_snap)
            self._cache.merge_inc(inc_snap)

    def get_all_new_ids(self) -> dict:
        return self._cache.new_ids_d

    def stats(self) -> str:
        return self._cache.stats()

    # -- Private --

    def _create_parents(
        self, layer: int, ccs: list, graph_ids: np.ndarray,
        child_to_parent: dict, deferred_roots: list,
    ) -> None:
        c = self._cache
        size_map = defaultdict(int)
        cc_info = {}
        root_layer = self.meta.layer_count

        for i, cc_idx in enumerate(ccs):
            cc_ids = graph_ids[cc_idx]
            parent_layer = layer + 1
            if len(cc_ids) == 1:
                cx_d = c.raw_cx_edges.get(int(cc_ids[0]), {})
                for l in range(layer + 1, root_layer):
                    if l in cx_d and len(cx_d[l]) > 0:
                        parent_layer = l
                        break
                else:
                    parent_layer = root_layer
            cc_info[i] = (parent_layer, cc_ids)
            if parent_layer < root_layer:
                chunk_id = int(chunk_hierarchy.get_parent_chunk_id(self.meta, cc_ids[0], parent_layer))
                size_map[chunk_id] += 1
                cc_info[i] = (parent_layer, cc_ids, chunk_id)

        chunk_new_ids = batch_create_ids(self.cg, size_map) if size_map else {}

        for i, cc_idx in enumerate(ccs):
            cc_ids = cc_info[i][1]
            parent_layer = cc_info[i][0]
            if parent_layer == root_layer:
                deferred_roots.append(cc_ids)
                continue

            parent = chunk_new_ids[cc_info[i][2]].pop()
            c.new_ids_d[parent_layer].append(parent)

            parent_cx_raw = defaultdict(set)
            for child in cc_ids:
                child_cx = c.raw_cx_edges.get(int(child), {})
                for l, edges in child_cx.items():
                    if l >= parent_layer and len(edges) > 0:
                        for e in edges:
                            parent_cx_raw[l].add((int(parent), int(e[1])))

            merged_raw = {}
            for l, pairs in parent_cx_raw.items():
                if pairs:
                    merged_raw[l] = np.array(list(pairs), dtype=basetypes.NODE_ID)
            c.raw_cx_edges[int(parent)] = merged_raw

            for child in cc_ids:
                child_to_parent[int(child)] = int(parent)

            c.children_cache[parent] = cc_ids
            tree.update_parents_cache(c, cc_ids, parent)

    def _allocate_roots(self, deferred_roots: list) -> None:
        if not deferred_roots:
            return
        c = self._cache
        root_layer = self.meta.layer_count
        root_chunk = int(chunk_hierarchy.get_parent_chunk_id(self.meta, deferred_roots[0][0], root_layer))
        root_ids = batch_create_ids(
            self.cg, {root_chunk: len(deferred_roots)}, root_chunks={root_chunk},
        )[root_chunk]

        for i, cc_ids in enumerate(deferred_roots):
            rid = root_ids[i]
            c.new_ids_d[root_layer].append(rid)
            c.children_cache[rid] = cc_ids
            tree.update_parents_cache(c, cc_ids, rid)

    def _ensure_cached(self, node_ids: np.ndarray, label: str) -> None:
        uncached = node_ids[~self._cache.has_batch(node_ids)]
        if len(uncached) == 0:
            return
        layers = self.cg.get_chunk_layers(uncached)
        svs = uncached[layers <= 1]
        l2s = uncached[layers == 2]
        higher = uncached[layers > 2]
        n = len(node_ids)
        if len(svs) > 0:
            self._read_and_cache(svs, [attributes.Hierarchy.Parent], f"{label}_sv", n)
        if len(l2s) > 0:
            acx_props = [
                attributes.Connectivity.AtomicCrossChunkEdge[l]
                for l in range(2, max(3, self.cg.meta.layer_count))
            ]
            self._read_and_cache(
                l2s, [attributes.Hierarchy.Parent, attributes.Hierarchy.Child] + acx_props,
                f"{label}_l2", n, with_acx=True,
            )
        if len(higher) > 0:
            self._read_and_cache(higher, [attributes.Hierarchy.Parent, attributes.Hierarchy.Child], label, n)

    def _read_and_cache(
        self, node_ids: np.ndarray, props: list, label: str,
        n_total: int, with_acx: bool = False,
    ) -> None:
        t0 = time.time()
        raw = self.cg.client.read_nodes(node_ids=node_ids, properties=props)
        self.rpc_log.append((label, n_total, len(node_ids), time.time() - t0))
        for n in node_ids:
            n_int = int(n)
            data = raw.get(n, {})
            parent_cells = data.get(attributes.Hierarchy.Parent, [])
            if parent_cells:
                self._cache.put_parent(n_int, int(parent_cells[0].value))
            child_cells = data.get(attributes.Hierarchy.Child, [])
            if child_cells:
                self._cache.put_children(n_int, child_cells[0].value)
            if with_acx:
                acx_d = {}
                for layer in range(2, max(3, self.cg.meta.layer_count)):
                    prop = attributes.Connectivity.AtomicCrossChunkEdge[layer]
                    cells = data.get(prop, [])
                    if cells:
                        acx_d[layer] = cells[0].value.copy()
                self._cache.put_acx(n_int, acx_d)
