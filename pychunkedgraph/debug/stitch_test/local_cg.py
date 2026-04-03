# pylint: disable=no-member
"""
LocalChunkedGraph: composite cached ChunkedGraph.

Owns WaveCache. Delegates to stateless operation modules (hierarchy, topology).
External code calls lcg methods — never touches cache directly.
"""

import datetime
import logging
import os
import pickle
import random
import time
from collections import defaultdict

import fastremap
import numpy as np
from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes, types
from pychunkedgraph.graph.chunks import hierarchy as chunk_hierarchy
from pychunkedgraph.graph.edges.utils import get_cross_chunk_edges_layer
from pychunkedgraph.graph.utils import flatgraph
from kvdbclient import serializers

from . import resolver
from . import tree
from .id_allocator import batch_create as batch_create_ids
from .utils import RpcEntry, batch_get_l2children, stitch_sanity_check, timed
from .wave_cache import WaveCache

log = logging.getLogger(__name__)


_worker_meta = None
_worker_cv_info = None
SANITY_CHECK = False




def _setup_env() -> None:
    os.environ.setdefault("BIGTABLE_PROJECT", "zetta-proofreading")
    os.environ.setdefault("BIGTABLE_INSTANCE", "pychunkedgraph")


class LocalChunkedGraph:

    MARKER_PREFIX = b"stitch_marker:"

    def __init__(self, graph_id: str, preloaded: tuple = None, incremental: tuple = None, meta=None) -> None:
        _setup_env()
        self.cg = ChunkedGraph(graph_id=graph_id, meta=meta) if meta else ChunkedGraph(graph_id=graph_id)
        self.cg.client._max_row_key_count = 16_000
        self._cache = WaveCache(
            lambda ids: self._ensure_cached(ids, "cache_miss"),
            preloaded, incremental=incremental,
        )
        self.rpc_log: list[RpcEntry] = []
        self._read_row_keys: set[int] = set()
        self.stitch_timestamp: datetime.datetime | None = None

    def acquire_stitch_timestamp(self, edge_file: str) -> datetime.datetime:
        """Write or read marker row for this edge file. Returns stitch timestamp."""
        marker_key = self.MARKER_PREFIX + edge_file.encode()
        row = self.cg.client._table.read_row(marker_key)
        if row is not None:
            cell = row.cells[0]
            ts = datetime.datetime.fromtimestamp(cell.timestamp_micros / 1e6, tz=datetime.timezone.utc)
            self.stitch_timestamp = ts
            return ts
        from google.cloud.bigtable.data.mutations import SetCell
        mutation = SetCell(family="0", qualifier=b"status", new_value=b"started")
        self.cg.client._table.mutate_row(marker_key, mutation)
        row = self.cg.client._table.read_row(marker_key)
        cell = row.cells[0]
        ts = datetime.datetime.fromtimestamp(cell.timestamp_micros / 1e6, tz=datetime.timezone.utc)
        self.stitch_timestamp = ts
        return ts

    def release_stitch_timestamp(self, edge_file: str) -> None:
        """Delete marker row after successful stitch."""
        marker_key = self.MARKER_PREFIX + edge_file.encode()
        from google.cloud.bigtable.data.mutations import DeleteAllFromRow
        self.cg.client._table.mutate_row(marker_key, DeleteAllFromRow())
        self.stitch_timestamp = None

    def prepare_pool_init(self) -> tuple:
        meta_bytes = pickle.dumps(self.cg.meta)
        ws_cv = getattr(self.cg.meta, '_ws_cv', None)
        cv_info = ws_cv.info if ws_cv is not None else CloudVolume(self.cg.meta.data_source.WATERSHED, mip=0).info
        return meta_bytes, cv_info

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

    def sanity_check(self, roots: np.ndarray, atomic_edges: np.ndarray) -> None:
        if SANITY_CHECK:
            stitch_sanity_check(self, roots, atomic_edges)

    # -- Lifecycle --

    def begin_stitch(self) -> None:
        self._cache.begin_stitch()
        self.rpc_log = []
        self._read_row_keys = set()

    def end_stitch(self) -> None:
        c = self._cache
        c.save_wave_state(
            old_to_new=c.old_to_new, new_node_ids=c.new_node_ids,
            sibling_ids=c.sibling_ids, unresolved_acx=c.unresolved_acx,
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
        return self.cg.get_roots(node_ids, time_stamp=self.stitch_timestamp)

    def mutate_rows(self, rows: dict, time_stamp=None) -> None:
        ts = time_stamp or self.stitch_timestamp
        entries = [self.cg.client.mutate_row(rk, vd, time_stamp=ts) for rk, vd in rows.items()]
        self.cg.client.write(entries)

    # -- Cached reads (delegate to cache batch API) --

    def get_parents(self, node_ids: np.ndarray) -> np.ndarray:
        return self._cache.get_parents(np.asarray(node_ids, dtype=basetypes.NODE_ID))

    def get_children(self, node_ids: np.ndarray) -> dict:
        return self._cache.get_children_batch(np.asarray(node_ids, dtype=basetypes.NODE_ID))

    def get_acx(self, l2_ids: np.ndarray) -> dict:
        return self._cache.get_acx_batch(np.asarray(l2_ids, dtype=basetypes.NODE_ID))

    def read_l2(self, l2_ids: np.ndarray) -> tuple:
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        return self._cache.get_children_batch(l2_ids), self._cache.get_acx_batch(l2_ids)

    def bulk_read_parent_child(self, node_ids: np.ndarray) -> tuple:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        return self._cache.get_parents(node_ids), self._cache.get_children_batch(node_ids)

    # -- Stitch phases (called by stitch.py orchestrator) --

    def read_upfront(self, atomic_edges: np.ndarray, perf: dict) -> None:
        c = self._cache

        with timed(perf, "read_sv_parents"):
            svs = np.unique(atomic_edges)
            sv_parents = self.get_parents(svs)
            sv_to_l2 = dict(zip(svs, sv_parents))
            edge_layers = get_cross_chunk_edges_layer(self.meta, atomic_edges)

        l2_edges_list = []
        l2_cx_edges = defaultdict(lambda: defaultdict(list))
        for i, edge in enumerate(atomic_edges):
            layer = edge_layers[i]
            sv0, sv1 = int(edge[0]), int(edge[1])
            p0, p1 = sv_to_l2[sv0], sv_to_l2[sv1]
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
        for l2id, ch in children_d.items():
            c.put_children(int(l2id), ch)
        for l2id, acx in atomic_cx.items():
            c.put_acx(int(l2id), acx)

        with timed(perf, "read_partner_chains"):
            partner_svs = set()
            for layer_d in atomic_cx.values():
                for edges in layer_d.values():
                    partner_svs.update(edges[:, 1])
            for layer_d in l2_cx_edges.values():
                for edge_list in layer_d.values():
                    partner_svs.update(np.array(edge_list, dtype=basetypes.NODE_ID)[:, 1])

            partner_sv_arr = np.array(list(partner_svs), dtype=basetypes.NODE_ID)
            partner_parents = c.get_parents(partner_sv_arr)
            unique_partner_l2s = np.unique(partner_parents[partner_parents != 0])

            all_chain_l2s = np.concatenate([c.l2ids, unique_partner_l2s]) if len(unique_partner_l2s) > 0 else c.l2ids
            all_chains = tree.get_all_parents_filtered(self, np.unique(all_chain_l2s))

            l2id_set = set(int(x) for x in c.l2ids)
            self._store_old_hierarchy({k: v for k, v in all_chains.items() if k in l2id_set})


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
        unresolved_acx = {}

        with timed(perf, "merge_l2_loop"):
            all_old_l2s = np.array(
                [int(nid) for cc in components for nid in graph_ids[cc]],
                dtype=basetypes.NODE_ID,
            )
            all_ch = c.get_children_batch(all_old_l2s)
            all_acx = c.get_acx_batch(all_old_l2s)
            all_cx = c.get_cx_batch(all_old_l2s)

            cx_merge_stats = defaultdict(int)
            for cc in components:
                old_ids = graph_ids[cc]
                new_id = chunk_ids_map[self.get_chunk_id(old_ids[0])].pop()
                new_l2_ids.append(new_id)
                for old_id in old_ids:
                    old_to_new[int(old_id)] = int(new_id)

                merged_children = np.concatenate([all_ch[int(l2id)] for l2id in old_ids]).astype(basetypes.NODE_ID)
                c.put_children(int(new_id), merged_children)
                tree.update_parents_cache(c, merged_children, new_id)

                merged_acx = defaultdict(list)
                cx_merged = defaultdict(list)
                for old_l2 in old_ids:
                    for layer, acx_edges in all_acx.get(int(old_l2), {}).items():
                        merged_acx[layer].append(acx_edges)
                    for layer, edge_list in c.l2_cx_edges.get(int(old_l2), {}).items():
                        merged_acx[layer].append(np.array(edge_list, dtype=basetypes.NODE_ID))
                    for layer, edges in all_cx.get(int(old_l2), {}).items():
                        if len(edges) > 0:
                            cx_merged[layer].append(edges)

                raw_acx = {}
                for layer, arrs in merged_acx.items():
                    raw_acx[layer] = np.unique(np.concatenate(arrs).astype(basetypes.NODE_ID), axis=0)
                c.put_acx(int(new_id), raw_acx)
                unresolved_acx[int(new_id)] = resolver.acx_to_cx(raw_acx, new_id)

                if cx_merged:
                    merged_cx = {}
                    for layer, arrs in cx_merged.items():
                        combined = np.concatenate(arrs).astype(basetypes.NODE_ID)
                        combined = fastremap.remap(combined, old_to_new, preserve_missing_labels=True)
                        merged_cx[layer] = np.unique(combined, axis=0)
                        cx_merge_stats[layer] += 1
                    c.put_cx(int(new_id), merged_cx)
            perf["merge_l2_cx_layers"] = dict(cx_merge_stats)

        c.old_to_new = {**self._cache.accumulated_replacements, **old_to_new}
        for old_l2, new_l2 in old_to_new.items():
            c.new_to_old.setdefault(new_l2, set()).add(old_l2)
        c.unresolved_acx = unresolved_acx
        c.new_ids_d[2] = list(new_l2_ids)
        return new_l2_ids

    def discover_siblings(self, perf: dict) -> None:
        c = self._cache
        old_parents = self.get_parents(c.l2ids)
        all_old_parents = set(int(x) for x in np.unique(old_parents))

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
            else:
                sib_ch, sib_acx = {}, {}

        with timed(perf, "siblings_partner_chains"):
            # Resolve partner SVs so dirty check can detect replaced L2 identities.
            # Unknown siblings first (few), then only known siblings with missing partners.
            all_acx = {**sib_acx}
            if len(known) > 0:
                known_acx = c.get_acx_batch(np.array(list(known), dtype=basetypes.NODE_ID))
                all_acx.update(known_acx)
            if all_acx:
                resolver.ensure_partners_cached(c, all_acx)

        with timed(perf, "siblings_setup"):
            for l2id in unknown:
                l2id_int = int(l2id)
                c.put_children(l2id_int, sib_ch.get(l2id, np.array([], dtype=basetypes.NODE_ID)))
                c.put_acx(l2id_int, sib_acx.get(l2id, {}))
                c.unresolved_acx[l2id_int] = resolver.acx_to_cx(sib_acx.get(l2id, {}), l2id)

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
        deferred_roots = []
        layer_perf = {}
        get_layer = lambda nid: self.get_chunk_layer(np.uint64(nid))

        for layer in range(2, self.meta.layer_count):
            new_nodes = c.new_ids_d[layer]
            sib_nodes = c.siblings_d.get(layer, [])
            all_nodes = list(new_nodes) + list(sib_nodes)
            if not all_nodes:
                continue
            if SANITY_CHECK and layer > 2:
                for n in all_nodes:
                    nl = self.get_chunk_layer(np.uint64(n))
                    assert nl == layer, (
                        f"node {n} is L{nl} but in all_nodes at layer={layer}, "
                        f"in_new_ids={n in set(new_nodes)}, "
                        f"in_siblings={n in set(sib_nodes)}"
                    )
            c.new_node_ids.update(new_nodes)
            lp = {}

            if layer > 2:
                with timed(lp, "prefetch_parents"):
                    all_svs = []
                    for node in all_nodes:
                        edges = c.unresolved_acx.get(int(node), {}).get(layer)
                        if edges is not None and len(edges) > 0:
                            all_svs.append(edges[:, 1])
                    if all_svs:
                        svs = np.unique(np.concatenate(all_svs))
                        c.get_parents(svs)

            with timed(lp, "resolve_cx"):
                all_cx = resolver.resolve_cx_at_layer(all_nodes, layer, c, get_layer)

            with timed(lp, "store_cx"):
                resolver.store_cx_from_resolved(c, all_cx, layer)

            with timed(lp, "cc"):
                nodes_arr = np.array(all_nodes, dtype=basetypes.NODE_ID)
                self_edges = np.vstack([nodes_arr, nodes_arr]).T
                all_edges = np.concatenate([all_cx, self_edges]).astype(basetypes.NODE_ID)
                graph, _, _, graph_ids = flatgraph.build_gt_graph(all_edges, make_directed=True)
                ccs = flatgraph.connected_components(graph)

            counts_before = {pl: len(c.new_ids_d[pl]) for pl in range(layer + 1, self.meta.layer_count)}
            with timed(lp, "create_parents"):
                self._create_parents(layer, ccs, graph_ids, deferred_roots)

            with timed(lp, "update_counterparts"):
                self._update_counterpart_cx(layer)

            with timed(lp, "discover_siblings"):
                for pl in range(layer + 1, self.meta.layer_count):
                    if len(c.new_ids_d[pl]) > counts_before.get(pl, 0):
                        self._discover_layer_siblings(pl, lp)
            layer_perf[layer] = lp

        with timed(perf, "phase3_alloc_resolve"):
            self._allocate_roots(deferred_roots)
            roots = np.array(c.new_ids_d[self.meta.layer_count], dtype=basetypes.NODE_ID)
            c.new_node_ids.update(roots.tolist())
            resolver.resolve_remaining_cx(c, self)
        return roots, layer_perf

    def build_rows(self) -> dict:
        c = self._cache
        rows = {}

        new_arr = np.array(list(c.new_node_ids), dtype=basetypes.NODE_ID)
        all_ch = c.get_children_batch(new_arr)
        all_cx = c.get_cx_batch(new_arr)
        all_acx = c.get_acx_batch(new_arr)

        for nid in c.new_node_ids:
            assert int(nid) != 0, "Zero node ID in new_node_ids"
            children = all_ch.get(int(nid))
            if children is None or len(children) == 0:
                continue
            rk = serializers.serialize_uint64(nid)
            vd = rows.setdefault(rk, {})
            vd[attributes.Hierarchy.Child] = children
            for layer, cx in all_cx.get(int(nid), {}).items():
                vd[attributes.Connectivity.CrossChunkEdge[layer]] = cx
            for layer, acx in all_acx.get(int(nid), {}).items():
                vd[attributes.Connectivity.AtomicCrossChunkEdge[layer]] = acx
            for child in children:
                assert int(child) != 0, f"Zero child ID for parent {nid}"
                crk = serializers.serialize_uint64(child)
                rows.setdefault(crk, {})[attributes.Hierarchy.Parent] = nid

        cp_arr = np.array(list(c.counterpart_ids), dtype=basetypes.NODE_ID)
        cp_cx = c.get_cx_batch(cp_arr) if len(cp_arr) > 0 else {}
        for nid in c.counterpart_ids:
            assert int(nid) != 0, "Zero counterpart ID"
            rk = serializers.serialize_uint64(np.uint64(nid))
            vd = rows.setdefault(rk, {})
            for layer, cx_edges in cp_cx.get(int(nid), {}).items():
                vd[attributes.Connectivity.CrossChunkEdge[layer]] = cx_edges

        return rows

    # -- Pool support --

    def preloaded(self) -> tuple:
        return self._cache.preloaded()

    def incremental_state(self) -> tuple:
        return self._cache.incremental_state()

    def wave_snapshot(self) -> tuple:
        c = self._cache
        return (
            c.local_snapshot(),
            c.inc_snapshot_from(
                old_to_new=c.old_to_new, new_node_ids=c.new_node_ids,
                sibling_ids=c.sibling_ids, unresolved_acx=c.unresolved_acx,
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

    def _populate_old_hierarchy(self, node_ids: np.ndarray) -> None:
        chains = tree.get_all_parents_filtered(self, node_ids)
        self._store_old_hierarchy(chains)

    def _store_old_hierarchy(self, chains: dict) -> None:
        c = self._cache
        for nid, chain in chains.items():
            if chain:
                c.old_hierarchy[int(nid)] = chain
                for parent_id in chain.values():
                    c.old_hierarchy[int(parent_id)] = chain

    def _build_raw_cx_from_children(self, node_ids: np.ndarray, parent_layer: int) -> None:
        c = self._cache
        node_ch = c.get_children_batch(node_ids)
        # Single pass: build child→node mapping and collect all children
        child_to_node = {}
        all_children = []
        for nid in node_ids:
            ch = node_ch.get(int(nid))
            if ch is not None and len(ch) > 0:
                nid_int = int(nid)
                for child in ch:
                    child_to_node[int(child)] = nid_int
                all_children.extend(int(x) for x in ch)
        if not all_children:
            return
        all_children_arr = np.array(all_children, dtype=basetypes.NODE_ID)
        all_child_acx = c.get_acx_batch(all_children_arr)
        # Build unresolved_acx per node, collecting edges into lists
        for child_int, node_int in child_to_node.items():
            child_acx = all_child_acx.get(child_int, {})
            for layer, edges in child_acx.items():
                if layer >= parent_layer and len(edges) > 0:
                    working = edges.copy()
                    working[:, 0] = node_int
                    node_acx = c.unresolved_acx.setdefault(node_int, {})
                    existing = node_acx.get(layer)
                    if existing is None:
                        node_acx[layer] = [working]
                    elif isinstance(existing, list):
                        existing.append(working)
                    else:
                        node_acx[layer] = [existing, working]
        # Deduplicate: concatenate lists into unique arrays
        for node_int in (int(n) for n in node_ids):
            cx_d = c.unresolved_acx.get(node_int, {})
            for layer in list(cx_d.keys()):
                val = cx_d[layer]
                if isinstance(val, list):
                    cx_d[layer] = np.unique(np.concatenate(val).astype(basetypes.NODE_ID), axis=0)

    def _update_counterpart_cx(self, layer: int) -> None:
        c = self._cache
        new_nodes = c.new_ids_d.get(layer, [])
        if not new_nodes:
            return

        get_layer = lambda nid: self.get_chunk_layer(np.uint64(nid))
        in_scope = set(c.new_node_ids)

        all_svs = set()
        for nid in new_nodes:
            raw = c.unresolved_acx.get(int(nid), {})
            for lyr in range(layer, self.meta.layer_count):
                edges = raw.get(lyr, types.empty_2d)
                all_svs.update(int(sv) for sv in edges[:, 1])

        if not all_svs:
            return
        sv_arr = np.array(list(all_svs), dtype=basetypes.NODE_ID)
        sv_map = resolver.resolve_svs_to_layer(sv_arr, layer, c, get_layer)

        counterpart_layers = {}
        for nid in new_nodes:
            raw = c.unresolved_acx.get(int(nid), {})
            for lyr in range(layer, self.meta.layer_count):
                edges = raw.get(lyr, types.empty_2d)
                for sv in edges[:, 1]:
                    resolved = sv_map.get(int(sv), int(sv))
                    if resolved not in in_scope:
                        counterpart_layers.setdefault(resolved, lyr)
        if not counterpart_layers:
            return

        cp_arr = np.array(list(counterpart_layers.keys()), dtype=basetypes.NODE_ID)
        cp_cx_batch = c.get_cx_batch(cp_arr)

        node_map = {}
        for nid in (*c.new_node_ids, *new_nodes):
            for old_id in c.new_to_old.get(int(nid), set()):
                node_map[old_id] = int(nid)

        for cp_int in counterpart_layers:
            cp_cx = cp_cx_batch.get(cp_int, {})
            remapped = {}
            for lyr in range(layer, self.meta.layer_count):
                edges = cp_cx.get(lyr, types.empty_2d)
                if edges.size == 0:
                    continue
                new_edges = fastremap.remap(edges.copy(), node_map, preserve_missing_labels=True)
                remapped[lyr] = np.unique(new_edges, axis=0)
            if remapped:
                c.put_cx(cp_int, remapped)
                c.counterpart_ids.add(cp_int)

    def _discover_layer_siblings(self, parent_layer: int, perf: dict) -> None:
        c = self._cache
        new_parents = c.new_ids_d.get(parent_layer, [])
        if not new_parents:
            return

        old_ids = set()
        for p in new_parents:
            old_ids.update(c.new_to_old.get(int(p), set()))
        if not old_ids:
            perf[f"_dls_l{parent_layer}_empty_old_ids"] = len(new_parents)
            return

        old_ids_arr = np.array(list(old_ids), dtype=basetypes.NODE_ID)
        old_layers = self.get_chunk_layers(old_ids_arr)
        perf[f"_dls_l{parent_layer}_old_ids"] = len(old_ids)
        perf[f"_dls_l{parent_layer}_old_layers"] = dict(zip(*np.unique(old_layers, return_counts=True)))

        old_parent_ids = self.get_parents(old_ids_arr)
        unique_parents = np.unique(old_parent_ids)
        parent_layers = self.get_chunk_layers(unique_parents)
        perf[f"_dls_l{parent_layer}_parents"] = dict(zip(*np.unique(parent_layers, return_counts=True)))

        _, ch_d = self.bulk_read_parent_child(unique_parents)
        all_children = set()
        for ch in ch_d.values():
            all_children.update(int(x) for x in ch)
        perf[f"_dls_l{parent_layer}_children"] = len(all_children)

        new_parent_set = set(int(x) for x in new_parents)
        siblings = all_children - old_ids - new_parent_set
        siblings = {c.old_to_new.get(s, s) for s in siblings}
        perf[f"_dls_l{parent_layer}_siblings_pre_filter"] = len(siblings)

        sibs_arr = np.array(list(siblings), dtype=basetypes.NODE_ID)
        if len(sibs_arr) > 0:
            layers = self.get_chunk_layers(sibs_arr)
            layer_counts = dict(zip(*np.unique(layers, return_counts=True)))
            perf[f"_dls_l{parent_layer}_sibling_layers"] = layer_counts
            sibs_arr = sibs_arr[layers == parent_layer]

        perf[f"_dls_l{parent_layer}_after_filter"] = len(sibs_arr)
        if len(sibs_arr) == 0:
            return

        self._ensure_cached(sibs_arr, f"siblings_l{parent_layer}")
        self._populate_old_hierarchy(sibs_arr)
        self._build_raw_cx_from_children(sibs_arr, parent_layer)

        new_acx = {}
        for sib in sibs_arr:
            acx = c.unresolved_acx.get(int(sib), {})
            if acx:
                new_acx[int(sib)] = acx
        if new_acx:
            resolver.ensure_partners_cached(c, new_acx)

        c.siblings_d.setdefault(parent_layer, []).extend(int(x) for x in sibs_arr)
        c.sibling_ids.update(int(x) for x in sibs_arr)
        perf[f"siblings_l{parent_layer}"] = len(sibs_arr)

    def _update_lineage(self, parent: int, cc_ids, new_at_layer: set, parent_layer: int) -> None:
        c = self._cache
        old_ids_at_layer = set()
        for child_int in (int(x) for x in cc_ids if int(x) in new_at_layer):
            for old_id in c.new_to_old.get(child_int, set()):
                entry = c.old_hierarchy.get(old_id)
                old_ids_at_layer.add(entry.get(parent_layer, old_id) if entry else old_id)
        if old_ids_at_layer:
            c.new_to_old[parent] = old_ids_at_layer

    # -- Private --

    def _create_parents(
        self, layer: int, ccs: list, graph_ids: np.ndarray,
        deferred_roots: list,
    ) -> None:
        c = self._cache
        size_map = defaultdict(int)
        cc_info = {}
        root_layer = self.meta.layer_count

        for i, cc_idx in enumerate(ccs):
            cc_ids = graph_ids[cc_idx]
            parent_layer = layer + 1
            if len(cc_ids) == 1:
                cx_d = c.unresolved_acx.get(int(cc_ids[0]), {})
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
        new_at_layer = set(int(x) for x in c.new_ids_d[layer])

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
                child_cx = c.unresolved_acx.get(int(child), {})
                for l, edges in child_cx.items():
                    if l >= parent_layer and len(edges) > 0:
                        for e in edges:
                            parent_cx_raw[l].add((int(parent), int(e[1])))

            merged_raw = {}
            for l, pairs in parent_cx_raw.items():
                if pairs:
                    merged_raw[l] = np.array(list(pairs), dtype=basetypes.NODE_ID)
            c.unresolved_acx[int(parent)] = merged_raw

            if SANITY_CHECK:
                for child in cc_ids:
                    existing_parent = int(c.get_parents(np.array([int(child)], dtype=basetypes.NODE_ID))[0])
                    assert existing_parent == 0 or existing_parent not in c.new_node_ids, (
                        f"Child {int(child)} (L{self.get_chunk_layer(np.uint64(child))}) "
                        f"already has new parent {existing_parent} "
                        f"(L{self.get_chunk_layer(np.uint64(existing_parent))}), "
                        f"being reassigned to {int(parent)} "
                        f"(L{self.get_chunk_layer(np.uint64(parent))}) at layer={layer}, "
                        f"parent_layer={parent_layer}, cc_size={len(cc_ids)}, "
                        f"acx_layers={list(c.unresolved_acx.get(int(child), {}).keys())}, "
                        f"old_hierarchy={c.old_hierarchy.get(int(child), 'NONE')}"
                    )
            tree.update_parents_cache(c, cc_ids, parent)
            c.put_children(int(parent), cc_ids)
            self._update_lineage(int(parent), cc_ids, new_at_layer, parent_layer)

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
            c.put_children(int(rid), cc_ids)
            tree.update_parents_cache(c, cc_ids, rid)

    def _ensure_cached(self, node_ids, label: str) -> None:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        uncached = node_ids[~self._cache.has_batch(node_ids)]
        if len(uncached) == 0:
            return
        lc = self.cg.meta.layer_count
        props = (
            [attributes.Hierarchy.Parent, attributes.Hierarchy.Child]
            + [attributes.Connectivity.AtomicCrossChunkEdge[l] for l in range(2, lc)]
            + [attributes.Connectivity.CrossChunkEdge[l] for l in range(2, lc)]
        )
        self._read_and_cache(uncached, props, label, len(node_ids))

    def _read_and_cache(
        self, node_ids: np.ndarray, props: list, label: str,
        n_total: int,
    ) -> None:
        t0_total = time.time()
        dupes = set(int(x) for x in node_ids) & self._read_row_keys
        assert not dupes, f"DUPLICATE BIGTABLE READ: {len(dupes)} rows read twice in {label}: {list(dupes)[:5]}"
        self._read_row_keys.update(int(x) for x in node_ids)

        t0 = time.time()
        raw = self.cg.client.read_nodes(node_ids=node_ids, properties=props, end_time=self.stitch_timestamp)
        t_read = time.time() - t0

        lc = self.cg.meta.layer_count
        t0 = time.time()
        for n in node_ids:
            n_int = int(n)
            data = raw.get(n, {})
            parent_cells = data.get(attributes.Hierarchy.Parent, [])
            if parent_cells:
                self._cache.put_parent(n_int, int(parent_cells[0].value))
            child_cells = data.get(attributes.Hierarchy.Child, [])
            if not child_cells:
                continue
            self._cache.put_children(n_int, child_cells[0].value)
            acx_d = {
                layer: cells[0].value.copy()
                for layer in range(2, lc)
                if (cells := data.get(attributes.Connectivity.AtomicCrossChunkEdge[layer], []))
            }
            if acx_d:
                self._cache.put_acx(n_int, acx_d)
            cx_d = {
                layer: cells[0].value
                for layer in range(2, lc)
                if (cells := data.get(attributes.Connectivity.CrossChunkEdge[layer], []))
                and len(cells[0].value) > 0
            }
            if cx_d:
                self._cache.put_cx(n_int, cx_d)
        t_cache = time.time() - t0

        self.rpc_log.append(RpcEntry(
            label=label, n_requested=n_total, n_read=len(node_ids),
            t_read=t_read, t_cache=t_cache,
            t_total=time.time() - t0_total,
        ))


