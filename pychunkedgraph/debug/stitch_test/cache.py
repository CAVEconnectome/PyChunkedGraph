"""
CachedReader for BigTable reads with cross-wave cache support.

Cache contract:
- Children and ACX for existing nodes are NEVER modified by stitch writes.
  Safe to cache indefinitely across waves.
- Parent pointers ARE modified by stitch writes (children of new nodes get new parents).
  Parents are cached within a single stitch (reads before writes) but the cross-wave
  preloaded cache for parents is treated as a hint: whenever a node IS read from
  BigTable (for any reason), the fresh Parent value always overwrites the preloaded one.

Wave cache structure: (parents_dict, children_dict, acx_dict)
- Populated by merging worker snapshots after each wave
- Two-pass merge: reader caches first (pre-stitch), ctx caches second (post-stitch)
- Workers inherit via fork COW — no serialization for reads
"""

import logging
import time
from collections import ChainMap, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from google.cloud.bigtable.data.row_filters import (
    CellsRowLimitFilter,
    RowFilterChain,
    StripValueTransformerFilter,
)
from kvdbclient.serializers import deserialize_uint64, serialize_uint64_batch

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes
from pychunkedgraph.graph.utils.generic import filter_failed_node_ids

log = logging.getLogger(__name__)

USE_BULK_READ = True
USE_SAMPLING = True
SAMPLING_THRESHOLD = 25000
SAMPLE_SIZE = 2500
SAMPLING_STOP = 10000

_EXIST_FILTER = RowFilterChain(filters=[
    CellsRowLimitFilter(1),
    StripValueTransformerFilter(True),
])


class CachedReader:
    """Caches BigTable reads. Supports preloaded cache from prior waves.

    Thread-safe under CPython GIL for concurrent reads producing same values.
    """

    def __init__(self, cg: ChunkedGraph, preloaded: tuple = None):
        self.cg = cg
        self._parents_local = {}
        self._children_local = {}
        self._acx_local = {}

        ro_p = preloaded[0] if preloaded else {}
        ro_c = preloaded[1] if preloaded else {}
        ro_a = preloaded[2] if preloaded else {}

        self._parents = ChainMap(self._parents_local, ro_p)
        self._children = ChainMap(self._children_local, ro_c)
        self._acx = ChainMap(self._acx_local, ro_a)
        self.rpc_log: list[tuple] = []

    def _uncached(self, node_ids: np.ndarray, *caches) -> np.ndarray:
        return np.array(
            [n for n in node_ids if any(int(n) not in c for c in caches)],
            dtype=basetypes.NODE_ID,
        )

    def _read_and_cache(
        self, node_ids: np.ndarray, props: list, label: str,
        n_total: int, with_acx: bool = False,
    ) -> None:
        t0 = time.time()
        raw = self.cg.client.read_nodes(node_ids=node_ids, properties=props)
        self.rpc_log.append((label, n_total, len(node_ids), time.time() - t0))
        self._populate(node_ids, raw, with_acx=with_acx)

    def _populate(
        self, node_ids: np.ndarray, raw: dict, with_acx: bool = False,
    ) -> None:
        for n in node_ids:
            n_int = int(n)
            data = raw.get(n, {})

            parent_cells = data.get(attributes.Hierarchy.Parent, [])
            if parent_cells:
                self._parents[n_int] = int(parent_cells[0].value)

            child_cells = data.get(attributes.Hierarchy.Child, [])
            if child_cells:
                self._children[n_int] = child_cells[0].value

            if with_acx:
                acx_d = {}
                for layer in range(2, max(3, self.cg.meta.layer_count)):
                    prop = attributes.Connectivity.AtomicCrossChunkEdge[layer]
                    cells = data.get(prop, [])
                    if cells:
                        acx_d[layer] = cells[0].value.copy()
                self._acx[n_int] = acx_d

    def _ensure_cached(self, node_ids: np.ndarray, label: str) -> None:
        """Read all columns for uncached nodes. SVs get Parent only.
        L2 nodes get Parent+Child+ACX. Higher nodes get Parent+Child.
        Each row is read exactly once with all its columns.
        Parent cache is the single source of truth for "has been read"."""
        uncached = self._uncached(node_ids, self._parents)
        if len(uncached) == 0:
            return
        layers = self.cg.get_chunk_layers(uncached)
        svs = uncached[layers <= 1]
        l2s = uncached[layers == 2]
        higher = uncached[layers > 2]
        n = len(node_ids)
        if len(svs) > 0:
            self._read_and_cache(
                svs, [attributes.Hierarchy.Parent], f"{label}_sv", n,
            )
        if len(l2s) > 0:
            acx_props = [
                attributes.Connectivity.AtomicCrossChunkEdge[l]
                for l in range(2, max(3, self.cg.meta.layer_count))
            ]
            self._read_and_cache(
                l2s,
                [attributes.Hierarchy.Parent, attributes.Hierarchy.Child] + acx_props,
                f"{label}_l2", n, with_acx=True,
            )
        if len(higher) > 0:
            self._read_and_cache(
                higher,
                [attributes.Hierarchy.Parent, attributes.Hierarchy.Child],
                label, n,
            )

    def get_parents(self, node_ids: np.ndarray, **kw) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(node_ids, "get_parents")
        return np.array(
            [self._parents[int(n)] for n in node_ids], dtype=basetypes.NODE_ID,
        )

    def get_children(self, node_ids: np.ndarray) -> dict:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(node_ids, "get_children")
        return {n: self._children[int(n)] for n in node_ids}

    def get_acx(self, l2_ids: np.ndarray) -> dict:
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(l2_ids, "get_acx")
        return {n: self._acx[int(n)] for n in l2_ids}

    def bulk_read_parent_child(self, node_ids: np.ndarray) -> tuple:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(node_ids, "bulk_pc")
        parents = np.array(
            [self._parents.get(int(n), 0) for n in node_ids],
            dtype=basetypes.NODE_ID,
        )
        return parents, {n: self._children[int(n)] for n in node_ids}

    def bulk_read_l2(self, l2_ids: np.ndarray) -> tuple:
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        self._ensure_cached(l2_ids, "bulk_l2")
        return (
            {n: self._children[int(n)] for n in l2_ids},
            {n: self._acx[int(n)] for n in l2_ids},
        )

    def snapshot(self) -> tuple:
        """Return local caches for merging into shared cache."""
        return (
            self._parents_local,
            self._children_local,
            self._acx_local,
        )


def read_l2(reader: CachedReader, l2ids: np.ndarray) -> tuple:
    if USE_BULK_READ:
        return reader.bulk_read_l2(l2ids)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fc = ex.submit(reader.get_children, l2ids)
        fa = ex.submit(reader.get_acx, l2ids)
        return fc.result(), fa.result()


def get_all_parents_filtered(
    reader: CachedReader, node_ids: np.ndarray,
) -> dict:
    """Walk parent chains with orphan filtering at each layer."""
    cg = reader.cg
    result = {int(n): {} for n in node_ids}
    nodes = np.array(node_ids, dtype=basetypes.NODE_ID)
    child_parent = {}
    layer_map = {}

    while nodes.size > 0:
        parents = reader.get_parents(nodes)
        parent_layers = cg.get_chunk_layers(parents)

        remap = {}
        unique_parents = np.unique(parents)
        if len(unique_parents) > 0:
            _, ch_d = reader.bulk_read_parent_child(unique_parents)
            max_ch = [
                int(np.max(ch_d[p])) if len(ch_d.get(p, [])) > 0 else 0
                for p in unique_parents
            ]
            seg_ids = np.array(
                [cg.get_segment_id(p) for p in unique_parents]
            )
            valid = set(
                int(x) for x in filter_failed_node_ids(
                    unique_parents, seg_ids, max_ch,
                )
            )
            mcid_valid = {}
            for p, mc in zip(unique_parents, max_ch):
                if int(p) in valid:
                    mcid_valid[mc] = int(p)
            for p, mc in zip(unique_parents, max_ch):
                if int(p) not in valid:
                    remap[int(p)] = mcid_valid.get(mc, int(p))

        for node, parent, layer in zip(nodes, parents, parent_layers):
            resolved = remap.get(int(parent), int(parent))
            layer_map[resolved] = int(layer)
            child_parent[int(node)] = resolved

        nxt = []
        for parent, layer in zip(parents, parent_layers):
            resolved = remap.get(int(parent), int(parent))
            if int(layer) < cg.meta.layer_count:
                nxt.append(resolved)
        nodes = (
            np.unique(np.array(nxt, dtype=basetypes.NODE_ID))
            if nxt else np.array([], dtype=basetypes.NODE_ID)
        )

    for n in node_ids:
        cur = int(n)
        chain = {}
        while cur in child_parent:
            par = child_parent[cur]
            chain[layer_map[par]] = par
            cur = par
        result[int(n)] = chain
    return result


def filter_orphaned_nodes(
    cg: ChunkedGraph, node_ids: np.ndarray, children_d: dict,
) -> np.ndarray:
    if len(node_ids) == 0:
        return node_ids
    max_ch = np.array([
        int(np.max(children_d[n])) if len(children_d.get(n, [])) > 0 else 0
        for n in node_ids
    ])
    seg_ids = np.array([cg.get_segment_id(n) for n in node_ids])
    return filter_failed_node_ids(node_ids, seg_ids, max_ch)


def resolve_partner_sv_parents(
    reader: CachedReader, unknown_svs: set,
) -> dict:
    """Resolve partner SVs to L2 parents. Sampling for large sets."""
    if not unknown_svs:
        return {}
    arr = np.array(list(unknown_svs), dtype=basetypes.NODE_ID)

    if not USE_SAMPLING or len(arr) <= SAMPLING_THRESHOLD:
        parents = reader.get_parents(arr)
        return {int(sv): int(l2) for sv, l2 in zip(arr, parents)}

    rng = np.random.default_rng()
    remaining = set(int(x) for x in arr)
    resolved = {}
    known_l2s = set()

    while len(remaining) > SAMPLING_STOP:
        rem = np.array(list(remaining), dtype=basetypes.NODE_ID)
        sample = rng.choice(rem, size=min(SAMPLE_SIZE, len(rem)), replace=False)
        parents = reader.get_parents(sample)
        for sv, l2 in zip(sample, parents):
            resolved[int(sv)] = int(l2)
        remaining -= set(int(x) for x in sample)

        new_l2s = set(int(x) for x in np.unique(parents)) - known_l2s
        if new_l2s:
            _, ch = reader.bulk_read_parent_child(
                np.array(list(new_l2s), dtype=basetypes.NODE_ID)
            )
            for l2_int in new_l2s:
                for sv in ch.get(np.uint64(l2_int), ch.get(l2_int, [])):
                    sv_int = int(sv)
                    if sv_int in remaining:
                        resolved[sv_int] = l2_int
                        remaining.discard(sv_int)
            known_l2s.update(new_l2s)

    if remaining:
        rem = np.array(list(remaining), dtype=basetypes.NODE_ID)
        parents = reader.get_parents(rem)
        for sv, l2 in zip(rem, parents):
            resolved[int(sv)] = int(l2)

    return resolved


def batch_create_node_ids(
    cg: ChunkedGraph, size_map: dict, root_chunks: set = None,
) -> dict:
    """Allocate node IDs. Collision check for root chunks."""
    if root_chunks is None:
        root_chunks = set()

    def _alloc(chunk_id: int) -> tuple:
        count = size_map[chunk_id]
        is_root = chunk_id in root_chunks
        if not is_root:
            return chunk_id, list(cg.id_client.create_node_ids(
                np.uint64(chunk_id), size=count, root_chunk=False,
            ))
        batch_size = count
        new_ids = []
        while len(new_ids) < count:
            candidates = cg.id_client.create_node_ids(
                np.uint64(chunk_id), size=batch_size, root_chunk=True,
            )
            rows = cg.client._read(
                row_keys=serialize_uint64_batch(candidates),
                row_filter=_EXIST_FILTER,
            )
            existing = {deserialize_uint64(k) for k in rows}
            new_ids.extend(set(candidates) - existing)
            batch_size = min(batch_size * 2, 2**16)
        return chunk_id, new_ids[:count]

    result = {}
    n_workers = min(len(size_map), 16)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_alloc, c): c for c in size_map}
        for fut in as_completed(futs):
            cid, ids = fut.result()
            result[cid] = ids
    return result


def collect_and_resolve_partners(
    reader: CachedReader, acx_source: dict,
    known_svs: set, resolver: dict,
) -> np.ndarray:
    """Collect partner SVs from ACX, resolve unknowns, update resolver."""
    partner_svs = set()
    for layer_d in acx_source.values():
        for edges in layer_d.values():
            if len(edges) > 0:
                partner_svs.update(edges[:, 0])
                partner_svs.update(edges[:, 1])

    unknown = np.array(
        list(partner_svs - known_svs), dtype=basetypes.NODE_ID,
    )
    if len(unknown) > 0:
        chains = get_all_parents_filtered(reader, unknown)
        for sv, chain in chains.items():
            resolver[int(sv)] = {int(l): int(p) for l, p in chain.items()}
    return unknown


class WaveCache:
    """Manages shared cache between waves. Thread-safe for fork COW."""

    def __init__(self) -> None:
        self._data: tuple = None  # (parents, children, acx)

    @property
    def data(self) -> tuple:
        return self._data

    def _ensure(self) -> tuple:
        if self._data is None:
            self._data = ({}, {}, {})
        return self._data

    def merge_reader(self, snapshot: tuple) -> None:
        """Merge one reader snapshot (pre-stitch) immediately."""
        parents, children, acx = self._ensure()
        rp, rc, ra = snapshot
        parents.update(rp)
        children.update(rc)
        acx.update(ra)

    def merge_all_ctx(self, ctx_snapshots: list[tuple]) -> None:
        """Merge all ctx snapshots (post-stitch) after wave. Ctx always wins."""
        parents, children, acx = self._ensure()
        for cp, cc, ca in ctx_snapshots:
            parents.update(cp)
            children.update(cc)
            acx.update(ca)

    def merge_single(
        self,
        reader_snapshot: tuple,
        ctx_parents: dict, ctx_children: dict, ctx_acx: dict,
    ) -> None:
        """Merge for in-process single-file waves. Reader first, ctx second."""
        self.merge_reader(reader_snapshot)
        self.merge_all_ctx([(ctx_parents, ctx_children, ctx_acx)])

    def merge_wave(
        self, reader_snapshots: list[tuple], ctx_snapshots: list[tuple],
    ) -> None:
        """Full wave merge: readers first, ctx second."""
        for snap in reader_snapshots:
            self.merge_reader(snap)
        self.merge_all_ctx(ctx_snapshots)

    def stats(self) -> str:
        if not self._data:
            return "empty"
        p, c, a = self._data
        return f"{len(p)} parents, {len(c)} children, {len(a)} acx"
