"""
Cached BigTable reader and ID allocation utilities for the proposed stitch.
Separated from proposed.py to keep the core algorithm focused.
"""

import time
from collections import ChainMap
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from google.cloud.bigtable.data.row_filters import CellsRowLimitFilter, RowFilterChain, StripValueTransformerFilter

from kvdbclient.serializers import deserialize_uint64, serialize_uint64_batch

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes
from pychunkedgraph.graph.utils.generic import filter_failed_node_ids

_EXIST_FILTER = RowFilterChain(filters=[
    CellsRowLimitFilter(1),
    StripValueTransformerFilter(True),
])


USE_BULK_READ = True
USE_SAMPLING = True
VERBOSE = False

_SAMPLING_THRESHOLD = 25000
_SAMPLE_SIZE = 1000
_SAMPLING_STOP = 10000


class CachedReader:
    """Caches BigTable reads across multiple calls within a single stitch.
    Caches: get_parents, get_children, get_atomic_cross_edges.
    Thread-safe under CPython GIL for concurrent reads with convergent keys.
    Cache is never invalidated — all reads happen before any writes.
    """

    def __init__(self, cg, preloaded: tuple = None):
        self.cg = cg
        ro_parents = preloaded[0] if preloaded else {}
        ro_children = preloaded[1] if preloaded else {}
        ro_acx = preloaded[2] if preloaded else {}
        self._parents_local = {}
        self._children_local = {}
        self._acx_local = {}
        self._parents = ChainMap(self._parents_local, ro_parents)
        self._children = ChainMap(self._children_local, ro_children)
        self._atomic_cx = ChainMap(self._acx_local, ro_acx)
        self.rpc_log = []

    def _populate_from_raw(self, node_ids: np.ndarray, raw: dict, with_acx: bool = False) -> None:
        """Parse raw read_nodes response into caches."""
        for n in node_ids:
            n_int = int(n)
            node_data = raw.get(n, {})

            if n_int not in self._parents:
                parent_cells = node_data.get(attributes.Hierarchy.Parent, [])
                if parent_cells:
                    self._parents[n_int] = int(parent_cells[0].value)

            if n_int not in self._children:
                child_cells = node_data.get(attributes.Hierarchy.Child, [])
                if child_cells:
                    self._children[n_int] = child_cells[0].value

            if with_acx and n_int not in self._atomic_cx:
                acx_d = {}
                for l in range(2, max(3, self.cg.meta.layer_count)):
                    prop = attributes.Connectivity.AtomicCrossChunkEdge[l]
                    cells = node_data.get(prop, [])
                    if cells:
                        acx_d[l] = cells[0].value.copy()
                self._atomic_cx[n_int] = acx_d

    def _read_and_cache(self, node_ids: np.ndarray, props: list, label: str, n_total: int, with_acx: bool = False) -> None:
        """Read properties from BigTable for uncached nodes and populate caches."""
        t0 = time.time()
        raw = self.cg.client.read_nodes(node_ids=node_ids, properties=props)
        self.rpc_log.append((label, n_total, len(node_ids), time.time() - t0))
        self._populate_from_raw(node_ids, raw, with_acx=with_acx)

    def _uncached(self, node_ids: np.ndarray, *caches) -> np.ndarray:
        """Return node IDs missing from any of the specified caches."""
        return np.array(
            [n for n in node_ids if any(int(n) not in c for c in caches)],
            dtype=basetypes.NODE_ID,
        )

    def get_parents(self, node_ids, **kwargs) -> np.ndarray:
        """Read Parent for uncached nodes. For non-SVs, also reads Child."""
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        uncached = self._uncached(node_ids, self._parents)
        if len(uncached) > 0:
            layers = self.cg.get_chunk_layers(uncached)
            svs = uncached[layers <= 1]
            non_svs = uncached[layers > 1]
            if len(svs) > 0:
                self._read_and_cache(svs, [attributes.Hierarchy.Parent], "get_parents_sv", len(node_ids))
            if len(non_svs) > 0:
                self._read_and_cache(non_svs, [attributes.Hierarchy.Parent, attributes.Hierarchy.Child], "get_parents", len(node_ids))
        return np.array([self._parents[int(n)] for n in node_ids], dtype=basetypes.NODE_ID)

    def get_children(self, node_ids) -> dict:
        """Read Parent+Child for uncached nodes. Always caches both."""
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        uncached = self._uncached(node_ids, self._parents, self._children)
        if len(uncached) > 0:
            props = [attributes.Hierarchy.Parent, attributes.Hierarchy.Child]
            self._read_and_cache(uncached, props, "get_children", len(node_ids))
        return {n: self._children[int(n)] for n in node_ids}

    def get_atomic_cross_edges(self, l2_ids) -> dict:
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        uncached = self._uncached(l2_ids, self._atomic_cx)
        if len(uncached) > 0:
            props = [attributes.Connectivity.AtomicCrossChunkEdge[l]
                     for l in range(2, max(3, self.cg.meta.layer_count))]
            self._read_and_cache(uncached, props, "get_atomic_cx", len(l2_ids), with_acx=True)
        return {n: self._atomic_cx[int(n)] for n in l2_ids}

    def bulk_read_parent_child(self, node_ids: np.ndarray) -> tuple:
        """Read Parent + Child in one RPC. Returns (parents_array, children_dict)."""
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        uncached = self._uncached(node_ids, self._parents, self._children)
        if len(uncached) > 0:
            props = [attributes.Hierarchy.Parent, attributes.Hierarchy.Child]
            self._read_and_cache(uncached, props, "bulk_parent_child", len(node_ids))
        parents = np.array([self._parents.get(int(n), 0) for n in node_ids], dtype=basetypes.NODE_ID)
        children_d = {n: self._children[int(n)] for n in node_ids}
        return parents, children_d

    def bulk_read_l2(self, l2_ids) -> tuple:
        """Read Parent + Child + AtomicCrossChunkEdge in one RPC for L2 nodes."""
        l2_ids = np.asarray(l2_ids, dtype=basetypes.NODE_ID)
        uncached = self._uncached(l2_ids, self._parents, self._children, self._atomic_cx)
        if len(uncached) > 0:
            props = [attributes.Hierarchy.Parent, attributes.Hierarchy.Child] + [
                attributes.Connectivity.AtomicCrossChunkEdge[l]
                for l in range(2, max(3, self.cg.meta.layer_count))
            ]
            self._read_and_cache(uncached, props, "bulk_read_l2", len(l2_ids), with_acx=True)
        children_d = {n: self._children[int(n)] for n in l2_ids}
        acx_d = {n: self._atomic_cx[int(n)] for n in l2_ids}
        return children_d, acx_d


def get_all_parents_filtered(
    cg_or_reader, node_ids: np.ndarray, time_stamp=None
) -> dict:
    """
    Like get_all_parents_dict_multiple but applies filter_failed_node_ids
    at every layer to discard orphaned nodes from failed stitch attempts.
    Orphaned parents are remapped to their valid counterparts.

    cg_or_reader: ChunkedGraph or CachedReader. Using CachedReader avoids
    re-reading nodes already seen in prior calls.
    """
    reader = (
        cg_or_reader
        if isinstance(cg_or_reader, CachedReader)
        else CachedReader(cg_or_reader)
    )
    cg = reader.cg
    result = {int(node): {} for node in node_ids}
    nodes = np.array(node_ids, dtype=basetypes.NODE_ID)
    child_parent_map = {}
    layers_map = {}

    while nodes.size > 0:
        parents = reader.get_parents(nodes, time_stamp=time_stamp)
        parent_layers = cg.get_chunk_layers(parents)

        remap = {}
        unique_parents = np.unique(parents)
        if len(unique_parents) > 0:
            # read Parent+Child together — Children needed for orphan check,
            # Parent cached for next iteration's get_parents call
            _, children_d = reader.bulk_read_parent_child(unique_parents)
            max_children_ids = [
                int(np.max(children_d[p])) if len(children_d.get(p, [])) > 0 else 0
                for p in unique_parents
            ]
            segment_ids = np.array([cg.get_segment_id(p) for p in unique_parents])
            valid = set(
                int(x)
                for x in filter_failed_node_ids(
                    unique_parents, segment_ids, max_children_ids
                )
            )

            mcid_to_valid = {}
            for p, mcid in zip(unique_parents, max_children_ids):
                if int(p) in valid:
                    mcid_to_valid[mcid] = int(p)

            for p, mcid in zip(unique_parents, max_children_ids):
                if int(p) not in valid:
                    remap[int(p)] = mcid_to_valid.get(mcid, int(p))

        for node, parent, layer in zip(nodes, parents, parent_layers):
            resolved_parent = remap.get(int(parent), int(parent))
            layers_map[resolved_parent] = int(layer)
            child_parent_map[int(node)] = resolved_parent

        next_nodes = []
        for parent, layer in zip(parents, parent_layers):
            resolved = remap.get(int(parent), int(parent))
            if int(layer) < cg.meta.layer_count:
                next_nodes.append(resolved)
        nodes = (
            np.unique(np.array(next_nodes, dtype=basetypes.NODE_ID))
            if next_nodes
            else np.array([], dtype=basetypes.NODE_ID)
        )

    for node in node_ids:
        current = int(node)
        node_result = {}
        while current in child_parent_map:
            parent = child_parent_map[current]
            parent_layer = layers_map[parent]
            node_result[parent_layer] = parent
            current = parent
        result[int(node)] = node_result
    return result


def filter_orphaned_nodes(cg, node_ids: np.ndarray, children_d: dict) -> np.ndarray:
    """Filter out orphaned nodes from prior failed stitch attempts."""
    if len(node_ids) == 0:
        return node_ids
    max_children_ids = np.array(
        [
            int(np.max(children_d[n])) if len(children_d.get(n, [])) > 0 else 0
            for n in node_ids
        ]
    )
    segment_ids = np.array([cg.get_segment_id(n) for n in node_ids])
    return filter_failed_node_ids(node_ids, segment_ids, max_children_ids)


def resolve_partner_sv_parents(reader: CachedReader, unknown_svs: set) -> dict:
    """Resolve partner SVs → L2 parents. Uses sampling when > threshold.
    Returns {sv_int: l2_int}.
    """

    if not unknown_svs:
        return {}

    unknown_arr = np.array(list(unknown_svs), dtype=basetypes.NODE_ID)

    if not USE_SAMPLING or len(unknown_arr) <= _SAMPLING_THRESHOLD:
        t0 = time.time()
        parents = reader.get_parents(unknown_arr)
        if VERBOSE:
            unique_l2s = len(np.unique(parents))
            print(f"    [resolve_partners] brute, {len(unknown_arr)} SVs → {unique_l2s} L2s, {time.time() - t0:.1f}s")
        return {int(sv): int(l2) for sv, l2 in zip(unknown_arr, parents)}

    t_start = time.time()
    rng = np.random.default_rng()
    remaining = set(int(x) for x in unknown_arr)
    resolved = {}
    known_l2s = set()
    iteration = 0

    while len(remaining) > _SAMPLING_STOP:
        rem_arr = np.array(list(remaining), dtype=basetypes.NODE_ID)
        sample = rng.choice(
            rem_arr, size=min(_SAMPLE_SIZE, len(rem_arr)), replace=False
        )
        parents = reader.get_parents(sample)

        for sv, l2 in zip(sample, parents):
            resolved[int(sv)] = int(l2)
        remaining -= set(int(x) for x in sample)

        new_l2s = set(int(x) for x in np.unique(parents)) - known_l2s
        if new_l2s:
            _, ch = reader.bulk_read_parent_child(np.array(list(new_l2s), dtype=basetypes.NODE_ID))
            for l2_int in new_l2s:
                l2_uint = np.uint64(l2_int)
                for sv in ch.get(l2_uint, ch.get(l2_int, [])):
                    sv_int = int(sv)
                    if sv_int in remaining:
                        resolved[sv_int] = l2_int
                        remaining.discard(sv_int)
            known_l2s.update(new_l2s)

        iteration += 1
        if VERBOSE:
            print(
                f"    [sampling] iter {iteration}: {len(sample)} sampled, {len(new_l2s)} L2s, {len(remaining)} remaining"
            )

    if remaining:
        rem_arr = np.array(list(remaining), dtype=basetypes.NODE_ID)
        parents = reader.get_parents(rem_arr)
        for sv, l2 in zip(rem_arr, parents):
            resolved[int(sv)] = int(l2)
        if VERBOSE:
            print(
                f"    [sampling] leftovers: {len(rem_arr)} SVs, total {len(resolved)} resolved"
            )

    if VERBOSE:
        unique_l2s = len(set(resolved.values()))
        print(f"    [resolve_partners] sampling, {len(unknown_arr)} SVs → {unique_l2s} L2s, {time.time() - t_start:.1f}s")
    return resolved


def batch_create_node_ids(cg, size_map: dict, root_chunks: set = None) -> dict:
    """Allocate node IDs for multiple chunks in parallel threads.
    Checks for collisions with existing rows for root chunks.
    """
    if root_chunks is None:
        root_chunks = set()

    def _alloc(chunk_id):
        count = size_map[chunk_id]
        is_root = chunk_id in root_chunks
        if not is_root:
            return chunk_id, list(
                cg.id_client.create_node_ids(
                    np.uint64(chunk_id),
                    size=count,
                    root_chunk=False,
                )
            )
        batch_size = count
        new_ids = []
        while len(new_ids) < count:
            candidate_ids = cg.id_client.create_node_ids(
                np.uint64(chunk_id),
                size=batch_size,
                root_chunk=True,
            )
            row_keys = serialize_uint64_batch(candidate_ids)
            existing_rows = cg.client._read(row_keys=row_keys, row_filter=_EXIST_FILTER)
            existing_ids = {deserialize_uint64(k) for k in existing_rows}
            non_existing = set(candidate_ids) - existing_ids
            new_ids.extend(non_existing)
            batch_size = min(batch_size * 2, 2**16)
        return chunk_id, new_ids[:count]

    result = {}
    with ThreadPoolExecutor(max_workers=min(len(size_map), 16)) as executor:
        futures = {executor.submit(_alloc, c): c for c in size_map}
        for fut in as_completed(futures):
            chunk_id, ids = fut.result()
            result[chunk_id] = ids
    return result


def collect_and_resolve_partner_svs(
    reader: CachedReader, acx_source: dict, known_svs: set, resolver: dict
) -> np.ndarray:
    """Collect partner SVs from atomic cx edges, resolve unknown ones.
    acx_source: dict of {node_id: {layer: edges}} (atomic cx).
    Updates resolver in-place with new chains.
    """
    partner_svs = set()
    for layer_d in acx_source.values():
        for layer, edges in layer_d.items():
            if len(edges) > 0:
                partner_svs.update(edges[:, 0])
                partner_svs.update(edges[:, 1])

    unknown_svs = np.array(list(partner_svs - known_svs), dtype=basetypes.NODE_ID)
    if len(unknown_svs) > 0:
        chains = get_all_parents_filtered(reader, unknown_svs)
        for sv, chain in chains.items():
            resolver[int(sv)] = {int(l): int(p) for l, p in chain.items()}
    return unknown_svs


def read_l2_parallel(reader: CachedReader, l2ids: np.ndarray) -> tuple:
    """Read children + atomic cx in two parallel RPCs. Faster for small batches."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_children = executor.submit(reader.get_children, l2ids)
        fut_acx = executor.submit(reader.get_atomic_cross_edges, l2ids)
        return fut_children.result(), fut_acx.result()


def read_l2_bulk(reader: CachedReader, l2ids: np.ndarray) -> tuple:
    """Read parent + children + atomic cx in one combined RPC. Fewer RPCs, caches parent."""
    return reader.bulk_read_l2(l2ids)


def read_l2(reader: CachedReader, l2ids: np.ndarray) -> tuple:
    """Dispatch to bulk or parallel based on USE_BULK_READ flag."""
    mode = "bulk" if USE_BULK_READ else "parallel"
    t0 = time.time()
    result = read_l2_bulk(reader, l2ids) if USE_BULK_READ else read_l2_parallel(reader, l2ids)
    if VERBOSE:
        print(f"    [read_l2] {mode}, {len(l2ids)} nodes, {time.time() - t0:.1f}s")
    return result
