import threading
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timedelta, UTC
from functools import reduce
from unittest.mock import MagicMock

import numpy as np

from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph import basetypes
from ..ingest.create.atomic_layer import add_atomic_chunk
from ..ingest.create.parent_layer import add_parent_chunk


def fake_timestamp():
    """A timestamp safely in the past, for edits/lineage tests needing an old parent_ts."""
    return datetime.now(UTC) - timedelta(days=10)


class CloudVolumeBounds(object):
    def __init__(self, bounds=[[0, 0, 0], [0, 0, 0]]):
        self._bounds = np.array(bounds)

    @property
    def bounds(self):
        return self._bounds

    def __repr__(self):
        return self.bounds

    def to_list(self):
        return list(np.array(self.bounds).flatten())


class CloudVolumeMock(object):
    def __init__(self):
        self.resolution = np.array([1, 1, 1], dtype=int)
        self.bounds = CloudVolumeBounds()


class _TSRead:
    def __init__(self, arr):
        self._arr = arr

    def read(self):
        return self

    def result(self):
        return self._arr


class TensorStoreMock:
    """Stand-in for a tensorstore neuroglancer_precomputed handle.

    ``handle[slices].read().result()`` returns a slice of the backing ``seg``
    array, or a constant ``fill`` block shaped to the slice when ``seg`` is None.
    """

    def __init__(self, seg=None, fill=0, dtype=np.uint64):
        self._seg = seg
        self._fill = fill
        self._dtype = dtype

    def __getitem__(self, key):
        if self._seg is not None:
            return _TSRead(self._seg[key])
        shape = tuple(s.stop - s.start for s in key)
        return _TSRead(np.full(shape, self._fill, dtype=self._dtype))


def mock_ws_info(resolution=(1, 1, 1), voxel_offset=(0, 0, 0), size=(0, 0, 0)):
    """A single-scale watershed ``info`` dict for ``meta._ws_info_d``."""
    return {
        "scales": [
            {
                "resolution": list(resolution),
                "voxel_offset": list(voxel_offset),
                "size": list(size),
            }
        ]
    }


def create_chunk(cg, vertices=None, edges=None, timestamp=None):
    """
    Helper function to add vertices and edges to the chunkedgraph - no safety checks!
    """
    edges = edges if edges else []
    vertices = vertices if vertices else []
    vertices = np.unique(np.array(vertices, dtype=np.uint64))
    edges = [(np.uint64(v1), np.uint64(v2), np.float32(aff)) for v1, v2, aff in edges]
    isolated_ids = [
        x
        for x in vertices
        if (x not in [edges[i][0] for i in range(len(edges))])
        and (x not in [edges[i][1] for i in range(len(edges))])
    ]

    chunk_edges_active = {}
    for edge_type in EDGE_TYPES:
        chunk_edges_active[edge_type] = Edges([], [])

    for e in edges:
        if cg.get_chunk_id(e[0]) == cg.get_chunk_id(e[1]):
            sv1s = np.array([e[0]], dtype=basetypes.NODE_ID)
            sv2s = np.array([e[1]], dtype=basetypes.NODE_ID)
            affs = np.array([e[2]], dtype=basetypes.EDGE_AFFINITY)
            chunk_edges_active[EDGE_TYPES.in_chunk] += Edges(
                sv1s, sv2s, affinities=affs
            )

    chunk_id = None
    if len(chunk_edges_active[EDGE_TYPES.in_chunk]):
        chunk_id = cg.get_chunk_id(chunk_edges_active[EDGE_TYPES.in_chunk].node_ids1[0])
    elif len(vertices):
        chunk_id = cg.get_chunk_id(vertices[0])

    for e in edges:
        if not cg.get_chunk_id(e[0]) == cg.get_chunk_id(e[1]):
            # Ensure proper order
            if chunk_id is not None:
                if not chunk_id == cg.get_chunk_id(e[0]):
                    e = [e[1], e[0], e[2]]
            sv1s = np.array([e[0]], dtype=basetypes.NODE_ID)
            sv2s = np.array([e[1]], dtype=basetypes.NODE_ID)
            affs = np.array([e[2]], dtype=basetypes.EDGE_AFFINITY)
            if np.isinf(e[2]):
                chunk_edges_active[EDGE_TYPES.cross_chunk] += Edges(
                    sv1s, sv2s, affinities=affs
                )
            else:
                chunk_edges_active[EDGE_TYPES.between_chunk] += Edges(
                    sv1s, sv2s, affinities=affs
                )

    all_edges = reduce(lambda x, y: x + y, chunk_edges_active.values())
    cg.mock_edges += all_edges

    isolated_ids = np.array(isolated_ids, dtype=np.uint64)
    add_atomic_chunk(
        cg,
        cg.get_chunk_coordinates(chunk_id),
        chunk_edges_active,
        isolated=isolated_ids,
        time_stamp=timestamp,
    )


def to_label(cg, l, x, y, z, segment_id):
    return cg.get_node_id(np.uint64(segment_id), layer=l, x=x, y=y, z=z)


def get_layer_chunk_bounds(
    n_layers: int, atomic_chunk_bounds: np.ndarray = np.array([])
) -> dict:
    if atomic_chunk_bounds.size == 0:
        limit = 2 ** (n_layers - 2)
        atomic_chunk_bounds = np.array([limit, limit, limit])
    layer_bounds_d = {}
    for layer in range(2, n_layers):
        layer_bounds = atomic_chunk_bounds / (2 ** (layer - 2))
        layer_bounds_d[layer] = np.ceil(layer_bounds).astype(int)
    return layer_bounds_d


SV = namedtuple("SV", ["x", "y", "z", "seg"], defaults=(0, 0, 0, 0))
BuiltGraph = namedtuple("BuiltGraph", ["cg", "sv"])


def label(cg, sv, layer=1):
    """Node id for supervoxel coordinate ``sv`` at ``layer`` (layer 1 = the supervoxel)."""
    return to_label(cg, layer, sv.x, sv.y, sv.z, sv.seg)


def build_graph(gen_graph, n_layers, supervoxels, edges=(), *, timestamp=None, atomic_chunk_bounds=None):
    """Build a test graph from named supervoxels and edges; parents derived, at one ts.

    supervoxels maps a name to its (x, y, z, seg) atomic coordinate; edges are
    (name, name, affinity). Returns BuiltGraph(cg, sv); sv maps each name to its node id.
    """
    bounds = np.array([]) if atomic_chunk_bounds is None else atomic_chunk_bounds
    cg = gen_graph(n_layers=n_layers, atomic_chunk_bounds=bounds)
    ts = fake_timestamp() if timestamp is None else timestamp
    sv = {name: to_label(cg, 1, *coord) for name, coord in supervoxels.items()}
    chunk = {name: tuple(coord[:3]) for name, coord in supervoxels.items()}
    members = {}
    for name in supervoxels:
        members.setdefault(chunk[name], []).append(sv[name])
    for coord, labels in members.items():
        chunk_edges = [
            (sv[a], sv[b], aff) for a, b, aff in edges if coord in (chunk[a], chunk[b])
        ]
        create_chunk(cg, vertices=labels, edges=chunk_edges, timestamp=ts)
    fanout = cg.meta.graph_config.FANOUT
    for layer in range(3, n_layers + 1):
        pcoords = {tuple(np.array(c) // fanout ** (layer - 2)) for c in members}
        for pcoord in sorted(pcoords):
            add_parent_chunk(cg, layer, list(pcoord), time_stamp=ts, n_threads=1)
    return BuiltGraph(cg, sv)


@contextmanager
def assert_graph_unchanged(cg):
    """Assert the (rejected) edit run inside the block leaves every stored row untouched."""
    res_old = cg.client.read_all_rows()
    res_old.consume_all()
    yield
    res_new = cg.client.read_all_rows()
    res_new.consume_all()
    assert res_new.rows == res_old.rows


class RowKeyLockRegistry:
    """Thread-safe in-memory stand-in for kvdbclient's row-key lock API.

    Matches the full `cg.client.lock_by_row_key*` / `unlock_by_row_key*`
    / `renew_lock_by_row_key` surface — including the indefinite-column
    variants — so row-key-based lock primitives (DownsampleBlockLock,
    L2ChunkLock, IndefiniteL2ChunkLock, …) can be exercised without a
    bigtable emulator.

    Two separate maps, one per column. The "with_indefinite" temporal
    acquire refuses if either map holds the row, mirroring the filter
    union that `lock_by_row_key_with_indefinite` uses on bigtable.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._held = {}
        self._held_indefinite = {}

    def lock_by_row_key(self, row_key, operation_id):
        with self._lock:
            if row_key in self._held:
                return False
            self._held[row_key] = operation_id
            return True

    def lock_by_row_key_with_indefinite(self, row_key, operation_id):
        with self._lock:
            if row_key in self._held or row_key in self._held_indefinite:
                return False
            self._held[row_key] = operation_id
            return True

    def lock_by_row_key_indefinitely(self, row_key, operation_id):
        with self._lock:
            if row_key in self._held_indefinite:
                return False
            self._held_indefinite[row_key] = operation_id
            return True

    def unlock_by_row_key(self, row_key, operation_id):
        with self._lock:
            if self._held.get(row_key) == operation_id:
                del self._held[row_key]
                return True
            return False

    def unlock_indefinitely_locked_by_row_key(self, row_key, operation_id):
        with self._lock:
            if self._held_indefinite.get(row_key) == operation_id:
                del self._held_indefinite[row_key]
                return True
            return False

    def renew_lock_by_row_key(self, row_key, operation_id):
        with self._lock:
            return self._held.get(row_key) == operation_id


def make_cg_with_row_key_lock_registry(registry: RowKeyLockRegistry):
    """Attach a `RowKeyLockRegistry` to a `MagicMock` cg.client."""
    cg = MagicMock()
    cg.client.lock_by_row_key = registry.lock_by_row_key
    cg.client.lock_by_row_key_with_indefinite = registry.lock_by_row_key_with_indefinite
    cg.client.lock_by_row_key_indefinitely = registry.lock_by_row_key_indefinitely
    cg.client.unlock_by_row_key = registry.unlock_by_row_key
    cg.client.unlock_indefinitely_locked_by_row_key = (
        registry.unlock_indefinitely_locked_by_row_key
    )
    cg.client.renew_lock_by_row_key = registry.renew_lock_by_row_key
    return cg
