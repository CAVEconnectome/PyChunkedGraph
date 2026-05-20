"""Internal helpers for the OCDBT package.

Path builders, schema extraction, populate-marker IO, layer-bbox math.
Not part of the public API except for the marker IO and ``_layer_bbox``
which the ingest worker uses across the package boundary.
"""

import json
from typing import Optional

import numpy as np
import tensorstore as ts


def _ensure_trailing_slash(path: str) -> str:
    """Ensure kvstore paths end with / so they're treated as directories."""
    return path if path.endswith("/") else path + "/"


def _base_ocdbt_path(ws_path: str) -> str:
    return _ensure_trailing_slash(f"{ws_path.rstrip('/')}/ocdbt/base")


def _populate_markers_path(ws_path: str) -> str:
    return _ensure_trailing_slash(f"{ws_path.rstrip('/')}/ocdbt/.populated")


def _marker_key(layer: int, coords) -> str:
    return f"l{int(layer)}_{int(coords[0])}_{int(coords[1])}_{int(coords[2])}"


def _read_source_scales(ws_path: str):
    """Read the source precomputed ``info`` JSON to get scale count and resolutions.

    The leading '/' in '/info' is required for GCS — without it the read
    returns empty.
    """
    kvs = ts.KvStore.open(ws_path).result()
    info = json.loads(kvs.read("/info").result().value)
    return info["scales"]


def _open_precomputed_scale(
    kvstore, scale_index: int, create: bool = False, **schema_kw
):
    """Open one neuroglancer_precomputed scale on top of a kvstore spec."""
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": kvstore,
        "scale_index": scale_index,
    }
    return ts.open(spec, create=create, **schema_kw).result()


def _schema_from_src(src_handle) -> dict:
    """Extract the schema kwargs needed to open a matching destination.

    ``domain`` already carries both extent and origin (voxel_offset). Passing
    ``shape`` alongside conflicts with non-zero-origin sources because shape
    implies origin=0 — tensorstore refuses to merge ``[0, N)`` with
    ``[offset, offset+N)``.
    """
    s = src_handle.schema
    return dict(
        rank=s.rank,
        dtype=s.dtype,
        codec=s.codec,
        domain=s.domain,
        chunk_layout=s.chunk_layout,
        dimension_units=s.dimension_units,
    )


def is_chunk_populated(ws_path: str, layer: int, coords) -> bool:
    """Check whether this chunk's precomputed→OCDBT copy has already completed.

    Markers live outside the OCDBT keyspace at
    ``<ws>/ocdbt/.populated/l<layer>_<x>_<y>_<z>`` so retried ingest tasks
    don't re-copy chunks and bloat the database with redundant versioned
    writes.
    """
    kvs = ts.KvStore.open(_populate_markers_path(ws_path)).result()
    result = kvs.read(_marker_key(layer, coords)).result()
    return result.value is not None and len(result.value) > 0


def mark_chunk_populated(ws_path: str, layer: int, coords) -> None:
    """Record that this chunk's precomputed→OCDBT copy completed."""
    kvs = ts.KvStore.open(_populate_markers_path(ws_path)).result()
    kvs.write(_marker_key(layer, coords), b"1").result()


def read_populate_meta(ws_path: str) -> Optional[dict]:
    """Return the per-base populate config dict, or None if not yet written."""
    kvs = ts.KvStore.open(_populate_markers_path(ws_path)).result()
    r = kvs.read("meta.json").result()
    if r.value is None or len(r.value) == 0:
        return None
    return json.loads(r.value)


def write_populate_meta(ws_path: str, meta: dict) -> None:
    """Persist the per-base populate config (layer, etc.) alongside markers."""
    kvs = ts.KvStore.open(_populate_markers_path(ws_path)).result()
    kvs.write("meta.json", json.dumps(meta).encode()).result()


def base_exists(ws_path: str) -> bool:
    """Check if the base OCDBT has already been created for this watershed."""
    base = _base_ocdbt_path(ws_path)
    kvs = ts.KvStore.open(base).result()
    result = kvs.read("manifest.ocdbt").result()
    return result.value is not None and len(result.value) > 0


def fork_exists(ws_path: str, graph_id: str) -> bool:
    """Check if this ChunkedGraph's fork has been initialized."""
    fork_dir = _ensure_trailing_slash(f"{ws_path.rstrip('/')}/ocdbt/{graph_id}")
    kvs = ts.KvStore.open(fork_dir).result()
    result = kvs.read("manifest.ocdbt").result()
    return result.value is not None and len(result.value) > 0


def _layer_bbox(meta, layer: int, coords) -> tuple:
    """Base-resolution voxel bbox of a chunk at this layer."""
    chunk_size = np.array(meta.graph_config.CHUNK_SIZE, dtype=int)
    layer_chunk_size = chunk_size * (1 << (layer - 2))
    coords = np.array(coords, dtype=int)
    vol_start = meta.voxel_bounds[:, 0]
    vol_end = meta.voxel_bounds[:, 1]
    lo = coords * layer_chunk_size + vol_start
    hi = np.minimum(lo + layer_chunk_size, vol_end)
    return lo, hi
