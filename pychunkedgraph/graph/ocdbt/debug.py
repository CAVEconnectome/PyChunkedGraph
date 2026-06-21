"""Diagnostic plumbing for OCDBT failures.

Humanize-count for log lines, generic failure envelope (host/pod/versions/
traceback/timestamp), bbox-failure payload builder, and a GCS dump helper
that writes per-task forensic JSON under ``$ERROR_DUMP/<dump_tag>__<utc>.json``.
Kept out of ``main.py`` and ``utils.py`` so the core OCDBT code stays
free of import bloat that's only used on failure paths.
"""

import json
import logging
import os
import socket
import sys
import traceback
from datetime import datetime, timezone
from os import environ
from typing import Optional

import tensorstore as ts

_logger = logging.getLogger(__name__)


def humanize_count(n: int) -> str:
    """Compact count for log lines: 1234567 → '1.2M', 950 → '950'."""
    for unit, scale in (("G", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return str(n)


def failure_envelope(exc: BaseException, dump_tag: Optional[str]) -> dict:
    """Generic metadata for any failure dump — host, pod, versions,
    timestamp, traceback, coordinator env. Caller merges with the
    failure-specific fields to build the final payload.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dump_tag": dump_tag,
        "host": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "pod_name": environ.get("MY_POD_NAME"),
            "pod_ip": environ.get("MY_POD_IP"),
            "node_name": environ.get("MY_NODE_NAME"),
        },
        "versions": {
            "tensorstore": getattr(ts, "__version__", None),
            "python": sys.version,
        },
        "ocdbt_coordinator_env": {
            "OCDBT_COORDINATOR_HOST": environ.get("OCDBT_COORDINATOR_HOST"),
            "OCDBT_COORDINATOR_PORT": environ.get("OCDBT_COORDINATOR_PORT"),
        },
        "exception": {
            "type": type(exc).__name__,
            "module": type(exc).__module__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def bbox_failure_payload(
    exc: BaseException,
    dump_tag: Optional[str],
    bbox_lo,
    bbox_hi,
    resolutions,
    per_scale,
    dst_handle,
    src_handle,
) -> dict:
    """Build the full diagnostic dict for a ``copy_ws_bbox_multiscale``
    commit failure.

    Merges generic ``failure_envelope`` metadata with bbox-specific
    fields (per-scale shape / chunk / key-count / raw-bytes, src+dst
    kvstore specs). Spec dumps are wrapped in try/except so a malformed
    handle doesn't shadow the original exception.
    """
    try:
        dst_spec = dst_handle.kvstore.spec().to_json()
    except Exception as e:
        dst_spec = f"<spec-dump-error: {e!r}>"
    try:
        src_spec = src_handle.kvstore.spec().to_json()
    except Exception as e:
        src_spec = f"<spec-dump-error: {e!r}>"
    total_voxels = sum(p[2] for p in per_scale)
    total_raw = sum(p[3] for p in per_scale)
    total_keys = sum(p[5] for p in per_scale)
    return {
        **failure_envelope(exc, dump_tag),
        "bbox_lo": [int(c) for c in bbox_lo],
        "bbox_hi": [int(c) for c in bbox_hi],
        "resolutions": [list(map(int, r)) for r in resolutions],
        "n_scales": len(per_scale),
        "total_voxels": total_voxels,
        "total_raw_bytes": total_raw,
        "total_keys": total_keys,
        "per_scale": [
            {
                "scale_index": i,
                "dims": list(dims),
                "voxels": nvox,
                "raw_bytes": raw_bytes,
                "chunk_shape": list(chunk_shape),
                "n_keys": n_keys,
                "max_raw_per_key_bytes": max_per_key,
            }
            for i, dims, nvox, raw_bytes, chunk_shape, n_keys, max_per_key in per_scale
        ],
        "dst_kvstore_spec": dst_spec,
        "src_kvstore_spec": src_spec,
    }


def dump_failure_to_gcs(payload: dict, dump_tag: str) -> Optional[str]:
    """Write a per-task failure report to ``$ERROR_DUMP/<dump_tag>__<utc>.json``.

    Returns the full path or None (env unset, dump_tag empty, or write
    error). ``dump_tag`` carries the calling-context identifier (graph
    id, layer, coords, …) so multiple experiments can share one
    ``ERROR_DUMP`` bucket without collisions.
    """
    root = environ.get("ERROR_DUMP", "").strip()
    if not root or not dump_tag:
        return None
    if not root.endswith("/"):
        root += "/"
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    rel = f"{dump_tag}__{utc}.json"
    full = root + rel
    try:
        ts.KvStore.open(root).result().write(
            rel, json.dumps(payload, indent=2).encode("utf-8")
        ).result()
        return full
    except Exception as e:
        _logger.warning("failed to write ERROR_DUMP at %s: %r", full, e)
        return None
