"""Best-effort error artifact dumps for edit operations.

Writes a JSON blob per failed operation to
`{WATERSHED}/graphene_errors/{cg.graph_id}/{op_id}.json` so logs stay
concise while the full payload (ids, edges, traceback) survives for
later inspection. `read_err_artifact(cg, op_id)` reads it back.
"""

import traceback
from datetime import datetime

import numpy as np
from cloudfiles import CloudFiles

from pychunkedgraph import get_logger

logger = get_logger(__name__)

_DUMP_ATTRS = (
    "source_ids",
    "sink_ids",
    "source_coords",
    "sink_coords",
    "added_edges",
    "removed_edges",
    "atomic_edges",
    "affinities",
    "bbox_offset",
)


def _json_safe(obj):
    """Recursively coerce numpy + datetime into JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _err_dir(cg):
    """The `{WATERSHED}/graphene_errors/{cg.graph_id}` base; None if WATERSHED unset."""
    ws = getattr(cg.meta.data_source, "WATERSHED", None)
    if not ws:
        return None
    return f"{ws.rstrip('/')}/graphene_errors/{cg.graph_id}"


def payload_summary(op) -> str:
    """One-line operation source/sink summary for diagnostic log lines."""
    src = getattr(op, "source_ids", None)
    snk = getattr(op, "sink_ids", None)
    if src is None and snk is None:
        return ""
    return (
        f" source_ids={src.tolist() if src is not None else None}"
        f" sink_ids={snk.tolist() if snk is not None else None}"
    )


def build_err_payload(op, op_id, err) -> dict:
    """Structured operation snapshot capturing inputs + traceback for replay."""
    payload = {
        "op_type": type(op).__name__,
        "op_id": int(op_id),
        "exception_class": type(err).__name__,
        "exception_message": str(err),
        "traceback": traceback.format_exc(),
        "user_id": getattr(op, "user_id", None),
        "parent_ts": getattr(op, "parent_ts", None),
    }
    for attr in _DUMP_ATTRS:
        val = getattr(op, attr, None)
        if val is not None:
            payload[attr] = val
    return payload


def dump_err_artifact(cg, op_id, payload):
    """Write `{err_dir}/{op_id}.json`. Returns URL or None; never raises."""
    cf_dir = _err_dir(cg)
    if cf_dir is None:
        return None
    try:
        filename = f"{op_id}.json"
        CloudFiles(cf_dir).put_json(filename, _json_safe(payload))
        return f"{cf_dir}/{filename}"
    except Exception as e:
        logger.warning(f"err_dump failed for op={op_id}: {e}")
        return None


def read_err_artifact(cg, op_id):
    """Return the artifact dict for this `op_id`, or None if missing."""
    cf_dir = _err_dir(cg)
    if cf_dir is None:
        return None
    try:
        return CloudFiles(cf_dir).get_json(f"{op_id}.json")
    except Exception as e:
        logger.warning(f"err_dump read failed for op={op_id}: {e}")
        return None
