"""Best-effort error artifact dumps for edit operations.

Writes a JSON blob per failed operation to the diagnostics store, as
`graphene_diagnostics/{cg.graph_id}/errors/{op_id}.json` beside the graph's
mesh dir, so logs stay concise while the full payload (ids, edges,
traceback) survives for later inspection. `read_err_artifact(cg, op_id)`
reads it back.
"""

import traceback
from datetime import datetime

import numpy as np

from pychunkedgraph import get_logger
from pychunkedgraph.utils import diagnostics

logger = get_logger(__name__)

# The diagnostics store's kind for a failed operation's dump.
ERRORS = "errors"

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
    """Write the operation's `errors` artifact. Returns URL or None; never raises."""
    try:
        safe = _json_safe(payload)
    except Exception as e:
        logger.warning(f"err_dump failed for op={op_id}: {e}")
        return None
    return diagnostics.put_json(cg, ERRORS, f"{op_id}.json", safe)


def read_err_artifact(cg, op_id):
    """Return the artifact dict for this `op_id`, or None if missing."""
    return diagnostics.get_json(cg, ERRORS, f"{op_id}.json")
