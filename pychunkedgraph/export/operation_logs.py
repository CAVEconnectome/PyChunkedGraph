from typing import Optional
from typing import Iterable
from datetime import datetime


from ..graph import ChunkedGraph
from ..graph.attributes import OperationLogs


def _parse_attr(attr, val) -> str:
    try:
        if isinstance(val, str):
            return (attr.key, val)
        return (attr.key, attr.serialize(val))
    except AttributeError:
        return (attr, val)


def get_parsed_logs(
    cg: ChunkedGraph, start_time: Optional[datetime] = None
) -> Iterable[dict]:
    """Parse logs for compatibility with destination platform."""
    logs = cg.client.read_log_entries(start_time=start_time)
    result = []
    for _id, _log in logs.items():
        log = {"id": int(_id)}
        for attr, val in _log.items():
            attr, val = _parse_attr(attr, val)
            log[attr] = val
        log["type"] = "Merge" if OperationLogs.AddedEdge.key in log else "Split"
        result.append(log)
    return result
