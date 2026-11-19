"""Refusals for a fork's writes that land outside its table."""

from ..graph.exceptions import Forbidden


def refuse_on_fork(cg, action: str) -> None:
    """Raise `Forbidden` when `cg` is a fork, which writes nothing outside its table."""
    if cg.client.marker is not None:
        raise Forbidden(f"{cg.graph_id} is a read-only fork: {action} is refused")
