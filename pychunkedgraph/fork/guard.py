"""Refusals for a read-only fork's writes that land outside its table."""

from kvdbclient.fork.client import ForkClient

from ..graph.exceptions import Forbidden


def refuse_on_fork(cg, action: str) -> None:
    """Raise `Forbidden` when `cg` is a fork.

    A fork writes nothing until fork edits land; its client refuses only table writes.
    """
    if isinstance(cg.client, ForkClient) and cg.client.marker is not None:
        raise Forbidden(f"{cg.graph_id} is a read-only fork: {action} is refused")
