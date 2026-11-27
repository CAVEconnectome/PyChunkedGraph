"""One timeline per edit request: its spans from the request's arrival to its answer.

`init_app` is the only call the app makes. With `PCG_PROFILER_ENABLED` unset it does
nothing. Set, it wraps the functions `spans.SPANS` names, opens a profiler scope around
each edit request, and writes the scope's blocks to the diagnostics store once the
response has been sent, so no answer waits on the bucket.

All of it is light by construction: a scope never traces allocations, starts one RSS
sampler thread per request and none per span, and outside an edit request a wrapped
call only reads one context variable. Thread tallies assume the process serves one
request at a time, as uWSGI's process workers do.
"""

import functools
import os
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Callable, NamedTuple, Optional

from flask import current_app, g, request

from pychunkedgraph import __version__, get_logger
from pychunkedgraph.utils import diagnostics

from . import main, spans
from .hierarchical import BlockMetrics

logger = get_logger(__name__)

# The views whose requests are edits, by the last part of their endpoint's name.
EDIT_VIEWS = frozenset(
    {
        "handle_merge",
        "handle_merge_admin",
        "handle_split",
        "handle_undo",
        "handle_redo",
        "handle_rollback",
    }
)
ROOT_BLOCK = "request"
# The diagnostics store's kind for an edit's timeline.
KIND = "timing"
VERSION = 1


class Block(NamedTuple):
    """One block of a timeline, as stored; RSS is 0 where the block read none."""

    path: str
    elapsed_s: float
    calls: int
    end_s: float
    rss_start_bytes: int
    rss_peak_bytes: int

    @classmethod
    def of(cls, block: BlockMetrics) -> "Block":
        return cls(
            block.path,
            round(block.elapsed_s, 6),
            block.call_count,
            round(block.wall_end_s, 6),
            block.rss_start_bytes,
            block.rss_peak_bytes,
        )


class Timeline(NamedTuple):
    """One edit request's record, as the diagnostics store holds it."""

    started: datetime
    graph_id: Optional[str]
    endpoint: str
    status: int
    operation_id: Optional[int]
    user_id: str
    blocks: tuple[Block, ...]
    tallies: tuple[main.Tally, ...]
    # False when another scope overlapped this one, so a pool thread's calls may be
    # missing from `tallies`.
    tallies_complete: bool

    @property
    def name(self) -> str:
        """The record's place in the store: its day, then its operation id, or its
        time and process when the request set none."""
        day = self.started.strftime("%Y-%m-%d")
        if self.operation_id is not None:
            return f"{day}/{self.operation_id}.json"
        return f"{day}/{self.started.strftime('%H%M%S%f')}_{os.getpid()}.json"

    def payload(self) -> dict[str, object]:
        """The record as JSON-ready values."""
        total = next((b.elapsed_s for b in self.blocks if b.path == ROOT_BLOCK), 0.0)
        return {
            "version": VERSION,
            "started": self.started.isoformat(),
            "graph_id": self.graph_id,
            "endpoint": self.endpoint,
            "status": self.status,
            "operation_id": self.operation_id,
            "user_id": self.user_id,
            "total_s": total,
            "pid": os.getpid(),
            "pcg_version": __version__,
            "blocks": [block._asdict() for block in self.blocks],
            "tallies": [
                {"name": t.name, "calls": t.calls, "elapsed_s": round(t.elapsed_s, 6)}
                for t in self.tallies
            ],
            "tallies_complete": self.tallies_complete,
        }


class _Open(NamedTuple):
    """An edit request in flight: its profiler, its scope and its entered root block."""

    profiler: main.ScopeProfiler
    scope: main.Scope
    root: AbstractContextManager
    started: datetime


def _begin() -> None:
    """Open a scope when the request is an edit; any other request is left alone."""
    if (request.endpoint or "").rpartition(".")[2] not in EDIT_VIEWS:
        return
    profiler = main.ScopeProfiler()
    # The root block is entered before the scope opens: a failure here leaves none open.
    root = profiler.profile(ROOT_BLOCK, with_rss=True)
    root.__enter__()
    scope = main.open_scope(profiler)
    g.pcg_timeline = _Open(profiler, scope, root, datetime.now(timezone.utc))


def _close(status: int) -> Optional[Timeline]:
    """End the request's scope, if it has one open, and return its timeline."""
    opened: Optional[_Open] = g.pop("pcg_timeline", None)
    if opened is None:
        return None
    try:
        opened.root.__exit__(None, None, None)
    finally:
        exclusive = main.close_scope(opened.scope)
    operation_id = opened.profiler.operation_id
    if operation_id is None:
        # Set by the merge and split handlers alone, and only once their edit is done.
        operation_id = getattr(current_app, "operation_id", None)
    return Timeline(
        started=opened.started,
        graph_id=getattr(current_app, "table_id", None),
        endpoint=request.endpoint or "",
        status=status,
        operation_id=None if operation_id is None else int(operation_id),
        user_id=str(getattr(current_app, "user_id", "")),
        blocks=tuple(Block.of(block) for block in opened.profiler.blocks),
        tallies=opened.profiler.tallies,
        tallies_complete=exclusive,
    )


def _dump(timeline: Timeline, graph_of: Callable) -> None:
    """Write a timeline beside its graph's meshes; with no such graph open in this
    process there is no store to write to."""
    cg = graph_of(timeline.graph_id)
    if cg is None:
        logger.warning(f"timeline dropped: no graph open for {timeline.graph_id}")
        return
    diagnostics.put_json(cg, KIND, timeline.name, timeline.payload())


def _safely(hook: Callable) -> Callable:
    """`hook`, unable to fail its request: a raise inside it is logged and swallowed,
    and its first argument, a response when it has one, is returned."""

    @functools.wraps(hook)
    def safe(*args):
        try:
            return hook(*args)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.warning(f"timeline hook {hook.__name__} failed: {err}")
            return args[0] if args else None

    return safe


def init_app(app, graph_of: Callable) -> bool:
    """Time every edit request `app` serves, when the profiler's switch is on.

    `graph_of` returns the graph this process holds open for a table id, or None.
    Returns whether the hooks were installed.
    """
    if not main.PROFILER_ENABLED:
        return False

    def end(response):
        """Close an edit's scope; its timeline is written once the response is sent."""
        timeline = _close(response.status_code)
        if timeline is not None:
            response.call_on_close(functools.partial(_dump, timeline, graph_of))
        return response

    def teardown(_error):
        """Close a scope `end` never reached, which an unhandled raise leaves open."""
        timeline = _close(500)
        if timeline is not None:
            _dump(timeline, graph_of)

    try:
        spans.install()
    except Exception as err:  # pylint: disable=broad-exception-caught
        spans.uninstall()
        logger.error(f"edit timelines stay off: the span table did not install: {err}")
        return False
    app.before_request(_safely(_begin))
    app.after_request(_safely(end))
    app.teardown_request(_safely(teardown))
    return True
