import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import NamedTuple, Optional

from .hierarchical import BlockMetrics, HierarchicalProfiler

PROFILER_ENABLED = os.environ.get("PCG_PROFILER_ENABLED", "0") == "1"
_profiler: HierarchicalProfiler = None


class Tally(NamedTuple):
    """The calls one span name got from threads other than its scope's owner."""

    name: str
    calls: int
    elapsed_s: float


class Entered(NamedTuple):
    """An open span, as `ScopeProfiler.leave` needs it back."""

    start: float
    rss: int
    owned: bool


class ScopeProfiler(HierarchicalProfiler):
    """The profiler of one scope, such as a request's: timing, and RSS only where a
    block asks for it.

    The owner's thread builds the path tree. A block from any other thread goes to a
    flat tally, since it must never touch the owner's open path.
    """

    def __init__(self):
        super().__init__(enabled=True, with_memory=False, with_rss=False)
        self._owner = threading.get_ident()
        self._pid = os.getpid()
        self._tally_lock = threading.Lock()
        self._tallies: dict[str, Tally] = {}
        # The operation the scope's edit runs as, once its lock has given it an id.
        self.operation_id: Optional[int] = None

    @property
    def tallies(self) -> tuple[Tally, ...]:
        """Every tally so far, in first-seen order."""
        with self._tally_lock:
            return tuple(self._tallies.values())

    def _tally(self, name: str, elapsed: float) -> None:
        with self._tally_lock:
            seen = self._tallies.get(name, Tally(name, 0, 0.0))
            self._tallies[name] = Tally(name, seen.calls + 1, seen.elapsed_s + elapsed)

    def enter(self, name: str, memory: bool) -> Entered:
        """Open a span; `leave` takes back what this returns. The cheap path of a
        wrapped call: no record is made until the span's path is first left."""
        owned = threading.get_ident() == self._owner
        # Read before the path is touched: a failed read must leave it as it was.
        rss = self.sample_rss() if memory and owned else 0
        if owned:
            self.current_path.append(name)
        start = time.perf_counter()
        if owned and self._base_perf is None:
            self._base_perf = start
        return Entered(start, rss, owned)

    def leave(self, name: str, entered: Entered) -> None:
        """Close a span, folding it into its path's block as `_record` folds one."""
        end = time.perf_counter()
        if not entered.owned:
            self._tally(name, end - entered.start)
            return
        path = ".".join(self.current_path)
        self.current_path.pop()
        block = self._agg.get(path)
        if block is None:
            block = BlockMetrics(path=path, elapsed_s=0.0, call_count=0, pid=self._pid)
            self._agg[path] = block
            self._order.append(path)
        block.elapsed_s += end - entered.start
        block.call_count += 1
        block.wall_end_s = end - self._base_perf
        if entered.rss:
            block.rss_peak_bytes = max(
                block.rss_peak_bytes, entered.rss, self.sample_rss()
            )
            block.rss_start_bytes = min(
                block.rss_start_bytes or entered.rss, entered.rss
            )

    @contextmanager
    def profile(self, name: str, **kwargs):
        """Time a block written as a `with`: the owner's through the full block path,
        any other thread's into the tally."""
        if threading.get_ident() == self._owner:
            with super().profile(name, **kwargs):
                yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._tally(name, time.perf_counter() - start)


class ProcessProfiler(HierarchicalProfiler):
    """The process-wide profiler `get_profiler` hands out.

    A block opened while a scope is open belongs to that scope, whichever profiler
    object the caller holds: a module that bound this instance at import still lands
    in the scope, timed only, and never in this profiler's own memory tracing.
    """

    def profile(self, name: str, **kwargs):
        scope = active_scope()
        if scope is not None:
            return scope.profile(name)
        return super().profile(name, **kwargs)


# The profiler of the scope open in this context, such as one request's.
_scoped: ContextVar[Optional[ScopeProfiler]] = ContextVar(
    "pcg_profiler_scope", default=None
)
# The scope a pool thread records into, since a new thread starts with an empty
# context: the process's one open scope, None while none or several are open.
_shared: Optional[ScopeProfiler] = None
_scope_lock = threading.Lock()
_open_scopes = 0
_overlaps = 0


class Scope(NamedTuple):
    """An open scope, as `close_scope` needs it."""

    token: Token
    overlaps: int
    alone: bool


def get_profiler() -> HierarchicalProfiler:
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = ProcessProfiler(enabled=PROFILER_ENABLED)
    return _profiler


def reset_profiler():
    """Reset the global profiler."""
    global _profiler
    if _profiler is not None:
        _profiler.reset()


def active_scope() -> Optional[ScopeProfiler]:
    """The scope a call records into: this context's, else the process's only one."""
    scoped = _scoped.get()
    return scoped if scoped is not None else _shared


def open_scope(profiler: ScopeProfiler) -> Scope:
    """Make `profiler` this context's scope until `close_scope` takes it back."""
    global _shared, _open_scopes, _overlaps
    with _scope_lock:
        alone = _open_scopes == 0
        if not alone:
            _overlaps += 1
        _open_scopes += 1
        _shared = profiler if alone else None
        return Scope(_scoped.set(profiler), _overlaps, alone)


def close_scope(scope: Scope) -> bool:
    """End a scope. True when no other scope was open at any time in its life, so
    every call a pool thread made in that time was recorded into it."""
    global _shared, _open_scopes
    with _scope_lock:
        _open_scopes -= 1
        _shared = None
        try:
            _scoped.reset(scope.token)
        except ValueError:
            # A token made in another context cannot be reset here; clear instead.
            _scoped.set(None)
        return scope.alone and scope.overlaps == _overlaps
