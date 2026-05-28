from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import os
import threading
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil


def _fmt_time(s: float) -> str:
    """Auto-scale seconds → ``12.3 ms`` / ``1.23 s``."""
    if s < 1.0:
        return f"{s * 1000:.1f} ms"
    return f"{s:.2f} s"


def _fmt_bytes(n: int, *, signed: bool = False) -> str:
    """Auto-scale bytes (binary) → ``512 B`` / ``1.5 KB`` / ``45.6 MB`` / ``1.23 GB``.

    With ``signed=True``, positive values get a ``+`` prefix (for delta columns).
    """
    if signed:
        sign = "+" if n > 0 else "-" if n < 0 else ""
    else:
        sign = "-" if n < 0 else ""
    n = abs(int(n))
    if n < 1024:
        return f"{sign}{n} B"
    if n < 1024**2:
        return f"{sign}{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{sign}{n / 1024**2:.1f} MB"
    return f"{sign}{n / 1024**3:.2f} GB"


def _fmt_count(n: int) -> str:
    """Thousands-separator integer: ``1234567`` → ``1,234,567``."""
    return f"{int(n):,}"


@dataclass
class BlockMetrics:
    """Per-block metrics captured by HierarchicalProfiler.profile()."""

    path: str
    elapsed_s: float
    py_heap_peak_bytes: int = 0
    rss_start_bytes: int = 0
    rss_peak_bytes: int = 0
    counter_deltas: Dict[str, int] = field(default_factory=dict)
    # Wall-clock time when this block finished, measured relative to
    # the first profile() entry since the profiler was reset.
    wall_end_s: float = 0.0


class _RSSSampler:
    """Daemon thread that samples process RSS and tracks the max.

    Internal to this module; not part of the profiler's public API.
    """

    def __init__(self, interval_s: float = 0.050):
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()
        self.start_rss = 0
        self.peak_rss = 0

    def start(self) -> None:
        self.start_rss = self._proc.memory_info().rss
        self.peak_rss = self.start_rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            self._stop.wait(self._interval)


class HierarchicalProfiler:
    """
    Hierarchical profiler for detailed timing breakdowns.
    Tracks timing at multiple levels and prints a breakdown at the end.

    Optional per-block memory + IO collection is opt-in via kwargs on
    profile(); see BlockMetrics for the captured fields.
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        with_memory: bool = True,
        with_rss: bool = True,
    ):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.stack: List[Tuple[str, float]] = []
        self.current_path: List[str] = []
        self.blocks: List[BlockMetrics] = []
        # perf_counter at the first profile() entry since reset.
        # Used to stamp each block's wall_end_s for cumulative-wall view.
        self._base_perf: Optional[float] = None
        # Per-instance defaults so inline profile() call sites stay short
        # (no per-block kwargs). Callers can override per-call.
        self.with_memory_default = with_memory
        self.with_rss_default = with_rss
        # Optional caller-set default counters dict used when profile()
        # is called without an explicit `counters=` kwarg. Lets inline
        # profile() blocks in production code pick up an outer harness's
        # IO counters without needing to thread them through.
        self.default_counters: Optional[Dict[str, int]] = None

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        with_memory: Optional[bool] = None,
        with_rss: Optional[bool] = None,
        counters: Optional[Dict[str, int]] = None,
    ):
        """Context manager for profiling a code block.

        Default behavior (no kwargs) records only timing into
        `self.timings` / `self.call_counts`, matching the original
        implementation.

        Optional kwargs collect extra metrics into `self.blocks`:
        - with_memory: tracemalloc Python heap peak per block.
        - with_rss: psutil RSS peak via a 50 ms sampler thread.
        - counters: caller-supplied dict; per-key deltas recorded
          (after - before for keys present at exit).
        """
        if not self.enabled:
            yield
            return

        if with_memory is None:
            with_memory = self.with_memory_default
        if with_rss is None:
            with_rss = self.with_rss_default
        if counters is None:
            counters = self.default_counters

        full_path = ".".join(self.current_path + [name])
        self.current_path.append(name)

        started_tracemalloc = False
        if with_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                started_tracemalloc = True
            tracemalloc.reset_peak()

        sampler: Optional[_RSSSampler] = None
        if with_rss:
            sampler = _RSSSampler()
            sampler.start()

        counters_before = dict(counters) if counters is not None else None

        start_time = time.perf_counter()
        if self._base_perf is None:
            self._base_perf = start_time
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            self.timings[full_path].append(elapsed)
            self.call_counts[full_path] += 1
            self.current_path.pop()

            py_peak = 0
            if with_memory:
                _curr, py_peak = tracemalloc.get_traced_memory()
                if started_tracemalloc:
                    tracemalloc.stop()

            rss_start = 0
            rss_peak = 0
            if sampler is not None:
                sampler.stop()
                rss_start = sampler.start_rss
                rss_peak = sampler.peak_rss

            counter_deltas: Dict[str, int] = {}
            if counters is not None and counters_before is not None:
                for k, v_after in counters.items():
                    counter_deltas[k] = v_after - counters_before.get(k, 0)

            self.blocks.append(
                BlockMetrics(
                    path=full_path,
                    elapsed_s=elapsed,
                    py_heap_peak_bytes=int(py_peak),
                    rss_start_bytes=int(rss_start),
                    rss_peak_bytes=int(rss_peak),
                    counter_deltas=counter_deltas,
                    wall_end_s=end_time - self._base_perf,
                )
            )

    def print_report(self, operation_id=None):
        """Print a detailed timing breakdown."""
        if not self.timings:
            return

        print("\n" + "=" * 80)
        print(
            f"PROFILER REPORT{f' (operation_id={operation_id})' if operation_id else ''}"
        )
        print("=" * 80)

        # Group by depth level
        by_depth: Dict[int, List[Tuple[str, float, int]]] = defaultdict(list)
        for path, times in self.timings.items():
            depth = path.count(".")
            total_time = sum(times)
            count = self.call_counts[path]
            by_depth[depth].append((path, total_time, count))

        # Sort each level by total time
        for depth in sorted(by_depth.keys()):
            items = sorted(by_depth[depth], key=lambda x: -x[1])
            for path, total_time, count in items:
                indent = "  " * depth
                avg_time = total_time / count if count > 0 else 0
                if count > 1:
                    print(
                        f"{indent}{path}: {total_time*1000:.2f}ms total "
                        f"({count} calls, {avg_time*1000:.2f}ms avg)"
                    )
                else:
                    print(f"{indent}{path}: {total_time*1000:.2f}ms")

        # Print summary
        print("-" * 80)
        top_level_total = sum(
            sum(times) for path, times in self.timings.items() if "." not in path
        )
        print(f"Total top-level time: {top_level_total*1000:.2f}ms")

        # Print top 10 slowest operations
        print("\nTop 10 slowest operations:")
        all_ops = [
            (path, sum(times), self.call_counts[path])
            for path, times in self.timings.items()
        ]
        all_ops.sort(key=lambda x: -x[1])
        for i, (path, total_time, count) in enumerate(all_ops[:10]):
            pct = (total_time / top_level_total * 100) if top_level_total > 0 else 0
            print(f"  {i+1}. {path}: {total_time*1000:.2f}ms ({pct:.1f}%)")

        print("=" * 80 + "\n")

    # Counter keys that are uninformative for the SV-split flow and
    # only add visual noise to the report.
    _SKIP_COUNTERS = ("ocdbt_reads",)

    def metrics_report(self, operation_id=None) -> None:
        """Print a compact, human-readable table over self.blocks.

        Columns: stage, wall, cum_wall, py_peak, rss_start, rss_peak,
        rss_Δ (signed), plus one column per counter key that has a
        non-zero value in at least one block. ``cum_wall`` is wall
        time elapsed from the first ``profile()`` block since reset.
        rss_start / rss_peak are absolute process RSS; rss_Δ is the
        new-allocation delta inside the block.
        """
        if not self.blocks:
            return

        # Collect counter keys in first-seen order; drop ones that are
        # zero in every block (e.g., bt_log_reads is usually 0 for SV
        # splits and only adds noise) or in the static skip list.
        counter_keys: List[str] = []
        seen_keys: set = set()
        for b in self.blocks:
            for k in b.counter_deltas:
                if k not in seen_keys:
                    seen_keys.add(k)
                    counter_keys.append(k)
        counter_keys = [
            k
            for k in counter_keys
            if k not in self._SKIP_COUNTERS
            and any(b.counter_deltas.get(k, 0) for b in self.blocks)
        ]

        cols = [
            "stage",
            "wall",
            "cum_wall",
            "py_peak",
            "rss_start",
            "rss_peak",
            "rss_Δ",
        ] + counter_keys

        def fmt_counter(key: str, val: int) -> str:
            if key.endswith("_bytes"):
                return _fmt_bytes(val)
            return _fmt_count(val)

        rows: List[List[str]] = []
        for b in self.blocks:
            row = [
                b.path,
                _fmt_time(b.elapsed_s),
                _fmt_time(getattr(b, "wall_end_s", 0.0)),
                _fmt_bytes(b.py_heap_peak_bytes),
                _fmt_bytes(b.rss_start_bytes),
                _fmt_bytes(b.rss_peak_bytes),
                _fmt_bytes(b.rss_peak_bytes - b.rss_start_bytes, signed=True),
            ]
            for k in counter_keys:
                row.append(fmt_counter(k, b.counter_deltas.get(k, 0)))
            rows.append(row)

        widths = [len(c) for c in cols]
        for row in rows:
            for i, v in enumerate(row):
                if len(v) > widths[i]:
                    widths[i] = len(v)

        def line(values: List[str]) -> str:
            return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

        title = "metrics report"
        if operation_id is not None:
            title = f"{title} (operation_id={operation_id})"
        print(title)
        print(line(cols))
        print(line(["-" * w for w in widths]))
        for row in rows:
            print(line(row))

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()
        self.stack.clear()
        self.current_path.clear()
        self.blocks.clear()
        self._base_perf = None


# Global profiler instance - enable via environment variable
PROFILER_ENABLED = os.environ.get("PCG_PROFILER_ENABLED", "0") == "1"
_profiler: HierarchicalProfiler = None


def get_profiler() -> HierarchicalProfiler:
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = HierarchicalProfiler(enabled=PROFILER_ENABLED)
    return _profiler


def reset_profiler():
    """Reset the global profiler."""
    global _profiler
    if _profiler is not None:
        _profiler.reset()
