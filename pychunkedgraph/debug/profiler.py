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


@dataclass
class BlockMetrics:
    """Per-block metrics captured by HierarchicalProfiler.profile()."""

    path: str
    elapsed_s: float
    py_heap_peak_bytes: int = 0
    rss_start_bytes: int = 0
    rss_peak_bytes: int = 0
    counter_deltas: Dict[str, int] = field(default_factory=dict)


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

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.stack: List[Tuple[str, float]] = []
        self.current_path: List[str] = []
        self.blocks: List[BlockMetrics] = []

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        with_memory: bool = False,
        with_rss: bool = False,
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
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
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
                )
            )

    def print_report(self, operation_id=None):
        """Print a detailed timing breakdown."""
        if not self.enabled or not self.timings:
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

    def metrics_report(self, operation_id=None) -> None:
        """Print a fixed-width table over self.blocks (memory + IO view).

        Columns: path, elapsed_s, py_peak_MB, rss_start_MB, rss_peak_MB,
        plus one column per counter key that appears in any block.
        Counterpart to print_report (which covers timing-only).
        """
        if not self.enabled or not self.blocks:
            return

        counter_keys: List[str] = []
        seen_keys: set = set()
        for b in self.blocks:
            for k in b.counter_deltas:
                if k not in seen_keys:
                    seen_keys.add(k)
                    counter_keys.append(k)

        cols = [
            "path",
            "elapsed_s",
            "py_peak_MB",
            "rss_start_MB",
            "rss_peak_MB",
        ] + counter_keys
        widths = [len(c) for c in cols]
        rows: List[List[str]] = []
        for b in self.blocks:
            row = [
                b.path,
                f"{b.elapsed_s:.3f}",
                f"{b.py_heap_peak_bytes / 1e6:.1f}",
                f"{b.rss_start_bytes / 1e6:.1f}",
                f"{b.rss_peak_bytes / 1e6:.1f}",
            ]
            for k in counter_keys:
                row.append(str(b.counter_deltas.get(k, 0)))
            rows.append(row)
        for row in rows:
            for i, v in enumerate(row):
                if len(v) > widths[i]:
                    widths[i] = len(v)

        def fmt(values: List[str]) -> str:
            return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

        print("\n" + "=" * 80)
        print(
            f"METRICS REPORT{f' (operation_id={operation_id})' if operation_id else ''}"
        )
        print("=" * 80)
        print(fmt(cols))
        print(fmt(["-" * w for w in widths]))
        for row in rows:
            print(fmt(row))
        print("=" * 80 + "\n")

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()
        self.stack.clear()
        self.current_path.clear()
        self.blocks.clear()


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
