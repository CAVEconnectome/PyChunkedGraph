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

from .utils import _fmt_time, _fmt_bytes, _fmt_count


@dataclass
class BlockMetrics:
    """Per-block metrics captured by HierarchicalProfiler.profile()."""

    path: str
    elapsed_s: float
    # Number of profile() calls folded into this block (1 per call, summed).
    call_count: int = 1
    py_heap_peak_bytes: int = 0
    rss_start_bytes: int = 0
    rss_peak_bytes: int = 0
    counter_deltas: Dict[str, int] = field(default_factory=dict)
    # Wall-clock time when this block finished, measured relative to
    # the first profile() entry since the profiler was reset.
    wall_end_s: float = 0.0
    # Process that recorded this block; lets a cross-process rollup compute
    # per-worker lifetime peak RSS. 0 when unset (single-process use).
    pid: int = 0


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
        self.current_path: List[str] = []
        # One folded BlockMetrics per path (bounded by distinct paths, not call
        # count) so a hot per-call loop never grows an unbounded block list.
        self._agg: Dict[str, BlockMetrics] = {}
        self._order: List[str] = []
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
        self._proc: Optional[psutil.Process] = None

    @property
    def blocks(self) -> List[BlockMetrics]:
        """One folded block per path, in first-seen order."""
        return [self._agg[p] for p in self._order]

    def __getstate__(self):
        # _proc is a live psutil.Process handle — not picklable / not portable.
        state = self.__dict__.copy()
        state["_proc"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _record(self, block: BlockMetrics) -> None:
        """Fold ``block`` into the per-path aggregate (sum elapsed/counters, max
        peaks, min-nonzero rss_start). Same math as ``from_blocks`` so in-process
        and cross-process folds compose."""
        cur = self._agg.get(block.path)
        if cur is None:
            self._order.append(block.path)
            self._agg[block.path] = block
            return
        cur.elapsed_s += block.elapsed_s
        cur.call_count += block.call_count
        cur.py_heap_peak_bytes = max(cur.py_heap_peak_bytes, block.py_heap_peak_bytes)
        cur.rss_peak_bytes = max(cur.rss_peak_bytes, block.rss_peak_bytes)
        if block.rss_start_bytes:
            cur.rss_start_bytes = (
                block.rss_start_bytes
                if not cur.rss_start_bytes
                else min(cur.rss_start_bytes, block.rss_start_bytes)
            )
        cur.wall_end_s = block.wall_end_s
        for k, v in block.counter_deltas.items():
            cur.counter_deltas[k] = cur.counter_deltas.get(k, 0) + v

    def sample_rss(self) -> int:
        """Current process RSS in bytes, or 0 when profiling is disabled. Gated
        here so hot-loop call sites need no ``if enabled`` of their own; a
        disabled profiler makes this a no-op like ``profile()``."""
        if not self.enabled:
            return 0
        if self._proc is None:
            self._proc = psutil.Process()
        return self._proc.memory_info().rss

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        with_memory: Optional[bool] = None,
        with_rss: Optional[bool] = None,
        sampled_rss: bool = False,
        counters: Optional[Dict[str, int]] = None,
    ):
        """Context manager for profiling a code block.

        Default behavior (no kwargs) records only timing into
        `self.timings` / `self.call_counts`, matching the original
        implementation.

        Optional kwargs collect extra metrics into `self.blocks`:
        - with_memory: tracemalloc Python heap peak per block.
        - with_rss: psutil RSS peak via a 50 ms sampler thread.
        - sampled_rss: take RSS at enter/exit only (two memory_info calls, no
          thread) — for hot, tight blocks where a sampler thread per call would
          thrash; ``rss_peak`` is then max(enter, exit), not a true peak.
        - counters: caller-supplied dict; per-key deltas recorded
          (after - before for keys present at exit).

        Disabled (or when nested under a disabled profiler) this is a no-op,
        so call sites need no ``if enabled`` guards.
        """
        if not self.enabled:
            yield
            return

        if with_memory is None:
            with_memory = self.with_memory_default
        if with_rss is None and not sampled_rss:
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
        rss_enter = self.sample_rss() if sampled_rss else 0

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
            elif sampled_rss:
                rss_start = rss_enter
                rss_peak = max(rss_enter, self.sample_rss())

            counter_deltas: Dict[str, int] = {}
            if counters is not None and counters_before is not None:
                for k, v_after in counters.items():
                    counter_deltas[k] = v_after - counters_before.get(k, 0)

            self._record(
                BlockMetrics(
                    path=full_path,
                    elapsed_s=elapsed,
                    py_heap_peak_bytes=int(py_peak),
                    rss_start_bytes=int(rss_start),
                    rss_peak_bytes=int(rss_peak),
                    counter_deltas=counter_deltas,
                    wall_end_s=end_time - self._base_perf,
                    pid=os.getpid(),
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

    @staticmethod
    def _tree_preorder(paths, sort_key):
        """Parent-first pre-order over dotted ``paths`` (e.g. ``stitch.decode``),
        siblings and roots ordered by ``sort_key`` (a ``path -> comparable``).
        Returns the ordered path list — the shared layout for both reports."""
        children: Dict[str, List[str]] = defaultdict(list)
        roots: List[str] = []
        for path in paths:
            if "." in path:
                children[path.rsplit(".", 1)[0]].append(path)
            else:
                roots.append(path)
        roots.sort(key=sort_key)
        for kids in children.values():
            kids.sort(key=sort_key)

        ordered: List[str] = []

        def _visit(path):
            ordered.append(path)
            for child in children.get(path, []):
                _visit(child)

        for root in roots:
            _visit(root)
        return ordered

    @staticmethod
    def _print_tree_table(title, ordered_paths, metric_headers, metric_cells):
        """Render a table whose row labels are ``ordered_paths`` split into one
        name column per nesting depth (L0, L1, …) — a path's leaf name sits in
        the column matching its depth — followed by ``metric_headers`` columns
        filled from ``metric_cells`` (``path -> list[str]``). Shared by both
        reports so the tree/column layout lives in one place."""
        max_depth = max(p.count(".") for p in ordered_paths)
        level_cols = [f"L{i}" for i in range(max_depth + 1)]
        cols = level_cols + list(metric_headers)

        rows: List[List[str]] = []
        for path in ordered_paths:
            level_cells = [""] * len(level_cols)
            level_cells[path.count(".")] = path.rsplit(".", 1)[-1]
            rows.append(level_cells + list(metric_cells(path)))

        widths = [len(c) for c in cols]
        for row in rows:
            for i, value in enumerate(row):
                widths[i] = max(widths[i], len(value))

        def line(values):
            return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

        print(title)
        print(line(cols))
        print(line(["-" * w for w in widths]))
        for row in rows:
            print(line(row))
        print()

    def metrics_report(self, operation_id=None) -> None:
        """Print a compact, human-readable table over self.blocks.

        Columns: stage, wall, cum_wall, py_peak, rss_start, rss_peak,
        rss_Δ (signed), plus one column per counter key that has a
        non-zero value in at least one block. ``cum_wall`` is wall
        time elapsed from the first ``profile()`` block since reset.
        rss_start / rss_peak are absolute process RSS; rss_Δ is the
        new-allocation delta inside the block.

        Rows are laid out as a tree: parent-first pre-order so each
        group reads top-down (rollup, then per-step breakdown). Each
        nesting level gets its own column (L0, L1, …); a block's name
        sits in the column matching its depth, deeper columns blank.
        Top-level groups keep execution order. No blank separator
        rows.
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

        def fmt_counter(key: str, val: int) -> str:
            if key.endswith("_bytes"):
                return _fmt_bytes(val)
            return _fmt_count(val)

        # self.blocks is in completion order (children before parents); use it as
        # the sibling/root tie-break so groups keep execution order.
        by_path = {b.path: b for b in self.blocks}
        order_idx = {b.path: i for i, b in enumerate(self.blocks)}
        ordered = self._tree_preorder(by_path, sort_key=lambda p: order_idx[p])

        metric_headers = [
            "wall",
            "cum_wall",
            "py_peak",
            "rss_start",
            "rss_peak",
            "rss_Δ",
        ] + counter_keys

        def metric_cells(path):
            b = by_path[path]
            cells = [
                _fmt_time(b.elapsed_s),
                _fmt_time(b.wall_end_s),
                _fmt_bytes(b.py_heap_peak_bytes),
                _fmt_bytes(b.rss_start_bytes),
                _fmt_bytes(b.rss_peak_bytes),
                _fmt_bytes(b.rss_peak_bytes - b.rss_start_bytes, signed=True),
            ]
            cells += [fmt_counter(k, b.counter_deltas.get(k, 0)) for k in counter_keys]
            return cells

        title = "metrics report"
        if operation_id is not None:
            title = f"{title} (operation_id={operation_id})"
        self._print_tree_table(title, ordered, metric_headers, metric_cells)

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()
        self.current_path.clear()
        self._agg.clear()
        self._order.clear()
        self._base_perf = None

    @staticmethod
    def percentile_report(blocks: List["BlockMetrics"], operation_id=None) -> None:
        """Print per-stage distribution over ``blocks`` (one block per stage per
        unit of work, e.g. per worker-batch). Sums hide stragglers; this shows
        the spread. Columns: stage, n, wall p50/p90/p99/max, then ABSOLUTE
        rss_peak p50/p99/max (memory is never summed). A stage whose wall max ≫
        p50 is the straggler; whose rss max ≫ p50 is the memory spike. Follows
        with per-worker lifetime peak RSS so a leak (one worker climbing) shows."""
        if not blocks:
            return
        by_path: Dict[str, List["BlockMetrics"]] = defaultdict(list)
        for b in blocks:
            by_path[b.path].append(b)

        def pct(sorted_vals, q):
            if not sorted_vals:
                return 0.0
            idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
            return sorted_vals[idx]

        total_of = {p: sum(b.elapsed_s for b in bl) for p, bl in by_path.items()}
        ordered = HierarchicalProfiler._tree_preorder(
            by_path, sort_key=lambda p: -total_of[p]
        )

        metric_headers = [
            "n",
            "total",
            "calls",
            "t/call",
            "wall p50",
            "wall p90",
            "wall p99",
            "wall max",
            "rss p50",
            "rss p99",
            "rss max",
        ]

        def metric_cells(path):
            bl = by_path[path]
            walls = sorted(b.elapsed_s for b in bl)
            peaks = sorted(b.rss_peak_bytes for b in bl)
            total_s = total_of[path]
            calls = sum(b.call_count for b in bl)
            return [
                str(len(bl)),
                _fmt_time(total_s),
                _fmt_count(calls) if calls else "-",
                _fmt_time(total_s / calls) if calls else "-",
                _fmt_time(pct(walls, 0.50)),
                _fmt_time(pct(walls, 0.90)),
                _fmt_time(pct(walls, 0.99)),
                _fmt_time(walls[-1]),
                _fmt_bytes(pct(peaks, 0.50)),
                _fmt_bytes(pct(peaks, 0.99)),
                _fmt_bytes(peaks[-1]),
            ]

        title = "percentile report"
        if operation_id is not None:
            title = f"{title} (operation_id={operation_id})"
        HierarchicalProfiler._print_tree_table(
            title, ordered, metric_headers, metric_cells
        )

        worker_peak: Dict[int, int] = defaultdict(int)
        for b in blocks:
            worker_peak[b.pid] = max(worker_peak[b.pid], b.rss_peak_bytes)
        peaks = sorted(worker_peak.values())
        if peaks:
            print(
                f"per-worker lifetime peak RSS ({len(peaks)} workers): "
                f"min {_fmt_bytes(peaks[0])}, p50 {_fmt_bytes(pct(peaks, 0.50))}, "
                f"max {_fmt_bytes(peaks[-1])}"
            )

    @classmethod
    def from_blocks(cls, blocks: List["BlockMetrics"]) -> "HierarchicalProfiler":
        """Build a profiler whose report rolls up ``blocks`` collected across
        multiple processes (each worker profiles locally and ships back its
        ``self.blocks``). Same-path blocks are folded into one row: ``elapsed_s``
        and ``counter_deltas`` summed (total work across workers, which overlaps
        in wall-clock); memory is absolute, never summed — ``rss_start`` is the
        min (process floor when the stage first ran) and ``rss_peak`` the max
        (worst the stage reached in any worker). ``timings`` / ``call_counts``
        are repopulated so ``print_report`` works too."""
        prof = cls(enabled=True)
        for b in blocks:
            prof.timings[b.path].append(b.elapsed_s)
            prof.call_counts[b.path] += 1
            # copy so folding doesn't mutate the caller's blocks in place.
            prof._record(
                BlockMetrics(
                    path=b.path,
                    elapsed_s=b.elapsed_s,
                    call_count=b.call_count,
                    py_heap_peak_bytes=b.py_heap_peak_bytes,
                    rss_start_bytes=b.rss_start_bytes,
                    rss_peak_bytes=b.rss_peak_bytes,
                    counter_deltas=dict(b.counter_deltas),
                )
            )

        cum = 0.0
        for path in prof._order:
            cum += prof._agg[path].elapsed_s
            prof._agg[path].wall_end_s = cum
        return prof
