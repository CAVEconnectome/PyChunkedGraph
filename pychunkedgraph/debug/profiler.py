from typing import Dict
from typing import List
from typing import Tuple

import os
import time
from collections import defaultdict
from contextlib import contextmanager


class HierarchicalProfiler:
    """
    Hierarchical profiler for detailed timing breakdowns.
    Tracks timing at multiple levels and prints a breakdown at the end.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.stack: List[Tuple[str, float]] = []
        self.current_path: List[str] = []

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block."""
        if not self.enabled:
            yield
            return

        full_path = ".".join(self.current_path + [name])
        self.current_path.append(name)
        start_time = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timings[full_path].append(elapsed)
            self.call_counts[full_path] += 1
            self.current_path.pop()

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

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()
        self.stack.clear()
        self.current_path.clear()


# Global profiler instance - enable via environment variable
PROFILER_ENABLED = os.environ.get("PCG_PROFILER_ENABLED", "1") == "1"
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
