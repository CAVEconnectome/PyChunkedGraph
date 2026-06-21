import os

from .hierarchical import HierarchicalProfiler

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
