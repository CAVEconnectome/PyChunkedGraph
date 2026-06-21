import os
from typing import Optional


def cgroup_peak_bytes() -> Optional[int]:
    """Peak memory of the whole control group (cgroup) in bytes, or ``None`` when
    not running under a memory cgroup (e.g. a dev box or macOS).

    This is the high-water mark the kernel tracks for the entire cgroup — the
    parent process plus every forked worker plus page cache it accounts — which
    is exactly the figure a Kubernetes pod's memory limit / OOM killer enforces.
    Use it to size pod memory requests/limits for the stitching job, since an
    in-process per-worker RSS reading cannot see the concurrent whole-pod total.

    Reads cgroup v2 ``memory.peak`` first, then the v1
    ``memory.max_usage_in_bytes`` fallback. Returns ``None`` if neither is
    readable so callers can degrade gracefully.
    """
    for path in (
        "/sys/fs/cgroup/memory.peak",  # cgroup v2 (the container's own cgroup root)
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",  # cgroup v1
    ):
        try:
            with open(path) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            continue
    return None


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
