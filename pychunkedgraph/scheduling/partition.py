"""Generic load-balanced partitioning for parallel work.

Domain-agnostic: callers describe each unit of work as a ``WorkItem`` carrying
its payload and a cost ``weight``; the partitioner returns bins balanced by
summed weight. Used to split heavy-tailed workloads (per-node mesh stitching
today; connected-component writes and chunk reads in ingest are natural future
callers) across workers so one expensive item doesn't strand a worker while the
rest sit idle.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, List, Sequence


@dataclass(frozen=True)
class WorkItem:
    """One unit of parallel work: the caller's ``payload`` plus its cost
    ``weight`` (a proxy used only to balance load — e.g. child count, component
    size). The scheduler never inspects ``payload``."""

    payload: Any
    weight: float = field(default=1.0)


def lpt_partition(items: Sequence[WorkItem], n_bins: int) -> List[List[WorkItem]]:
    """Partition ``items`` into ``n_bins`` bins balanced by summed ``weight``
    using Longest-Processing-Time-first (LPT / greedy number partitioning):
    sort heaviest-first, then place each item in the currently-lightest bin.

    Makespan (heaviest bin) is within 4/3 of optimal, and closer in practice as
    the item count grows. A single dominant item seeds its own bin and stays
    nearly alone — the point: it runs on one worker while the rest pack the
    others, instead of stranding a worker behind it.

    Bins are returned heaviest-load first, so a pull-based pool dispatches the
    dominant bin first. Empty bins (more bins than items) are dropped; ``n_bins``
    is clamped to ``[1, len(items)]``.
    """
    if not items:
        return []
    n_bins = max(1, min(n_bins, len(items)))

    ordered = sorted(items, key=lambda it: it.weight, reverse=True)
    bins: List[List[WorkItem]] = [[] for _ in range(n_bins)]
    loads = [0.0] * n_bins
    heap = [(0.0, i) for i in range(n_bins)]
    heapq.heapify(heap)
    for item in ordered:
        load, idx = heapq.heappop(heap)
        bins[idx].append(item)
        loads[idx] = load + item.weight
        heapq.heappush(heap, (loads[idx], idx))

    order = sorted(range(n_bins), key=lambda i: -loads[i])
    return [bins[i] for i in order if bins[i]]


def n_bins_for(n_items: int, target_per_bin: int) -> int:
    """Bin count keeping each bin near ``target_per_bin`` items
    (``ceil(n_items / target_per_bin)``), at least 1."""
    return max(1, math.ceil(n_items / max(1, target_per_bin)))
