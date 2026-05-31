# scheduling

Generic, domain-agnostic load-balancing primitives for parallel work. Nothing
here knows about meshes, the graph, or any specific workload — callers describe
work and supply weights; this package decides how to split it across workers.

## Why

PyChunkedGraph workloads are **heavy-tailed**: most units are small, a few are
enormous (one supervoxel/component/parent can be orders of magnitude larger than
the median). Splitting such work into equal-*count* chunks strands a worker on
the one giant chunk while the rest sit idle — the classic distributed
**straggler**. The fix is to balance by *cost*, not count.

## What it provides

- **`WorkItem(payload, weight)`** — the shared unit of work. `payload` is the
  caller's data (opaque to the scheduler); `weight` is a cheap cost proxy used
  only to balance load (child count, component size, byte size, …).
- **`lpt_partition(items, n_bins)`** — Longest-Processing-Time-first
  partitioning: sort heaviest-first, greedily place each item in the
  currently-lightest bin. Makespan ≤ 4/3 × optimal, closer in practice. A
  dominant item seeds its own bin and runs solo. Bins are returned
  heaviest-load-first, so a pull-based pool dispatches the dominant bin first.
- **`n_bins_for(n_items, target_per_bin)`** — bin-count helper.

## Usage

```python
from pychunkedgraph.scheduling import WorkItem, lpt_partition, n_bins_for

items = [WorkItem(payload=obj, weight=cost(obj)) for obj in work]
for bin_items in lpt_partition(items, n_bins_for(len(items), target_per_bin=32)):
    dispatch([it.payload for it in bin_items])   # one worker batch
```

## Scope

Partitioning only. It does **not** own the process pool, dispatch, result
collection, or any fork/teardown handling — those differ per caller (e.g.
mesh stitching streams results via `imap_unordered`; ingest writes directly via
`pool.map`). Each caller keeps its own pool and attaches its own shared context;
this package just decides which items go together.

## Callers

- `meshing/stitch` — balances per-parent stitch work by immediate-child count.
- ingest connected-component writes / chunk reads are natural future callers
  (currently count-chunked); they can swap to weight-aware partitioning by
  wrapping their units in `WorkItem`.
