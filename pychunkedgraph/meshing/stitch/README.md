# stitch — parallel sharded mesh stitching

Parallel rewrite of `meshgen.chunk_initial_sharded_stitching_task` (layer-3+ mesh stitching).
Output is **mesh-equivalent** to the single-threaded `meshgen` shard (crc32c / per-label
decoded-mesh gate); correctness is the hard constraint, speed is secondary.

- Dispatched by both drivers: `meshing_sqs.MeshTask` (taskqueue) and `pipeline.meshing.worker`
  (k8s Indexed Job). The old `meshgen` function is kept (deprecated), no longer called.
- Per layer-L chunk: each valid multi-child parent's descendant fragments are fetched from the
  lower-layer `initial` shards, transformed into the parent's quantization, merged across the
  chunk boundary, re-encoded as draco, written to one output shard.
- This file is the maintained design + performance reference. Keep it in sync when the
  scheduler, profiler integration, or the byte path changes.

## Module map

| file | role |
|------|------|
| `task.py` | parent orchestration: whole-chunk scan → LPT batches → fork pool → collect → synthesize → upload. Public: `chunk_initial_sharded_stitching_task_mp`. |
| `worker.py` | per-process work: rebuilds its own `cg`/`cv`, resolves descendants, byte-range fetches fragments, stitches. Pure `with prof.profile(...)` instrumentation. |
| `utils.py` | `cg`-free geometry/transform/merge helpers (the byte math, mirrors `meshgen`), `MipInfo`, `derive_batch_size`, `sample_debug`. |
| `profile.py` | notebook-driven harness: run + crc32c/mesh compare against production + metrics report. Never writes production. |
| `_forkdiag.py` | one-shot fork-safety diagnostic (debug only). |

Batch balancing lives in `pychunkedgraph/scheduling` (`lpt_partition`) — see its README.

## Design decisions (the load-bearing ones)

| decision | why |
|---|---|
| **Fork safety** — workers never receive a live `cg`/`cv`; they rebuild from `cg.get_serialized_info()` + the graphene `info` dict. `mp.Pool` forks **before** the parent touches the network. Plain `fork`, no spawn. | tensorstore installs a fork-hostile guard; a forked child that touches it `abort()`s. `_forkdiag.py` localizes it. |
| **Memory** — byte-range fetch only the needed labels (`readers[layer].get_data`), never whole shards, no cache. | whole-shard fetches raise the per-worker footprint by orders of magnitude. |
| **Anti-straggler scheduling (LPT)** — `_make_batches` uses `scheduling.lpt_partition`, heaviest-first by immediate-child count into the least-loaded bin; bins yielded heaviest-first. | per-parent cost is heavy-tailed; a fixed-size batch holding the giant strands one worker while others idle. The giant seeds its own bin and runs solo, dispatched first. |
| **Read/compute overlap** — each batch split in two; the second half prefetches on a 1-thread `ThreadPoolExecutor` while the first stitches. | GCS read is GIL-released IO, stitch is CPU. One fetch in flight → at most two halves resident. |
| **Shard upload** — part size derived from the shard so the part count stays at or under the single-compose limit; small shards fall under the floor and upload as one PUT. | cloudfiles uses one value as both composite trigger and part size. Any shard size then costs exactly one GCS `compose` instead of a tree. GCS 429s that endpoint under many concurrent pods, and cloudfiles composes without `if_generation_match`, which disables the storage client's conditional retry — one 429 is otherwise fatal to the chunk. |
| **Profiling** — all measurement in `HierarchicalProfiler`; call sites are pure `with prof.profile("name")`. RSS sampled **once per batch**; hot substages timing-only. Blocks folded on record. | production (env unset) is a true no-op; block list stays bounded by distinct stages, not call count. |

## Measured profile (full run, 16 workers, LPT)

Output `MESH-EQUIVALENT` to production. Shares are of aggregate worker CPU time:

| stage | share | nature |
|------|------|------|
| encode | ~47 % | one `DracoPy.encode_mesh_to_buffer` per parent (C, the floor) |
| decode | ~29 % | one `DracoPy.decode_buffer_to_mesh` per descendant fragment (C) |
| read | ~14 % | GCS byte-range fetch (IO; overlapped) |
| merge | ~4 % | numpy concat + dedup per parent |
| transform | ~2 % | numpy affine per fragment |
| graph | <1 % | bigtable descendant resolution |

- The wall is the giant straggler's encode — a double-digit percentage of total wall on its own.
- LPT isolates it, but one mesh's encode is irreducible: DracoPy has no batch/streaming encode,
  and the fragment can't be split without changing output bytes.
- Compute (encode+decode+merge) ≈ 80 %: cores are saturated, the residual is this tail.

## Batchability of decode/transform/merge/encode

| stage | batchable? | byte-safe? | payoff (cores saturated) | verdict |
|------|------|------|------|------|
| **decode** | yes — `decode_many` C++ loop in DracoPy, nogil/OpenMP; per-buffer calls self-contained | yes (results in input order) | ~0 common case; only the giant's fragment decodes vs 15 idle cores in its solo tail | **straggler-only, likely not worth it** |
| **encode** | no — giant straggler is ONE mesh = one un-batchable `encode_mesh` | n/a | none | **no** |
| **transform** | partial — group by source layer, stacked affine via `np.repeat` origins | yes (element-wise, no reduction) | ~1 % ceiling, eroded by grouping | **not worth it** |
| **merge** | already 1 op/parent; two work-reducing trims applied (below) | yes (byte-gated) | real — cuts CPU + transient allocs on big parents | **done** |

A DracoPy batch API only re-parallelizes work the pool already does. Full source-cited
feasibility study lives outside the repo at `dist/meshing-perf/dracopy-batch-feasibility.md`.

## Applied byte-gated optimizations

In `utils.merge_draco_meshes_across_boundaries_pure`, each verified mesh-equivalent:

1. **Dense lookup-table gather for the index remap** — `lut = np.empty(N, uint32)` indexed by old
   vertex id, gathered as `new_faces = lut[faces]`, replacing a per-vertex dict + `fastremap.remap`.
   Same dense integer remap, no multi-million-entry dict on big parents.
2. **float32 dedup, index carried out of band** — the index column rides alongside as `uint32`
   instead of `hstack`ed onto the vertices (which would upcast to float64 and double the
   `np.unique` working set). Vertices stay float32 through encode.
3. `meshgen.decode_draco_mesh_buffer` uses `np.asarray` (zero-copy view of the DracoPy buffer).
4. Descendant resolution and per-layer bucketing batched: one `get_downstream_multi_child_nodes`
   and one `get_chunk_layers` per call.

## Verification

Offline gates (pcg env, no live cg) — recreate under `/tmp` or promote to pytest:

1. `scheduling` LPT: `pychunkedgraph/tests/scheduling/test_partition.py` (committed).
2. `_make_batches` LPT contract: parent set + child lists preserved, giant isolated solo.
3. `sample_debug` medium..tiny selection.
4. prefetch `_split_in_two` / `_group_labels`: each parent stitched once with its own labels,
   overlap occurs, ≤2 halves resident.
5. profiler: disabled = no-op (0 `memory_info`); substage blocks spawn no sampler thread;
   fold-on-record (sum elapsed/call_count, max rss, min-nonzero start); `from_blocks` composes;
   pickle drops `_proc`; `print_report`/`metrics_report` intact.

Live (notebook):

```python
profile.run_parallel(cg, chunk_id, mip=0, out_subdir="test_par")
profile.compare_against_production(cg, chunk_id, "test_par")   # must be MESH-EQUIVALENT
```

- crc32c fast path, else per-label decoded-mesh parity.
- Fails fast on the first mismatched label; `fail_fast=False` scans all and reports the count.
- `max_parents=` gives a fast medium preview (`out_subdir="test_debug"`, `subset=True` compare →
  **SUBSET-EQUIVALENT**).

## Config knobs

- `n_processes` arg, else `PCG_N_PROCESSES` — the one pool-size contract every worker reads, set
  by the pipeline from the pod's cpu request. Never `cpu_count()`: a container reads the *node's*
  cores, so that fallback oversubscribes the pod and CFS-throttles it.
- `PCG_PROFILER_ENABLED=1`, or the harness flipping `get_profiler().enabled`, enables the report.
