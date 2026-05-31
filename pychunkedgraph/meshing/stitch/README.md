# stitch — parallel sharded mesh stitching

Parallel rewrite of `meshgen.chunk_initial_sharded_stitching_task` (layer-3+ mesh
stitching). Output is **mesh-equivalent** to the single-threaded `meshgen` shard
(crc32c / per-label decoded-mesh gate); correctness is the hard constraint, speed
is secondary. `meshing.meshing_sqs.MeshTask` dispatches this for layer-3+ chunks;
the old `meshgen` function is kept (deprecated) but no longer called.

This file is the maintained design + performance reference for the package. Keep
it in sync when the scheduler, profiler integration, or the byte path changes.

## What it does

For a layer-L chunk, each valid multi-child parent's descendant fragments are
fetched from the lower-layer `initial` sharded meshes, transformed into the
parent's quantization, merged across the chunk boundary, re-encoded as draco, and
written to one output shard.

## Module map

| file | role |
|------|------|
| `task.py` | parent orchestration: whole-chunk scan → LPT batches → fork pool → collect → synthesize → upload. Public: `chunk_initial_sharded_stitching_task_mp`. |
| `worker.py` | per-process work: rebuilds its own `cg`/`cv`, resolves descendants, byte-range fetches fragments, stitches. Pure `with prof.profile(...)` instrumentation. |
| `utils.py` | `cg`-free geometry/transform/merge helpers (the byte math, mirrors `meshgen`), `MipInfo`, `derive_batch_size`, `sample_debug`. |
| `profile.py` | notebook-driven harness: run + crc32c/mesh compare against production + metrics report. Never writes production. |
| `_forkdiag.py` | one-shot fork-safety diagnostic (debug only). |

Batch balancing lives in the generic `pychunkedgraph/scheduling` package
(`lpt_partition`), not here — see its README.

## Design decisions (the load-bearing ones)

**Fork safety.** tensorstore (the watershed `cv`) installs a fork-hostile guard;
a forked child that touches it `abort()`s. So workers never receive a live `cg`/
`cv` — they rebuild from `cg.get_serialized_info()` (`{graph_id}`) + the graphene
`info` dict (the ingest pattern). The `mp.Pool` is forked **before** the parent
touches the network. Plain `fork`, no spawn. (`_forkdiag.py` localizes this.)

**Memory.** Workers byte-range-fetch only the labels they need
(`readers[layer].get_data`), never whole shards, no cache — fetching whole shards
blows the per-worker footprint up by orders of magnitude.

**Anti-straggler scheduling (LPT).** Per-parent cost is heavy-tailed: most parents
are small, a few are orders of magnitude larger (a single giant fragment can dwarf
the rest combined). A fixed-size batch that happens to hold the giant strands one
worker while the others idle. `_make_batches` uses `scheduling.lpt_partition`
(Longest-Processing-Time: sort heaviest-first by immediate-child count, assign each
to the least-loaded bin) so the giant **seeds its own bin and runs solo** while the
rest pack the others. Bins are yielded heaviest-first so the giant dispatches first.
Makespan floor is the giant's own solo encode — irreducible (see profile below).

**Read/compute overlap.** Each batch is split into two halves; the second half's
fragments are prefetched on a single-thread `ThreadPoolExecutor` while the first
half stitches (GCS read is GIL-released IO, stitch is CPU). One fetch in flight →
at most two halves' bytes resident (lower peak than fetching the whole batch).

**Profiling.** All measurement lives in `HierarchicalProfiler`; call sites are pure
`with prof.profile("name")` blocks. The worker times decode/transform/merge/encode
(per fragment/parent) and samples RSS **once per batch** (the `stitch` block,
`sampled_rss=True`) — the hot substages are timing-only. The harness flips
`get_profiler().enabled = True` (and `with_memory_default=False`,
`with_rss_default=False`) once before the fork; workers inherit it. Production
(env unset) is a true no-op. Per-batch profiler blocks are folded on record, so the
block list is bounded by the number of distinct stages, not the call count.

## Measured profile (full run, 16 workers, LPT)

Output `MESH-EQUIVALENT` to production. Shares are of aggregate worker CPU time
(sum across the process pool):

| stage | share | nature |
|------|------|------|
| encode | ~47 % | one `DracoPy.encode_mesh_to_buffer` per parent (C, the floor) |
| decode | ~29 % | one `DracoPy.decode_buffer_to_mesh` per descendant fragment (C) |
| read | ~14 % | GCS byte-range fetch (IO; overlapped) |
| merge | ~4 % | numpy concat + dedup per parent |
| transform | ~2 % | numpy affine per fragment |
| graph | <1 % | bigtable descendant resolution |

**The wall is the giant straggler's encode.** The single largest fragment's encode
is a double-digit percentage of total wall on its own — that one parent holds the
job open after the rest of the pool finishes. LPT isolates it (own solo bin,
dispatched first), but a single mesh's encode is irreducible: DracoPy has no
batch/streaming encode, and the fragment can't be split without changing output
bytes. Compute (encode+decode+merge) ≈ 80 % — cores are saturated; the residual is
this tail.

## Batchability of decode/transform/merge/encode

Researched (incl. feasibility of adding a batch API to DracoPy, which we own).
Bottom line: cores are already saturated by the process pool, so thread-level
re-parallelization of CPU work the pool already parallelizes is theater. Only
work-reducing or genuinely-idle-core changes help.

| stage | batchable? | byte-safe? | payoff (cores saturated) | verdict |
|------|------|------|------|------|
| **decode** | yes — `decode_many` C++ loop in DracoPy, nogil/OpenMP; per-buffer calls are self-contained, no shared state | yes (results in input order) | ~0 common case; only the giant's thousands of fragment decodes vs 15 idle cores during its solo tail | **straggler-only, likely not worth it** |
| **encode** | no — giant straggler is ONE mesh = one un-batchable `encode_mesh` | n/a | none | **no** |
| **transform** | partial — group by source layer, stacked affine via `np.repeat` origins | yes (element-wise, no reduction) | ~1 % ceiling, eroded by grouping | **not worth it** |
| **merge** | already 1 op/parent; two work-reducing trims applied (below) | yes (byte-gated) | real — cuts CPU + transient allocs on big parents | **done** |

Net: cores are already saturated by the process pool, so a DracoPy batch API
(`decode_many`/`encode_many`) only re-parallelizes work the pool already does —
worthless in the common case. Encode can never be batched (the giant straggler is a
single mesh). Decode batching is defensible only for the giant's fragment decodes
during its solo tail, and even that is undercut by its un-batchable encode. The full
source-cited feasibility study (DracoPy `.pyx`/`.h`/draco internals, the GIL/nogil
analysis, the cheaper zero-copy `np.frombuffer` alternative) lives outside the repo
at `dist/meshing-perf/dracopy-batch-feasibility.md` for review.

## Applied byte-gated optimizations

In `utils.merge_draco_meshes_across_boundaries_pure` (each verified mesh-equivalent
against production):
1. **Dense lookup-table gather for the index remap** — a `lut = np.empty(N, uint32)`
   indexed by old vertex id and gathered as `new_faces = lut[faces]`, instead of a
   per-vertex Python dict + `fastremap.remap`. Same result (dense integer remap),
   no multi-million-entry dict on big parents.
2. **float32 dedup, index carried out of band** — the index column rides alongside
   as `uint32` rather than `hstack`ed onto the vertices (which would upcast them to
   float64 and double the `np.unique` working set); `np.unique` runs on float32 and
   the vertices stay float32 through encode.

Also: `meshgen.decode_draco_mesh_buffer` uses `np.asarray` (zero-copy view of the
DracoPy buffer, not a copy); descendant resolution and per-layer bucketing are
batched (one `get_downstream_multi_child_nodes` / one `get_chunk_layers` per call).

## Verification

Offline gates (pcg env, no live cg) — recreate under `/tmp` or promote to pytest:
- `scheduling` LPT: `pychunkedgraph/tests/scheduling/test_partition.py` (committed).
- `_make_batches` LPT contract (parent set + child lists preserved, giant isolated solo).
- `sample_debug` medium..tiny selection.
- prefetch `_split_in_two` / `_group_labels` (each parent stitched once with its own labels, overlap occurs, ≤2 halves resident).
- profiler: disabled = no-op (0 `memory_info`); substage blocks spawn no sampler thread; fold-on-record (sum elapsed/call_count, max rss, min-nonzero start); `from_blocks` composes; pickle drops `_proc`; consumers (`print_report`/`metrics_report`) intact.

Live (notebook): `profile.run_parallel(cg, chunk_id, mip=0, out_subdir="test_par")`
then `profile.compare_against_production(cg, chunk_id, "test_par")` → must be
**MESH-EQUIVALENT** (crc32c fast path, else per-label decoded-mesh parity). The
compare fails fast on the first mismatched label by default; pass `fail_fast=False`
to scan all and report the total mismatch count. Use `max_parents=` for a fast
medium-sized preview (`out_subdir="test_debug"`, `subset=True` compare →
**SUBSET-EQUIVALENT**).

## Config knobs

- `n_processes` arg, or `PCG_MESH_STITCH_WORKERS` env (arg wins; else env; else cpu_count).
- `PCG_PROFILER_ENABLED=1` or the harness flipping `get_profiler().enabled` enables the report.
