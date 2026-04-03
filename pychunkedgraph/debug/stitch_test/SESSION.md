# Stitch Redesign Session Log

## Dataset
- hsmith_mec 100GVx exp16 0.26
- 661 edge files across 32 waves (606 in wave 0)
- Backup: hsmith-mec-100gvx-exp16-0.26-backup
- Edges: gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges

## Module Structure

```
row_cache.py   — CacheRow + RowCache (row-based, COW local→preloaded)
wave_cache.py  — WaveCache (uses RowCache, SiblingEntry, dirty check, _collect_partner_svs)
local_cg.py    — LocalChunkedGraph (stitch phases, timestamp, SANITY_CHECK global)
tree.py        — get_all_parents_filtered, update_parents_cache, restore_known_siblings
resolver.py    — resolve_svs_to_layer, resolve_cx_at_layer, ensure_partners_cached, acx_to_cx
stitch.py      — Orchestrator (thin wrapper calling lcg methods)
runner.py      — BaselineRun / ProposedRun (StitchRun base), run_proposed, run_baseline
utils.py       — RpcEntry, timed, batch_get_l2children, stitch_sanity_check, extraction, comparison
id_allocator.py — batch_create with ThreadPool, root chunk collision check
tables.py      — restore_test_table, warm_cache, set_autoscaling
test_helpers.py — noop_read, make_cache, resolve_sv (batch wrapper)
```

## Cache Design (CACHE_DESIGN.md)
- Row-based: CacheRow with parent/children/acx/cx columns
- Two-dict lookup: _local → _preloaded (COW on put)
- Single batch read per _ensure_cached call (all node types together)
- CX derived from ACX during resolution, not read from BigTable for own nodes
- Partner resolution for ALL siblings BEFORE dirty check
- _collect_partner_svs shared by compute_dirty_siblings and _build_partner_ids

## Timestamp-Based Retry Safety
- `acquire_stitch_timestamp(edge_file)` writes marker row, reads back server timestamp T
- All BigTable reads use `end_time=T` — only see pre-stitch state
- All writes use `time_stamp=T`
- `release_stitch_timestamp(edge_file)` deletes marker on success
- On retry: marker exists → same T → failed writes invisible
- Replaces filter_failed_node_ids / filter_orphaned entirely

## Key Bug Fixes

### resolve_svs_to_layer stuck check (2026-04-03)
Root cause of duplicate L2s under two roots. The stuck check `len(still_active) == len(active)` treated no-resolution as stuck even when SVs moved to new parents. All remaining SVs got dumped at pre-move positions. Fix: track actual progress with `progressed` flag — only dump if truly stuck (`parent == current` for all).

### Resolver elimination (2026-03-31)
Eliminated resolver dict. All resolution via cache.get_parents (single source of truth). Fixed stale parent chains causing wrong CX connections at L5+.

### Fixture extraction (2026-04-01)
BFS from roots, multiprocessing, gzip. Edge SVs + ACX source/target SVs kept as L2 children.

## Code Review Cleanup (2026-04-02/03)
- Removed debug instrumentation (_resolve_stats, _partner_stats) from resolver.py
- Deduplicated partner SV collection in wave_cache.py (_collect_partner_svs)
- Merged passes in _build_raw_cx_from_children (4→3)
- Early exit for SVs in _read_and_cache cache loop (skip ACX/CX checks if no children)
- resolve_svs_to_layer: clear names (sv/current/parent_of), docstring, proper progress tracking
- resolve_sv_to_layer moved from tree.py to test_helpers.py as resolve_sv (batch wrapper)
- debug_single.py: batch API usage, fixed broken resolve_cx_at_layer call
- RpcEntry dataclass in utils.py replaces positional tuples for rpc_log
- Fixed double baseline extraction in compare_with
- stitch_sanity_check moved from stitch.py to utils.py
- SANITY_CHECK global in local_cg.py, controlled by run_proposed(sanity_check=False)
- Parent reassignment assert in _create_parents (gated on SANITY_CHECK)

## Performance
- Parallel gRPC chunks in kvdbclient: 16K chunks + ThreadPoolExecutor (6.2x speedup on 900K SVs)
- Single file (task_0_591.edges): stitch=134s, gRPC=93s (69%), cache=14s (10%), other=27s
- Wave 0 (606 files): ~492s wall (pre-cleanup), ~570s with thread pool (needs re-measurement)

## Runner
- StitchRun base class: get_waves(), get_wave_files(), use_pool, _autoscale(min_nodes)
- experiment modes: "single" (1 file, no pool), "wave" (wave 0, pool), "multiwave" (all waves)
- Single file: task_0_591.edges (worst case from profiling)
- bt_min_nodes param on entry points, _autoscale context manager
- Baseline extraction: compare_with → batched_extract_and_compare (no double extraction)

## E2E Test
- Fixtures: `fixtures/e2e_fixture_{N}e.pkl.gz`
- `_WaveChecker` context manager: pre-stitch CC count, post-stitch root count + edge SV sharing
- Workers use `acquire_stitch_timestamp` / `release_stitch_timestamp`
- All 3 fixtures pass (1e: 87s, 2e: 124s, 3e: 158s)

## Real Table Wave 0 Results (2026-04-03, run 94655d5e)
**MATCH** — 311,916 roots, all layers L2-L6 CX match. 419s wall (15% faster than previous 492s).

### Bottleneck Analysis

Aggregate (606 files, 80,166s total stitch across all workers):
- **BigTable gRPC: 39,161s (49%)** — IO bound, parallel chunks helping
- **Algorithm/other: 40,906s (51%)** — dominated by ID allocation + partner chain walking
- **Cache population: 98s (~0%)** — negligible after SV early exit optimization

Worst file (task_0_591.edges, 292s stitch, 1024 edges):
- **merge_l2_alloc: 72s (25%)** — ID allocation for new L2s. ThreadPool(4) in id_allocator.py. Each chunk = 1 RPC. Increase pool size for speedup.
- **siblings_partner_chains: 63s (22%)** — Python dict walking through cached partner chains. Could vectorize.
- **read_partner_chains: 46s (16%)** — initial partner chain walk. Mix of cache hits and BigTable reads.
- **L2_discover_siblings: 32s (11%)** — sibling discovery at L3.
- **gRPC 310K SVs: 61s** — largest single BigTable read. 5x faster than pre-optimization.
- **L2_create_parents: 13.5s** — ID allocation for L3 parents.
- **Small batch cold reads (2K rows): 16-28s** — cold tablet latency on first reads.

### Optimization Priorities
1. **ID allocation parallelism** — increase id_allocator ThreadPool from 4 to 16+. Largest non-IO bottleneck.
2. **Partner chain walking** — 109s in worst file. Pure Python dict ops on cached data. Vectorize with numpy.
3. **Small batch cold reads** — 2K rows taking 16-28s due to tablet scatter. Consider warm-up reads or reordering.

## For Next Session

### Setup
```bash
# Emulator must be running for e2e tests:
gcloud beta emulators bigtable start --host-port=localhost:8540

# Working directory:
cd /home/akhilesh/opt/zetta_utils/.env/pcg

# Run e2e (all 3 fixtures sequentially):
for n in 1 2 3; do E2E_MODE=wave E2E_EDGES=$n python -m pytest pychunkedgraph/debug/stitch_test/test_e2e.py -x -v -s; done

# Run on real BigTable (single file):
# In notebook: mode="proposed", experiment="single"

# Run on real BigTable (wave 0, 606 files):
# In notebook: mode="proposed", experiment="wave"

# Notebook: .env/pcg/.env/stitching/hsmith_mec.ipynb
```

### Key Files
- `local_cg.py` — main stitch logic, SANITY_CHECK global, timestamp methods
- `resolver.py` — resolve_svs_to_layer (progress tracking fix was critical)
- `runner.py` — BaselineRun/ProposedRun, run_proposed(sanity_check=False, bt_min_nodes=5)
- `stitch.py` — thin orchestrator
- `wave_cache.py` — cache layer with _collect_partner_svs
- `tree.py` — chain walks (simplified, no filter_failed)
- `utils.py` — RpcEntry, stitch_sanity_check, extraction, comparison
- `kvdbclient/bigtable/client.py` — parallel gRPC chunks (16K + ThreadPool), has debug print

### Context
- Read `SESSION.md`, `CACHE_DESIGN.md`, `fixtures/README.md`
- Memory: `~/.claude/projects/-home-akhilesh-opt-zetta-utils/memory/project_stitch_redesign.md`
- All correctness tests pass. Focus is on performance optimization.
- NEVER speculate — always verify with data. Say "I haven't verified" when unsure.
- Run pylint after EVERY edit. No exceptions.

### TODO
1. Increase id_allocator thread pool (4 → 16+) — biggest non-IO bottleneck
2. Vectorize partner chain walking in resolve_svs_to_layer and discover_siblings
3. Remove kvdbclient debug print in _read
4. _max_row_key_count = 16_000 hardcoded in local_cg.py — make configurable
5. Run multiwave e2e
6. Consider: can merge_l2_alloc and L2_create_parents ID allocations be batched into fewer RPCs?
