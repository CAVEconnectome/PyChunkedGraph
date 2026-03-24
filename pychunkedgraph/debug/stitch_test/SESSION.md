# Stitch Redesign — Session Context

For a new Claude session to pick up this work, tell it:
> Read pychunkedgraph/debug/stitch_test/SESSION.md to understand the stitch redesign state.

## Key files

- `pychunkedgraph/debug/stitch_test/proposed.py` — the proposed stitch implementation
- `pychunkedgraph/debug/stitch_test/wave.py` — unified test runner (single/wave/multiwave experiments)
- `pychunkedgraph/debug/stitch_test/utils.py` — structure extraction, batched parallel extraction, per-node comparison
- `pychunkedgraph/debug/stitch_test/compare.py` — orchestration, persistence helpers
- `pychunkedgraph/debug/stitch_test/current.py` — wrapper for current `add_edges` baseline
- `pychunkedgraph/debug/stitch_test/tables.py` — BigTable backup/restore, env setup, autoscaling
- `.env/stitching/hsmith_mec.ipynb` — test notebook

## Module structure (no cycles)

`stitch_types` → `tables` → `reader` → `hierarchy` → `proposed` → `compare` → `wave`
                  `tables` → `utils` → `{current, proposed}` → `compare` → `wave`

- `stitch_types.py` — shared dataclasses: StitchResult, StitchContext, RunResult
- `reader.py` — CachedReader (bulk_read_l2, get_parents/children/acx with cache), get_all_parents_filtered, filter_orphaned_nodes, resolve_partner_sv_parents (sampling), batch_create_node_ids, collect_and_resolve_partner_svs
- `hierarchy.py` — resolve_cx_at_layer, resolve_sv_to_layer, create_parents, allocate_deferred_roots, resolve_cx_for_write
- `proposed.py` — stitch(), run_proposed_stitch, _read_upfront, _merge_l2, _discover_siblings, _build_hierarchy, _build_entries
- `utils.py` — extract_structure, compare_structures, batched extraction, RunResult (re-exported from stitch_types)
- `compare.py` — orchestration + persistence
- `wave.py` — unified entry points (single/wave/multiwave)

## Current status (2026-03-23)

### What works
- Proposed algorithm implemented and structurally verified (wave 0 MATCH)
- Single file test: proposed ~129s vs current ~211s (1.63x speedup on this VM)
- Wave 0 current baseline: 606 files, 311K roots, ~1050s wall with 512 workers
- Wave 0 proposed: 680s wall (1.54x speedup), 311916 roots, 617K edges — **MATCH at all layers**
- Comparison: canonical ID based — L2=frozenset(svs), L3+=frozenset(child canonical IDs). Components + cx edge counts match.
- Extraction: ~51s for 311K roots (1248 shards, 250 roots/shard)

### Root ID collision bug (CRITICAL, fixed 2026-03-23)
- **Symptom**: proposed wave extraction showed truncated hierarchy (roots → L2 directly, missing layers 3-6). Root had Child column with timestamp 2026-02-10 (from backup), not the stitch's write.
- **Root cause**: `_batch_create_node_ids` in proposed.py called `id_client.create_node_ids` without checking if IDs already existed in the table. New root IDs collided with existing backup roots, so the stitch's `Child` write went to a row that already had a `Child` from ingest.
- **ID allocation mechanism**:
  - Non-root layers: single atomic counter per chunk via `read_modify_write_row`. Sequential, always beyond existing max. No collision.
  - Root layer: 256 sharded counters per root chunk (for write throughput). Random counter: `segment_id = base * 256 + counter`. When backup is restored, counters are copied too. New allocations from same counters collide with existing roots.
  - Code: `kvdbclient.BigTableClient.create_node_ids` → `_get_root_segment_ids_range` (root) vs `_get_ids_range` (non-root)
- **Fix**: Added collision check (`read_nodes` to verify non-existence) in `_batch_create_node_ids`, but only for `root_chunks`. Matches the pattern in `edits.py:_get_new_ids` (line 738-749).
### Cross edge data flow in proposed stitch (proposed.py)

**What gets written per node type:**

| Node type | Child | CrossChunkEdge | AtomicCrossChunkEdge | Parent (on children) |
|-----------|-------|---------------|---------------------|---------------------|
| New L2 | SVs | Resolved at L2 layer | Raw [sv, sv] | Yes |
| L3-L6 (new) | children | Resolved at that layer | — | Yes |
| Root (new) | children at L6 | Empty | — | Yes |
| L2 siblings (existing) | — (unchanged) | Resolved (updated) | — (unchanged) | Updated via new parent |

**Data flow:**
1. **Read**: `atomic_cx` from old L2 nodes via `get_atomic_cross_edges` (line 246). Format: `{l2_id: {layer: [[sv, partner_sv]]}}`
2. **Merge (new L2)**: `l2_atomic_cx[new_l2]` = merged edges from all old L2s in component + stitch edges, deduplicated (lines 408-426). Raw `[sv, sv]`.
3. **Hierarchy propagation**: `node_cx[node]` — col 0 = node's own ID, col 1 = raw partner SV. Parent inherits child edges at layers >= parent_layer (lines 746-760).
4. **Resolution** (`_resolve_cx_for_write`): col 1 resolved from partner SV → partner identity at node's layer via `_resolve_sv_to_layer` (lines 791-808). Uses resolver (SV→parent chain), old_to_new (merged L2 remap), child_to_parent (hierarchy walk).
5. **Write**: `CrossChunkEdge[L]` = resolved edges. `AtomicCrossChunkEdge[L]` = raw [sv, sv] (L2 only). (lines 833-842)

**Key invariant**: AtomicCrossChunkEdge is immutable raw data — never transformed, only merged. CrossChunkEdge is the resolved version. Future proposed stitches read AtomicCrossChunkEdge only.

### Extraction and comparison design
- **Canonical IDs**: L2 = frozenset(svs), L3+ = frozenset(child canonical IDs). Table-independent.
- **CrossChunkEdge comparison**: read CrossChunkEdge from all nodes (L2-L6), compare unique edge counts per node per cx_layer using canonical IDs for both endpoints.
- **Storage**: gzip pickle (canonical IDs are nested frozensets)
- **Dynamic sharding**: 250 roots per shard (scales with input), pool of `min(n_shards, 4 * cpu_count)` workers
- **Fresh extraction**: clears all `batch_*` dirs before extracting (no stale cache)
- **Parallel comparison loading**: both sides loaded concurrently via ThreadPoolExecutor
- **Reuse current baseline**: `batched_extract_and_compare` accepts `current_extract_dir` to skip re-extracting current
- **No table deletion**: user manages cleanup

### Retry safety
- **Two-phase writes**: `_build_entries` returns `(node_entries, parent_entries)`. Node rows written first, then Parent pointers.
- **No FormerParent**: proposed path does not write FormerParent/deprecation entries.
- **Crash recovery**: `stitch_results.json` saved immediately after stitch completes. Pass `run_id` to resume.

### Architecture decisions
- **No neighbor CrossChunkEdge updates**: stale is OK — future proposed stitches read AtomicCrossChunkEdge (immutable). Human edits via add_edges handle staleness via LatestEdgesFinder.
- **No locks**: lock-free, enables true parallelism within waves.
- **Deferred root ID allocation**: root IDs allocated in single batch after hierarchy loop, with collision check. Non-root IDs allocated per-layer (sequential counters, no collision).
- **Sibling write optimization**: L2 siblings are existing nodes. Only write CrossChunkEdge (updated) + Parent (via new parent). Skip Child + AtomicCrossChunkEdge (unchanged). Reduces write volume significantly for files with many siblings.
- **No cg.cache**: proposed stitch uses `StitchContext` dataclass for all state. Does not touch `cg.cache` (edit path's cache). Clean separation.
- **Dataclasses in stitch_types.py**: `StitchResult`, `StitchContext`, `RunResult`. Shared across all modules, prevents circular deps and dict key typos.
- **CachedReader**: caches get_parents, get_children, get_atomic_cross_edges across calls. `bulk_read_l2` combines Parent+Child+AtomicCrossChunkEdge in one RPC. Shared between Phase 1 concurrent threads and Phase 2b.
- **Partner SV sampling**: when >25K partner SVs, samples 500 at a time, gets their L2 parents, discovers sibling SVs via L2 children. Reduces `get_parents` reads by ~5x for large files.
- **Pool initializer**: meta + CloudVolume info read once in parent process, passed to workers via Pool initializer. Workers skip BigTable metadata read + GCS CloudVolume info read.
- **BigTable autoscaling**: min_nodes=5 + cpu=25% for wave/multiwave, reverts to min_nodes=1 + cpu=60% after.
- **Task randomization**: edge files shuffled before pool dispatch to spread BigTable load across tablets.
- **ThreadPoolExecutor capped**: max_workers=min(len(tasks), 16) for ID allocation to avoid BigTable counter row contention.

### Test infrastructure
- **Entry points**: `run_current(experiment)` and `run_proposed_and_compare(experiment, run_id=None)`
- **Experiment types**: "single" (one file), "wave" (wave 0), "multiwave" (all waves)
- **Extraction**: 500K root batches, 250 roots/shard, `4 * cpu_count` pool
- **Retries**: tenacity on extraction reads (3 attempts, exponential backoff)
- **Workers**: `min(n_files, 4 * cpu_count)` for wave processing
- **Progress**: tqdm for extraction and wave processing (not during parallel comparison to avoid interleaving)
- **PostToolUse hook**: `.claude/check_edit.sh` checks for nested imports + import validity on every edit to stitch_test/*.py

### Performance

**Single file (task_0_0.edges, 1024 edges)**:
- Proposed ~129s vs current ~211s (1.63x)

**Single file straggler (task_0_591.edges, 1024 edges, 17143 L2 CCs, 49942 siblings)**:
- Current: 303.7s
- Proposed: 261.3s (stitch+write, with sampling+bulk) — 1.16x
- RPC log shows: bulk_read_l2 49942 nodes 54s, get_parents 2048 SVs 34s (cold start latency)
- BigTable cold start after restore causes elevated read latency for first few reads

**Wave 0 (606 files, 311K roots, 617K edges)**:
- Current: ~1050s wall (with lock contention from 512 workers)
- Proposed: ~340s wall (3.1x, lock-free) — **MATCH verified**
- Per-file: mean=206s, median=228s, p95=246s, max=329s
- Extraction: ~37-51s for 311K roots (1248 shards)
- Comparison: ~143-157s (parallel shard loading, canonical ID based)

### BigTable cold start latency
- After backup restore, BigTable needs time to optimize + warm block cache
- No direct cache priming API — must send actual read traffic
- First RPC after restore consistently 30-40s for small batches (2048 rows)
- Subsequent reads on same/nearby rows are fast (block cache populated)
- Sampling partner SVs helps warm cache for subsequent reads
- `time.sleep(10)` after restore is NOT sufficient — need read traffic to warm
- TODO: add warm-up read step after restore to prime block cache before stitch starts

### Read optimization flags (reader.py)
- `USE_BULK_READ=True` — combined Parent+Child+AtomicCX in one RPC (default)
- `USE_SAMPLING=True` — sample partner SVs to reduce get_parents calls (default)
- `VERBOSE=False` — logging for sampling/read dispatch (set True for single mode)
- `_SAMPLE_SIZE=1000`, `_SAMPLING_THRESHOLD=25000`, `_SAMPLING_STOP=10000`

### Remaining work
- Warm-up reads after restore to mitigate cold start latency
- Run multiwave test (wave 0 validated)
- Further optimize straggler file (task_0_591)
- Add incremental file result saving during wave runs

## User preferences (critical)
- **Never describe how code works without reading it first** — use Read/Grep, or say "I haven't verified this"
- **Never use nested/inline imports** — all imports at module top level, design modules to avoid circular deps
- **Never create commits** — user does them
- **Vectorized numpy** — no Python loops where numpy works
- **Keep notebooks simple** — short function calls only, all logic in modules
- **No patchwork** — design complete algorithms from first principles
- **No fat VMs** — hard constraint
- **No mocks** — only mocker fixture
- **Never modify user's code without asking**
- **Terse responses** — no trailing summaries
- **Never delete tables** — user manages cleanup via prefix
- **Never add destructive operations without being asked**

## Dataset
- **hsmith_mec**: 7 layers, ~600k edges, 1095 total files
  - Wave 0: 606 files
  - Edge source: `gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges`
  - Backup table: `hsmith-mec-100gvx-exp16-0.26-backup`
  - BigTable project: `zetta-proofreading`, instance: `pychunkedgraph`
