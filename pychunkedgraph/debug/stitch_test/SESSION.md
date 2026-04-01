# Stitch Redesign Session Log

## Dataset
- hsmith_mec 100GVx exp16 0.26
- 661 edge files across 32 waves (606 in wave 0)
- Backup: hsmith-mec-100gvx-exp16-0.26-backup
- Edges: gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges

## Module Structure

```
row_cache.py   — CacheRow + RowCache (row-based, no ChainMap)
wave_cache.py  — WaveCache (uses RowCache, SiblingEntry, dirty check)
local_cg.py    — LocalChunkedGraph (all stitch phases)
tree.py        — resolve_sv_to_layer, restore_known_siblings, etc
topology.py    — resolve_cx_at_layer, resolve_remaining_cx (batched), etc
stitch.py      — Orchestrator
runner.py      — run_proposed, compare_run
CACHE_DESIGN.md
```

Tests: test_stitch.py (26), test_wave_cache.py (28), test_local_cg.py (4), test_cache_perf.py (10), test_e2e.py (1) = 69

## Cache Design (CACHE_DESIGN.md)
- Row-based: CacheRow with parent/children/acx/cx columns
- Two-dict lookup: _local → _preloaded (no ChainMap, 7x faster has_batch)
- Resolve ALL nodes fresh at all layers (no stale written_cx in CC graph)
- Clean siblings: cx_cache popped after store → skip writes
- Partner resolution for ALL siblings BEFORE dirty check
- accumulated_replacements in old_to_new
- Batched resolve_remaining_cx (layers > 2)

## Runs

### 695be87b (pre-refactor reference)
- 347,471 roots MATCH, 3069s wall

### Pending run (row-based cache, all fixes)
- 68 tests passing, 0 errors
- No written_cx in CC graph (correctness by construction)
- Clean sibling write skip (performance)

## Bugs Found (13 total)
1. cx_edges undefined (renamed to all_cx)
2. partition_siblings broken (old_hierarchy missing sibling chains)
3. Stale cx_cache (resolved against prior wave's node IDs)
4. Pool workers missing incremental (_shared_inc)
5. known_svs removed incorrectly (10M extra reads)
6. Pool worker count uncapped
7. accumulated_replacements missing from old_to_new
8. children_d vs ChainMap in snapshots
9. Layer 3+ skip (store/resolve must not skip at layer 3+)
10. written_cx stale at layer 3 (child_to_parent changes)
11. written_cx carry-forward (clean siblings overwritten with empty)
12. Partner resolution ordering (must run before dirty check)
13. written_cx in CC graph fundamentally unsound (resolver/old_to_new/child_to_parent all change per wave)

## E2E Test (test_e2e.py)
- Fixtures: `fixtures/e2e_fixture_{N}e.pkl.gz` from `sample_test_data.py`
- Stats: `fixtures/STATS.md` (auto-updated on extraction)
- BigTable emulator on port 8540, parallel populate (2 processes, 16K batch)
- `set_max_node_id` per chunk after populate (skip root chunks)
- Production `add_edges` on `e2e_prod` vs our stitch on `e2e_ours`, compare components at all layers
- Pool(2 workers, spawn), `_e2e_worker` returns traceback strings for pickling
- `_WaveChecker` context manager: pre-stitch CC count from root-level edge graph, post-stitch verifies root count matches CCs and all edge SVs share a root

### Refactors (2026-03-30)
1. Eliminated `child_to_parent` dict — single source of truth is `cache.parents` (via batch API)
2. Batched cache API: `get_parents`, `get_children_batch`, `get_acx_batch`, `get_cx_batch` — each handles cache misses internally via `_ensure_read` → BigTable batch read. No direct `_ColView` property access.
3. `resolve_svs_to_layer` replaces `resolve_sv_to_layer` — batch all SVs, walk level by level
4. Production `add_edges` passes all 32 waves on emulator (5222 roots, 0 errors) — fixture is complete
5. Build frontier: newly-created nodes have `parent=None` in cache, `get_parents` returns 0, walks stop naturally

6. Failed node filtering in `_read_and_cache` — `filter_failed_node_ids` on L2..root-1 before caching
7. Baseline comparison fix: pkl had stale roots → empty extraction. `_bl_extract_dir` now points to permanent cache dir.

8. Runner redesign: `StitchRun` → `BaselineRun` / `ProposedRun` class hierarchy. Baseline cached permanently (roots.pkl + extraction shards), protected with `force` param. `batched_extract_structure` skips if shards exist.
9. Deleted 13 unused modules (baseline.py, compare.py, current.py, hierarchy.py, inspect.py, proposed.py, proposed_v2.py, reader.py, topology.py, types.py, wave.py, worker_utils.py, stitch_types.py). Moved `StitchResult` into stitch.py.
10. Stitch phases renamed for clarity: read_upfront, merge_l2, discover_siblings, compute_dirty, build_hierarchy, build_rows.

### Status (2026-03-31)
- 86 unit tests pass
- **Real table wave 0: MATCH** — 311916 roots, all layers L2-L6 components match
- E2e emulator: prod 5222 vs ours 3401 (subset fixture, not full graph — separate issue)
- Baseline cached at `runs/wave/baseline/` (roots.pkl, extract/, stitch_results.json)

### Resolver elimination (2026-03-31)
Root cause of e2e mismatch: resolver stored stale parent chains from BigTable. When build_hierarchy created new parents, resolver entries stayed old → resolve_svs_to_layer returned old_L5 instead of new_L5 → missing CX connections at L5+ → over-merging.

Fix: eliminated resolver dict entirely. All resolution goes through cache.get_parents which has live hierarchy (including newly-created nodes from _create_parents). Also eliminated known_svs, resolver_entries in SiblingEntry, resolve_partner_sv_parents.

### Fixture extraction rewrite (2026-04-01)
- BFS from roots with multiprocessing Pool (fork) + tqdm
- L2 children = ACX source SVs (all layers) + edge SVs from wave files
- ACX target SVs added to their L2's children list + parent rows
- CX preserved for ALL nodes including L2s
- Table restored once, reused across edge counts
- `extract(edges_range=(1,4))` generates 1e, 2e, 3e fixtures
- Fixtures saved as gzip-compressed pickle (`.pkl.gz`)
- `FIXTURE_DIR` defined once in `sample_test_data.py`, imported by test
- Auto-generated `STATS.md` with per-fixture statistics
- `README.md` documents generation algorithm and fixture contents

### E2E results (2026-04-01)
**All 3 fixtures pass wave mode with full hierarchy match + sanity checks:**
- 1e: 1212 roots → 606 CCs, prod=606 ours=606 ✓ (90s)
- 2e: 2423 roots → 1211 CCs, prod=1211 ours=1211 ✓ (128s)
- 3e: 3633 roots → 1815 CCs, prod=1815 ours=1815 ✓ (162s)

Components match at ALL layers (L2-L7) for all fixtures.

Sanity checks per wave (`_WaveChecker`):
1. Pre-stitch: root-level edge graph → expected CC count
2. Post-stitch: root count == expected CCs
3. Post-stitch: every edge's SVs share a root

### TODO
- Run multiwave e2e
- Re-run on real table to verify no regression
- Fix L3 sibling discovery when skip connections bypass L3 (if needed — currently passing)
