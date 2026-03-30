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
- Fixture: `test_data/e2e_fixture.pkl` (836K nodes, ~26 MB) from `sample_test_data.py`
- BigTable emulator on port 8540, parallel populate (2 processes, 16K batch, ~49s)
- `set_max_node_id` per chunk after populate (skip root chunks)
- Production `add_edges` on `e2e_prod` vs our stitch on `e2e_ours`, compare components at all layers
- Pool(2 workers, spawn), `_e2e_worker` returns traceback strings for pickling

### Current blocker (2026-03-30)
Wave 0 fails: `KeyError: 266556802944993234` in `_discover_layer_siblings` → `get_all_parents_filtered` → `get_parents`.
- Node is L3, segment 2002. Fixture max segment in that chunk = 2000. Not in fixture.
- No fixture node has it as parent. Zero references in fixture ACX/CX.
- Created by our stitch's `batch_create_ids` during `build_hierarchy`.
- Then encountered as a sibling at a higher layer during `_discover_layer_siblings` before its parent is assigned.
- **Root cause**: algorithm ordering bug — newly-created node appears as sibling before its parent chain exists.
- **TODO**: trace `build_hierarchy` → `_discover_layer_siblings` ordering to fix.
