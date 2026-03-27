# Stitch Redesign Session Log

## Dataset
- hsmith_mec 100GVx exp16 0.26
- 661 edge files across 32 waves (606 in wave 0)
- Backup: hsmith-mec-100gvx-exp16-0.26-backup
- Edges: gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges

## Module Structure

```
local_cg.py    — LocalChunkedGraph (owns WaveCache, all stitch phases)
wave_cache.py  — WaveCache (ChainMap cache, SiblingEntry, dirty check, flush_created)
tree.py        — Stateless: resolve_sv_to_layer, get_all_parents_filtered, restore_known_siblings
topology.py    — Stateless: acx_to_cx, resolve_cx_at_layer, store_cx_from_resolved, resolve_remaining_cx
stitch.py      — Orchestrator: phases in order
runner.py      — run_baseline, run_proposed, compare_run
CACHE_DESIGN.md — Hard requirements + design doc
```

Tests: test_stitch.py (26), test_wave_cache.py (41), test_local_cg.py (4) = 71 total

## Runs

### Baseline (production reference)
- Table: stitch_redesign_test_hsmith_mec_baseline_multiwave
- 347,471 roots, 32 waves

### Run 695be87b (pre-refactor reference)
- 347,471 roots MATCH, 3069s wall

### Run cc919417 (wave 0 matched, crashed wave 1 — written_cx bug)
- Wave 0: 285s, 50333s stitch — matches 695be87b

### Pending run (all fixes)
- accumulated_replacements, children_d, written_cx carry-forward, partner resolution before dirty check, batched resolve_remaining_cx

## Key Bugs Found (chronological)
1. **cx_edges undefined**: renamed to all_cx but store call not updated
2. **partition_siblings broken**: old_hierarchy missing sibling chains
3. **Stale cx_cache**: resolved against prior wave's node IDs
4. **Pool workers missing incremental**: _shared_inc not passed
5. **known_svs removed incorrectly**: caused 10M extra reads
6. **Pool worker count uncapped**: 96 workers for 9 files
7. **accumulated_replacements**: per-stitch old_to_new must include all prior waves
8. **children_d vs ChainMap**: wave_snapshot/end_stitch passed wrong dict
9. **Layer 3+ skip**: store/resolve must not skip siblings at layer 3+
10. **written_cx at layer 3**: stale due to child_to_parent changes
11. **written_cx carry-forward**: clean siblings' written_cx overwritten with empty cx_cache
12. **Partner resolution ordering**: collect_and_resolve_partners must run for ALL siblings BEFORE compute_dirty_siblings
