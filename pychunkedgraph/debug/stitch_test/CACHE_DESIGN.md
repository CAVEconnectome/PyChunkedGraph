# Cache Design

## Hard Requirements
1. No row read more than once from BigTable ever — once read, stays in memory across all waves
2. New rows never read from BigTable — we create them, we have them
3. No row written more than once per wave — all columns in one mutate_row call
4. Sibling CX only written when changed — dirty check prevents redundant writes

## Cache Structure

```
parents  = ChainMap(_parents_local, ro_p)
children = ChainMap(_children_local, ro_c)
acx      = ChainMap(_acx_local, ro_a)
```

`_*_local`: reads + created data. `ro_*`: fork COW preloaded (immutable).

`flush_created()` copies per-stitch caches into `_*_local` before snapshot/save.

## old_to_new: Accumulated

Per-stitch `old_to_new = {**accumulated_replacements, **current_wave}`.

Ensures `resolve_sv_to_layer` remaps partner SVs replaced in ANY prior wave.

## Sibling Categories

**known** (have SiblingEntry): restore raw_cx_edges + resolver_entries, skip BigTable reads.
**unknown** (new): full BigTable read.

**dirty** (partner L2 identity in `old_to_new.keys() | new_l2_ids`): CX changed, must resolve + store + write.
**clean**: CX unchanged AT LAYER 2 ONLY. Layer 3+ always changes (child_to_parent rebuilt each wave).

## Critical ordering in discover_siblings

```
1. split_known_siblings → known vs unknown
2. restore_known_siblings → populate resolver_entries, raw_cx_edges, children_d
3. read unknown siblings from BigTable
4. collect_and_resolve_partners for ALL siblings (known + unknown)
   → partner SVs get resolver entries
   → MUST run before compute_dirty_siblings
5. compute_dirty_siblings → uses resolver to check partner L2 identities
6. setup unknown siblings' per-stitch state
```

Step 4 before step 5 is critical: dirty check needs resolver entries for partner SVs to detect if their L2 identity was replaced. Without this, partner SVs default to raw SV ID → not found in old_to_new → false clean.

## build_hierarchy Per-Layer Logic

### Layer 2
- Resolve: dirty + new nodes only (skip clean siblings)
- CC graph: resolved edges + clean siblings' `written_cx[2]` injected
- Store to cx_cache: resolved edges only (clean siblings absent from cx_cache at layer 2)

### Layer 3+
- Resolve: ALL nodes (child_to_parent changes every wave)
- CC graph: all resolved edges
- Store to cx_cache: ALL nodes

### resolve_remaining_cx
Batched by layer (uses `resolve_cx_at_layer` + `store_cx_from_resolved`). Only processes layers > 2. Layer 2 handled by build_hierarchy layer loop.

## written_cx lifecycle

SiblingEntry stores `written_cx` — the CX last written to BigTable for this sibling.

**Carry-forward rule**: `save_wave_state` / `inc_snapshot_from` use `cx_cache.get(sib, prior_written_cx)`:
- Dirty sibling: in cx_cache → written_cx updated
- Clean sibling: NOT in cx_cache → written_cx carried forward from prior wave's SiblingEntry

## Snapshot

### Pool (wave_snapshot)
```
flush_created()
local_snap = (_parents_local, _children_local, _acx_local)
inc_snap = {old_to_new, new_node_ids, sibling_data with written_cx}
```
**children param = `children_d`** (per-stitch sibling children), NOT `children` ChainMap.

### In-process (end_stitch → save_wave_state)
Same — children param = `children_d`.

### Parent merge
```
merge_reader(local_snap)  → _*_local
merge_inc(inc_snap)       → accumulated_replacements, _siblings
```
