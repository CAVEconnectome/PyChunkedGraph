# Cache Design

## Reference Data
- **Pre-stitch backup**: `hsmith-mec-100gvx-exp16-0.26-backup` — READ ONLY, never edit. Use to explore pre-stitch trees.

## Core Principle
Derive CX from ACX (not stale BigTable CX). Components must match production at every layer. CX on each node serves two purposes: (1) same-layer CX for CC computation, (2) all-layer CX as cache for parent CX derivation in future edits.

## Hard Requirements
1. No row read more than once from BigTable (enforced by `_read_row_keys` assertion)
2. New rows never read from BigTable — we create them
3. No row written more than once per wave
4. No unnecessary writes — cx_hash comparison
5. Single source of truth — cache is the ONLY source for all data. No separate dicts.
6. No silent defaults — missing data must raise
7. All columns for a node read in one RPC via `_ensure_read`
8. Never use stale CX from BigTable — always derive from ACX
9. Never hardcode layer numbers — skip connections mean any layer can be skipped
10. No individual reads/writes — always batch. Design for batch is top priority to reduce IO latency.

## Data Structures

### RowCache (row_cache.py)
```
CacheRow(__slots__): parent, children, acx, cx
RowCache: _local → _preloaded (COW on put)
```

### WaveCache (wave_cache.py)
- Mandatory `read_fn` constructor arg — no None default
- Batch methods: `get_parents`, `get_children_batch`, `get_acx_batch`, `get_cx_batch`
- `_ensure_read` handles cache misses via batch BigTable read through `read_fn`
- `SiblingEntry`: `unresolved_acx`, `partner_ids` (no resolver_entries)
- `compute_dirty_siblings` uses `cache.get_parents` for SV→L2 lookup
- `_build_partner_ids` shared by `save_wave_state` and `inc_snapshot_from`

### Single Source of Truth
- **All data**: through WaveCache batch APIs only. No separate dicts (resolver, child_to_parent, etc.)
- **SV → L2**: `cache.get_parents(svs)` — walks parent chain from SV
- **Node → parent**: `cache.get_parents(nodes)` — direct parent lookup
- **Unresolved edges**: `unresolved_acx` — ACX with source remapped to node ID, target still raw SV
- **Lineage**: `new_to_old` — for sibling discovery. `old_to_new` — L2 only, for dirty detection.

## Algorithm

### Phase 1: read_upfront

| Step | BT read (batched) | Cache update |
|------|-------------------|--------------|
| Read SV parents | batch: all SVs → Parent | RowCache parent |
| Read L2 nodes | batch: all L2s → Parent+Child+ACX[2..N] | RowCache parent, children, acx |
| Walk parent chains | batch per level: parents → Parent+Child | RowCache |

### Phase 2: merge_l2

| Step | BT read | Cache update |
|------|---------|--------------|
| Build CC graph | none | — |
| Allocate IDs | batch: id_client | new_ids_d[2] |
| Merge children+ACX | none | cache children, unresolved_acx |
| Update cache parents | none | cache.get_parents returns new L2 for merged SVs |
| Set lineage | none | old_to_new, new_to_old |

### Phase 2b: discover_siblings

| Step | BT read (batched) | Cache update |
|------|-------------------|--------------|
| ensure_partners_cached | batch: unknown partner SVs → Parent+Child+ACX | RowCache |
| Walk partner chains | batch per level → Parent+Child | RowCache |
| Build unresolved_acx | none (from cached acx) | unresolved_acx[sib] |
| Dirty detection | none | dirty_siblings set |

### Phase 3: build_hierarchy (per layer)

**3a. Resolve CX**

| Step | BT read | Cache update |
|------|---------|--------------|
| acx_to_cx via resolve_svs_to_layer | none (all from cache.get_parents) | cx_cache |

**3b. CCs + create parents**

| Step | BT read | Cache update |
|------|---------|--------------|
| Build CC graph | none | — |
| Allocate IDs | batch: id_client | new_ids_d[parent_layer] |
| Set parent/children | none | cache parent, children |
| Merge unresolved_acx | none | unresolved_acx[parent] |
| Update lineage | none | new_to_old |

**3c. Update counterpart CX (matching production _update_neighbor_cx_edges)**

| Step | BT read (batched) | Cache update |
|------|-------------------|--------------|
| Collect ALL CX targets of new nodes | none (from cx_cache) | counterpart set |
| Filter to out-of-scope targets | none | counterpart set (remove siblings + new nodes) |
| Batch read counterparts' CX | batch: all counterparts → CX[all layers] | counterpart_cx dict |
| Remap old→new in counterpart CX | none | counterpart_rows (for build_rows) |

### Phase 4: build_rows (→ single batch BT write)

| What | Columns written | Condition |
|------|----------------|-----------|
| New nodes | Child, CX[layers], ACX[layers] (L2 only) | Always |
| Parent pointers | Parent (for children of new nodes) | Always |
| Counterpart nodes | CX[layers] | Sibling/external found as counterpart |
| Clean siblings | — | NOTHING |

**CX format per node:** CX at layer L on a node at layer M has targets at layer M (node's own layer). This matches production's `_update_cross_edge_cache_batched` which remaps to `parent_layer`.

**CX as cache for future edits:** Children's CX at all layers acts as pre-resolved cache. When a future edit creates a new parent at layer L, `_update_cross_edge_cache_batched` reads children's CX from BigTable/cache, aggregates, and remaps to L. Without correct children CX, the parent derivation fails or produces stale results.

**Children write-back:** Production writes resolved CX back to children after stale edge resolution. Our ACX-based derivation achieves the same result — we always resolve from immutable ACX, so children's CX is fresh by construction.

### Phase 5: mutate_rows

Single batch write of ALL rows from Phase 4.

## Multiwave Optimizations
- SiblingEntry: caches unresolved_acx, partner_ids across waves
- Dirty detection: vectorized searchsorted + partner_ids comparison
- accumulated_replacements: carries old_to_new across waves
- Known siblings restored from cache, unknown read from BigTable

## Emulator Test Setup
- After populating emulator table from fixture, must call `set_max_node_id(chunk_id, max_node_id)` per chunk to prevent `batch_create_ids` from allocating IDs that collide with pre-existing fixture nodes
- Skip root chunks — root layer handles collision internally via its own counter scheme

## Autoscaling
- Wave 0: min_nodes=5
- After wave 1: min_nodes=1
- Before extraction: min_nodes=5
