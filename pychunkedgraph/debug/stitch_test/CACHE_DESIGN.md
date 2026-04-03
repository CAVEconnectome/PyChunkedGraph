# Cache Design

## Reference Data
- **Pre-stitch backup**: `hsmith-mec-100gvx-exp16-0.26-backup` — READ ONLY, never edit.

## Core Principle
Derive CX from ACX (not stale BigTable CX). Components must match production at every layer.

## Hard Requirements
1. No row read more than once from BigTable (enforced by `_read_row_keys` assertion)
2. New rows never read from BigTable — we create them
3. No row written more than once per wave
4. No unnecessary writes — cx_hash comparison
5. Single source of truth — cache is the ONLY source for all data. No separate dicts.
6. No silent defaults — missing data must raise
7. All columns for a node read in one RPC via `_ensure_cached` (single batch, all node types)
8. Never use stale CX from BigTable — always derive from ACX
9. Never hardcode layer numbers — skip connections mean any layer can be skipped
10. No individual reads/writes — always batch.

## Retry Safety
- Timestamp-based: `acquire_stitch_timestamp` writes marker row, reads server timestamp T
- All reads use `end_time=T`, all writes use `time_stamp=T`
- On retry: same T, failed writes invisible
- `release_stitch_timestamp` deletes marker on success
- No filter_failed_node_ids needed — timestamp filtering replaces it

## Data Structures

### RowCache (row_cache.py)
```
CacheRow(__slots__): parent, children, acx, cx
RowCache: _local → _preloaded (COW on put)
```

### WaveCache (wave_cache.py)
- Mandatory `read_fn` constructor arg
- Batch methods: `get_parents`, `get_children_batch`, `get_acx_batch`, `get_cx_batch`
- `_ensure_read` handles cache misses via batch BigTable read through `read_fn`
- `SiblingEntry`: `unresolved_acx`, `partner_ids`
- `_collect_partner_svs` shared by `compute_dirty_siblings` and `_build_partner_ids`

### Single Source of Truth
- **All data**: through WaveCache batch APIs only. No separate dicts.
- **SV → L2**: `cache.get_parents(svs)` — walks parent chain from SV
- **Node → parent**: `cache.get_parents(nodes)` — direct parent lookup
- **Unresolved edges**: `unresolved_acx` — ACX with source remapped to node ID, target still raw SV
- **Lineage**: `new_to_old` — for sibling discovery. `old_to_new` — L2 only, for dirty detection.

### RpcEntry (utils.py)
```
@dataclass: label, n_requested, n_read, t_read, t_cache, t_total
```

## Algorithm

### Phase 1: read_upfront
- Read SV parents (batch)
- Read L2 nodes: Parent+Child+ACX+CX (single batch, all node types)
- Walk partner chains: batch per level

### Phase 2: merge_l2
- Build CC graph from L2 edges
- Allocate IDs (batch)
- Merge children+ACX+CX, set lineage

### Phase 2b: discover_siblings
- Get old parents' L2 children
- Split known/unknown siblings
- Read unknown siblings, ensure partners cached
- Dirty detection via _collect_partner_svs + searchsorted

### Phase 3: build_hierarchy (per layer)
- Prefetch partner SVs' parents
- resolve_cx_at_layer → resolve_svs_to_layer (proper progress tracking)
- Store resolved CX
- Build CC graph, create parents (skip connections for single-node CCs)
- Update counterpart CX
- Discover layer siblings

### Phase 4: build_rows → single batch BT write
- New nodes: Child, CX, ACX (L2 only)
- Parent pointers for children of new nodes
- Counterpart nodes: CX only
- Clean siblings: nothing written

### Phase 5: mutate_rows
Single batch write with stitch_timestamp.

## Sanity Checks (optional, SANITY_CHECK global)
- Layer check: all_nodes at each layer must be correct layer
- Parent reassignment: no child gets two new parents
- Post-stitch: no duplicate L2s across roots, all edge SVs covered
- Controlled by `run_proposed(sanity_check=True)`

## Multiwave
- SiblingEntry caches unresolved_acx, partner_ids across waves
- Dirty detection: vectorized searchsorted + partner_ids comparison
- accumulated_replacements carries old_to_new across waves
- Known siblings restored from cache, unknown read from BigTable

## Autoscaling
- Managed by `_autoscale(min_nodes)` context manager in StitchRun
- `bt_min_nodes` param on `run_proposed`/`run_baseline` entry points
- No-op when min_nodes=1
