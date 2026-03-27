# Stitch Redesign

Lock-free, batch-read stitch using immutable AtomicCrossChunkEdge.
Replaces `cg.add_edges` proofreading path.

## Algorithm (stitch.py)

### Phase 1: Upfront reads
1. `get_parents(svs)` — SV → L2 parent
2. Classify edges by layer
3. `read_l2(l2ids)` — children + ACX for affected L2 nodes (all columns in one read)
4. `filter_orphaned_nodes` — discard L2 nodes from failed prior stitches
5. `resolve_partner_sv_parents` — partner SVs → L2 parents (sampling for >25K)
6. `get_all_parents_filtered` — full parent chains for our L2s + partner L2s

Result: `StitchContext` with resolver `{sv: {layer: identity}}`, children, ACX, old hierarchy.

### Phase 2: Merge L2
- Build L2 connectivity graph from within-chunk edges, find CCs
- Allocate new L2 IDs, merge children + ACX per CC
- Update resolver with new SV→L2 mappings

### Phase 2b: Discover siblings
- Walk old hierarchy to find all L2 descendants of old parents
- Read sibling L2 data (children + ACX)
- **Incremental**: partition siblings into affected (under replaced parents) vs unchanged (reuse cached CX from prior wave)
- Resolve partner chains only for affected siblings

### Phase 3: Build hierarchy
- Layer by layer (2 → root):
  - `_resolve_cx_at_layer`: resolve partner SVs to layer-L identity
  - `store_cx`: sort+split resolved edges into `cx_cache` per node (for writing)
  - Build CC graph from resolved cross edges
  - `_create_parents`: allocate parent IDs, propagate raw CX upward (sets for dedup)
- `_resolve_remaining_cx`: resolve higher CX layers for nodes that only participated at lower layers (e.g., L2 nodes' CrossChunkEdge[3,4,5])
- `_allocate_roots`: batch root ID allocation with collision check

### Phase 4: Build entries
- One mutation per unique row key
- New nodes: Child + CrossChunkEdge + AtomicCrossChunkEdge + Parent pointers
- Siblings: CrossChunkEdge only (pre-filtered to non-empty)

## Cache (cache.py)

`CachedReader`: ChainMap(local, preloaded) for parents/children/ACX.

Contract:
- Every row read once with ALL columns for its layer (SVs: Parent only, L2: Parent+Child+ACX, L3+: Parent+Child)
- `_ensure_cached`: single entry point, checks `_parents` as source of truth for "has been read"
- No partial cache — no row is ever re-read

`WaveCache`: manages `_shared_cache` between waves.
- Two-pass merge: reader caches first, ctx caches second (ctx wins)
- Workers inherit via fork COW

## Incremental (incremental.py)

`IncrementalState`: persists across waves.
- Tracks `previous_cx_cache`, `previous_siblings`, `previous_old_to_new`, `previous_new_node_ids`
- Pool workers return `inc_snap`, parent merges after each wave
- `partition_siblings`: affected = under replaced parents or new. Unchanged = reuse cached CX.
- Wave 11 data: 4% affected, 96% unchanged, partner chains halved

## Runner (runner.py)

- `run_proposed(experiment)`: restore → warm cache → wave 0 (large pool) → waves 1+ (pool or in-process)
- New Pool per wave (workers must see latest `_shared_cache`)
- `_check_roots`: validates roots_before + new_roots against production reference
- `compare_run(run_id)`: re-run comparison without re-stitching
- Memory tracking via psutil RSS

## Graph init (graph_init.py)

- `prepare_shared_init`: read meta + cv_info once
- `pool_init`: deserialize meta in worker
- `create_cg`: create CG from cached meta (no BigTable/GCS reads)

## Reference data

- `runs/reference.json`: 661 per-file results from production stitch
- `runs/multiwave/baseline/`: baseline extraction for comparison

## Test coverage (43 tests)

- `test_cache.py` (13): preloaded values, merge ordering, cache hits
- `test_graph_init.py` (5): no remote reads in threads/processes, pickle roundtrip
- `test_stitch.py` (15): SV resolution, CX resolution, ACX conversion, store_cx, CX propagation
- `test_incremental.py` (10): partition logic, affected parents, multi-wave evolution
