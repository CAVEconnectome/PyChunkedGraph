# Edge updates after a supervoxel split

## Context

A supervoxel split rewrites voxels inside a bbox: a single old SV is replaced by N new fragments (one per chunk × per side of the cut). Every atomic edge that referenced the old SV — to neighbors inside the same root, to neighbors in a different root, and to other pieces of the same physical supervoxel — must now reference an appropriate fragment instead, or the graph hierarchy diverges from the new segmentation.

Edge update is the second half of `split_supervoxel`. The first half produced a labeled bbox, an `old_new_map` (`old_sv_id → set[new_sv_ids]`), and a `new_id_label_map` (`new_sv_id → cut-side label`). This document covers what happens from there.

## Algorithm overview

```
inputs from voxel-level split
  ├─ new_seg           bbox volume with new SV IDs in place of old
  ├─ old_new_map       which old SVs got split, and into which new IDs
  └─ new_id_label_map  for each new ID, which side of the cut it's on

update_edges (edges_sv.py):
  1. fetch atomic subgraph inside bbox, rooted at the rep's root
  2. dedupe edges, drop self-loops
  3. group by partner-root vs split-root  →  active / inactive
  4. for each old SV:
       inactive partners  → broadcast edge to every fragment
       active partners    → expand split partners, match by label/proximity
       intra-fragment     → low-affinity edges between every fragment pair
  5. validate (no cross-label inf bridges, no self-loops, completeness)
  6. return new (edges, affinities, areas)

add_new_edges (edges_sv.py):
  1. duplicate bidirectional, group by L2 parent chunk
  2. per chunk: append to SplitEdges (history) and rewrite
     CompactedSplitEdges (snapshot, with stale rows filtered)
```

## Inputs to `update_edges`

- `cg, root_id, bbox` — the rep's root and the bbox the voxel-level cut acted on.
- `new_seg` — segmentation in the read window (bbox + 1-voxel shell). The shell is what makes anchor lookups work for unsplit pieces of the rep on the other side of a chunk boundary; without it, cross-chunk edges from those pieces would route to whatever happens to lie at the boundary face, not to the actual fragment the cut produced.
- `old_new_map` — drives which edges need re-routing.
- `new_id_label_map` — used to pair fragments with the same cut-side label across cross-chunk edges.

`update_edges` calls `cg.get_subgraph(root_id, bbox, bbox_is_coordinate=True)`. This returns every atomic edge whose endpoint sits in the bbox under the rep's root. That set already includes both intra-cut edges (between split SVs) and the cross-chunk-shell edges to neighbors outside the rewritten region.

After fetch, edges are sorted within each pair, deduped, and self-loops filtered. The remaining set is the input to classification.

## Classification

For each edge, the partner's root determines the routing path. `sv_root_map` is built from one batched `cg.get_roots(...)` over all unique partners.

### Inactive partner (`partner_root != root_id`)

The partner sits in a different agglomerated object. The split's cut-side has no semantic relationship to that neighbor — *any* fragment of the old SV that touched the neighbor's voxels still touches them after the split. **Broadcast**: for each old SV split into N fragments, copy the edge to every fragment, preserving affinity and area.

This intentionally over-creates edges. They cost nothing if both endpoints stay in different roots forever; they collapse harmlessly into a single root-level edge if the two roots later merge.

### Active partner (`partner_root == root_id`)

The partner is inside the same agglomerated object as the rep — the partner is either:

- another piece of the same physical SV (cross-chunk-connected),
- a different SV in the same root reachable via L2 hierarchy.

For active partners, edges are routed based on affinity type:

#### Inf-affinity, partner also split

The partner SV is itself in `old_new_map` (e.g. it's another piece of the rep that was rewritten). We need each new fragment of the old SV to connect to the *matching-label* new fragment of the partner — the one on the same cut-side. `_match_by_label` does this lookup via `new_id_label_map`. If no fragment of the partner shares the source SV's label (rare, indicates a partial split), fallback to the closest fragment by distance.

#### Inf-affinity, partner unsplit

This is the cross-chunk edge to a piece of the rep that the bbox didn't include — by construction with the seed-driven bbox, these are the rep's far-away pieces that keep their old IDs. The unsplit partner has no `new_id_label_map` entry.

**Critical**: do *not* broadcast this edge to all fragments. An unsplit partner connected via inf-affinity to fragments on both sides of the cut would form an uncuttable bridge — a future mincut on this object would route through `frag_a → unsplit_partner → frag_b` with infinite affinity and never separate them. So `_match_inf_unsplit` assigns the edge to exactly one fragment: the one closest to the partner.

`validate_split_edges` enforces this with check (A): no inf-affinity edge from an unsplit partner to fragments with different cut-side labels.

#### Finite-affinity (regular)

Real adjacency edges between SVs based on per-pair affinity. `_match_by_proximity` assigns the edge to *every* fragment within `cg.meta.sv_split_threshold` voxels of the partner, fallback to closest if none qualify. Multiple fragments may legitimately neighbor the partner; the threshold preserves the original adjacency where it actually exists.

### Intra-fragment edges

For each old SV split into multiple new fragments, add a low-affinity (0.001) edge between every pair of fragments. These are cuttable by future mincut operations — they record that the fragments share a graph-level neighborhood (they came from the same SV) without forcing them to stay agglomerated. Without these edges, an entirely-disconnected fragment of an old SV would have no link to the rest of the object; with them, the standard mincut machinery handles the relationship correctly.

## Distance computation

Distances drive both proximity matching and the closest-fragment fallback. Each fragment gets a `cKDTree` over its voxels (built from `build_coords_by_label(new_seg)`).

- **Partner inside bbox**: build a kdtree on the partner's voxels too. For each fragment, the smaller-tree-queries-larger heuristic minimizes work; result is the minimum voxel distance.
- **Active partner outside bbox**: the partner's voxels aren't in `new_seg`. `_compute_boundary_distances` uses the partner's chunk coordinate to determine which face of the source chunk the edge crosses, then measures each fragment's distance to that boundary plane. This is an over-estimate for non-boundary-aligned partners but it's the only signal available without extra reads.

## Validation

`validate_split_edges` checks four invariants and raises `PostconditionError` on any violation. Failures abort the operation cleanly under the indefinite L2 chunk lock; the recovery flow then handles cleanup.

| Check | Why |
|-------|-----|
| (A) No inf-affinity bridges between cut-sides via an unsplit partner | Would be uncuttable by future mincuts |
| (B) No self-loops | Indicates a routing bug; would skew degree counts and break some traversal assumptions |
| (C) Every old SV has at least one replacement edge from its fragments | Catches old SVs that vanished from the edge set entirely (would orphan them in the hierarchy) |
| (D) All fragment pairs of each old SV are connected | Confirms the intra-fragment low-affinity edges were emitted |

These run before any bigtable write, so the validation is the last line of defense before the writes commit under the lock.

## Persisting: `add_new_edges`

The new edges are batched into bigtable per L2 chunk. Two columns get written per chunk per op:

### `SplitEdges` (history)

An append-only column. Each split op writes its new edges as a fresh cell with the op's logical timestamp. Time-travel reads at any timestamp T walk all cells with `ts ≤ T`, then apply the stale-edge resolution path to filter out edges whose endpoints have been superseded by later ops. This is the authoritative store for historical reads.

### `CompactedSplitEdges` (snapshot)

A latest-only column for fast current-time reads. On each op:

1. Read the previous compacted cell (if any) plus its matching `CompactedAffinity` and `CompactedArea`.
2. Filter out rows whose endpoints reference any old SV in `old_new_map.keys()` (these are the SVs that just got split — their edges are stale).
3. Concatenate the new rows.
4. Write the whole thing as one fresh cell.

Current-time readers can take this single cell directly without history walks or stale-edge resolution.

The chunk grouping uses each edge's first endpoint's L2 parent chunk: `cg.get_chunk_ids_from_node_ids(cg.get_parents(nodes))`. Parent chunks (not the SV's own L1 chunk) is the correct routing — the edge belongs to the chunk where its endpoint lives in the L2 hierarchy. Bidirectional duplication ensures every edge is owned by both endpoints' chunks; readers picking up either side find it.

Both writes use `time_stamp=task.operation_ts`, so all rows from one op land at the same logical time. Concurrent SV-splits on disjoint chunks don't interfere because they write disjoint chunk rows.

## Invariants

- For every old SV in `old_new_map`, every atomic edge that referenced it in the pre-split graph has at least one corresponding edge among its fragments after the split.
- No inf-affinity edge crosses cut-sides through an unsplit partner.
- Every cross-chunk piece of the rep that the bbox didn't include keeps its old ID and its existing edges resolve unchanged (because no edge in those rows references the now-split SVs at endpoints — the routing only touches edges whose endpoints are in the bbox or its 1-voxel shell).
- `SplitEdges` and `CompactedSplitEdges` agree at the latest timestamp: the compacted snapshot is the result of replaying the history through the stale-edge filter.
