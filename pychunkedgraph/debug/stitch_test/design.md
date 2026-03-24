# Proposed Stitch Algorithm

## Phase 1: Upfront BigTable reads (done once)

1. **`get_parents(all_svs)`** — map each SV in the stitch edges to its L2 parent
2. **Classify edges by layer** — `get_cross_chunk_edges_layer` determines at which octree layer each edge becomes relevant (layer 1 = within chunk, layer 2+ = cross chunk)
3. **`get_children(l2ids)`** — get SV children of each affected L2 node
4. **`get_atomic_cross_edges(l2ids)`** — read immutable `AtomicCrossChunkEdge[layer]` for all affected L2 nodes. Format: `[sv, sv]` at each layer
5. **`get_all_parents_dict_multiple(partner_svs)`** — for every partner SV in column 1 of atomic cross edges (nodes outside our stitch scope), read their full parent chain `{layer: parent_id}` in one batch. This is the key read that lets us resolve partners at any layer without further IO.

**Result:** `resolver` dict — `{sv: {layer: identity_at_that_layer}}` for all SVs (ours + partners).

## Phase 2: L2 merge (in-memory + ID allocation)

- Build graph from layer-1 L2 edges, find connected components
- Allocate new L2 IDs per component
- For each new L2 node:
  - Merge SV children from old L2 nodes
  - Merge `AtomicCrossChunkEdge` from old L2 nodes. Store raw `[sv, sv]` in `l2_atomic_cx` for BigTable write.
  - For hierarchy building: `node_cx[new_id]` has `[new_id, partner_sv]` (col 0 = new L2 ID, col 1 = raw partner SV)
  - Stitch cross edges also stored in same format
  - Track `old_to_new` mapping

## Phase 2b: Discover siblings

The old parents of affected L2 nodes have other L2 descendants (siblings). ALL of them must participate in the hierarchy rebuild because the old parents are being invalidated.

We get **L2 descendants** (not direct children) of old parents. This ensures we capture siblings at L2 regardless of how many intermediate layers exist or skip connections. We never read stored `CrossChunkEdge` from L3+ nodes — all connectivity is derived from L2 `AtomicCrossChunkEdge`.

1. **`get_all_parents_dict_multiple(l2ids)`** — full parent chains for affected L2 nodes → find all old parents at every layer
2. **`get_l2children(old_parents)`** — get ALL L2 descendants of old parents → these are the siblings
3. **`get_atomic_cross_edges(l2_siblings)`** — siblings' immutable cross edges
4. **`get_children(l2_siblings)`** — siblings' SV children
5. **`get_all_parents_dict_multiple(sibling_partner_svs)`** — resolve siblings' partner SVs for the resolver

Siblings are added to `siblings_d[2]` (tracked separately from `new_ids_d`). Their `node_cx`, `resolver`, `atomic_cx`, and `children_d` entries are populated so they participate in hierarchy building alongside new L2 nodes. Their `AtomicCrossChunkEdge` is also stored in `l2_atomic_cx` for writing.

## Phase 3: Layer-by-layer parent propagation (in-memory)

For each layer L from 2 to root:

### Nodes at this layer
- `new_ids_d[L]` — newly created nodes at this layer (from prior iteration or L2 merge)
- `siblings_d[L]` — existing sibling nodes at this layer (only populated at L2 from Phase 2b)
- Combined into `all_nodes` for this layer

### Step 1 — Resolve cross edges at layer L
- For each node, look up `node_cx[node][L]` — its cross edges at this layer
- Column 0 is already the current node ID
- Column 1 is a partner SV. Resolve it via `_resolve_sv_to_layer`:
  1. `resolver[sv]` gives `{layer: identity}` from upfront read
  2. For our own SVs: resolver has `{2: old_l2}`. Apply `old_to_new` to get new_l2, then walk `child_to_parent` to reach current layer.
  3. For external SVs: resolver has the full chain from `get_all_parents_dict_multiple`.
  4. Stop walking `child_to_parent` if it would overshoot `target_layer` (skip connections).
- Result: cross edges as `[our_node_at_L, partner_node_at_L]`

### Step 2 — Connected components
- Build graph from resolved cross edges + self-edges for all nodes at this layer
- Find connected components

### Step 3 — Create parents
- For singletons: check `node_cx` for cross edges at any higher layer (from cache, no IO). Skip to the lowest layer that has them. If none, skip to root.
- Allocate parent IDs (`id_client.create_node_ids` — small IO)
- Parent inherits children's cross edges at layers >= parent_layer, with column 0 remapped to parent ID
- Update `child_to_parent[child] = parent` so future layer resolution works
- Store in `cg.cache` (children, parents, cross edges)
- New parents go into `new_ids_d[parent_layer]` for the next iteration

## Phase 3b: Resolve cross edges for writing

After hierarchy is built, `node_cx` still has raw partner SVs in column 1. For writing to BigTable as `CrossChunkEdge`, column 1 must be resolved to the partner's identity at the appropriate layer.

Uses `_resolve_sv_to_layer` with the fully-built `child_to_parent` map. Stores resolved edges in `cg.cache.cross_chunk_edges_cache`.

## Phase 4: Write mutations

Two-phase write for crash safety:

**1. Node entries (first):**

| Node type | Child | CrossChunkEdge | AtomicCrossChunkEdge |
|-----------|-------|---------------|---------------------|
| New L2 | SVs | Resolved at L2 layer | Raw [sv, sv] (merged) |
| L3-L6 (new) | children | Resolved at that layer | — |
| Root (new) | children at L6 | Empty | — |
| L2 siblings (existing) | — (skip, unchanged) | Resolved (updated) | — (skip, unchanged) |

**2. Parent entries (second):** For all children of new nodes (includes siblings as children of new L3+ parents).

Root nodes are added to `ctx.new_node_ids` after the hierarchy loop.

All state stored in `StitchContext` dataclass — no `cg.cache` usage.

### Deferred root ID allocation
Root IDs are NOT allocated during the layer-by-layer hierarchy build. Instead, CCs that skip to root are collected in `deferred_roots`. After the hierarchy loop, `_allocate_deferred_roots` batch-allocates all root IDs in a single `create_node_ids` call + one collision check.

This is safe because roots have no cross edges (empty at root layer), are never resolved by `_resolve_sv_to_layer` (walk always breaks before root), and don't participate in CC building.

**Why deferred:** Root layer uses 256 sharded counters. On backup-restored tables, new allocations collide with existing roots. The collision check (`read_nodes`) is an RPC — deferring consolidates multiple per-layer RPCs into one. All root IDs share one chunk (layer_count, 0, 0, 0).

Non-root layers use sequential per-chunk counters — no collision possible, no deferral needed.

## Key design decisions

### Only AtomicCrossChunkEdge, never stored CrossChunkEdge
Stored `CrossChunkEdge` on L3+ nodes can go stale from prior stitches. `AtomicCrossChunkEdge` on L2 nodes is immutable. All connectivity at every layer is derived from L2 atomic edges by resolving both SV columns to identities at the target layer.

### Siblings are always L2
We get L2 descendants of old parents, not direct children. This means siblings at L3+ (which would have stale `CrossChunkEdge`) are never read. Instead, their L2 descendants are included, and the hierarchy is rebuilt from L2 up.

### AtomicCrossChunkEdge written on new L2 nodes
Future stitches will call `get_atomic_cross_edges` on these new L2 nodes. The merged atomic edges must be present.

### No neighbor CrossChunkEdge updates
Partner nodes' `CrossChunkEdge` is NOT updated (goes stale). This is acceptable because:
- Future proposed stitches read `AtomicCrossChunkEdge` (immutable), not `CrossChunkEdge`
- Human proofreading edits via `add_edges` already handle stale edges via `LatestEdgesFinder`

### No locks
Lock-free design enables true parallelism within waves. Edge files within a wave touch independent boundary regions with independent roots — no coordination needed.

### No FormerParent
Proposed path does not write FormerParent/deprecation entries. Only relevant for proofreading edit history (bookmarks, annotations), not bulk stitching.

## What gets written to BigTable per node type

| Node type | Child | CrossChunkEdge | AtomicCrossChunkEdge | Parent (on children) |
|-----------|-------|---------------|---------------------|---------------------|
| New L2 | SVs | Resolved at L2 layer | Raw [sv, sv] merged from old L2s + stitch edges | Yes |
| L3-L6 | children at layer below | Resolved at that layer | — | Yes |
| Root (L7) | children at L6 | Empty (no cx at root) | — | Yes |

**CrossChunkEdge format**: `[node_id, resolved_partner_id]` — partner resolved to their identity at the node's layer. Written for the current `add_edges` path's consumption (proofreading).

**AtomicCrossChunkEdge format**: `[sv, sv]` — immutable, raw. Only on L2 nodes. This is what the proposed path reads for future stitches.

## Cross edge data flow through the algorithm

1. **Read** (Phase 1): `get_atomic_cross_edges(l2ids)` → `{l2: {layer: [[sv, partner_sv]]}}`
2. **Merge** (Phase 2): old L2 edges + stitch edges → `l2_atomic_cx[new_l2]` (raw, deduplicated)
3. **Propagate** (Phase 3): `node_cx[node] = {layer: [[node_id, partner_sv]]}` — col 0 remapped to current node, col 1 still raw SV. Parent inherits child edges at layers >= parent_layer.
4. **Resolve** (Phase 3b): `_resolve_sv_to_layer` transforms col 1 from raw SV → partner identity at node's layer. Uses resolver (upfront parent chains) + old_to_new + child_to_parent.
5. **Write** (Phase 4): `CrossChunkEdge[L]` = resolved. `AtomicCrossChunkEdge[L]` = raw (L2 only).

## What makes this different from the current path

| | Current | Proposed |
|---|---|---|
| Hierarchy init | `_init_old_hierarchy`: traverse L2→root for all nodes, warm cache | `get_all_parents_dict_multiple` + `get_l2children` for sibling discovery |
| Cross edge source | Stored `CrossChunkEdge` (goes stale) + `LatestEdgesFinder` | `AtomicCrossChunkEdge` (immutable) + resolver |
| Partner resolution | Per-layer `get_roots` or stale search | One upfront `get_all_parents_dict_multiple`, then in-memory |
| Sibling handling | `get_children(old_parents)` at each layer, read their `CrossChunkEdge` | Get all L2 descendants once, read their `AtomicCrossChunkEdge` |
| Per-layer IO | Cross edge reads + stale resolution | Only `create_node_ids` |
| Neighbor updates | `_update_neighbor_cx_edges` patches counterparts | None — partners resolved from immutable source |
