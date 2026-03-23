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

Build all BigTable mutations and write in one batch:
- **New nodes (all layers):** `Child`, `CrossChunkEdge[layer]`, `Parent` columns
- **New L2 nodes:** also `AtomicCrossChunkEdge[layer]` (raw `[sv, sv]` format) — critical for future stitches
- **All children of new nodes:** `Parent` pointer updated to new parent

## Key design decisions

### Only AtomicCrossChunkEdge, never stored CrossChunkEdge
Stored `CrossChunkEdge` on L3+ nodes can go stale from prior stitches. `AtomicCrossChunkEdge` on L2 nodes is immutable. All connectivity at every layer is derived from L2 atomic edges by resolving both SV columns to identities at the target layer.

### Siblings are always L2
We get L2 descendants of old parents, not direct children. This means siblings at L3+ (which would have stale `CrossChunkEdge`) are never read. Instead, their L2 descendants are included, and the hierarchy is rebuilt from L2 up.

### AtomicCrossChunkEdge written on new L2 nodes
Future stitches will call `get_atomic_cross_edges` on these new L2 nodes. The merged atomic edges must be present.

## What makes this different from the current path

| | Current | Proposed |
|---|---|---|
| Hierarchy init | `_init_old_hierarchy`: traverse L2→root for all nodes, warm cache | `get_all_parents_dict_multiple` + `get_l2children` for sibling discovery |
| Cross edge source | Stored `CrossChunkEdge` (goes stale) + `LatestEdgesFinder` | `AtomicCrossChunkEdge` (immutable) + resolver |
| Partner resolution | Per-layer `get_roots` or stale search | One upfront `get_all_parents_dict_multiple`, then in-memory |
| Sibling handling | `get_children(old_parents)` at each layer, read their `CrossChunkEdge` | Get all L2 descendants once, read their `AtomicCrossChunkEdge` |
| Per-layer IO | Cross edge reads + stale resolution | Only `create_node_ids` |
| Neighbor updates | `_update_neighbor_cx_edges` patches counterparts | None — partners resolved from immutable source |
