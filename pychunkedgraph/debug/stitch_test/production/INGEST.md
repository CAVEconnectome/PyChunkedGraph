# Production Ingest Algorithm

How the ChunkedGraph is built from scratch during initial ingestion.

## Entry Point
`pychunkedgraph/ingest/create/` — processed chunk-by-chunk, layer-by-layer.

## Phase 1: Atomic Layer (L2 creation)
**File:** `ingest/create/atomic_layer.py`

```
INPUT: chunk_edges_d = {in_chunk, between_chunk, cross_chunk}, isolated SVs

# 1. Collect all SVs and in-chunk edges
chunk_node_ids = unique SVs from all edge types + isolated
chunk_edge_ids = in_chunk edges + self-edges for all SVs
    # Self-edges ensure isolated SVs form their own component

# 2. Connected components → L2 nodes
graph = build_gt_graph(chunk_edge_ids)
ccs = connected_components(graph)
    # Each CC = one L2 node. SVs connected by in-chunk edges merge.

# 3. Allocate L2 IDs
parent_ids = id_client.create_node_ids(parent_chunk_id, count=len(ccs))

# 4. For each CC:
for parent_id, sv_ids in zip(parent_ids, ccs):
    # Write SV → L2 parent pointer
    for sv in sv_ids:
        write(sv, {Parent: parent_id})

    # Collect outgoing edges (between_chunk + cross_chunk) for all SVs
    out_edges = concat([get_outgoing_edges(sv) for sv in sv_ids])
        # out_edges format: [[SV_self, SV_partner], ...]
        # These are raw SV-level edges crossing chunk boundaries

    # Determine ACX layer for each edge
    # Layer = lowest layer where the two SVs' chunks differ
    cce_layers = get_cross_chunk_edges_layer(out_edges)
        # Algorithm: start at L1 coords, divide by FANOUT each layer,
        # increment layer while coords differ.
        # L2 edge: adjacent L1 chunks. L3 edge: different L2 chunks. etc.

    # Store L2 node with children + ACX
    write(parent_id, {
        Child: sv_ids,
        AtomicCrossChunkEdge[layer]: out_edges_at_layer  # for each layer
    })
```

**Key: L2 nodes store ACX (raw SV edges), NOT CX. CX is derived later during parent creation.**

**ACX column family (family_id=3) has a BigTable garbage collection rule.** ACX columns are eventually deleted (by version or age). This is why production edits cannot rely on ACX — by the time an edit runs, ACX may no longer exist. Edits use children's CX (family_id=4, no GC rule) instead. Our stitch runs immediately after ingest when ACX is still available, so we CAN use ACX as source of truth.

## Phase 2: Parent Layer Creation (L3+)
**File:** `ingest/create/parent_layer.py`

Processed one chunk at a time, layer by layer from L3 to root.

```
INPUT: layer_id, coords of parent chunk

# 1. Read children from child chunks
children_ids = read all nodes from child chunks at (layer_id - 1)

# 2. Get cross-chunk edges connecting children
if layer_id == 3:
    # L2 children only have ACX (no CX yet) — read atomic edges
    cx_edges_d = get_atomic_cross_edges(children_ids)
else:
    # L3+ children already have CX from prior layer processing
    cx_edges_d = get_cross_chunk_edges(children_ids, raw_only=True)
    # raw_only=True: read directly from BigTable, no cache

# 3. Build CC graph
edges = cx_edges_d values + self-edges for all children
graph = build_gt_graph(edges)
ccs = connected_components(graph)
    # Children connected by cross-chunk edges form CCs → one parent each

# 4. Skip connection detection
for each CC:
    if single-node CC:
        parent_layer = node_layer_d.get(node, root_layer)
            # node_layer_d tracks the LOWEST layer where this node
            # has cross-chunk edges. If none → root.
            # This is how skip connections work: a node jumps directly
            # to the layer where it first has cross-chunk connectivity.
    else:
        parent_layer = layer_id  # normal: parent at this layer

# 5. For each CC: create parent, lift CX from children
parent_id = allocate_id(parent_chunk_at_parent_layer)

# 5a. Process children: lift their CX to parent's level
_children_rows(parent_id, children_in_cc, cx_edges_d):
    for each child:
        for layer in range(child_layer, layer_count):
            edges = cx_edges_d[child][layer]

            # KEY: Remap targets to child's own layer identity
            # This is because CX targets should be at the child's layer,
            # not at the raw SV level.
            nodes = unique(edges)
            parents = get_roots(nodes, stop_layer=child_layer, ceil=False)
                # get_roots with stop_layer: walks UP from each node
                # to find its identity at child_layer.
                # For L2 children: stop_layer=2, so targets stay at L2.
                # For L3 children: stop_layer=3, targets remapped to L3.
            edge_parents_d = dict(zip(nodes, parents))
            edges = fastremap.remap(edges, edge_parents_d)
            edges = unique(edges)

            # Write lifted CX back to child
            write(child, {
                Parent: parent_id,
                CrossChunkEdge[layer]: edges
            })
                # WHY write CX to children: children's CX serves as
                # PRE-RESOLVED CACHE for future edits. When a future
                # add_edges creates a new parent, _update_cross_edge_cache_batched
                # reads children's CX (not ACX) to derive parent CX.
                # Writing fresh CX here means future reads get current data.

# 5b. Aggregate children CX → parent CX
parent_cx = concatenate_cross_edge_dicts(all_children_cx, unique=True)
write(parent_id, {
    Child: children_in_cc,
    CrossChunkEdge[layer]: parent_cx[layer]  # for each layer >= parent_layer
})
```

## CX Format After Ingest

| Node Layer | CX Stored At | Target Format | Source |
|-----------|-------------|---------------|--------|
| L2 | L2, L3, ..., root-1 | Targets at L2 (child's own layer) | From `_children_rows` with `stop_layer=2` |
| L3 | L3, L4, ..., root-1 | Targets at L3 | From `_children_rows` with `stop_layer=3` |
| L4 | L4, L5, ..., root-1 | Targets at L4 | From `_children_rows` with `stop_layer=4` |

**Rule: CX at any layer L on a node at layer M has targets at layer M.**

This is because `_children_rows` calls `get_roots(nodes, stop_layer=node_layer)` which resolves ALL targets to the child's own layer identity.

## ACX Layer Determination

```python
# Determines at which layer a cross-chunk edge becomes relevant
# (edges/utils.py)

cross_chunk_edge_layers = ones(n_edges)  # start at layer 1
coords0 = chunk_coords(edge[:, 0])      # L1 chunk coords of source SV
coords1 = chunk_coords(edge[:, 1])      # L1 chunk coords of target SV

for layer in range(2, layer_count):
    diff = sum(abs(coords0 - coords1))
    cross_chunk_edge_layers[diff > 0] += 1
    coords0 = coords0 // FANOUT         # move up to next layer's coords
    coords1 = coords1 // FANOUT

# Result: layer at which the two SVs first belong to different chunks
# L2 edge: SVs in adjacent L1 chunks (same L2 chunk or adjacent)
# L3 edge: SVs in different L2 chunks
# L4 edge: SVs in different L3 chunks
```

## Key Properties

1. **ACX is immutable** — set once during L2 creation, never modified
2. **CX is mutable** — written to children during parent creation, updated during edits
3. **Children CX = cache** — future edits read children's CX to derive parent CX
4. **No cache service during ingest** — all reads are direct BigTable (`raw_only=True`)
5. **Chunk-parallel** — each chunk processed independently, no cross-chunk coordination needed
6. **Layer-sequential** — must process L2 before L3, L3 before L4, etc.
