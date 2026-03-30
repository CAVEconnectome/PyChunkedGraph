# Production Edits Algorithm

How add_edges (merge) and remove_edges (split) work in production.
**File:** `pychunkedgraph/graph/edits.py`

## Merge: add_edges

### Phase 1: Analyze edges and create L2 nodes

```
INPUT: atomic_edges (SV pairs to merge), parent_ts (read timestamp)

# 1. Convert SV edges to L2 edges
edges, l2_cross_edges_d = _analyze_affected_edges(atomic_edges, parent_ts)
    # For each SV pair: get L2 parents
    # Determine edge layer (which chunk boundary crossed)
    # Build cross_edges_d: {l2_id: {layer: [[l2_id, partner_l2]]}}

# 2. Build L2 merge graph
l2ids = unique L2 IDs from edges
old_hierarchy_d = {l2_id: {layer: parent_at_layer, ...}}  # full chain to root
cross_edges_d = merge(get_cross_chunk_edges(l2ids), l2_cross_edges_d)
    # Combines existing BigTable CX with new cross-edges from atomic_edges
graph = build_gt_graph(edges)
components = connected_components(graph)
    # Each CC = L2 nodes that will merge into one new L2

# 3. Allocate new L2 IDs
chunk_ids_map = batch allocate IDs per chunk (threaded)

# 4. For each CC: create new L2 node
for new_id, old_ids in zip(new_l2_ids, components):
    new_old_id_d[new_id] = set(old_ids)
    old_new_id_d[old_id] = {new_id} for each old
    cache.children_cache[new_id] = concat all old L2s' SVs
    cache.parents_cache[sv] = new_id for each SV

# 5. Set CX for new L2 nodes
#    KEY: CX comes from merging old L2s' CX, NOT from ACX.
for new_id, old_ids in zip(new_l2_ids, components):
    cx_edges = [cross_edges_d[old_l2] for old_l2 in old_ids]
    merged = concatenate_cross_edge_dicts(cx_edges, unique=True)

    # Remap: replace old L2 IDs with new L2 IDs in ALL edge columns
    temp_map = {old: new for old, new in old_new_id_d.items() if len(v)==1}
    for layer, edges in merged.items():
        edges = fastremap.remap(edges, temp_map, preserve_missing_labels=True)
        # Result: edges[:, 0] = new_id (remapped from old)
        #         edges[:, 1] = targets (L2 IDs, remapped where applicable)
        assert edges[:, 0] == new_id
    cache.cross_chunk_edges_cache[new_id] = merged
    # WHY merge old CX instead of re-deriving from ACX:
    # ACX column family (family_id=3) has a BigTable GC rule — ACX columns
    # are eventually deleted. By the time an edit runs, ACX may not exist.
    # CX column family (family_id=4) has NO GC rule — always available.
    # Children's CX is a pre-resolved cache kept fresh by write-backs.
    # Merging CX is cheaper, and ACX may not exist (GC'd).
```

### Phase 2: CreateParentNodes.run()

Per-layer loop from L2 to root-1:

```
for layer in range(2, layer_count):
    new_nodes = new_ids_d[layer]
    if empty: continue
    cache.new_ids.update(new_nodes)
        # Mark as new — prevents get_stale_nodes from flagging them

    # STEP A: _update_cross_edge_cache_batched(new_nodes)
    # Sets CX for new parent nodes AND writes back resolved CX to children.
    # SKIPS layer==2 (L2 CX already set in Phase 1).
    _update_cross_edge_cache_batched(new_nodes):
        children_d = get_children(new_nodes)          # cache hit
        all_children = concat(children_d.values())
        cx_raw = get_cross_chunk_edges(all_children)   # from CACHE (not BigTable)
            # WHY cache: children's CX was set by prior layer's processing.
            # Cache has the freshest data.
        combined = concatenate_cross_edge_dicts(cx_raw.values())

        # Resolve stale edges
        updated, edge_nodes = get_latest_edges_wrapper(combined)
            # Checks: are any edge targets stale (replaced by prior edits)?
            # If yes: finds current identity via SV tracking.
            # In clean graph (first edit): no-op.

        # WRITE BACK resolved CX to children
        # WHY: prevents stale edge accumulation across edits. Without this,
        # each edit adds stale references that future get_latest_edges_wrapper
        # calls must resolve — compounding over time.
        for child, cx_map in children_cx_edges.items():
            cache[child] = cx_map
            write(child, {CrossChunkEdge[l]: edges for l, edges in cx_map})

        # Derive parent CX by remapping children → parent identity
        edge_parents = get_new_nodes(edge_nodes, parent_layer)
            # For each node in CX edges: find its parent at parent_layer.
            # WHY parent_layer (not the CX layer): the parent's CX should
            # have targets at the parent's own layer. An L3 parent's L4 CX
            # has L3 targets, not L4 targets. This is because the parent
            # represents the entire subtree — its CX connects to other
            # subtrees at the same level.
        edge_parents_d = dict(zip(edge_nodes, edge_parents))

        for new_id in new_nodes:
            for layer >= parent_layer:
                edges = updated[layer] filtered to new_id's children
                edges = fastremap.remap(edges, edge_parents_d)
                    # Column 0: child → new_id (the parent)
                    # Column 1: target child → target's parent at parent_layer
                assert edges[:, 0] == new_id
            cache[new_id] = parent_cx_edges_d

    # STEP B: _update_neighbor_cx_edges(new_nodes)
    # Updates CX of counterpart nodes (nodes in other chunks whose
    # CX targets were replaced by the merge).
    _update_neighbor_cx_edges(new_nodes):
        # Read new nodes' CX from cache (ALL layers, set by Step A)
        cx_d = get_cross_chunk_edges(new_nodes)  # cache hit

        # Build cumulative node_map
        node_map = {old: new for old, new in old_new_id_d if 1-to-1}
            # Includes L2 old→new AND all higher-layer mappings accumulated so far

        # Find counterparts: ALL CX targets NOT in new_ids
        for new_id in new_nodes:
            counterparts = _get_counterparts(new_id, cx_d[new_id])
                # Scans CX at ALL layers >= node_layer
                # Extracts targets (column 1)
                # Since Step A remapped targets to parent_layer,
                # ALL counterparts are at the same layer as new_id.
                # WHY: counterpart discovery is layer-consistent.
            all_cps.update(counterparts)

        # Read counterpart CX from BigTable (at parent_ts — BEFORE this edit)
        cp_cx = get_cross_chunk_edges(all_cps, time_stamp=parent_ts)
        descendants = _get_descendants_batch(new_nodes)

        # Remap each counterpart's CX
        for new_id in new_nodes:
            # Add per-new_id mappings to node_map
            node_map.update({old: new_id for old in flip_ids(new_old_id, [new_id])})

            for counterpart in counterparts_of[new_id]:
                for layer >= node_layer:
                    edges = cp_cx[counterpart][layer]
                    edges = fastremap.remap(edges, node_map)

                    if layer == counterpart_layer:
                        # Add flip edge: [counterpart, new_id]
                        # WHY: ensures bidirectional connectivity.
                        # The new node has CX→counterpart (from Step A).
                        # The counterpart needs CX→new_node too.
                        edges = concat(edges, [[counterpart, new_id]])

                        # Collapse descendants: if edge target is a descendant
                        # of new_id, replace with new_id itself.
                        # WHY: the descendant was absorbed into new_id.
                        # The counterpart should point to new_id, not its parts.
                        mask = isin(edges[:, 1], descendants[new_id])
                        edges[mask, 1] = new_id

                    edges = unique(edges)
                cache[counterpart] = remapped_cx
                write(counterpart, {CrossChunkEdge[l]: edges})

    # STEP C: _create_new_parents(layer)
    _create_new_parents(layer):
        # Discover siblings
        node_ids = _get_layer_node_ids(new_ids_d[layer], layer)
            old_ids = flip_ids(new_old_id_d, new_ids)
            old_parents = get_parents(old_ids)
            siblings = get_children(old_parents)  # ALL children of same parents
            # Replace old IDs with new, filter to target layer
            # Result: new nodes + unchanged siblings at this layer

        # Build CCs from cross-chunk edges AT THIS LAYER ONLY
        ccs, graph_ids = _get_connected_components(node_ids, layer)
            cx_d = get_cross_chunk_edges(node_ids)  # cache or BigTable
            edges = [cx_d[id].get(layer) for id in node_ids] + self_edges
            graph = build_gt_graph(concat(edges))
            # WHY only this layer: connectivity at layer L determines
            # which nodes share an L-level parent. Higher-layer CX
            # determines higher-level grouping, handled at later iterations.

        # Create parents
        for each CC:
            parent_layer = determine_parent_layer(CC)
                # Single-node CC: skip to layer where it has CX
                # Multi-node CC: parent_layer = layer + 1
            parent = allocate_id(chunk at parent_layer)
            new_ids_d[parent_layer].append(parent)
            _update_id_lineage(parent, cc_ids, layer, parent_layer)
                # Track: new_to_old[parent] = old parents of new children
                # Only traces through NEW children (not siblings)
                # Uses old_hierarchy to find old identity at parent_layer
            cache.children[parent] = cc_ids
            cache.parents[child] = parent for each child
```

### Phase 3: Write to BigTable

```
create_new_entries():
    # Read final CX from cache for all new nodes
    val_dicts = _get_cross_edges_val_dicts()
        for layer in 2..root:
            cx_d = get_cross_chunk_edges(new_ids_d[layer])  # FROM CACHE
            # WHY cache: cache has the freshest CX from Step A processing.

    # Write each new node
    for layer in 2..root:
        for new_id in new_ids_d[layer]:
            children = get_children(new_id)
            write(new_id, {Child: children, CrossChunkEdge[l]: cx})
            write(child, {Parent: new_id}) for each child

    # Root lineage (skip if stitch_mode)
    write FormerParent/NewParent for root transitions
```

## Split: remove_edges

```
INPUT: atomic_edges (SV pairs to cut), parent_ts

# 1. Analyze affected edges
edges, _ = _analyze_affected_edges(atomic_edges, parent_ts)
l2ids = unique L2 IDs
assert: all L2s belong to SAME root (splitting within one segment)

# 2. Split each L2 node
for each affected l2_id:
    _split_l2_agglomeration(l2_id, removed_edges):
        # Remove specified SV edges from agglomeration
        # Build CC graph from remaining in-chunk edges
        # Filter cross-chunk edges to only active ones (within same root)
        # Returns: CCs (new groupings), graph_ids, remaining cross_edges

    # Allocate new L2 IDs (one per CC within the split)
    for each CC:
        new_id = allocate_id
        new_old_id_d[new_id] = {old_l2_id}
        old_new_id_d[old_l2_id].add(new_id)
        cache.children[new_id] = SVs in CC
        cache.parents[sv] = new_id

# 3. Update cross-chunk edges (similar to merge)
for new_id in new_l2_ids:
    cx_d = get_cross_chunk_edges([new_id])
    for layer, edges in cx_d.items():
        edges = fastremap.remap(edges, old_to_new_map)
        assert edges[:, 0] == new_id
    cache[new_id] = cx_d

# 4. Create parent hierarchy (same as merge)
CreateParentNodes(new_l2_ids, ...).run()
create_new_entries()
```

## CX Format

```
CX at layer L on node at layer M:
  edges[:, 0] = node_id (always the owning node)
  edges[:, 1] = targets at layer M (the node's own layer)

WHY targets at M, not L:
  _update_cross_edge_cache_batched remaps via edge_parents_d which maps
  ALL nodes → their parent at parent_layer (= M, the node's layer).
  During ingest, _children_rows uses get_roots(stop_layer=node_layer).
  Both produce the same result: targets at the node's own layer.

WHY this matters for future edits:
  When _update_cross_edge_cache_batched reads children's CX to derive
  parent CX, it remaps targets via edge_parents_d to the parent's layer.
  The input target layer doesn't matter — fastremap maps any node to
  its parent identity. But having targets at a consistent level means
  the stale edge resolver can efficiently check for staleness.
```

## Data Structures

```
new_old_id_d: {new_id: set(old_ids)}
    Merge: one new → many old
    Split: many new → one old each

old_new_id_d: {old_id: set(new_ids)}
    Inverse of above

old_hierarchy_d: {node_id: {layer: parent_at_layer}}
    Full chain from node to root. Used by _update_id_lineage
    to find old parents at each layer during hierarchy creation.

node_map (in counterpart update):
    Cumulative {old: new} across all layers processed so far.
    Built from old_new_id_d (1-to-1 entries) + per-new_id flip_ids.
    Grows each layer as new parents are created.

cache.cross_chunk_edges_cache: {node_id: {layer: edges_array}}
    In-memory cache. Updated by:
    - add_edges: sets new L2 CX
    - _update_cross_edge_cache_batched: sets children + parent CX
    - _update_neighbor_cx_edges: sets counterpart CX
    Read by:
    - get_cross_chunk_edges (when cache exists)
    - _get_cross_edges_val_dicts (for final BigTable write)

cache.new_ids: set
    Tracks recently created node IDs. Prevents get_stale_nodes
    from flagging them as stale (they don't exist at parent_ts).
```

## Column Families and GC

```
Family 3: AtomicCrossChunkEdge (ACX)
  - Has BigTable garbage collection rule
  - ACX columns deleted after time/version threshold
  - Set during ingest only, never updated by edits
  - WHY edits can't use ACX: may not exist by edit time

Family 4: CrossChunkEdge (CX)
  - NO garbage collection rule — permanent
  - Written during ingest (on children), updated during edits
  - Serves as pre-resolved cache for parent CX derivation
  - This is why _update_cross_edge_cache_batched reads children's CX, not ACX
```

## Assertions

1. `edges[:, 0] == node_id` — CX source must be the owning node
2. Merge: L2s from different roots
3. Split: L2s from same root
4. Parent layer > max children layer
5. Root at root_layer
