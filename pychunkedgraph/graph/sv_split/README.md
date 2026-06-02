# Supervoxel splitting

Reference for the SV-split sub-flow of a multicut edit. Covers what the
operation does, where it sits in the edit pipeline, the voxel-level cut
algorithm, edge re-routing afterwards, the concurrency / storage contracts,
recovery, performance characteristics, and the failure modes.

---

## 1. Overview

A *supervoxel split* bisects one physical supervoxel — a connected region in
the raw segmentation — along a user-seeded cut. The user supplies source and
sink seeds inside one SV; the system finds a cut surface, mints fresh L0 ids
for each side, rewrites the affected OCDBT segmentation chunks, and reroutes
every graph edge that used to reference the old SV.

The SV-split runs only when the multicut detects that source and sink resolve
to the same physical SV across chunk boundaries — i.e. they sit in the same
INF-connected component of the chunk graph. A multicut cannot sever an
INF-affinity edge, so the only way to honour the user's cut is to actually
split the underlying voxels and produce new ids the graph mincut can then cut
between.

Only graphs whose segmentation is OCDBT-backed (`meta.ocdbt_seg=True`) can
run SV-splits — the operation needs a writable segmentation store. On
read-only segmentation backends the multicut surfaces a `PreconditionError`
instead of attempting the split.

## 2. End-to-end flow

```
Split request (source coords, sink coords, source/sink ids)
  │
  ▼
Resolve coords → current L0 SV ids at those voxels
  │
  ▼
┌───────────────────────────────────────────────────────────────────────┐
│  ROOT LOCK (held across the whole operation)                          │
│                                                                       │
│  multicut #1:                                                         │
│      build local subgraph around source / sink                        │
│      merge cross-chunk INF edges into "rep" super-nodes               │
│      run mincut between source and sink                               │
│      → Cut | SvSplitRequired(sv_remapping) | PreconditionError        │
│                                                                       │
│  if SvSplitRequired:                                                  │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  L2 CHUNK LOCK (temporal, per-chunk, 1-CG-chunk margin)       │   │
│  │                                                               │   │
│  │  plan_sv_splits:                                              │   │
│  │      enumerate one task per cross-chunk rep that bridges      │   │
│  │      src ↔ sink                                               │   │
│  │      bbox = full envelope of seed coords + 1 CG-chunk margin  │   │
│  │                                                               │   │
│  │  split_supervoxels (pure, no IO):                             │   │
│  │      for each task:                                           │   │
│  │          read seg in [bbs − 1, bbe + 1]   ← 1-voxel shell     │   │
│  │          geodesic cut → label map {0, 1, 2}                   │   │
│  │          per chunk, ≥ 2 labels  → mint fresh L1 ids           │   │
│  │                        1 label  → keep original (no OCDBT)    │   │
│  │          route edges incident to split SVs (§4)               │   │
│  │      aggregate per-task results into SplitResult              │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────┐     │   │
│  │  │  INDEFINITE L2 CHUNK LOCK (durable scope record)    │     │   │
│  │  │     write_seg_chunks(seg_writes)  → OCDBT           │     │   │
│  │  │     _persist_rows(bigtable_rows)  → BT              │     │   │
│  │  └─────────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  refresh src/sink ids to fresh fragments (from SplitResult)           │
│                                                                       │
│  multicut #2 (retry against post-split graph):                        │
│      → Cut | SvSplitRequired                                          │
│      if SvSplitRequired: raise PreconditionError                      │
│                                                                       │
│  commit cut:                                                          │
│      edits.remove_edges(atomic_edges)                                 │
│      mint new roots, write hierarchy + op-log                         │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
Release root lock — edit is durable
  │
  ▼
Publish pubsub message; on SV-split, carries seg_bboxes of rewritten regions
  │
  ▼
Async mesh / downsample workers consume bboxes under their own pyramid-block locks
```

Key timing rules:

- `SvSplitRequired` is a **return value**, not an exception. The root lock
  stays held across detect → split → retry → commit; nothing unwinds.
- SV-split writes (`write_seg_chunks` + `_persist_rows`) land **before** the
  retry multicut. If the retry rejects the cut, the SV writes stay durable
  (orphan fragments) — the op aborts to the user but the segmentation has
  moved.
- SV-split is per-rep. Each cross-chunk-connected rep that bridges src↔sink
  is handled by its own task under the same L2 lock set.

## 3. Voxel-level cut

The cut is a **geodesic region grow**, not a voxel graph mincut. Each voxel
of the SV being split is assigned to whichever seed it reaches first in an
anisotropy-aware geodesic cost field. Lives in
[`sv_split/cutting.py`](../cutting.py) (`split_supervoxel_growing`,
`_enforce_single_component`) and the helper `build_coords_by_label`.

### Inputs

- `binary_seg` — `(Z, Y, X)` boolean mask of the SV's voxels in the task bbox.
- `seed_source`, `seed_sink` — global mip0 voxel seed coords.
- `sampling` — voxel size in nm `(x, y, z)`; supplies anisotropy.

### Steps

1. **Foreground bbox crop.** Tight bbox of `binary_seg`'s True voxels;
   geodesic runs on this sub-volume only. MCP_Geometric array allocations
   scale with foreground extent, not task bbox.
2. **Seed ridge bridge + snap.** `connect_both_seeds_via_ridge` builds a
   path from each seed to a foreground ridge; each seed snaps to the
   nearest in-mask voxel via kdtree. Seeds need not land on a foreground
   voxel.
3. **EDT → speed → travel cost.**
   `dist = _compute_edt(sv, sampling)`;
   `speed = clip((dist / dist.max()) ** gamma_neck, eps, 1)` with
   `gamma_neck ≈ 1.6`; `travel_cost = 1 / speed`. Interior is cheap, neck
   is expensive — the cut tracks the thinnest section of the SV.
4. **Geodesic arrival.** `MCP_Geometric(travel_cost, sampling=sampling_ds)`
   from each seed-set; each voxel goes to the side it reaches more cheaply.
   `sampling_ds = sampling × downsample_geodesic` so the geodesic stays in
   physical-nm units regardless of stride.
5. **Narrow-band proximity boost.** Voxels within `narrow_band_rel` of the
   opposing side get a cost boost so the boundary tracks the midline.
6. **Single-CC enforcement** (`_enforce_single_component`). For each side
   keep the largest 26-connected component containing a seed; relabel
   strays to a transient label 3. Each label-3 component is dilated within
   its own bbox and reassigned to side 1 or 2 by border-count majority,
   EDT tiebreak.
7. **Embed back.** Result is a `uint16` map with values `{0, 1, 2}` over
   the original task bbox; voxels outside the foreground crop are 0.

### Single-CC invariant

`_update_chunks` ([`edits.py`](../edits.py)) mints **one fresh id per
distinct label value per L1 chunk** and does **no CC of its own**. If a side
left two disconnected pieces inside one chunk, both would collapse into one
new id and the chunk graph would gain a spuriously-connected fragment.
Therefore every label out of the geodesic must already be a single
26-connected component per chunk — enforced at full resolution because
upsampling under a foreground mask can fragment.

### Design rationale

- **Geodesic region grow over voxel graph mincut.** A geodesic in an
  anisotropy-aware cost field follows the SV's medial geometry and yields a
  smooth midline cut without building / solving a per-voxel adjacency graph
  on every split.
- **Snap seeds to the foreground.** Seeds from operator clicks or upstream
  mincut output need not land on a true voxel; snapping keeps both arrival
  fields rooted inside the SV.
- **Resolve label-3 strays, don't drop them.** Dropping voxels loses mass;
  merging blindly can bridge sides. Border-count reassignment keeps every
  voxel while respecting the cut.
- **Stray-resolution dilation confined to per-component bbox.** Per-
  component border counts are identical to a full-volume dilation, so
  confining the dilation is exact, not approximate.
- **Reads pinned to `parent_ts`.** Every graph read during a split
  (parents, cross-chunk edges, sv_root_map) is pinned to the op's
  `parent_ts` so a replay sees the same graph state and allocates the same
  fresh SV ids — what makes interrupted splits safe to re-run under
  recovery.

## 4. Edge re-routing

After `_apply_and_capture` produces `old_new_map` (old SV → fresh fragments)
and `new_id_label_map` (fresh id → cut side `{1, 2}`),
`_route_edges_and_rows` updates every atomic edge that used to reference an
old SV. Lives in [`sv_split/edges.py`](../edges.py).

### Sketch

```
update_edges:
  1. fetch atomic subgraph in [bbs, bbe] via cg.get_subgraph
  2. dedup, drop self-loops
  3. resolve partner roots via one batched cg.get_roots
  4. for each old SV in old_new_map:
       inactive partner   → broadcast edge to every fragment
       active partner
         INF + partner-split    → _match_by_label (by cut side)
         INF + partner-unsplit  → _match_inf_unsplit (closest only)
         finite                 → _match_by_proximity (within threshold)
  5. inter-fragment 0.001 bridges between every fragment pair
  6. validate_split_edges  (4 invariants — §4c)
  7. return (edges, affinities, areas)

add_new_edges:
  1. duplicate bidirectional
  2. group by partner's L2 parent chunk
  3. per chunk: append to SplitEdges (history),
                rewrite CompactedSplitEdges (latest, stale-filtered)
```

### Routing rules

For each edge incident to a split SV, the partner's root determines the
path:

- **Inactive partner** (different root). Broadcast: every fragment gets a
  copy of the edge with affinity / area preserved. Costless if the roots
  stay apart; collapses harmlessly to one root-level edge if they later
  merge.
- **Active partner + INF + partner also split.** `_match_by_label` connects
  each fragment to the partner's fragment with the same cut-side label.
  Fallback (no matching label in this iteration's `new_ids`) writes
  closest-fragment INF — a known class-C bridge risk (§9).
- **Active partner + INF + partner unsplit.** `_match_inf_unsplit` writes
  the edge to the single closest fragment only. Broadcasting INF to both
  sides would form an uncuttable bridge by retry mincut.
- **Active partner + finite affinity.** `_match_by_proximity` connects
  every fragment within `cg.meta.sv_split_threshold` voxels; fallback
  closest.

### Inter-fragment 0.001 bridges

For each old SV with ≥ 2 fragments, every fragment pair gets a finite-0.001
edge. These are **cuttable** by the retry multicut — they are the route the
mincut actually severs to separate label-1 from label-2.

### `validate_split_edges` (post-route)

| check | invariant | raises |
|---|---|---|
| A | No cross-label INF edges to unsplit partners | `PostconditionError` |
| B | No self-loops | `PostconditionError` |
| C | Every old SV has at least one replacement edge | `PostconditionError` |
| D | Every fragment pair from same old SV has the 0.001 bridge | `PostconditionError` |

Failures abort before any write lands.

### Distance computation

- **Partner inside bbox.** `_compute_partner_distances` — kdtree over the
  partner's voxels in the seg crop; per fragment, smaller-tree-queries-
  larger heuristic returns minimum voxel distance.
- **Partner outside bbox.** `_compute_boundary_distances` — fragment
  kdtree to the partner's chunk boundary face. Over-estimate for non-
  boundary-aligned partners but the only signal available without extra
  reads.

### Persistence

`add_new_edges` writes per L2 chunk:

- `Connectivity.SplitEdges / Affinity / Area` — append-only history. Time-
  travel reads at any timestamp `T` walk cells with `ts ≤ T` and apply
  stale-edge resolution.
- `Connectivity.CompactedSplitEdges / CompactedAffinity / CompactedArea` —
  single fresh cell per op. Previous compacted rows are loaded, rows
  referencing any SV in `old_new_map.keys()` filtered out, new rows
  appended, and the union written back. Current-time readers take this
  single cell directly.

Edges are written at `time_stamp = op.time_stamp` (logical) so a
`parent_ts`-filtered reader sees atomic visibility.

## 5. Concurrency

Three lock layers, each scoped to the smallest window that closes its race
class:

| lock | scope | duration | races closed |
|---|---|---|---|
| `RootLock` | the op's root id(s) | entire op | same-root edit interleaving |
| `L2ChunkLock` (temporal) | every L1 chunk the SV-split touches | SV-split + retry | cross-root spatial races during SV-split read / compute |
| `IndefiniteL2ChunkLock` | same chunks | OCDBT + BT writes | crash-mid-write isolation (§7) |

### L2 chunk-set computation

`_l2_chunks_for_splits` walks each task's `[bbs − 1, bbe + 1]` bbox (the 1-
voxel shell the edge router needs to read), maps it to overlapping L1
chunks, unions across tasks, and returns a deterministically sorted list so
workers with overlapping sets never acquire in opposing orders.

### Write-scope minimization

Only chunks with ≥ 2 distinct labels (geodesic actually split the rep
there) get OCDBT writes. Annulus pieces — chunks where the rep is single-
label — keep their original L0 ids and skip the seg write entirely; the
segmentation backend is append-only so writing unchanged bytes would inflate
the on-disk delta for no value.

### Post-split id refresh without an extra read

After the SV-split lands, the original `source_ids` / `sink_ids` reference
now-superseded SVs. The retry multicut needs the *current* ids at the seed
voxels. The in-memory seg block produced by `split_supervoxels` is bitwise
identical to what just landed on OCDBT — the write is synchronous and
happens under the L2 lock, so nothing else mutated those voxels — and
`SplitResult.source_ids_fresh / sink_ids_fresh` carries the lookup result.
No extra round-trip.

## 6. Storage substrate

### Edges

| source | column | role |
|---|---|---|
| Bucket (GCS protobuf) | `EdgesMsg.in_chunk` | both endpoints in one L1 chunk |
| Bucket | `EdgesMsg.cross_chunk` | endpoints in different L1 chunks; INF-affinity (watershed's "same SV across this face" claim) |
| Bigtable | `Connectivity.FakeEdges` | operator-added merge edges |
| Bigtable | `Connectivity.SplitEdges / Affinity / Area` | append-only SV-split history |
| Bigtable | `Connectivity.CompactedSplitEdges / CompactedAffinity / CompactedArea` | latest-only snapshot |

`get_l2_agglomerations` merges bucket and bigtable edges, then
`_filter_stale_svs` drops edges incident to SVs whose row carries
`Hierarchy.NewIdentity` and are not in the live agglomeration's
`sv_parent_d`. On a clean table the filter is a no-op (no bigtable L0
edges, no NewIdentity rows).

### Hierarchy rows (per L0 SV)

- `Hierarchy.Parent` — single L1 parent id.
- `Hierarchy.FormerIdentity` — array of node ids this row was created from.
- `Hierarchy.NewIdentity` — array of node ids replacing this row (set on
  split).
- `Connectivity.CrossChunkEdge[layer]` — per-layer cross-chunk edges.

`copy_parents_and_add_lineage` writes the FormerIdentity / OperationID on
each new fragment, copies the parent pointer (preserving the parent cell's
timestamp), updates the parent's `Hierarchy.Child` list to replace the old
SV with the new fragments, and writes NewIdentity on the old SV.

### OCDBT segmentation

`get_local_segmentation(meta, bbox_start, bbox_end, mip=0)` reads from
`meta.ws_ocdbt` (or `meta.ws_ocdbt_scales[mip]`) when `meta.ocdbt_seg=True`,
falling back to CloudVolume otherwise.

`write_seg_chunks(meta, seg_writes)` aggregates all `(voxel_slices, data)`
across tasks into one flat list and issues all tensorstore futures in
parallel; per-task / per-rep loops would serialize wall time. On-disk
OCDBT config is authoritative — opening an existing OCDBT must not pass a
top-level `"config"` key; the manifest is the source of truth.

### ID allocation

`cg.id_client.create_node_ids(chunk_id, size, root_chunk=False)` atomically
increments a per-chunk counter (`Concurrency.Counter`, `max_versions=1`),
claims a contiguous range, and OR-masks with `chunk_id`. Failed ops leak
ids — the counter never rolls back.

### Locks

Per-row columns: `Concurrency.Lock` (temporal, value = op id) and
`Concurrency.IndefiniteLock` (durable, family `0`). Per-chunk rows are
hash-prefixed for write distribution. `lock_by_row_key_with_indefinite`
refuses if either column is set. Renewal via `renew_lock_by_row_key`.
Value-matched release on `__exit__` prevents an op from clearing a lock
another op holds.

`OperationLogs.L2ChunkLockScope` durably records the chunk set the
indefinite lock covers — recovery reads this to know which chunks need
cleanup (§7).

## 7. Recovery — worker crash mid-write

Both writes inside the indefinite L2 chunk lock — OCDBT seg + BT rows —
must land for the op to be consistent. A worker death inside that block
leaves:

- Indefinite lock cells set on the affected chunks.
- `OperationLogs.L2ChunkLockScope` durably populated with the chunk ids.
- Possibly partial OCDBT writes and zero / partial BT rows.

Future ops on any of those chunks refuse to start (the lock blocks them) —
the crashed state is isolated, not amplified.

The authoritative signal that an op is stuck is `L2ChunkLockScope` non-
empty past the clean exit point. A minimum-age threshold (≈10 min) filters
in-flight ops from definitively-dead ones.

### Why a single pinned read is not enough

The SV-split reads a 1-voxel shell around each chunk; the shell's
neighbouring chunks may have been mutated by other ops since the crash.
A single pinned read of the world at `op.parent_ts` would return stale
neighbour values; routing fresh edges to those stale ids corrupts the
graph. Recovery cannot rely on a single pinned view.

### Cleanup-then-replay

1. **Cleanup.** For each chunk in `L2ChunkLockScope`: read the chunk's
   voxels at `op.parent_ts` (pinned handle), write those values back to
   the latest (unpinned) handle. The crashed op's chunks now show pre-op
   state at the latest manifest; neighbour chunks and any concurrent ops'
   work are untouched.
2. **Replay.** Re-run the op under the privileged-repair path. Reads see
   pre-op values on the op's chunks + current state on every other chunk
   — a consistent world. Allocates fresh ids (the crashed op's ids leak),
   writes new seg + hierarchy, lands the op-log row at `SUCCESS`. The
   indefinite lock's `__exit__` value-matched-releases the cells the
   crashed op originally set (replay reuses the operation id), freeing
   the chunks.

### Orphan history

OCDBT is append-only — the crashed op's partial writes still exist in
OCDBT commit history, just overshadowed at the latest manifest. Readers
pinning a historical version between crash and replay still see the
partial state; readers at latest never observe it. Orphan ids are never
referenced by any hierarchy row.

## 8. Performance

### Geodesic

- **Foreground crop.** `_compute_split` reduces the geodesic's working
  volume to the tight bbox of the SV's True voxels. Big win for thin
  reps; near no-op for dense reps that fill the bbox.
- **Downsample.** `split_supervoxel_growing` accepts `downsample_geodesic`
  (axis-wise strides). `sampling_ds = sampling × stride` keeps the
  geodesic in physical-nm units regardless of stride choice. Right
  derivation is `meta.resolution.max() // meta.resolution`, which makes
  the operation isotropic in nm space across any voxel anisotropy.
- **`narrow_band_rel`.** Refines the cut surface near the boundary at
  full resolution regardless of the global downsample.
- **`enforce_cc` runs at full res.** Single-CC must hold at chunk
  granularity; running it on the DS grid would let upsampling fragment a
  side into disconnected pieces.

### Edge re-routing

- One `cg.get_subgraph` call per task at the task's `[bbs, bbe]`. Subgraph
  walks down from the rep's root.
- Distances are kdtree queries on per-fragment voxel sets — no full-volume
  scans.
- `add_new_edges` writes per L2 chunk; bidirectional duplication; one
  parallel batch.

### Writes

- OCDBT: only changed chunks (geodesic produced ≥ 2 labels in the chunk).
  Annulus / shell-only chunks are untouched.
- Bigtable: lineage + new edges per affected chunk, one batched write.
- `write_seg_chunks` aggregates all `(slices, data)` across reps into one
  flat tensorstore future list — never per-rep loops.

### Lock hold time

- Temporal `L2ChunkLock` covers the read + geodesic + edge-route compute —
  the longest window in the SV-split. Released as soon as the compute
  finishes.
- `IndefiniteL2ChunkLock` covers only the persist block — short, but
  durable if the worker dies inside it.

## 9. Failure modes

| trigger | source | message / type |
|---|---|---|
| src and sink resolve to different roots | `assert_same_root` | `PreconditionError("Supervoxels must belong to the same object ...")` |
| retry multicut still returns SvSplitRequired | `_apply` retry branch | `PreconditionError("Supervoxel split succeeded but source and sink remain connected; place source and sink farther apart.")` |
| mincut produced no removable edges | `_apply` post-cut | `PostconditionError("Mincut could not find any edges to remove.")` |
| no edges in retry's local subgraph | `_run_multicut` | `PreconditionError("No local edges found.")` |
| src and sink in different CCs of retry's local subgraph | `_filter_graph_connected_components` | `PreconditionError("Sinks and sources are not connected through the local graph.")` |
| in-mask seed connection failed | `split_supervoxel_helper` | `RuntimeError("In-mask connection failed for at least one team; skipping split.")` |
| new fragment landed in different chunk than its old SV | `_assert_same_chunk` | `PreconditionError("new supervoxel landed in a different chunk than the SV it split from")` |
| cross-label INF / self-loop / missing replacement / missing 0.001 bridge | `validate_split_edges` | `PostconditionError(...)` |

### Bridge classes (residual)

After the retry mincut runs, the only structural paths from src-side to
sink-side are:

- **0.001 inter-fragment bridges** (cuttable; intended).
- **Cross-chunk INF edges** routed by `_match_by_label` /
  `_match_inf_unsplit` (uncuttable; should be label-pure by construction).

Known residual risks:

- **C — `_match_by_label` cross-label fallback.** When the iteration's
  `new_ids` contains fragments of only one label and the partner has the
  other label, the fallback writes closest-fragment INF. Bridge survives.
  Fix candidates: drop the edge or write finite cuttable affinity.
- **Annulus↔annulus bucket INF.** Two unsplit annulus pieces routed to
  opposite labels by their respective `_match_inf_unsplit` calls retain
  their direct bucket INF edge (the routing loop only iterates
  `old_new_map.items()` — never sees unsplit↔unsplit pairs). Fix
  candidates: force-split the bridge endpoints, or detect-and-abort with
  a dedicated `SvSplitBridgeError`.

## 10. Invariants

- SV-split + retry multicut + commit are one atomic operation under a
  single root lock.
- Within the SV-split step, concurrent splits on overlapping L2 chunks
  serialize via the temporal `L2ChunkLock`.
- OCDBT writes touch only chunks whose voxels actually changed. Annulus
  and shell-only chunks are untouched.
- Every fresh L1 id minted in the SV-split has at least one voxel in OCDBT
  carrying that id. OCDBT seg's per-voxel id ≡ the chunk graph's
  `Hierarchy.Child`-derived L1 set at that voxel.
- Every label out of the geodesic is a single 26-connected component per
  L1 chunk.
- After commit, readers at the op's timestamp see new SV ids in the cut
  region and new roots reflecting the cut.
- Coarser MIP levels are eventually consistent with mip0, lagging at most
  until the async downsample worker processes the pubsub message.

## 11. Source map

Entry points and key functions, by file.

```
graph/operation.py
   ::MulticutOperation.execute            top-level edit flow, root lock, op-log
   ::MulticutOperation._apply             multicut #1, SV-split branch, retry, commit
   ::MulticutOperation._run_multicut      local subgraph, LocalMincutGraph
   ::MulticutOperation._refresh_after_sv_split    fresh src/sink ids for retry

graph/multicut/cutting.py
   ::LocalMincutGraph                     INF-merge, sv_remapping, SvSplitRequired
   ::run_multicut                         Cut | SvSplitRequired wrapping

graph/sv_split/state.py
   ::SvSplitTask, SplitCtx, ApplyResult,
   ::SvSplitOutcome, SplitResult          per-stage data containers (one source)

graph/sv_split/edits.py
   ::_coords_bbox                         bbox = src+sink envelope + 1 CG-chunk margin
   ::_overlapping_reps, plan_sv_splits    task enumeration
   ::split_supervoxels                    orchestrator over tasks (pure, no IO)
   ::split_supervoxel                     per-task: read seg, geodesic, route edges
   ::_compute_split                       foreground crop + geodesic call
   ::_update_chunks                       mint fresh ids per chunk (≥ 2 labels)
   ::_apply_and_capture                   assemble old_new_map, new_id_label_map
   ::_pick_fresh_source_sink_ids          same-old-SV label-1/2 pair for retry mincut
   ::_route_edges_and_rows                update_edges → lineage → add_new_edges
   ::copy_parents_and_add_lineage         lineage rows + L2.Child update (cache + BT)

graph/sv_split/cutting.py
   ::connect_both_seeds_via_ridge         seed ridge bridge + snap
   ::split_supervoxel_growing             geodesic cut + single-CC enforcement
   ::_enforce_single_component            per-side largest 26-CC, label-3 strays
   ::_resolve_label3_touching_vectorized  border-count reassignment

graph/sv_split/edges.py
   ::update_edges                         subgraph fetch + classify + route
   ::_get_new_edges                       per-old-SV iteration
   ::_match_by_label                      INF + partner-split → same cut side
   ::_match_inf_unsplit                   INF + partner-unsplit → closest fragment
   ::_match_by_proximity                  finite → within threshold
   ::validate_split_edges                 4-check post-route validation
   ::add_new_edges                        per-chunk SplitEdges + Compacted snapshot

graph/chunkedgraph.py
   ::get_subgraph                         L2 walk + edges fetch
   ::get_l2_agglomerations                bucket + BT edges merge
   ::_filter_stale_svs                    drop NewIdentity-marked SV edges

graph/locks.py
   ::RootLock                             temporal root lock
   ::L2ChunkLock                          temporal per-chunk lock
   ::IndefiniteL2ChunkLock                durable scope + value-matched release

graph/ocdbt/main.py
   ::write_seg_chunks                     flat parallel tensorstore writes
```

Coordinate units across the pipeline:

- HTTP boundary: nm.
- After `_get_sources_and_sinks`: mip0 voxels (`int32`).
- `meta.split_bounding_offset`: mip0 voxels (default `(120, 120, 12)`).
- `meta.graph_config.CHUNK_SIZE`: L1 chunk dimensions in mip0 voxels.
- Chunk grid origin: `meta.voxel_bounds[:, 0]` — chunk (0,0,0) is at
  `voxel_bounds[:, 0]`, not 0.
