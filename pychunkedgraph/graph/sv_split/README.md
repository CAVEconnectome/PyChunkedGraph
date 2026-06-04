# Supervoxel splitting

Reference for the SV-split sub-flow of a multicut edit. Covers what the
operation does, where it sits in the edit pipeline, the voxel-level cut
algorithm, edge re-routing afterwards, the concurrency / storage contracts,
recovery, performance characteristics, and the failure modes.

Companion: `NOTES.md` (sibling) holds known issues, open failure modes, and
future-work threads. Any code or design change that alters a contract /
invariant / failure mode covered here must update both files in the same
commit.

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

Only graphs whose segmentation is OCDBT-backed can run SV-splits — the
operation needs a writable segmentation store. On read-only segmentation
backends the multicut surfaces a precondition error instead of attempting
the split.

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
│  MULTICUT #1:                                                         │
│      build local subgraph around source / sink                        │
│      merge cross-chunk INF edges into "rep" super-nodes               │
│      run mincut between source and sink                               │
│      → Cut | SvSplitRequired(sv_remapping) | PreconditionError        │
│                                                                       │
│  if SvSplitRequired:                                                  │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  L2 CHUNK LOCK (temporal, per-chunk, 1-CG-chunk margin)       │   │
│  │                                                               │   │
│  │  PLAN:                                                        │   │
│  │      enumerate one task per cross-chunk rep that bridges      │   │
│  │      src ↔ sink                                               │   │
│  │      bbox = full envelope of seed coords + 1 CG-chunk margin  │   │
│  │                                                               │   │
│  │  SPLIT (pure, no IO):                                         │   │
│  │      for each task:                                           │   │
│  │          read seg in [bbs − 1, bbe + 1]   ← 1-voxel shell     │   │
│  │          geodesic cut → label map {0, 1, 2}                   │   │
│  │          per chunk, ≥ 2 labels  → mint fresh L1 ids           │   │
│  │                        1 label  → keep original (no OCDBT)    │   │
│  │          route edges incident to split SVs (§4)               │   │
│  │      aggregate per-task results                               │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────┐     │   │
│  │  │  INDEFINITE L2 CHUNK LOCK (durable scope record)    │     │   │
│  │  │     write seg chunks → OCDBT                        │     │   │
│  │  │     persist BT rows  → BT                           │     │   │
│  │  └─────────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  refresh src/sink ids to fresh fragments                              │
│                                                                       │
│  MULTICUT #2 (retry against post-split graph):                        │
│      → Cut | SvSplitRequired                                          │
│      if SvSplitRequired: raise PreconditionError                      │
│                                                                       │
│  COMMIT CUT:                                                          │
│      remove the cut atomic edges                                      │
│      mint new roots, write hierarchy + op-log                         │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
Release root lock — edit is durable
  │
  ▼
Publish pubsub message; on SV-split, carries seg bboxes of rewritten regions
  │
  ▼
Async mesh / downsample workers consume bboxes under their own pyramid-block locks
```

Key timing rules:

- `SvSplitRequired` is a **return value**, not an exception. The root lock
  stays held across detect → split → retry → commit; nothing unwinds.
- SV-split writes (OCDBT seg + BT rows) land **before** the retry multicut.
  If the retry rejects the cut, the SV writes stay durable (orphan
  fragments) — the op aborts to the user but the segmentation has moved.
- SV-split is per-rep. Each cross-chunk-connected rep that bridges src↔sink
  is handled by its own task under the same L2 lock set.

## 3. Voxel-level cut

The cut is a **geodesic region grow**, not a voxel graph mincut. Each voxel
of the SV being split is assigned to whichever seed it reaches first in an
anisotropy-aware geodesic cost field.

### Inputs

- A 3D boolean mask of the SV's voxels in the task bbox.
- Source and sink seed coords (mip0 voxel space).
- Anisotropic voxel sampling in nm.

### Steps

1. **Foreground bbox crop.** Tight bbox of the True voxels; the geodesic
   runs on this sub-volume only. Geodesic-algorithm allocations scale with
   foreground extent, not task bbox.
2. **Seed ridge bridge + snap.** Build a path from each seed to a
   foreground ridge; snap each seed to the nearest in-mask voxel via
   kdtree. Seeds need not land on a foreground voxel.
3. **EDT → speed → travel cost.** Anisotropic Euclidean distance transform
   of the mask; speed = (distance / max distance)^γ_neck (γ ≈ 1.6),
   clipped to a small floor; travel cost = 1 / speed. Interior is cheap,
   neck is expensive — the cut tracks the thinnest section of the SV.
4. **Geodesic arrival.** Compute arrival times from each seed set on the
   travel-cost field with the anisotropic sampling baked in; each voxel
   goes to the side it reaches more cheaply. Optional axis-wise
   downsample keeps the geodesic in physical-nm units regardless of stride.
5. **Narrow-band proximity boost.** Voxels within a relative threshold of
   the opposing side get a cost boost so the boundary tracks the midline.
6. **Single-CC enforcement + stray resolution.** For each side keep the
   26-connected component(s) containing a seed; relabel orphan CCs to a
   transient label-3. Then for every label-3 voxel, look up its 26
   neighbours' labels and per-component assign the side with the majority
   of label-1 / label-2 neighbours; ties break on per-voxel anisotropic
   distance to the nearest source vs sink seed. Work scales with the
   label-3 voxel set, not the volume. See §12 for the full label-3
   lifecycle and why a second-pass enforce may leave some label-3
   fragments for the mincut to absorb.
7. **Embed back.** Result is a small-int label map over the original task
   bbox; voxels outside the foreground crop are 0.

### Single-CC invariant

The chunk-update step mints **one fresh id per distinct label value per L1
chunk** and does **no CC of its own**. If a side left two disconnected
pieces inside one chunk, both would collapse into one new id and the chunk
graph would gain a spuriously-connected fragment. Therefore every label
out of the geodesic must already be a single 26-connected component per
chunk — enforced at full resolution because upsampling under a foreground
mask can fragment.

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
  voxel while respecting the cut. Label-3 fragments that survive after the
  second-pass enforce propagate through to the mincut as their own SV ids
  and the cut resolves them topology-first; see §12.
- **Per-voxel scan over full-volume dilation.** Stray border counts visit
  only the label-3 voxel set and their 26 neighbours; work scales with
  the size of that set, not the volume.
- **Reads pinned to the operation's parent timestamp.** Every graph read
  during a split (parents, cross-chunk edges, SV-to-root map) is pinned so
  a replay sees the same graph state and allocates the same fresh SV ids —
  what makes interrupted splits safe to re-run under recovery.

## 4. Edge re-routing

After the chunk-write step produces an `old → fresh-fragments` map and a
`fragment id → cut-side label ∈ {1, 2, 3}` map (see §12 for label-3),
update every atomic edge that used to reference an old SV.

### Sketch

```
update_edges:
  1. fetch atomic subgraph in [bbs, bbe]
  2. dedup, drop self-loops
  3. resolve partner roots via one batched call
  4. for each old SV being split:
       inactive partner   → broadcast edge to every fragment
       active partner
         INF + partner-split    → match by cut-side label
         INF + partner-unsplit  → closest-fragment-only
         finite                 → connect every fragment within threshold;
                                  fall back to closest
  5. inter-fragment 0.001 bridges between every fragment pair
  6. validate (4 invariants — §4c)
  7. return (edges, affinities, areas)

add_new_edges:
  1. duplicate bidirectional
  2. group by partner's L2 parent chunk
  3. per chunk: append to the append-only split-edge history,
                rewrite the latest-only compacted snapshot (stale-filtered)
```

### Routing rules

For each edge incident to a split SV, the partner's root determines the
path:

- **Inactive partner** (different root). Broadcast: every fragment gets a
  copy of the edge with affinity / area preserved. Costless if the roots
  stay apart; collapses harmlessly to one root-level edge if they later
  merge.
- **Active partner + INF + partner also split.** Connect each fragment to
  the partner's fragment with the same cut-side label. Fallback (no
  matching label in this iteration's new ids) writes closest-fragment INF
  — a known class-C bridge risk (§9).
- **Active partner + INF + partner unsplit.** Write the edge to the
  single closest fragment only. Broadcasting INF to both sides would form
  an uncuttable bridge for the retry mincut.
- **Active partner + finite affinity.** Connect every fragment within the
  threshold (configurable per-graph); fall back to closest.

### Inter-fragment 0.001 bridges

For each old SV with ≥ 2 fragments, every fragment pair gets a finite-0.001
edge. These are **cuttable** by the retry multicut — they are the route the
mincut actually severs to separate the source side from the sink side.

### Post-route validation

| check | invariant | raises |
|---|---|---|
| A | No `{1, 2}` (source-side ↔ sink-side) INF bridge through any unsplit partner | `PostconditionError` |
| B | No self-loops | `PostconditionError` |
| C | Every old SV has at least one replacement edge | `PostconditionError` |
| D | Every fragment pair from same old SV has the 0.001 bridge | `PostconditionError` |

Check A only fires on `{1, 2}` — bridges involving label-3 fragments are
valid (the unresolved fragment rides with whichever seeded side the
inf-cluster joins; see §12 deep dive).

Failures abort before any write lands.

### Distance computation

- **Partner inside bbox.** Kdtree over the partner's voxels in the seg
  crop; per fragment, query against it and return the minimum voxel
  distance.
- **Partner outside bbox.** Fragment kdtree to the partner's chunk
  boundary face. Over-estimate for non-boundary-aligned partners but the
  only signal available without extra reads.

### Persistence

Edges write per L2 chunk into two parallel columns:

- **Append-only split-edge history.** Time-travel reads at any timestamp
  `T` walk cells with `ts ≤ T` and apply stale-edge resolution.
- **Latest-only compacted snapshot.** Single fresh cell per op. Previous
  compacted rows are loaded, rows referencing any SV in the old→new map
  are filtered out, new rows are appended, and the union is written
  back. Current-time readers take this single cell directly.

Edges write at the operation's logical timestamp so a parent-timestamp-
filtered reader sees atomic visibility.

## 5. Concurrency

Three lock layers, each scoped to the smallest window that closes its race
class:

| lock | scope | duration | races closed |
|---|---|---|---|
| Root lock | the op's root id(s) | entire op | same-root edit interleaving |
| L2-chunk lock (temporal) | every L1 chunk the SV-split touches | SV-split + retry | cross-root spatial races during SV-split read / compute |
| L2-chunk lock (indefinite) | same chunks | OCDBT + BT writes | crash-mid-write isolation (§7) |

### L2 chunk-set computation

Walk each task's `[bbs − 1, bbe + 1]` bbox (the 1-voxel shell the edge
router needs to read), map it to overlapping L1 chunks, union across
tasks, and return a deterministically sorted list so workers with
overlapping sets never acquire in opposing orders.

### Write-scope minimization

Only chunks with ≥ 2 distinct labels (geodesic actually split the rep
there) get OCDBT writes. Annulus pieces — chunks where the rep is single-
label — keep their original L0 ids and skip the seg write entirely; the
segmentation backend is append-only so writing unchanged bytes would
inflate the on-disk delta for no value.

### Post-split id refresh without an extra read

After the SV-split lands, the original source/sink ids reference now-
superseded SVs. The retry multicut needs the *current* ids at the seed
voxels. The in-memory seg block produced during split is bitwise
identical to what just landed on OCDBT — the write is synchronous and
happens under the L2 lock, so nothing else mutated those voxels — and the
fresh source/sink ids fall out of the same in-memory lookup. No extra
round-trip.

## 6. Storage substrate

### Edges

| source | role |
|---|---|
| Bucket (GCS protobuf), in-chunk channel | both endpoints in one L1 chunk |
| Bucket, cross-chunk channel | endpoints in different L1 chunks; INF-affinity (watershed's "same SV across this face" claim) |
| Bigtable, fake-edges column | operator-added merge edges |
| Bigtable, split-edge history columns | append-only SV-split history (edges + affinities + areas) |
| Bigtable, compacted split-edge columns | latest-only snapshot |

The subgraph fetch merges bucket and bigtable edges, then drops edges
incident to SVs whose row carries a "new-identity" marker and are not in
the live agglomeration's parent map. On a clean table the filter is a
no-op (no bigtable L0 edges, no new-identity rows).

### Hierarchy rows (per L0 SV)

- Parent: single L1 parent id.
- Former-identity: array of node ids this row was created from.
- New-identity: array of node ids replacing this row (set on split).
- Per-layer cross-chunk edges.

On split, the lineage step writes former-identity and the operation id on
each new fragment, copies the parent pointer (preserving the parent cell's
timestamp), updates the parent's child list to replace the old SV with
the new fragments, and writes new-identity on the old SV.

### OCDBT segmentation

Reads pull from the OCDBT-backed segmentation store when available,
falling back to CloudVolume otherwise. Writes aggregate all
`(voxel slices, data)` pairs across tasks into one flat list and issue
all tensorstore futures in parallel; per-task / per-rep loops would
serialize wall time. On-disk OCDBT config is authoritative — opening an
existing OCDBT must not pass a top-level `"config"` key; the manifest is
the source of truth.

### ID allocation

A per-chunk counter atomically increments, claims a contiguous range, and
OR-masks with the chunk id. Failed ops leak ids — the counter never rolls
back.

### Locks

Per-row columns for temporal and durable locks. Per-chunk rows are hash-
prefixed for write distribution. Acquire refuses if either column is set.
Value-matched release on `__exit__` prevents an op from clearing a lock
another op holds. The op log durably records the chunk set the indefinite
lock covers — recovery reads this to know which chunks need cleanup (§7).

## 7. Recovery — worker crash mid-write

Both writes inside the indefinite L2 chunk lock — OCDBT seg + BT rows —
must land for the op to be consistent. A worker death inside that block
leaves:

- Indefinite lock cells set on the affected chunks.
- The op log's chunk-set record durably populated with the chunk ids.
- Possibly partial OCDBT writes and zero / partial BT rows.

Future ops on any of those chunks refuse to start (the lock blocks them)
— the crashed state is isolated, not amplified.

The authoritative signal that an op is stuck is the chunk-set record
non-empty past the clean exit point. A minimum-age threshold (≈ 10 min)
filters in-flight ops from definitively-dead ones.

### Why a single pinned read is not enough

The SV-split reads a 1-voxel shell around each chunk; the shell's
neighbouring chunks may have been mutated by other ops since the crash.
A single pinned read of the world at the op's parent timestamp would
return stale neighbour values; routing fresh edges to those stale ids
corrupts the graph. Recovery cannot rely on a single pinned view.

### Cleanup-then-replay

1. **Cleanup.** For each chunk in the crashed op's lock scope: read the
   chunk's voxels at the op's parent timestamp (pinned handle), write
   those values back to the latest (unpinned) handle. The crashed op's
   chunks now show pre-op state at the latest manifest; neighbour chunks
   and any concurrent ops' work are untouched.
2. **Replay.** Re-run the op under the privileged-repair path. Reads see
   pre-op values on the op's chunks + current state on every other chunk
   — a consistent world. Allocates fresh ids (the crashed op's ids leak),
   writes new seg + hierarchy, lands the op-log row at success. The
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

- **Foreground crop.** Reducing the geodesic's working volume to the
  tight bbox of the SV's True voxels is the biggest single lever; big
  win for thin reps; near no-op for dense reps that fill the bbox.
- **Downsample.** Axis-wise stride is configurable; the geodesic stays
  in physical-nm units regardless of the stride choice. The principled
  derivation is `resolution.max() // resolution`, which makes the
  operation isotropic in nm space across any voxel anisotropy.
- **Narrow-band refinement.** Refines the cut surface near the boundary
  at full resolution regardless of the global downsample.
- **Single-CC at full res.** The single-CC invariant must hold at chunk
  granularity; running it on the DS grid would let upsampling fragment
  a side into disconnected pieces.
- **Parallel arrival fields.** Source-side and sink-side arrival times
  are computed concurrently in a two-worker fork pool. The underlying
  geodesic kernel holds the GIL, so process-level parallelism is the
  only lever; roughly halves the wall vs sequential.
- **Configurable backend.** `PYCG_GEODESIC_BACKEND=dj3d` (default) is the
  faster kernel; it approximates anisotropy via a single mean-scale factor
  on the cost grid. `=mcp` is anisotropy-correct via per-axis sampling; the
  cut surface diverges by a small amount on highly-anisotropic graphs
  depending on which backend is enabled.

### Edge re-routing

- One subgraph fetch per task at the task's `[bbs, bbe]`.
- Distances are kdtree queries on per-fragment voxel sets — no full-
  volume scans.
- Edge writes per L2 chunk; bidirectional duplication; one parallel
  batch.

### Writes

- OCDBT: only changed chunks. Annulus / shell-only chunks are untouched.
- Bigtable: lineage + new edges per affected chunk, one batched write.
- Seg-chunk writes aggregate all `(slices, data)` across reps into one
  flat tensorstore future list — never per-rep loops.

### Lock hold time

- Temporal L2 lock covers the read + geodesic + edge-route compute — the
  longest window in the SV-split. Released as soon as the compute
  finishes.
- Indefinite L2 lock covers only the persist block — short, but durable
  if the worker dies inside it.

## 9. Failure modes

| trigger | message / type |
|---|---|
| src and sink resolve to different roots | `PreconditionError("Supervoxels must belong to the same object ...")` |
| retry multicut still returns SvSplitRequired | `PreconditionError("Supervoxel split succeeded but source and sink remain connected; place source and sink farther apart.")` |
| mincut produced no removable edges | `PostconditionError("Mincut could not find any edges to remove.")` |
| no edges in retry's local subgraph | `PreconditionError("No local edges found.")` |
| src and sink in different CCs of retry's local subgraph | `PreconditionError("Sinks and sources are not connected through the local graph.")` |
| in-mask seed connection failed | `RuntimeError("In-mask connection failed for at least one team; skipping split.")` |
| new fragment landed in different chunk than its old SV | `PreconditionError("new supervoxel landed in a different chunk than the SV it split from")` |
| `{1, 2}` cross-side INF bridge via unsplit partner / self-loop / missing replacement / missing 0.001 bridge | `PostconditionError(...)` |

### Bridge classes (residual)

After the retry mincut runs, the only structural paths from src-side to
sink-side are:

- **0.001 inter-fragment bridges** (cuttable; intended).
- **Cross-chunk INF edges** routed by the by-label / closest-unsplit
  rules (uncuttable; should be label-pure by construction).

Known residual risks:

- **C — by-label cross-label fallback.** When the iteration's new-ids set
  contains fragments of only one label and the partner has the other
  label, the fallback writes closest-fragment INF. Bridge survives. Fix
  candidates: drop the edge or write finite cuttable affinity.
- **Annulus↔annulus bucket INF.** Two unsplit annulus pieces routed to
  opposite labels by their respective closest-unsplit calls retain their
  direct bucket INF edge (the routing loop only iterates over olds being
  split — never sees unsplit↔unsplit pairs). Fix candidates: force-split
  the bridge endpoints, or detect-and-abort with a dedicated error.

## 10. Invariants

- SV-split + retry multicut + commit are one atomic operation under a
  single root lock.
- Within the SV-split step, concurrent splits on overlapping L2 chunks
  serialize via the temporal L2 lock.
- OCDBT writes touch only chunks whose voxels actually changed. Annulus
  and shell-only chunks are untouched.
- Every fresh L1 id minted in the SV-split has at least one voxel in
  OCDBT carrying that id. OCDBT seg's per-voxel id ≡ the chunk graph's
  child-derived L1 set at that voxel.
- Every label out of the geodesic is a single 26-connected component per
  L1 chunk.
- After commit, readers at the op's timestamp see new SV ids in the cut
  region and new roots reflecting the cut.
- Coarser MIP levels are eventually consistent with mip0, lagging at most
  until the async downsample worker processes the pubsub message.

## 11. Coordinate units

- HTTP boundary: nm.
- After source/sink resolution: mip0 voxels.
- Bounding-box offset: mip0 voxels (default ≈ `(120, 120, 12)`).
- L1 chunk dimensions: mip0 voxels (graph config).
- Chunk grid origin: `voxel_bounds[:, 0]` — chunk `(0,0,0)` is at
  `voxel_bounds[:, 0]`, not 0.

## 12. Deep dive — the label-{1, 2, 3} lifecycle

The geodesic cut produces a per-voxel label volume. Its valid values
evolve as the cut pipeline runs. Understanding why each step exists, and
what label-3 means at each stage, is the key to reading the routing and
the post-route validation.

### Label legend (within the SV being split)

| value | meaning at the end of a step |
|---|---|
| `0` | background — voxel outside the SV mask |
| `1` | source-side fragment voxel (source seed reaches it first) |
| `2` | sink-side fragment voxel (sink seed reaches it first) |
| `3` | **transient unresolved voxel** — a fragment the system has not yet decided belongs to source or sink |

Label-3 is *internal* to the cut pipeline. After commit, no graph node
carries label 3 — every voxel ends up rooted under either the source-side
or sink-side new root after the retry mincut.

### Step-by-step rationale

**1) Geodesic cut → labels `{0, 1, 2}`.** `argmin(arrival_from_source,
arrival_from_sink)` is the geodesic Voronoi partition: each foreground
voxel is assigned to whichever seed set reaches it first along the cost-
weighted shortest path in the anisotropic field. This is the cut surface
as a voxel-level partition.

**2-3) 1st-pass single-CC enforcement, each side, with stray-demotion
enabled.** The Voronoi partition can leave **islands** of a side stranded
— e.g. a thin pocket the geodesic happens to assign to side A but with no
internal path to any source seed without crossing label-2. The chunk-
update step commits one new SV per `(old_sv × connected_component ×
label)`, so each `(old_sv, label)` must be a single CC inside one L1
chunk.

The rule: keep CCs that contain at least one **seed of that side** (those
represent the legitimate side territory). Demote every other CC. Demote
to **3** rather than back to the opposite side, because we don't actually
know if the island belongs to the opposite side — it might be
geographically inside one side's main mass with the geodesic having taken
a sub-optimal arc through it. Label 3 is the contract: *"unresolved —
let the smarter downstream resolver decide."*

**4) Stray resolver.** For each connected component of label-3:

- **Border vote (majority).** Count how many 26-neighbours of the CC's
  voxels carry label 1 vs label 2. Majority side wins. Rationale: an
  island's natural home is whatever it's geographically nestled inside;
  local topology is the strongest signal for which side it *should*
  belong to.
- **Seed-distance tiebreak.** If border counts are equal, fall back to
  the per-voxel anisotropic distance to the nearest seed of each side
  (kdtree on the seed point sets, scaled by sampling). Closer side
  wins. Global fallback when local context is symmetric.

After this step every voxel originally demoted in steps 2-3 has been
reassigned. **Invariant at exit of the resolver: labels ∈ `{0, 1, 2}`.**

**5-6) 2nd-pass single-CC enforcement, each side, with stray-demotion
enabled.** Step 4 just moved potentially many voxels from label-3 to
label-1 / label-2. Those newly-reassigned voxels can form **new**
disconnected CCs of their new side — an island resolved to label-1 may
be sitting far from the main label-1 mass with no internal label-1 path
between them. The single-CC-per-chunk invariant must hold at the end of
the pipeline; this second pass re-checks.

Same contract as steps 2-3: orphan CCs (no seed inside them) get
demoted to label 3.

**7) End — label-3 fragments may survive.** Unlike the first round, no
third resolver pass runs. The label-3 voxels that the second pass
produced do not get reassigned. This is intentional — see step 8.

**8) Per-chunk fresh-id minting.** The chunk-update step mints one fresh
L1 SV id per distinct label value per L1 chunk. Label 3 is treated like
any other label: every label-3 CC inside a chunk becomes its own new SV
id, and the routing step records the fragment-id → label-3 mapping so
edge routing knows how to handle it.

**9) Edge routing treats label-3 fragments as first-class new SVs:**

- Low-affinity inter-fragment edges (affinity `0.001`) are emitted for
  *every pair* of fragments of the same old SV — including all pairings
  with label-3 fragments. These edges are cuttable by the mincut.
- INF-affinity edges to **split partners** are routed by same-label
  match — a label-3 fragment of this SV connects via INF to a label-3
  fragment of a split partner, if one exists. Two unresolved fragments
  inf-joined is fine; both ride together.
- INF-affinity to **unsplit partners** uses closest-fragment-only. The
  partner can land inf-edged to a label-1 fragment via one cross-chunk
  face and inf-edged to a label-3 fragment via another, producing a
  `{1, 3}` bridge through the partner.
- Inactive partners (different root) broadcast to every fragment,
  including label-3.

**10) Post-route validation accepts label-3 bridges.** The cross-label
check only raises on `{1, 2}` — bridges through an unsplit partner
involving only seeded-side labels. `{1, 3}`, `{2, 3}`, and `{3}` bridges
are valid because the unresolved fragment has no seed; the mincut places
the entire inf-connected cluster on whichever side carries the seed.

**11) Retry multicut places label-3 fragments.** The mincut builds a
graph where:

- Source nodes = the new SV ids at each source seed voxel — anchored by
  *physical seed voxel location*, not by the label the fragment carries.
- Sink nodes = the new SV ids at each sink seed voxel.
- INF edges are uncuttable; the `0.001` inter-fragment edges are cheap
  to cut.

The cut algorithm finds the minimum-cost edge set to remove that
disconnects source nodes from sink nodes. Label-3 fragments end up on
whichever side has lower cut cost. For a partner-cluster `{label-1 frag,
label-3 frag, unsplit partner}` joined by INF, the cluster is one
indivisible unit; it joins the source side via the label-1 fragment's
source seed; the label-3 fragment rides along. Symmetrically for `{label-2
frag, label-3 frag, partner}` → sink side.

When a label-3 fragment has no INF anchor to either side, the cut runs
through its `0.001` neighbours: it ends up wherever the global cost
optimum places it.

### Why this works without a third resolver pass

Re-running the resolver after the second-pass enforce is *one option* —
it would produce labels `{0, 1, 2}` once again and the validate-routing
chain could keep the old "no cross-label" rule. The model we use instead
is: **the heuristic resolver and the global mincut decide different
questions, and the mincut is qualified to absorb whatever the heuristic
left undecided.**

- The resolver's border vote answers *"based on the local 26-
  neighbourhood, which side does this voxel belong to?"* — strong signal
  when the neighbourhood is asymmetric, no signal when it is symmetric
  or the fragment sits in a thin neck.
- The mincut answers *"given the global graph topology (INF clusters,
  inter-fragment 0.001 edges, partner connectivity), what's the minimum-
  cost cut that separates source seeds from sink seeds?"* — uses
  information the resolver doesn't have (cross-chunk INF edges, partner
  identity, multi-fragment routing).

When the second-pass enforce produces label-3 voxels, the local signal
has *already failed once* (the resolver placed them, and the placement
created a new disconnected CC). Re-applying the same border vote is
unlikely to be more correct than letting the mincut weigh the global
topology. So we publish the label-3 fragment as a real SV, route it, and
let the multicut place it.

### Mental model — labels are a routing hint, not a cut constraint

The fragment-id → label-{1, 2, 3} mapping is consumed by *one* thing:
the edge router, to decide which fragment of *this* split SV gets
inf-edged to each fragment of a *split partner* SV via the by-label
match rule. After routing, the multicut treats every fragment as just a
node in a weighted graph — the label has no further role. Source and
sink anchors come from the *physical seed voxel position*, not the label
of the fragment that ended up at that voxel. This is what lets label-3
fragments propagate through harmlessly: they are nodes with no seed
anchor, sitting wherever the cut places them.
