# Supervoxel splitting

## What it is

A *supervoxel split* bisects one physical supervoxel — a connected region in the raw segmentation — along a user-seeded cut. The user supplies a source coordinate and a sink coordinate inside one supervoxel; the system finds a cut surface separating them and assigns new supervoxel IDs to each half, writing the updated segmentation and the corresponding graph hierarchy.

This only runs on segmentations stored in OCDBT (a writable, append-only segmentation backend). With a read-only segmentation backend the split path is never entered; the multicut instead surfaces a precondition error asking the user to pick different source/sink points.

## Why it's needed

The graph is stored in **chunks**: the segmentation volume is partitioned into a regular 3D grid, and each chunk owns its own set of supervoxel IDs. When a physical supervoxel spans a chunk boundary, it is artificially cut into multiple graph-level supervoxel IDs — one per chunk — with infinite-affinity *cross-chunk edges* connecting the pieces so the graph still represents one physical object.

The multicut algorithm runs on a local graph around source and sink. If it finds that source and sink sit inside the same cross-chunk-connected component — i.e., in the same physical supervoxel — a clean graph cut cannot separate them without first **splitting that physical supervoxel at the voxel level** and giving the resulting halves fresh IDs. That voxel-level cut is what the split flow does. The multicut runs again against the refreshed graph and produces the graph-level edges to remove.

## End-to-end flow

```
Split request (source coord, sink coord)
  │
  ▼
Resolve coords → current supervoxel IDs at those pixels
  │
  ▼
┌───────────────────────────────────────────────────────────────────────┐
│  ROOT LOCK (held across the whole operation)                          │
│                                                                       │
│  multicut:                                                            │
│      build local subgraph around source/sink                          │
│      stitch cross-chunk-connected SVs via inf-affinity edges          │
│      run mincut between source and sink                               │
│      result ─► one of:                                                │
│          ● clean cut      → edges to remove                           │
│          ● SV split needed → cross-chunk-representative mapping       │
│          ● precondition   → surface to user, abort                    │
│                                                                       │
│  if SV split needed:                                                  │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  L2 CHUNK LOCK (spatial; sparse set + 1-chunk margin)         │   │
│  │                                                               │   │
│  │  for each cross-chunk rep linking source to sink:             │   │
│  │      bbs/bbe ◄ envelope of src+sink seeds + 1-chunk margin   │   │
│  │      read seg in [bbs-1, bbe+1]                               │   │
│  │          (1-voxel shell → anchor voxels for edge routing)     │   │
│  │      compute voxel-level cut between seeds                    │   │
│  │      allocate fresh SV IDs per chunk to each half             │   │
│  │      route existing cross-chunk edges onto the new fragments  │   │
│  │      write seg — only chunks that actually received new IDs   │   │
│  │      write hierarchy rows (lineage + new cross-chunk edges)   │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  refresh source/sink IDs:                                             │
│      look up the new IDs in the in-memory split output                │
│      (bit-identical to what just landed on storage; no extra read)    │
│                                                                       │
│  multicut (retry against post-split graph):                           │
│      result ─► clean cut  → edges to remove                           │
│               │ still-split-needed → surface precondition error       │
│                                                                       │
│  commit the cut:                                                      │
│      remove graph-level edges                                         │
│      produce new roots                                                │
│      write hierarchy rows + operation log                             │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
Release root lock — edit is durable
  │
  ▼
Publish pubsub message; when an SV split ran it carries the list of
base-resolution bounding boxes that were rewritten
  │
  ▼
┌───────────────────────────────────────────────────────────────┐
│  Async downsample worker                                      │
│                                                               │
│  partition each published bbox into pyramid blocks            │
│      (cube regions aligned to the coarsest MIP's chunk grid;  │
│       two distinct blocks never share a storage chunk at      │
│       any MIP level)                                          │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  PYRAMID BLOCK LOCK (separate lock family from L2)   │    │
│  │                                                      │    │
│  │  for each pyramid block:                             │    │
│  │      read base resolution                            │    │
│  │      downsample through every coarser MIP            │    │
│  │      write only tiles whose footprint intersects     │    │
│  │      a published bbox                                │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

### Notes on the flow

- **"SV split required" is a return value, not an exception.** The multicut returns one of several tagged outcomes so the caller dispatches with a straight branch. Nothing uses raise/catch for control flow, which is what allows the root lock to stay held across the detect-then-split-then-commit sequence without the exception unwinding the lock.

- **The cross-chunk-representative mapping** comes out of the multicut for free: as part of building its local graph it stitches every cross-chunk-connected group of graph-level supervoxels into one node and records the mapping. That map tells the split step which supervoxels are artificially-cut pieces of one physical SV, and which of them sit on a source→sink bridge.

- **The split is per-representative.** If two unrelated physical supervoxels both need splitting in one edit (rare but possible), each is handled in its own pass under the same L2 chunk lock.

## Concurrency design

Two races exist at the segmentation layer even with root locks in place:

- **Same-root race.** Without care, the root lock could drop between "detect split needed" and "perform split", letting another edit on the same root slip in and race for the same supervoxel pieces.
- **Cross-root spatial race.** Two edits on entirely distinct roots can target supervoxels whose pieces live in overlapping chunks. Root locks don't serialize them; segmentation writes would clobber each other.

The split flow closes both:

- **Root lock scope covers the full operation.** Detection, supervoxel-level split, retry detection, commit — all under one root lock. Same-root interleaving is impossible; any other edit on the root waits for this one to finish.

- **L2 chunk lock covers the supervoxel-level split only.** Inside the root lock, the split step additionally acquires a spatial lock on every L2 chunk it will read or write. Keyed by chunk, so edits on different roots but overlapping chunks serialize here. Released as soon as the split writes land; the graph-level commit afterwards runs under the root lock alone.

### How the spatial lock set is computed

For each cross-chunk representative being split, the read/cut region is the base-voxel envelope of that rep's source and sink seed coordinates, padded by one CG chunk on each side. The cut surface lives between the seeds, so pieces of the rep far from both seeds never participate — the seed envelope is the region that gets read and rewritten, not the rep's full piece-set envelope, which for an SV cut across many chunks can be orders of magnitude larger. To derive the lock set, expand that envelope by one voxel (because the edge-routing step reads a 1-voxel shell outside the rewritten region to see neighboring supervoxels' labels), map it to the overlapping L2 chunks, union the per-representative chunk sets, and sort deterministically so workers with overlapping sets never acquire in opposite orders.

The chunks locked are exactly the chunks the split will touch, plus the 1-chunk margin the shell read requires.

### How the write scope is kept minimal

Only chunks that actually receive new supervoxel IDs get written to storage. Gap chunks that happen to sit inside an envelope but contain no cross-chunk-connected pieces, and neighbor chunks read only for the edge-routing shell, are never written. The segmentation backend is append-only, so writing unchanged bytes would inflate the on-disk delta for no real change.

### Why the post-split ID refresh is safe without an extra read

After the split lands, the caller-supplied source and sink supervoxel IDs reference now-superseded supervoxels. The retry multicut needs the *current* IDs at the source and sink pixels — the subgraph fetch returns only live supervoxels, so a mincut asking about superseded ones would fail to find its endpoints.

The in-memory segmentation block produced by the split is bitwise identical to what was just written to storage, and the storage write is synchronous (we wait for it) and happens under the L2 chunk lock (so nothing else can have mutated those voxels). Looking up source/sink coords in that block returns the same IDs a storage re-read would — no extra round-trip needed.

### Worker crash mid-write

A worker that dies — or raises from the persist block — inside the indefinite L2 chunk lock's scope leaves the lock cells set and the op-log row's `L2ChunkLockScope` populated with the exact chunks being written. Future ops on any of those chunks refuse to start — the crashed state is isolated, not amplified. An operator runs the recovery flow described in [recovery.md](recovery.md) to revert the partial writes and replay the op.

## Invariants

- A supervoxel split and its graph-level commit are one atomic operation. Either both land or neither does, under a single root lock.
- Within the supervoxel-split step, concurrent splits on overlapping L2 chunks serialize. No two operations write segmentation to the same chunk at the same time.
- Supervoxel-level writes touch only chunks whose voxels actually changed. Gap chunks between cross-chunk-connected pieces and neighbor chunks read for edge routing are untouched.
- After the commit, readers at the operation's timestamp see new supervoxel IDs in the cut region and new roots reflecting the cut.
- Coarser MIP levels are eventually consistent with the base scale, lagging at most until the async downsample worker processes the operation's pubsub message.

## Related docs

- [Algorithm](algorithm.md) — the voxel-level geodesic cut.
- [Design](design.md) — rationale behind the cut.
- [Edges](edges.md) — edge re-routing after the cut.
- [Recovery](recovery.md) — replay / recovery of interrupted splits.
