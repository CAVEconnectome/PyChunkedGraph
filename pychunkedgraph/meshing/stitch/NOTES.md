# stitch — notes (known issues, test handles, future work)

Secondary material that doesn't belong in `README.md` (the design reference).

## Large-chunk test handle

`experimental_pinky_lgsv_split_v0` layer-6 chunk `451485862643892224` — ~28k
parents, ~5.5 GB shard, an ~83M-vertex giant fragment. One of the larger shards in
that dataset; a good memory/perf optimization target. Reproduce with
`debug.repro_chunk(cg, 451485862643892224)`.

## Known issue — peak memory OOM on small nodes

Heavy layer-6 chunks drive whole-cgroup peak RSS above a small node's limit
(~40 GB peak at 24 workers vs a 28 GB node), OOM-ing a worker mid-run. Reproduces
only under that memory pressure — the chunk above completes cleanly with more
headroom / fewer workers.

Measured split (the chunk above, 16 workers): per-worker peak RSS p50 ≈ 0.75 GB,
max ≈ 2.2 GB (the giant's worker). The peak is dominated by the **parent synthesize
transient** — `acc.merged_meshes` accumulates every encoded mesh, then
`synthesize_shard` holds ~2-3× the ~5.5 GB shard — **plus** the broad sum of
concurrent per-worker working sets (~0.75 GB × N). No single giant worker dominates;
both terms stack to ~40 GB at 24 workers.

Eager `del` of merge/decoded-fragment intermediates (in `worker._stitch_one` and
`utils.merge_draco_meshes_across_boundaries_pure`) trims only a few GB — not enough
to close the gap on its own.

## Future levers (neither done; parent term is the bigger one)

- **Stream the shard via a tmp dir.** Workers write each encoded mesh to a
  `mesh_path/<tmp>/<label>` blob as it finishes; the parent never accumulates, and
  final synthesis streams blobs from the tmp dir into the shard, freeing each.
  Removes the parent accumulation + synthesize transient (the dominant term). Costs
  ~one extra GCS put/get per parent; needs tmp cleanup on success/failure and a
  byte-identity gate against the in-RAM path.
- **Throttle concurrency while a giant is in flight** (or size workers by memory,
  not cores) to bound the `N × per-worker` mid-run term without idling cores the
  rest of the time.

Measure the parent-vs-worker split before choosing — it decides which lever closes
the gap.
