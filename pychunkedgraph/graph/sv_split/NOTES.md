# sv_split — notes (known issues, test handles, future work)

Secondary material that doesn't belong in `README.md` (the design reference).

## Known issues

### `{1, 2}` cross-side INF bridge through an unsplit partner

Validation refuses the split when an L1 INF-affinity edge in an unsplit partner
SV connects a source-side fragment (label 1) to a sink-side fragment (label 2).
The carve inside the SV is correct in voxel space, but the cut surface
geometrically passes through the partner — INF means the partner is the same
segment as both fragments, and the chunked graph can't downgrade that affinity.
Mincut on the L2 graph would have an uncuttable source→partner→sink path.

Current behavior: post-condition raises with the offending partner id. The user
re-issues with seeds that avoid the corridor through the partner, or includes
the partner in the seed set. A cascade-split — recursively splitting bridging
partners — is the long-term fix but expands the unit-of-edit and touches
lineage / op-log / multicut handshake.

### Mincut precondition rejection after a successful SV-split

On large objects the multicut may reject with `Sinks and sources are not
connected through the local graph` even though the SV-split, edge writeback,
and validation all pass. Open question: bbox window too tight to keep the
seeded L2 nodes in one component after the partner splits, or one of the
partner splits removed a bridge edge the mincut needed.

## Future work

### Dedup seg reads + subgraph fetches across reps in `split_supervoxels`

When a multicut produces multiple cross-chunk reps that need splitting,
the orchestrator iterates `for task in tasks: split_supervoxel(task)`
and each rep independently (a) reads its bbox crop from OCDBT (~2–4 s
per call) and (b) fetches `cg.get_subgraph(root, bbox)` (~1.6 s per
call). When reps cluster spatially or share an L2 ancestor the same
OCDBT chunks and subgraph payload are fetched twice or more.

Two surgical dedups, opt-in conditional on `len(tasks) > 1` so the
single-rep path is byte-equivalent to today:

**Seg-read union per overlap cluster.** Cluster reps by overlap of
their padded voxel bboxes (`[bbs−1, bbe+1]`). Connected components over
that overlap relation = clusters. One `get_local_segmentation(union_bbs,
union_bbe)` per cluster; each rep takes a `.copy()` of its sub-slice at
the same point `_read_seg_and_ids` would have returned, so per-rep
mutation isolation is preserved (`_parse_results` and `mask_except`
both mutate seg in place). Saves the OCDBT I/O; per-rep copy cost
unchanged.

**Subgraph union per shared root.** After `cg.get_roots([sv_id], parent_ts)`
resolves each rep's root, group reps by root. One `cg.get_subgraph(root,
union_bbox)` per shared root; each rep filters the returned edges by its
own `[bbs, bbe]`. `get_subgraph` already filters by bbox post-read
(subgraph.py:212–216 via `mask_nodes_by_bounding_box`), so a wider union
bbox returns a superset of any per-rep bbox's edges. Endpoints touching
multiple reps' bboxes get processed in each rep — matches today's
behavior since `_get_new_edges` is idempotent across duplicate edges.

Correctness gates: a `TestMultiRepParity` covering both dedups against
the independent-per-rep baseline (byte-equal `seg_writes`, `bigtable_rows`,
`source_ids_fresh`, `sink_ids_fresh`, `old_new_map`, `new_id_label_map`)
plus the canonical pinky log lines remaining unchanged. Single-rep path
must take the original code path verbatim — the new helpers must not
fire when `n_tasks == 1`.

Risks: union seg buffer holds longer than per-rep (mitigation: per-rep
copy at the same boundary today's path uses; union buffer freed after
the cluster's last rep finishes `_route_edges_and_rows`). Cluster
detection bug → larger union buffer, never wrong output. Cap union
volume to bound RSS on dense N-rep clusters.

Not in scope: parallelizing reps, batching `id_client.create_node_ids`,
changing `_compute_split` / `_apply_and_capture` / `_parse_results` /
`_update_chunks`, modifying per-rep log lines.

## Architecture

The split algorithm lives in the external `supervoxel-splitter` package.
`splitter.get_splitter()` resolves an implementation class via the
`PCG_SV_SPLITTER` env var (dotted import path; default
`supervoxel_splitter.GeodesicSplitter`) and forwards `**kwargs` to its
constructor so call-site tuning propagates. `_coords.py` holds post-split
coord utilities consumed by `edges.py`.

## Geodesic backend — `backend` kwarg on `GeodesicSplitter`

`dj3d` (default) selects a faster geodesic kernel that has no anisotropy
parameter; the cost grid is pre-scaled by `mean(sampling_ds)` to approximate
per-axis anisotropy, and the cut surface diverges by a small amount on highly
anisotropic graphs where one axis is >5× the others. `mcp` selects the
anisotropy-correct kernel with per-axis sampling. Pass via
`get_splitter(backend="mcp")` from PCG to override (forwarded as a
`GeodesicSplitter` constructor kwarg).
