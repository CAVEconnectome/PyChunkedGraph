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

## Experiments that didn't ship

### `crackle.point_cloud` (surface-only) as a replacement for `fastremap.point_cloud` in `update_edges`

`crackle` exposes a `point_cloud` that returns only the surface voxels per label
— ideal in principle since `cKDTree` nearest-distance queries from outside a
label always land on its shell, so the interior coords carried by
`fastremap.point_cloud` are dead weight downstream.

A side-by-side bench in `build_coords_by_label` measured correctness and wall
on a real masked `new_seg` (~742 M-voxel read bbox, ~99 % zero after
`mask_except`):

- Shell / full coord ratio ≈ 0.12 (real downstream savings would be substantial).
- Bbox(shell) == bbox(full) on every label, shell ⊆ full on the top-3 largest
  labels — correctness gates pass.
- But `crackle.compress(new_seg)` itself ran ~3.3× slower than `fastremap.point_cloud`
  on the same input (compress dominates; the actual `crackle.point_cloud` call on
  the compressed buffer is fast). The downstream `_get_new_edges` savings did not
  recoup the compress cost on either the sparse-foreground or dense-foreground
  payloads we tried.

If revisited: the `parallel=N` kwarg on `crackle.compress` is unexplored;
worth measuring whether parallel scaling closes the 3× compress gap. A
foreground-bbox crop before compress is *not* a help here — `update_edges`
partners scatter across the full read bbox so the nonzero bbox approximates
the read bbox.

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
