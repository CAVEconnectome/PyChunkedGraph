# sv_split — notes (known issues, test handles, future work)

Secondary material that doesn't belong in `README.md` (the design reference).

## Known issues

### `{1, 2}` cross-side INF bridge through an unsplit partner

Validation refuses the split when an L0 INF-affinity edge in an unsplit partner
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

## Future work — break up `cutting.py`

`cutting.py` mixes geodesic backend dispatch, seed snapping, EDT + cost-grid
construction, label assignment + narrow-band refinement, single-CC enforcement,
the resolve3 label-3 stray reassignment, and the `split_supervoxel_growing`
driver in a single ~1900-line module. Refactor target: promote to a `cutting/`
package with one module per concern (`arrival.py`, `cost.py`, `seeds.py`,
`label.py`, `enforce.py`, `driver.py`), re-export the public entry points
(`split_supervoxel_growing`, `connect_both_seeds_via_ridge`) at
`sv_split.cutting`. Defer until the geodesic backend choice and the stray
mechanism have settled.

## Geodesic backend switch — `PYCG_GEODESIC_BACKEND`

`dj3d` (default) selects a faster geodesic kernel that has no anisotropy
parameter; the cost grid is pre-scaled by `mean(sampling_ds)` to approximate
per-axis anisotropy, and the cut surface diverges by a small amount on highly
anisotropic graphs where one axis is >5× the others. `mcp` selects the
anisotropy-correct kernel with per-axis sampling.
