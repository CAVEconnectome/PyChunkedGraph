# SV splitting — voxel-level cut algorithm

This is the algorithmic core of SV splitting: given the voxels of one
supervoxel and two cut seeds, partition every voxel into a *source* side and a
*sink* side. It lives in `graph/sv_split/cutting.py` (`split_supervoxel`,
`enforce_cc`) plus the coord extraction helper `build_coords_by_label`.

The cut is a **geodesic region grow**, not a voxel graph-mincut: each voxel is
assigned to whichever seed is nearer in geodesic (anisotropy-aware) travel
cost across the supervoxel's interior.

## Inputs

- `coords` — `(N, 3)` global voxel coordinates of the supervoxel.
- `seed_source`, `seed_sink` — global-voxel coordinates of the two cut seeds.
- `resolution` — voxel size in nm `(x, y, z)`; supplies anisotropy.
- `voxel_offset` — volume origin (chunk corner).

## Steps

1. **ROI restriction.** Compute the foreground bounding box from `coords` and
   work inside it only. A dense occupancy volume is allocated over the ROI, not
   over the whole chunk, so cost scales with foreground extent.
2. **Seed snapping.** Build a KDTree over the foreground voxel coordinates and
   snap each seed to its nearest occupied voxel. Seeds need not lie exactly on
   a foreground voxel.
3. **EDT → speed / travel-cost map.** A Euclidean distance transform (sampled
   by `resolution`) gives each voxel its distance to the boundary. Speed favors
   the interior; geodesic travel cost is `1 / speed`, so paths hug the medial
   region rather than skimming the surface.
4. **Geodesic arrival.** `MCP_Geometric` (with `sampling=resolution`, so the
   metric is anisotropy-aware) computes arrival cost from each seed to every
   voxel. Each voxel is assigned to the side whose seed it reaches more cheaply.
5. **Narrow-band proximity boost.** Voxels within `narrow_band_width` of the
   opposing side receive a cost boost (`proximity_boost`) so the boundary
   tracks the geometric midline more closely.
6. **Optional downsampled grid.** When `downsample` is set, the geodesic grow
   runs on a downsampled grid and is upsampled back. Upsampling happens *before*
   single-CC enforcement, because upsampling under the foreground mask can
   fragment a label into disconnected pieces.
7. **Single-CC enforcement** (`enforce_cc`, when `enforce_single_cc=True`).
   For each side, keep the largest / seeded 26-connected component; relabel
   stray components to a transient label `3`. Each label-`3` component is then
   dilated **within its own bounding box** and reassigned to side 1 or 2 by
   which side it borders more, with EDT distance as the tie-break. Confining
   the dilation to the component's bbox is exact: per-component border counts
   are identical to a full-volume dilation.

## The single-CC invariant

`_update_chunks` (`graph/sv_split/edits.py`) assigns **exactly one new
supervoxel id per distinct label value per chunk** and performs **no connected
-component analysis of its own**. It relies entirely on the cut having already
produced single-CC labels. Therefore every output label out of `enforce_cc`
**must already be a single connected component** — if a side were left in two
pieces, both pieces would collapse into one new supervoxel id and the graph
would gain a spuriously-connected supervoxel.

This is why single-CC is enforced at full resolution and why label-`3` strays
are resolved rather than dropped.

## Related docs

- [Overview](README.md) — where this fits in the edit pipeline.
- [Design](design.md) — rationale for the choices above.
- [Edges](edges.md) — edge re-routing after the cut.
- [Recovery](recovery.md) — replay-safety of splits.
