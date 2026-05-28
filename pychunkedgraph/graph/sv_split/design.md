# SV splitting — design rationale

Why the [cut](algorithm.md) is shaped the way it is.

## Geodesic region-grow, not a voxel graph-mincut

The split assigns each voxel to the nearer seed by geodesic travel cost rather
than solving a min-cut on a voxel adjacency graph. A region grow over an
anisotropy-aware geodesic metric follows the object's medial geometry and
produces a smooth boundary near the midline, without building and cutting a
large per-voxel graph for every split.

## Snap seeds to the boundary

Cut seeds come from operator clicks / mincut output and need not land exactly
on a foreground voxel. Snapping each seed (via KDTree) to the nearest occupied
voxel makes the grow well-defined regardless of where the seed falls, and keeps
the two arrival fields rooted inside the object.

## Single-CC enforced at full resolution

Each output side must be a single connected component because the downstream
writer assigns one new supervoxel id per label and does no CC of its own (see
the [single-CC invariant](algorithm.md#the-single-cc-invariant)).
Enforcement runs at full resolution: if the geodesic grow is done on a
downsampled grid, the result is upsampled *first*, because upsampling under the
foreground mask can fragment a label into disconnected pieces — enforcing CC
before that would let the fragments through.

## Resolve strays, don't drop them

Small components cut off from a seeded body are relabeled to a transient label
and reassigned to whichever side they border more (EDT tie-break) rather than
discarded. Dropping voxels would lose mass; merging blindly could bridge the
two sides. Border-count reassignment keeps every voxel while respecting the
cut.

## Label-3 dilation confined to its bounding box

Reassigning a stray component only needs its local border with each side, so
the dilation is run inside the component's own bounding box rather than over
the whole volume. The per-component border counts are identical either way, so
this is an exact optimization, not an approximation.

## Single-pass coord extraction, in-place masking

`build_coords_by_label` groups voxels by label in a single `np.unique` pass and
masks in place, instead of allocating a volume-sized boolean array per label.
The cut likewise works inside the foreground bounding box. Both choices bound
work to the foreground extent rather than the chunk volume.

## Reads pinned to `parent_ts`

All graph reads during a split are pinned to the operation's `parent_ts` so a
replay sees the same graph state and allocates the same new supervoxel ids.
This is what makes interrupted splits safe to re-run (see
[Recovery](recovery.md)).

## Related docs

- [Overview](README.md)
- [Algorithm](algorithm.md)
- [Edges](edges.md)
- [Recovery](recovery.md)
