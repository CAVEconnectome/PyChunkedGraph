# sv_lookup

Resolve voxel coordinates from interactive UI clicks (split, merge) to
the current supervoxel IDs that those coordinates physically belong to.
Every coord is answered from a **current segmentation read** —
segmentation-agnostic with respect to backend (OCDBT or precomputed
CloudVolume), dispatched by `graph.utils.generic.get_local_segmentation`.

The caller-supplied `node_ids` are interpreted **only as a layer hint**
to distinguish a 2D slice click from a 3D mesh click. No SV id, root
id, or other value from the client is trusted as the answer.

## Contract

| Input layer | Click origin | Behaviour |
|---|---|---|
| `layer == 1` | 2D slice canvas (NG attaches the L1 SV from the slice view) | Return the literal seg SV at that voxel. No constraint, no search. Stale L1 ids from the UI are ignored because the seg is the source of truth. |
| `layer >= 2` | 3D mesh pick (NG attaches the root the user clicked) | Read the literal seg SV. If its current root equals the supplied root → return the literal. Otherwise run a parent-constrained nearest-SV search using the supplied root, at growing radii. |

The cost ceiling, derived from the contract:

| Case | seg reads | `get_roots` | `get_atomic_ids_from_coords` |
|---|---|---|---|
| All coords are 2D | 1 | 0 | 0 |
| All coords are 3D-interior (literal root already matches) | 1 | 1 | 0 |
| 3D coord(s) need a search, all share one root | 1 | 1 | ≤ `len(max_dist_steps)` |
| 3D coords need a search across `k` distinct roots | 1 | 1 | one growing-radius sequence per root |

The growing-radius schedule defaults to
`np.array([4, 8, 14, 28]) * mean(meta.resolution)` nm — the loop breaks
as soon as one radius returns a result for a given root.

## Layout

```
pychunkedgraph/graph/sv_lookup/
├── __init__.py          # re-exports the public API
├── main.py              # resolve_supervoxels_at_coords (the orchestrator)
└── utils.py             # lookup_svs_from_seg, get_atomic_id_from_coord,
                         # get_atomic_ids_from_coords (the low-level lookups)
```

- `main.resolve_supervoxels_at_coords(cg, coordinates, node_ids,
  max_dist_steps=None)` — public entry point. Returns `(N,) uint64`.
  Raises `cg_exceptions.BadRequest` on invalid input or unresolvable
  coords.
- `utils.lookup_svs_from_seg(meta, coordinates)` — one batched seg read
  over the coords' bbox; returns the literal SV per coord.
- `utils.get_atomic_ids_from_coords(meta, coordinates, parent_id,
  parent_id_layer, parent_ts, get_roots, max_dist_nm)` — the
  parent-constrained nearest-SV search. Reads one bbox-sized seg block
  around the input coords, maps every voxel to its root via `get_roots`,
  then picks the nm-closest voxel whose root matches `parent_id` for
  each input coord. Returns `None` if no voxel within `max_dist_nm` matches.
- `utils.get_atomic_id_from_coord(...)` — single-coord variant retained
  for the `cg.get_atomic_id_from_coord` method wrapper.

## Wiring

```
app.app_utils.handle_supervoxel_id_lookup            # thin Flask-layer wrapper
        └── sv_lookup.resolve_supervoxels_at_coords  # the contract above
                ├── sv_lookup.utils.lookup_svs_from_seg
                ├── cg.get_chunk_layers
                ├── cg.get_roots
                └── cg.get_atomic_ids_from_coords    # for 3D-needs-search only
                        └── sv_lookup.utils.get_atomic_ids_from_coords
```

`cg.get_atomic_ids_from_coords` (defined on `ChunkedGraph`) is the
method wrapper around `utils.get_atomic_ids_from_coords`; it provides
`parent_ts` from the parent's node timestamps and short-circuits a
layer-1 parent to `[parent_id] * N`. The orchestrator only ever invokes
it with a root (layer ≥ 2), so the layer-1 short-circuit never fires
from this path.

## Tests

- `tests/graph/test_sv_lookup_main.py` — orchestrator behaviour using a
  real bigtable-backed `gen_graph`, with `meta._ws_cv` swapped for a
  small sliceable in-memory seg. Covers 2D-only, 3D-interior,
  3D-on-background, search-exhausted, growing-radius-breaks-on-success,
  mixed-batch-with-multiple-roots, and the all-2D / all-3D-interior
  cost-ceiling guarantees.
- `tests/graph/test_sv_lookup_utils.py` — the low-level
  `get_atomic_id_from_coord` and `get_atomic_ids_from_coords` tests.
