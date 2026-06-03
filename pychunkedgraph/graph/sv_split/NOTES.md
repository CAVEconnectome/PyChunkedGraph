# sv_split — notes (known issues, test handles, future work)

Secondary material that doesn't belong in `README.md` (the design reference).

## Future work — break up `cutting.py`

`cutting.py` mixes geodesic backend dispatch, seed snapping, EDT + cost-grid
construction, label assignment + narrow-band refinement, single-CC enforcement,
`_resolve_label3_touching_vectorized`, and the `split_supervoxel_growing` driver
in a single ~1900-line module. Refactor target: promote to a `cutting/` package
with one module per concern (`arrival.py`, `cost.py`, `seeds.py`, `label.py`,
`enforce.py`, `driver.py`), re-export the public entry points
(`split_supervoxel_growing`, `connect_both_seeds_via_ridge`) at
`sv_split.cutting`. Defer until the geodesic backend choice and the stray
mechanism have settled.

## Geodesic backend switch — `PCG_SV_SPLIT_GEODESIC_BACKEND`

`mcp` (default) selects `skimage.graph.MCP_Geometric`. `dj3d` selects
`dijkstra3d.distance_field` with the cost grid pre-scaled by `mean(sampling_ds)`
to approximate per-axis anisotropy (the function has no anisotropy parameter,
so the approximation degrades when one axis is >5× the others).
