# mesh_meta

Single, branch-agnostic home for **graphene mesh metadata**. Every value comes
from **`custom_data["mesh"]`, the single source of truth**. `MeshMeta(cg)` wraps a
ChunkedGraph and exposes, as properties, the mesh dir/bucket locations and the
per-dataset mesh params used by the v2 manifest and by remeshing; multi-resolution
(LOD) mesh meta will live here too as it lands, so this is the module to grow
rather than scattering mesh-meta reads across the code.

## Why it exists

Supervoxels (watershed) can move to a cheap, egress-waived bucket while meshes
stay put. The v2 graphene manifest therefore carries **absolute mesh bucket
paths** rather than paths composed relative to the watershed `data_dir`, so a
client fetches meshes straight from wherever they live. This module is the one
place that decides where meshes live, and what the mesh params are, for a given
ChunkedGraph.

## Two mesh locations

- **initial** — the sharded meshes produced at ingest. **Shared / graph-independent**:
  several ChunkedGraphs (e.g. copies) point at the *same* initial meshes, so this
  location is never namespaced by graph id.
- **dynamic** — the unsharded meshes produced by proofreading edits.
  **Per-graph**: each graph keeps its own, so on the pcgv3 line it defaults to a
  graph-id-derived dir (`dynamic_<graph_id>`) and stays isolated even while
  sharing initial meshes; on the pcgv2 line it is a plain subdir.

## Configuration

All read from `custom_data["mesh"]`:

- `dir` — the mesh root under the watershed (default `graphene_meshes`).
- `initial_mesh_dir` — the initial location (default `initial`).
- `dynamic_mesh_dir` — the dynamic location (default `dynamic`; pcgv3 setup fills
  `dynamic_<graph_id>`).
- `max_layer`, `mip`, `max_error` — mesh params (start layer for manifests, and
  the mip / max-error used by remeshing).
- `initial_ts` — the ingest timestamp boundary that splits initial (sharded) node
  ids from proofread (dynamic) ones.

Resolution rule for the two location values: a value containing a cloud scheme
(`gs://`, `s3://`, …) is treated as an **absolute** bucket and used as-is;
otherwise it is joined under `<watershed>/<dir>`. This lets a dataset move
initial and/or dynamic meshes to different buckets without touching anything else.

Branch-agnostic on purpose: this module reads only `custom_data["mesh"]` and the
watershed path — accessors identical on the pcgv2 and pcgv3 lines — so it
cherry-picks clean between them. How that config gets *populated* (e.g. pcgv3's
setup-time `MeshConfig` from the dataset yaml) is branch-specific and lives
upstream; this module only reads the result.

## API

`MeshMeta(cg)` exposes these properties:

- `initial_path` → absolute dir of the initial (sharded) meshes.
- `dynamic_path` → absolute dir of the dynamic (unsharded) meshes.
- `dir` → the sharded mesh dir name under the watershed.
- `max_layer`, `mip`, `max_error`, `initial_ts` → the mesh params.
- `needs_v2` → true when either location falls **outside** the watershed dir,
  i.e. meshes are not co-located with the watershed. Old clients compose paths
  relative to the watershed, so they cannot reach such meshes: the v2 manifest
  serves them correctly, and the v1 path must fail loudly rather than return
  unreachable paths.
- `reader_anchor` → the `(data_dir, mesh)` pair to hand a CloudVolume mesh reader.
  The reader appends `initial/` internally, so it must be anchored at the
  **parent** of the initial dir; this yields shard byte-ranges from the right
  bucket even for a migrated graph.
