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
  sharing initial meshes; on the pcgv2 line it is a plain subdir. A copy writes
  only its own, and also reads the dynamic dirs it inherits, for meshes its
  sources wrote before it was copied.

## Configuration

All read from `custom_data["mesh"]`:

- `dir` — the mesh root under the watershed. No default: setup always writes it, and a
  table without it has no meshes, so reading it raises `KeyError`.
- `initial_mesh_dir` — the initial location (default `initial`).
- `dynamic_mesh_dir` — the dynamic location (default `dynamic`; pcgv3 setup fills
  `dynamic_<graph_id>`).
- `inherited_dynamic` — a copy's sources' dynamic dirs, oldest first, each
  `{"path", "until"}`: an absolute dir, and the Unix second of the copy's first
  open, before which its source made a node. Only a copy's first open writes it,
  through `ChunkedGraphMeta.for_copied_graph`; any other table has none and reads
  as before.
- `max_layer`, `mip`, `max_error` — mesh params (start layer for manifests, and
  the mip / max-error used by remeshing).
- `uniform_draco_grid_size` — the draco grid clients deduplicate on, written by setup.
  Deprecated shim: a table whose block lacks it, or `max_meshed_layer` /
  `draco_grid_sizes`, gets the watershed info's value in `client_metadata` until its
  block is rewritten. Nothing else reads the watershed info for meshes.
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

`MeshMeta(cg)`, or `MeshMeta(meta)` for a graph meta alone, exposes these properties:

- `initial_path` → absolute dir of the initial (sharded) meshes.
- `dynamic_path` → absolute dir of the dynamic (unsharded) meshes.
- `inherited` → a copy's inherited dynamic dirs as `Layer(path, until)`, oldest
  first; empty for any other table.
- `dynamic_stack` → every dynamic dir a node's mesh may be in, the table's own first,
  then inherited dirs newest first: the order a missed lookup searches.
- `by_dynamic_dir(node_ids, created)` → new node ids grouped by the dynamic dir their
  creation times hint at. `manifest/dynamic.py` checks each hint with a probe and
  searches `dynamic_stack` on a miss, so a copy finds every mesh it or its sources
  wrote.
- `spec_dir` → the dir of the shard spec `info`, the parent of `initial_path`.
- `dir` → the sharded mesh dir name under the watershed.
- `dynamic_mesh_dir` → the dynamic dir as configured, a name or a cloud URL.
- `client_metadata` → the graphene info's `mesh_metadata` for this table.
- `pinned_block` → the block with both dirs absolute and its grid keys written in, so it
  reads the same under any watershed; `ChunkedGraphMeta.repoint_watershed` stores it.
- `max_layer`, `mip`, `max_error`, `initial_ts` → the mesh params.
- `needs_v2` → true when either location falls **outside** the watershed dir, or the
  table inherits dynamic dirs, i.e. meshes are not all co-located with the
  watershed. Old clients compose paths relative to the watershed, so they cannot
  reach such meshes: the v2 manifest serves them correctly, and the v1 path must
  fail loudly rather than return unreachable paths.
- `reader_anchor` → the `(data_dir, mesh)` pair to hand a CloudVolume mesh reader.
  The reader appends `initial/` internally, so it must be anchored at the
  **parent** of the initial dir; this yields shard byte-ranges from the right
  bucket even for a migrated graph.
