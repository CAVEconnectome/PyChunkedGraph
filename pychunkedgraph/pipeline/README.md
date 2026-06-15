# `pipeline` — Kubernetes-native chunk-batch pipeline

Runs chunk-grid workloads — **ingest**, **meshing**, and **migrate** — as **one
Kubernetes Indexed `Job` per layer**, with **no Redis/RQ** and no scheduler add-on.
The workload-agnostic core (scatter, lock, exit-code contract, worker harness,
`run_and_exit`) is the `cave-pipeline` distribution package, shared with every other
worker so the grid bijection and Job contract have a single source. This package
supplies the per-workload subpackages plus the PyChunkedGraph harness glue (a
ChunkedGraph `context_factory` and the per-layer `bounds_fn`). Each workload is its
own container entrypoint:

```
python -m pychunkedgraph.pipeline.ingest     # ingest worker
python -m pychunkedgraph.pipeline.meshing     # mesh worker
python -m pychunkedgraph.pipeline.migrate     # migrate worker
```

## Model

A layer's chunks form an `X·Y·Z` grid of size `N`. The Indexed Job's completion
index **is** the unit of work — there is no queue. To keep the job count within
Kubernetes limits at large `N` (10M+ chunks), each index processes a **batch** of
`B` chunks, so `completions = ceil(N / B)` and `parallelism` = the worker fleet
width. A pod walks its batch sequentially; ingest runs each chunk under a per-chunk
lock (one writer per chunk), while many pods run in parallel.

```
layer L (X·Y·Z = N chunks)
  └─ Indexed Job: completions = ceil(N/B), parallelism = fleet
       └─ pod (index i) -> B scattered chunk coords
            └─ for each chunk: process_one -> ok | done | transient | fatal
```

**One layer at a time, operator-gated.** Each layer is its own Job; the operator
launches the next only after the current Job reports `Complete` (resources
typically need tuning between layers). Nothing auto-advances.

## Layout

The chunk-distribution core — the grid scatter, per-chunk lock, exit-code contract,
worker harness, and `run_and_exit` — is the `cave-pipeline` `distribution` package, a
pinned dependency. This package holds only the workloads and the harness glue:

| Path | Responsibility |
|---|---|
| `__init__.py` | The harness glue injected into the shared core: a ChunkedGraph `context_factory` and the per-layer `bounds_fn`. |
| `ingest/` | Ingest workload: branch-aware `dispatch` (L2 atomic edges / L>2 agglomeration), `setup` (graph table + meta), `worker` (lock + heartbeat around dispatch). |
| `meshing/` | Mesh workload: `meta` (`MeshConfig`), `setup` (mesh-metadata, `setup_mesh_meta`), `worker` (marching cubes at L2, sharded stitching above; idempotent, no lock). |
| `migrate/` | Migrate workload: in-place chunk upgrade + `--clean` cleanup pass; idempotent, no lock. |

## Setup (run once per workload, before any Jobs)

Each workload has a one-shot setup step that runs the same image and reads the
dataset yaml from its mounted path (`PCG_DATASET`, not passed):

```
python -m pychunkedgraph.pipeline.ingest.setup <graph_id> [--raw]   # create table + graph meta
python -m pychunkedgraph.pipeline.meshing.setup <graph_id>          # write mesh.* metadata (needs `mesh_config:`)
```

Ingest setup creates the Bigtable table and folds the agglomeration source into
`meta.custom_data["agg"]`; after it, workers read everything from Bigtable.
Mesh setup writes the four `mesh.*` fields a graph needs to serve meshes (run once,
after ingest reaches the root layer), reading a `mesh_config:` block from the yaml.

**Credentials.** GCP clients use Application Default Credentials, so the cleanest
form is a one-shot in-cluster Job reusing the worker pods' Workload Identity.

## Worker contract (environment)

The Job template sets these on each pod:

| Variable | Meaning |
|---|---|
| `JOB_COMPLETION_INDEX` | batch index (set by Kubernetes) |
| `PCG_GRAPH_ID` | graph id; meta + workload state are read from Bigtable |
| `PCG_LAYER` | layer being built |
| `PCG_PERM_SEED` | permutation seed — **same across all pods and retries of a run** |
| `PCG_BATCH_SIZE` | `B`, chunks per index |
| `PCG_N_THREADS` | parallel sub-workers inside a parent-chunk build (default 1) |
| `PCG_LOCK_EXPIRY_SCALE` | (ingest) scales the per-layer claim TTL; default 1 |
| `PCG_LOCK_POLL_SEC` / `PCG_HELD_MAX_WAIT_SEC` | (ingest) poll interval / max wait before deferring a held chunk |
| `PCG_MESH_CACHE` | (meshing) `0` disables the mesh task's cloud cache; default on |

## Per-chunk lock (ingest)

Each chunk's lock row holds one of three states — *absent* / *claimed (token +
expiry)* / *done*. **acquire** succeeds only if neither `done` nor freshly claimed
elsewhere (an expired claim is stolen); a live worker **renews** in the background so
its in-progress chunk is never stolen; **mark-done**/**release** are value-matched on
the claim token (a fence) so a partitioned zombie can't clobber the new owner. On pod
death mid-batch the finished chunks stay `done`; the retried index skips them and
re-claims only the unfinished. Meshing needs no lock — it overwrites shards idempotently.

## Failure handling

- **Preemption** (spot reclaim): ignored by the failure policy (off the retry
  budget); the index is retried.
- **Transient failure**: counts toward the bounded per-index retry budget; the batch
  retries (ingest skips done chunks).
- **Non-transient failure** (`FatalChunkError`, exit 42): the index is failed fast and
  recorded for inspection. A batch finishes all its chunks before choosing an exit code.
- **Root verification** (ingest): after the root chunk is built, the pod runs the
  hierarchy sanity suite (`ingest.simple_tests`) as its final step — every ingest ends
  verified. A failed check fails the pod without re-opening the chunk, so re-submitting
  the root layer re-runs only the checks, never the build.

## Testing

- The chunk-distribution core (permutation, harness) is tested in `cave-pipeline`.
- Lock / workers: validated against the Bigtable emulator outside the committed suite,
  so the package stays importable on branches without that fixture.
