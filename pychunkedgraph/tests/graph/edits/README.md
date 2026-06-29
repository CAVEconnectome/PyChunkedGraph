# Edit-operation tests

The proofreading edit operations and their atomicity:

- **merge** (`add_edges`)
- **split** / **mincut** / **multicut** (`remove_edges`)
- **undo / redo**
- **supervoxel split**

Graphs are built with the `build_graph` factory (or explicit chunk construction
where the produced row structure is itself the subject). The core invariant under
test is that a **rejected** edit leaves every stored row untouched — expressed with
the `assert_graph_unchanged` context manager.

## Coverage matrix

| Operation | Topologies exercised | Accepted paths | Rejected paths (graph unchanged) |
|---|---|---|---|
| merge | same chunk, neighboring chunks, disconnected chunks, skip-connection across layers | new shared parent; cross-chunk edge re-mapping; multi-layer hierarchy correctness | already-connected pair; self-loop; abstract (non-L1) node; indirect cycle |
| split | same chunk, neighboring, disconnected; full-circle ↔ triple-chain; skip-connection components | component separation; pre-edit state still readable at the old timestamp | nonexistent edge; cross-layer / abstract-node edge |
| mincut | regular link, no link, previously-removed link, indivisible (`inf`) link | source/sink separation respecting affinities | disconnected source/sink; cut that isolates; cut disrespecting source/sink |
| multicut | path-augmented cut over real supervoxel data | multi-source / multi-sink cut | supervoxel-split-required is signalled |
| undo / redo | split ↔ merge inversion | state restoration; chain resolution; subgraph leaves preserved; edge re-validation on undo | — |
| supervoxel split | supervoxel-level split over real watershed data | split applied | — |

Per-operation **atomicity** (graph unchanged after a rejected edit) is asserted for
the merge, split, and mincut rejection cases.

## Gaps (→ edit-failure-handling follow-up)

These are the known coverage holes; each is filled by the edit-failure-handling and
single-edit-rollback follow-ups, ticking off rows here as they land.

- **Full-hierarchy invariant verifier after every edit** — layer monotonicity,
  parent↔child bidirectionality, one root per connected component, no duplicate
  children per layer, cross-edge validity. Today a rejected edit only compares the
  stored rows, not these structural invariants.
- **Fail-before-persist proven for every bad input** — zero hierarchy mutation *and*
  no `SUCCESS` operation-log row written. Two non-blocking rejection cases currently
  `warn` instead of asserting the graph is unchanged.
- **Mid-write crash + recovery** — interrupt the operation mid-persist and prove
  `repair_operation` restores a consistent state.
- **Concurrent-write isolation** — two edits on the same vs disjoint roots.
- **OCDBT-backed edit paths.**
- **The multicut scenario that currently skips itself** on a too-small graph.
