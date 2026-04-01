# E2E Stitch Test Fixtures

## Source
Extracted from BigTable backup `hsmith-mec-100gvx-exp16-0.26-backup` in `zetta-proofreading/pychunkedgraph`.

Edges from: `gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges`

## Generation
Run `sample_test_data.extract(edges_range=(start, end), force=True)` from the notebook.

### Algorithm
1. Restore backup to a temporary BigTable table
2. Sample N edges per wave file (661 files across 32 waves)
3. Get roots for all edge SVs via parallel `get_roots`
4. BFS top-down from roots: read all nodes with all data (parent, children, ACX, CX) using multiprocessing
5. Build fixture store:
   - L3+ nodes: copied as-is from BigTable
   - L2 nodes: children = ACX source SVs (all layers) + edge SVs. ACX filtered to kept SVs. CX preserved.
   - ACX target SVs added to their L2's children list
   - SV parent rows for every kept SV
6. Save as gzip-compressed pickle
7. Delete temporary table

### Key parameters
- `edges_range`: tuple `(start, end)` — generates fixtures for each edge count in `range(start, end)`
- `seed`: RNG seed (default 42) for reproducible sampling
- `SAMPLE_EDGES_PER_FILE`: default 2

## Fixture contents
Each `.pkl.gz` file contains:
- `edges_per_wave`: `{wave_id: [edge_arrays]}` — sampled atomic edges per wave
- `node_store`: `{node_id: {parent, children, acx, cx}}` — all nodes in the subgraph
- `meta_bytes`: pickled `ChunkedGraph.meta`
- `cv_info`: CloudVolume info dict
- `sample_config`: `{edges_per_file, seed}`

## Stats
See [STATS.md](STATS.md) for per-fixture statistics (auto-updated on extraction).
