# Stitch Redesign — Session Context

For a new Claude session to pick up this work, tell it:
> Read pychunkedgraph/debug/stitch_test/SESSION.md and design.md to understand the stitch redesign state.

## Key files

- `.env/stitching/design.md` — high-level algorithm design doc
- `pychunkedgraph/debug/stitch_test/design.md` — detailed algorithm description with phase breakdowns
- `pychunkedgraph/debug/stitch_test/proposed.py` — the proposed stitch implementation
- `pychunkedgraph/debug/stitch_test/wave.py` — unified test runner (single/wave/multiwave experiments)
- `pychunkedgraph/debug/stitch_test/utils.py` — structure extraction, batched parallel extraction, comparison functions
- `pychunkedgraph/debug/stitch_test/compare.py` — orchestration, persistence helpers
- `pychunkedgraph/debug/stitch_test/current.py` — wrapper for current `add_edges` baseline
- `pychunkedgraph/debug/stitch_test/tables.py` — BigTable backup/restore, env setup, autoscaling
- `.env/stitching/hsmith_mec.ipynb` — test notebook

## Module dependency order (no cycles)

tables → utils → {current, proposed} → compare → wave

- `utils.py` has pure functions: extract_structure, _compare_*, _convert_for_json, batched extraction, SV-based comparison
- `compare.py` has orchestration + persistence: imports from current, proposed, utils
- Never import from compare into utils

## Current status (2026-03-23)

### What works
- Proposed algorithm implemented and structurally correct (single file match verified)
- Single file test: proposed ~151s vs current ~205s (1.35x speedup on this VM)
- Wave 0 current baseline: 606 files, 311K roots, ~1050s wall with 512 workers
- Wave 0 proposed: completed 638s wall (1.64x speedup), structural comparison pending (comparison bug fixed, needs re-run)

### Extraction and comparison design
- **SV-based components**: `extract_structure` resolves L2 → SVs so components are frozensets of SV IDs (stable across tables, order-independent)
- **Compressed storage**: `np.savez_compressed` with flat arrays + offsets for variable-length SV sets
- **Independent extraction**: each side extracted into its own subdirectory (`current/`, `proposed/`)
- **Order-independent comparison**: uses sets of frozensets per layer, not sorted lists. No shard-to-shard matching needed.
- **No table deletion**: user manages table cleanup via prefix

### Retry safety
- **`_get_all_parents_filtered`**: replaces `get_all_parents_dict_multiple` for stitching. Applies `filter_failed_node_ids` at every layer during parent chain traversal to detect and remap orphaned nodes from prior failed stitch attempts.
- **`filter_failed_node_ids`** applied to both `l2ids` and `l2_siblings` after reading their children.
- **Two-phase writes**: `_build_entries` returns `(node_entries, parent_entries)`. Node rows written first, then Parent pointers. Ensures Parent pointers only reference rows that exist.
- **No FormerParent**: proposed path does not write FormerParent/deprecation entries.
- **Crash recovery**: `stitch_results.json` saved immediately after stitch completes. Pass `run_id` to resume.
- **Fresh runs**: `_clear_log_dir` deletes old results before restoring table.

### Architecture decisions
- **No neighbor CrossChunkEdge updates**: stale is OK — future proposed stitches read AtomicCrossChunkEdge (immutable) + Parent + Child.
- **No locks**: lock-free, enables true parallelism within waves.
- **No table deletion**: user manages cleanup.

### Test infrastructure
- **Entry points**: `run_current(experiment)` and `run_proposed_and_compare(experiment, run_id=None)`
- **Experiment types**: "single" (one file), "wave" (wave 0), "multiwave" (all waves)
- **Extraction**: 500K root batches, sharded across cpu_count workers, each saves own .npz
- **Retries**: tenacity on extraction reads (3 attempts, exponential backoff)
- **Workers**: `min(n_files, 4 * cpu_count)` for wave processing
- **Progress**: tqdm for wave file processing
- **Autoscaling**: for wave/multiwave, sets BigTable CPU target to 25% before, reverts to 60% after (in `finally`).

### Performance

**Single file (task_0_0.edges, 1024 edges)**:
- Proposed ~151s vs current ~205s (1.35x)

**Wave 0 (606 files, 311K roots)**:
- Current: ~1050s wall
- Proposed: ~638s wall (1.64x)
- Proposed per-file: mean=245s, median=272s, p95=295s, max=399s (task_0_591.edges)

### Remaining work
- Re-run wave 0 comparison with fixed SV-based extraction
- Add incremental file result saving during wave runs
- Optimize proposed further (straggler task_0_591 took 399s)
- Run multiwave test once wave 0 validates

## User preferences (critical)
- **Never describe how code works without reading it first** — use Read/Grep, or say "I haven't verified this"
- **Never use nested/inline imports** — all imports at module top level, design modules to avoid circular deps
- **Never create commits** — user does them
- **Vectorized numpy** — no Python loops where numpy works
- **Keep notebooks simple** — short function calls only, all logic in modules
- **No patchwork** — design complete algorithms from first principles
- **No fat VMs** — hard constraint
- **Max effort always**
- **No mocks** — only mocker fixture
- **Test end-to-end before presenting**
- **Never modify user's code without asking**
- **Terse responses** — no trailing summaries
- **Never delete tables** — user manages cleanup via prefix

## Dataset
- **hsmith_mec**: 7 layers, ~600k edges, 1095 total files
  - Wave 0: 606 files
  - Edge source: `gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges`
  - Backup table: `hsmith-mec-100gvx-exp16-0.26-backup`
  - BigTable project: `zetta-proofreading`, instance: `pychunkedgraph`
