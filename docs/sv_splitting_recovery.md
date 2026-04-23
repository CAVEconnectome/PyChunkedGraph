# Supervoxel split recovery

## What it is

A recovery path for supervoxel-split operations that crash mid-write, leaving partial state in segmentation and per-chunk locks held indefinitely. An operator runs a one-shot command that reverts the crashed op's partial segmentation writes and re-runs the op from scratch, producing a clean successful edit and freeing the affected chunks for future work.

## When it applies

Every supervoxel split writes two things atomically from the point of view of the operation:
- Segmentation chunks — the voxel-level split, where each L2 chunk touched by the split gets fresh supervoxel IDs at the voxels that moved to a new fragment.
- Graph hierarchy rows — lineage from the old supervoxels to the new ones plus the cross-chunk edges linking the new fragments.

Both writes happen under an indefinite L2 chunk lock covering the exact chunks being rewritten. If the worker running the op dies before the lock's context exits — process kill, pod eviction, hardware failure, OOM — the lock stays held. From that moment on, any new op whose chunk set overlaps the stuck op's chunks refuses to start, blocking further corruption.

The operator runs recovery when the lock has been held long enough that the worker is definitively gone, not merely slow. A minimum-age threshold (10 minutes is a reasonable default) distinguishes stuck ops from ops still in flight.

## Concurrent edits on other regions keep working

The indefinite lock is per L2 chunk. While op X is stuck on chunks `{C1, C2, C3}`, another op Y on chunks `{C4, C5}` sees no indefinite cell on its chunks and proceeds normally. Its writes advance the latest OCDBT manifest. By the time the operator gets to recovery, the manifest has moved past the stuck op's `OperationTimeStamp` and other regions of segmentation reflect Y's (and any subsequent ops') work.

This is important: the recovery must not undo Y's changes. It also cannot rely on a single "read pre-op segmentation" pin because that would return pre-Y state outside the stuck op's own chunks, and the replay would overwrite neighbor state with stale values.

## Why pre-op pinning is not enough on its own

A supervoxel split reads more than just its own chunks. To route existing cross-chunk edges onto the new fragments, the split reads a one-voxel shell around its chunk envelope — supervoxel IDs from neighboring chunks serve as anchors for the re-routed edges.

If the replay opened segmentation with a pin at the stuck op's `OperationTimeStamp` and then read that shell, the neighbor voxels would show their pre-op state, not their current state. If Y had split a supervoxel in one of those neighbor chunks in the interim, the pinned read would see the old neighbor IDs, and the replay would route its cross-chunk edges to supervoxel IDs that no longer exist. Graph corruption.

So the replay cannot read the world through a single pinned view. It must see latest state for the neighbor shell and clean pre-op state for its own chunks.

## Cleanup-then-replay

Recovery proceeds in two steps.

**Cleanup.** For each chunk in the stuck op's durably-recorded scope, the operator reads the chunk's pre-op voxel values from a segmentation handle pinned at the op's `OperationTimeStamp`, and writes those values back to the latest (unpinned) handle. The result: those chunks, at the latest manifest, now show pre-op segmentation — as if the crashed op had never started. Neighbor chunks and every chunk outside the stuck op's scope are untouched, so any concurrent op's work is preserved.

**Replay.** The operator then re-runs the op under the privileged-repair path. The run reads latest state, which is now consistent — clean pre-op values on the stuck op's own chunks, current state everywhere else — and goes through the normal edit flow. It allocates fresh supervoxel IDs, re-computes the split, writes new segmentation and hierarchy, and transitions the op-log row from `CREATED` through `WRITE_STARTED` to `SUCCESS`.

When the replay's indefinite lock context exits, it issues value-matched releases on every chunk in the scope. Because the replay re-uses the crashed op's operation ID, the value match succeeds and the pre-existing indefinite cells are deleted. The chunks are free for new ops again.

## Orphans in segmentation history

OCDBT is append-only. The crashed op's partial segmentation writes still exist in the store's commit history — they are not deleted, only overshadowed. At the latest manifest, the cleanup step has overwritten them with pre-op values, so a normal (unpinned) read returns the pre-op state and the replay's fresh writes take effect on top of that. Readers that explicitly pin a historical version between the crash and the replay will still see the partial writes as a snapshot, but readers at latest never observe them.

The orphan supervoxel IDs allocated by the crashed op are never referenced by any hierarchy row — the crashed op never wrote its hierarchy rows to completion, and the replay allocated a new set of IDs. From the graph's perspective those orphan IDs do not exist.

## Operator workflow

1. **List stuck ops.** The operator runs the list command with a minimum-age threshold. It returns op-log rows still at `CREATED` status past that age, along with each op's user ID, timestamp, age, and the number of chunks in its recorded scope. Ops too young to classify are skipped.

2. **Inspect.** For each candidate, confirm from logs or monitoring that the worker that submitted the op is definitively dead — not, for example, paused on a long-running multicut. The minimum-age threshold exists to reduce false positives but the operator retains final judgment.

3. **Replay.** The operator runs the replay command with the op ID. It performs the cleanup pass, then hands off to the privileged-repair path to rerun the op. On success, the op-log row shows `SUCCESS` and the previously-held indefinite lock cells are released.

4. **Verify.** A second list invocation should no longer include the op. Any new ops that were waiting on the affected chunks proceed.

If the replay itself fails — for example, the operator's judgment about the worker's status was wrong and the original worker comes back — the replay surfaces the error and leaves the op-log row and lock state as it found them. The operator investigates, potentially clears the lock manually via direct bigtable tools, and tries again.

## Invariants

- A stuck op's durable scope record (written before any segmentation or hierarchy write begins) lets recovery locate every chunk that might have received a partial write, without a bigtable-wide scan.
- Cleanup only touches chunks in the stuck op's scope. Neighbor state and any concurrent ops' changes are preserved byte-for-byte.
- The replay sees a consistent world: pre-op values on the stuck op's own chunks (from the cleanup), current state on every other chunk (from the latest manifest).
- After successful replay, the op-log row is at `SUCCESS`, all indefinite cells previously held by that op are released, and the affected chunks are available to new ops. The op's original intent — the edit the user asked for — is realized with a fresh set of supervoxel IDs.
