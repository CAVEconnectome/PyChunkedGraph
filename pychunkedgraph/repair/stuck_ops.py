"""Operator recovery for SV-split ops that crashed mid-write.

A crash inside `IndefiniteL2ChunkLock`'s critical section leaves the
per-chunk `Concurrency.IndefiniteLock` cells set *and* records the
chunk scope on the op-log row's `OperationLogs.L2ChunkLockScope` field.
Ops on other (non-overlapping) chunks continue to succeed and advance
the OCDBT manifest while the stuck op sits there blocking its own
chunks.

Recovery = cleanup + replay. The cleanup step reverts the stuck op's
partial OCDBT writes by copying pre-op voxel values (read from a
version-pinned OCDBT handle at the op's `OperationTimeStamp`) back to
the latest manifest. The replay then runs the op normally via the
existing `repair.edits.repair_operation` path — reads latest (clean on
the stuck op's chunks, current on everyone else's), writes fresh SV
IDs, and `IndefiniteL2ChunkLock`'s privileged-mode exit deletes the
crashed op's pre-existing cells.

See `pychunkedgraph/graph/sv_split/recovery.md` for the full
architecture and correctness argument.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np

from pychunkedgraph import get_logger
from pychunkedgraph.graph import ChunkedGraph, attributes
from pychunkedgraph.graph.chunks.utils import get_chunk_coordinates
from pychunkedgraph.graph.locks import _l2_chunk_lock_row_key
from pychunkedgraph.graph.ocdbt import get_seg_source_and_destination_ocdbt
from pychunkedgraph.repair.edits import repair_operation

logger = get_logger(__name__)


def _operation_ts_to_pin(operation_ts: datetime) -> str:
    """Convert an op-log `OperationTimeStamp` to the OCDBT `version`
    string format — ISO-8601 UTC with `Z` suffix, microsecond
    precision. OCDBT's binder rejects `+00:00`.
    """
    if operation_ts.tzinfo is None:
        operation_ts = operation_ts.replace(tzinfo=timezone.utc)
    else:
        operation_ts = operation_ts.astimezone(timezone.utc)
    return operation_ts.isoformat().replace("+00:00", "Z")


def _chunk_voxel_slices(cg: ChunkedGraph, chunk_id: int) -> tuple:
    """Voxel-space slice tuple covering one L2 chunk, clipped to volume bounds."""
    coords = get_chunk_coordinates(cg.meta, np.uint64(chunk_id))
    chunk_size = np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=int)
    voxel_bounds = cg.meta.voxel_bounds
    lo = coords * chunk_size + voxel_bounds[:, 0]
    hi = np.minimum(lo + chunk_size, voxel_bounds[:, 1])
    return tuple(slice(int(s), int(e)) for s, e in zip(lo, hi))


def list_stuck(cg: ChunkedGraph, min_age: timedelta = timedelta(minutes=10)) -> list:
    """Return op-log entries whose `L2ChunkLockScope` is set past `min_age`,
    excluding successfully-completed ops.

    The authoritative signal for a stuck op is "scope recorded" —
    `IndefiniteL2ChunkLock.__enter__` writes it before any seg/bigtable
    write and its clean `__exit__` clears it. An op whose scope is
    still populated is either a worker crash (Status=CREATED, Fix 1's
    `__exit__` short-circuit never ran) or an exception during the
    persist block (Status=EXCEPTION, Fix 1 held the cells on the way
    out). Either way it's still holding `Concurrency.IndefiniteLock`
    cells on the listed chunks and blocking any new op that overlaps.

    Ops that reach `SUCCESS` normally have scope cleared — we defensively
    filter them out in case `_clear_scope_on_op_log`'s best-effort write
    failed and logged. Failed ops that never touched the persist block
    (e.g. a PreconditionError from multicut) have no scope and don't
    show up here; they're not blocking anything.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - min_age
    entries = cg.client.read_log_entries()
    stuck = []
    success_code = attributes.OperationLogs.StatusCodes.SUCCESS.value
    for op_id, entry in entries.items():
        scope = entry.get(attributes.OperationLogs.L2ChunkLockScope)
        if scope is None or len(scope) == 0:
            continue
        if entry.get(attributes.OperationLogs.Status) == success_code:
            continue
        op_ts = entry.get(attributes.OperationLogs.OperationTimeStamp)
        if op_ts is None:
            continue
        if op_ts.tzinfo is None:
            op_ts = op_ts.replace(tzinfo=timezone.utc)
        if op_ts > cutoff:
            continue
        stuck.append(
            {
                "op_id": int(op_id),
                "operation_ts": op_ts,
                "age": now - op_ts,
                "user_id": entry.get(attributes.OperationLogs.UserID),
                "l2_chunk_scope": scope,
                "status": entry.get(attributes.OperationLogs.Status),
            }
        )
    stuck.sort(key=lambda r: r["op_id"])
    return stuck


def cleanup_partial_writes(cg: ChunkedGraph, op_id: int) -> int:
    """Revert a stuck op's partial OCDBT writes to pre-op voxel values.

    Reads each chunk in the op's `L2ChunkLockScope` through an OCDBT
    handle pinned at the op's `OperationTimeStamp` (which pre-dates any
    of its commits), then writes those pre-op values back to the latest
    manifest. Overwrites the crashed op's partial seg writes at the
    same chunk keys; neighbor chunks are untouched, preserving any
    concurrent ops' updates.

    Returns the number of chunks rewritten.
    """
    log_entries = cg.client.read_log_entries(operation_ids=[np.uint64(op_id)])
    if not log_entries:
        raise ValueError(f"No op-log row for op_id={op_id}")
    entry = log_entries[np.uint64(op_id)]

    scope = entry.get(attributes.OperationLogs.L2ChunkLockScope)
    if scope is None or len(scope) == 0:
        logger.info(f"op {op_id} has no L2ChunkLockScope — nothing to clean up")
        return 0

    operation_ts = entry.get(attributes.OperationLogs.OperationTimeStamp)
    if operation_ts is None:
        raise ValueError(f"op {op_id} has no OperationTimeStamp")
    pin_str = _operation_ts_to_pin(operation_ts)

    # Pinned read handle (read-only at pre-op version) vs. unpinned
    # write handle (latest). Tensorstore refuses writes on version-pinned
    # kvstores, so the two paths use separate handles.
    _, pinned_scales, _ = get_seg_source_and_destination_ocdbt(
        cg.meta.data_source.WATERSHED,
        cg.meta.graph_id,
        cg.meta.ocdbt_config,
        pinned_at=pin_str,
    )
    pinned_ws = pinned_scales[0]
    latest_ws = cg.meta.ws_ocdbt

    def _revert_chunk(chunk_id: int) -> None:
        voxel_slices = _chunk_voxel_slices(cg, int(chunk_id))
        pre_op = pinned_ws[voxel_slices + (slice(None),)].read().result()
        latest_ws[voxel_slices + (slice(None),)].write(pre_op).result()

    # Parallel read-then-write per chunk. Bounded pool so large scopes
    # don't saturate tensorstore's internal concurrency.
    max_workers = min(16, max(1, len(scope)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_revert_chunk, int(c)) for c in scope]
        for future in as_completed(futures):
            future.result()

    logger.info(f"op {op_id}: reverted {len(scope)} partial chunk writes")
    return len(scope)


def _verify_indefinite_cells(cg: ChunkedGraph, op_id: int, scope) -> list:
    """Check each chunk in `scope` actually has `Concurrency.IndefiniteLock`
    held by `op_id`. Returns the list of chunk IDs whose cell is missing
    or held by a different op_id — an empty list means everything is
    consistent.

    Guards `replay` against acting on a stale scope: if cells aren't
    actually held (operator already ran replay, manual intervention,
    any bug that released cells without clearing scope), `cleanup_
    partial_writes` would revert chunks that another op may have
    legitimately written to in the meantime. Refusing loudly is safer
    than assuming.
    """
    lock_column = attributes.Concurrency.IndefiniteLock
    expected = np.uint64(op_id)
    discrepancies = []
    for chunk_id in scope:
        row_key = _l2_chunk_lock_row_key(int(chunk_id))
        cells = cg.client._read_byte_row(row_key, columns=lock_column)
        if not cells:
            discrepancies.append(int(chunk_id))
            continue
        held_by = cells[0].value if hasattr(cells[0], "value") else None
        if held_by != expected:
            discrepancies.append(int(chunk_id))
    return discrepancies


def replay(cg: ChunkedGraph, op_id: int):
    """Recovery: verify locks, clean up partial OCDBT writes, then run
    the op normally.

    Before any destructive step, read back the per-chunk
    `Concurrency.IndefiniteLock` cells listed in the op's
    `L2ChunkLockScope` and confirm they're still held by `op_id`. If
    any are missing or held by another op, raise and do nothing —
    proceeding would have `cleanup_partial_writes` revert chunks we
    don't actually own.

    On clean verification, `cleanup_partial_writes` reverts the op's
    partial OCDBT writes, then `repair.edits.repair_operation` reruns
    `operation.execute(..., privileged_mode=True, parent_ts=<previous-
    edit ts>)`. `IndefiniteL2ChunkLock.__enter__` in privileged mode
    populates `acquired_keys` from the scope so `__exit__` releases the
    crashed op's pre-existing indefinite cells after the replay writes
    land.
    """
    log_entries = cg.client.read_log_entries(operation_ids=[np.uint64(op_id)])
    if not log_entries:
        raise ValueError(f"No op-log row for op_id={op_id}")
    entry = log_entries[np.uint64(op_id)]
    scope = entry.get(attributes.OperationLogs.L2ChunkLockScope)
    if scope is None or len(scope) == 0:
        raise RuntimeError(
            f"op {op_id} has no L2ChunkLockScope — not a stuck SV-split op. "
            "If the op failed cleanly, the client should re-submit under a "
            "fresh op_id rather than replay."
        )

    mismatched = _verify_indefinite_cells(cg, op_id, scope)
    if mismatched:
        raise RuntimeError(
            f"op {op_id}: L2ChunkLockScope lists chunks {[int(c) for c in scope]}, "
            f"but the following chunks do not have Concurrency.IndefiniteLock "
            f"held by op_id={op_id}: {mismatched}. Refusing to replay — the "
            "recorded scope disagrees with live lock state. Possible causes: "
            "replay already ran, cells were manually cleared, or a different "
            "op acquired these chunks. Investigate before retrying."
        )

    cleanup_partial_writes(cg, op_id)
    return repair_operation(cg, op_id, unlock=True)


def _main():
    parser = argparse.ArgumentParser(
        description="Recover stuck SV-split operations via cleanup + replay."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser(
        "list",
        help="List stuck ops (L2ChunkLockScope still populated past min-age).",
    )
    p_list.add_argument("--graph", required=True, help="Graph ID.")
    p_list.add_argument(
        "--min-age",
        type=int,
        default=10,
        help="Minimum age in minutes before an op is considered stuck (default: 10).",
    )

    p_replay = sub.add_parser(
        "replay", help="Clean up partial writes and replay a stuck op."
    )
    p_replay.add_argument("--graph", required=True, help="Graph ID.")
    p_replay.add_argument("--op-id", type=int, required=True, help="Op ID to replay.")

    args = parser.parse_args()
    cg = ChunkedGraph(graph_id=args.graph)

    if args.cmd == "list":
        stuck = list_stuck(cg, min_age=timedelta(minutes=args.min_age))
        if not stuck:
            print("No stuck ops.")
            return
        for row in stuck:
            scope_size = (
                len(row["l2_chunk_scope"]) if row["l2_chunk_scope"] is not None else 0
            )
            print(
                f"op {row['op_id']}: user={row['user_id']} "
                f"ts={row['operation_ts'].isoformat()} "
                f"age={row['age']} "
                f"l2_chunks={scope_size}"
            )
    elif args.cmd == "replay":
        result = replay(cg, args.op_id)
        print(f"replay complete: {result}")


if __name__ == "__main__":
    _main()
