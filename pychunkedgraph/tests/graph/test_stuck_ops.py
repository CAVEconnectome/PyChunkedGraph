"""Tests for pychunkedgraph.repair.stuck_ops — cleanup + replay path
for SV-split ops that crashed mid-write.

The heavy test (`test_cleanup_reverts_partial_writes_to_pre_op`)
exercises the full cleanup flow against a real local OCDBT store — it
writes a known pre-op state, snapshots the manifest, writes simulated
"partial" data, constructs an op-log row with `L2ChunkLockScope` and
`OperationTimeStamp`, and asserts that cleanup reverts the scoped
chunks to pre-op values while leaving neighbor chunks alone.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorstore as ts

from pychunkedgraph.graph import attributes, ocdbt as ocdbt_mod
from pychunkedgraph.graph.chunks.utils import get_chunk_coordinates
from pychunkedgraph.graph.locks import _l2_chunk_lock_row_key
from pychunkedgraph.graph.meta import ChunkedGraphMeta, DataSource, GraphConfig
from pychunkedgraph.repair import stuck_ops

# Pick up the shared `local_ocdbt` fixture from test_ocdbt.
from .test_ocdbt import local_ocdbt  # noqa: F401


class TestListStuck:
    """`list_stuck` surfaces ops with non-empty `L2ChunkLockScope` past
    `min_age` whose Status isn't SUCCESS — i.e. still holding
    `Concurrency.IndefiniteLock` cells somewhere."""

    def _entry(self, status, age_seconds, user="u", scope=None):
        now = datetime.now(timezone.utc)
        entry = {
            attributes.OperationLogs.Status: status,
            attributes.OperationLogs.OperationTimeStamp: now
            - timedelta(seconds=age_seconds),
            attributes.OperationLogs.UserID: user,
        }
        if scope is not None:
            entry[attributes.OperationLogs.L2ChunkLockScope] = np.asarray(
                scope, dtype=np.uint64
            )
        return entry

    def _cg(self, entries):
        cg = MagicMock()
        cg.client.read_log_entries.return_value = entries
        return cg

    def test_filters_out_success_with_scope(self):
        """Defensive: a SUCCESS op with stale scope (if
        `_clear_scope_on_op_log` ever failed silently) must not be
        listed as stuck."""
        success = attributes.OperationLogs.StatusCodes.SUCCESS.value
        created = attributes.OperationLogs.StatusCodes.CREATED.value
        cg = self._cg(
            {
                np.uint64(1): self._entry(success, 900, scope=[10, 20]),
                np.uint64(2): self._entry(created, 900, scope=[10, 20]),
            }
        )
        stuck = stuck_ops.list_stuck(cg, min_age=timedelta(minutes=1))
        assert [r["op_id"] for r in stuck] == [2]

    def test_filters_out_empty_scope(self):
        """Ops that never touched the persist block (no scope) are not
        stuck via L2 locks — they're outside `stuck_ops`' concern."""
        created = attributes.OperationLogs.StatusCodes.CREATED.value
        exception = attributes.OperationLogs.StatusCodes.EXCEPTION.value
        cg = self._cg(
            {
                np.uint64(1): self._entry(created, 900),  # no scope
                np.uint64(2): self._entry(exception, 900),  # no scope
                np.uint64(3): self._entry(created, 900, scope=[42]),
            }
        )
        stuck = stuck_ops.list_stuck(cg, min_age=timedelta(minutes=1))
        assert [r["op_id"] for r in stuck] == [3]

    def test_surfaces_exception_path_with_scope(self):
        """After Fix 1, a Python exception during the persist block
        leaves cells held + scope set but Status=EXCEPTION. The op must
        be listed so the operator can recover it."""
        exception = attributes.OperationLogs.StatusCodes.EXCEPTION.value
        cg = self._cg(
            {
                np.uint64(42): self._entry(
                    exception, 900, user="alice", scope=[100, 200]
                ),
            }
        )
        stuck = stuck_ops.list_stuck(cg, min_age=timedelta(minutes=1))
        assert len(stuck) == 1
        row = stuck[0]
        assert row["op_id"] == 42
        assert row["status"] == exception
        assert list(row["l2_chunk_scope"]) == [100, 200]

    def test_filters_out_young_ops(self):
        created = attributes.OperationLogs.StatusCodes.CREATED.value
        cg = self._cg(
            {
                np.uint64(1): self._entry(created, 10, scope=[1]),  # too young
                np.uint64(2): self._entry(created, 3600, scope=[2]),  # an hour old
            }
        )
        stuck = stuck_ops.list_stuck(cg, min_age=timedelta(minutes=10))
        assert [r["op_id"] for r in stuck] == [2]

    def test_returns_scope_and_user(self):
        created = attributes.OperationLogs.StatusCodes.CREATED.value
        cg = self._cg(
            {
                np.uint64(7): self._entry(created, 1800, user="op", scope=[100, 200]),
            }
        )
        stuck = stuck_ops.list_stuck(cg, min_age=timedelta(minutes=10))
        assert len(stuck) == 1
        row = stuck[0]
        assert row["op_id"] == 7
        assert row["user_id"] == "op"
        assert list(row["l2_chunk_scope"]) == [100, 200]
        assert row["age"] > timedelta(minutes=10)


class TestVerifyIndefiniteCells:
    """`_verify_indefinite_cells` reads each chunk's indefinite-lock cell
    and reports any that don't match the expected op_id."""

    class _Cell:
        def __init__(self, value):
            self.value = value

    def _cg(self, cells_by_row_key):
        cg = MagicMock()

        def read(row_key, columns=None):
            return cells_by_row_key.get(row_key, [])

        cg.client._read_byte_row.side_effect = read
        return cg

    def test_all_held_by_same_op(self):
        op_id = 42
        scope = [np.uint64(1), np.uint64(2)]
        cells = {
            stuck_ops._l2_chunk_lock_row_key(1): [self._Cell(np.uint64(op_id))],
            stuck_ops._l2_chunk_lock_row_key(2): [self._Cell(np.uint64(op_id))],
        }
        cg = self._cg(cells)
        assert stuck_ops._verify_indefinite_cells(cg, op_id, scope) == []

    def test_cell_missing_flagged(self):
        op_id = 42
        scope = [np.uint64(1), np.uint64(2)]
        cells = {
            stuck_ops._l2_chunk_lock_row_key(1): [self._Cell(np.uint64(op_id))],
            # chunk 2 has no cell
        }
        cg = self._cg(cells)
        discrepancies = stuck_ops._verify_indefinite_cells(cg, op_id, scope)
        assert discrepancies == [2]

    def test_cell_held_by_different_op_flagged(self):
        op_id = 42
        other_op = np.uint64(99)
        scope = [np.uint64(1), np.uint64(2)]
        cells = {
            stuck_ops._l2_chunk_lock_row_key(1): [self._Cell(other_op)],
            stuck_ops._l2_chunk_lock_row_key(2): [self._Cell(np.uint64(op_id))],
        }
        cg = self._cg(cells)
        discrepancies = stuck_ops._verify_indefinite_cells(cg, op_id, scope)
        assert discrepancies == [1]


class TestReplayVerifies:
    """`replay` refuses to call cleanup_partial_writes or repair_operation
    when the recorded scope disagrees with live indefinite-lock state."""

    def test_replay_refuses_when_cells_missing(self, monkeypatch):
        op_id = 77
        scope = np.asarray([1, 2], dtype=np.uint64)

        cg = MagicMock()
        cg.client.read_log_entries.return_value = {
            np.uint64(op_id): {
                attributes.OperationLogs.L2ChunkLockScope: scope,
                attributes.OperationLogs.OperationTimeStamp: datetime.now(timezone.utc),
            }
        }
        # No cells held on either chunk.
        cg.client._read_byte_row.return_value = []

        # Spy on the destructive steps — neither should be called.
        cleanup_called = {"v": False}
        repair_called = {"v": False}
        monkeypatch.setattr(
            stuck_ops,
            "cleanup_partial_writes",
            lambda *a, **k: cleanup_called.__setitem__("v", True),
        )
        monkeypatch.setattr(
            stuck_ops,
            "repair_operation",
            lambda *a, **k: repair_called.__setitem__("v", True),
        )

        with pytest.raises(RuntimeError, match="Refusing to replay"):
            stuck_ops.replay(cg, op_id)
        assert not cleanup_called["v"]
        assert not repair_called["v"]

    def test_replay_refuses_when_empty_scope(self, monkeypatch):
        op_id = 77
        cg = MagicMock()
        cg.client.read_log_entries.return_value = {
            np.uint64(op_id): {
                attributes.OperationLogs.OperationTimeStamp: datetime.now(timezone.utc),
            }
        }
        cleanup_called = {"v": False}
        monkeypatch.setattr(
            stuck_ops,
            "cleanup_partial_writes",
            lambda *a, **k: cleanup_called.__setitem__("v", True),
        )

        with pytest.raises(RuntimeError, match="not a stuck SV-split op"):
            stuck_ops.replay(cg, op_id)
        assert not cleanup_called["v"]


class TestCleanupPartialWrites:
    """Cleanup reverts partial OCDBT writes using pinned reads of pre-op state."""

    def _meta_with_fork(self, local_ocdbt_fixture, graph_id):
        """Build a real ChunkedGraphMeta pointing at the fixture's fork so
        `ws_ocdbt` reads/writes go through the same kvstack as production.

        Creates a matching source precomputed at the watershed root so
        `get_seg_source_and_destination_ocdbt` and `ws_cv` both work.
        Sets `layer_count` explicitly to bypass `ws_cv.bounds` inference.
        """
        ws = local_ocdbt_fixture["ws"]
        mm = {"type": "segmentation", "data_type": "uint64", "num_channels": 1}
        scale_metadata = {
            "size": [64, 64, 32],
            "resolution": [4, 4, 40],
            "encoding": "compressed_segmentation",
            "compressed_segmentation_block_size": [8, 8, 8],
            "chunk_size": [32, 32, 32],
        }
        ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": f"{ws}/",
                "multiscale_metadata": mm,
                "scale_metadata": scale_metadata,
            },
            create=True,
        ).result()

        local_ocdbt_fixture["make_fork"](graph_id)

        gc = GraphConfig(
            ID=graph_id,
            CHUNK_SIZE=np.array([32, 32, 32], dtype=int),
        )
        ds = DataSource(WATERSHED=f"{ws}/", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})
        meta.layer_count = 3  # avoids lazy cloudvolume layer inference
        return meta

    def _capture_fork_pin(self, local_ocdbt_fixture, graph_id):
        """Return an ISO-8601 `Z`-suffix pin string for the fork's current
        manifest commit — the pre-op timestamp for cleanup to pin on.
        """
        fork_manifest_kvs = ts.KvStore.open(
            f"{local_ocdbt_fixture['ws']}/ocdbt/{graph_id}/"
        ).result()
        manifest = ts.ocdbt.dump(fork_manifest_kvs).result()
        # commit_time is recorded as int ns since epoch; use a timestamp
        # just past the last commit as the pin so the upper-bound filter
        # picks up everything written so far.
        last_ns = manifest["versions"][-1]["commit_time"]
        return datetime.fromtimestamp(last_ns / 1e9 + 0.001, tz=timezone.utc)

    def test_cleanup_reverts_partial_writes_to_pre_op(self, local_ocdbt):
        """Write known pre-op state, snapshot time, write partial state to
        one chunk, simulate a stuck op with that chunk in scope, and
        confirm cleanup reverts the chunk while leaving a non-scoped
        neighbor chunk untouched.
        """
        fixture = local_ocdbt

        meta = self._meta_with_fork(fixture, "stuck_cg")
        fork_scale0 = fixture["make_fork"]("stuck_cg")

        # Pre-op state: chunk 0 region filled with 111, chunk 1 with 222.
        # Chunk grid is at base resolution with 32^3 voxels per chunk.
        fork_scale0[0:32, 0:32, 0:32, :] = np.full(
            (32, 32, 32, 1), 111, dtype=np.uint64
        )
        fork_scale0[32:64, 0:32, 0:32, :] = np.full(
            (32, 32, 32, 1), 222, dtype=np.uint64
        )

        # Snapshot pin timestamp just after the pre-op writes.
        pre_op_pin_dt = self._capture_fork_pin(fixture, "stuck_cg")

        # Partial "crash" writes: overwrite chunk 0 with garbage, touch
        # chunk 1 too to prove scope-boundedness (scope will only list
        # chunk 0, so chunk 1's garbage must persist after cleanup).
        fork_scale0[0:32, 0:32, 0:32, :] = np.full(
            (32, 32, 32, 1), 999, dtype=np.uint64
        )
        fork_scale0[32:64, 0:32, 0:32, :] = np.full(
            (32, 32, 32, 1), 888, dtype=np.uint64
        )

        # Chunk IDs for chunk-coord (0,0,0) and (1,0,0) at layer 2.
        chunk_id_0 = _chunk_id_from_coord(meta, layer=2, coord=(0, 0, 0))
        chunk_id_1 = _chunk_id_from_coord(meta, layer=2, coord=(1, 0, 0))

        # Sanity: scope chunk decodes back to the right coord.
        assert tuple(get_chunk_coordinates(meta, chunk_id_0)) == (0, 0, 0)

        # Synthetic op-log row with scope=[chunk_id_0] and OperationTimeStamp=pre_op_pin.
        op_id = 777
        op_log_row = {
            attributes.OperationLogs.L2ChunkLockScope: np.asarray(
                [chunk_id_0], dtype=np.uint64
            ),
            attributes.OperationLogs.OperationTimeStamp: pre_op_pin_dt,
        }

        cg = MagicMock()
        cg.meta = meta
        cg.client.read_log_entries.return_value = {np.uint64(op_id): op_log_row}

        # `_read_source_scales` reads `/info` from the watershed via
        # tensorstore's kvstore interface — fine on GCS, not on file://.
        # Bypass with a fake scale list matching the test's scale 0.
        fake_scales = [
            {
                "resolution": [4, 4, 40],
                "size": [64, 64, 32],
                "chunk_sizes": [[32, 32, 32]],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8],
            }
        ]
        # Patch the binding in `ocdbt.main` (where it's actually called).
        # The package re-export in `ocdbt/__init__.py` is a separate name
        # binding and patching it has no effect on main.py's local one.
        with patch.object(
            ocdbt_mod.main, "_read_source_scales", return_value=fake_scales
        ):
            reverted = stuck_ops.cleanup_partial_writes(cg, op_id)
        assert reverted == 1

        # Scoped chunk reverted to pre-op.
        scoped = fork_scale0[0:32, 0:32, 0:32, :].read().result()
        assert (
            scoped == 111
        ).all(), f"scoped chunk not reverted: unique={np.unique(scoped)}"
        # Non-scoped neighbor still has its post-crash "garbage" (888) —
        # cleanup does not touch it.
        neighbor = fork_scale0[32:64, 0:32, 0:32, :].read().result()
        assert (
            neighbor == 888
        ).all(), f"neighbor chunk erroneously reverted: unique={np.unique(neighbor)}"


def _chunk_id_from_coord(meta, layer, coord):
    """Encode (layer, x, y, z) into a chunk ID using the graph's bitmasks."""
    from pychunkedgraph.graph.chunks.utils import get_chunk_id

    return get_chunk_id(
        meta, layer=layer, x=int(coord[0]), y=int(coord[1]), z=int(coord[2])
    )
