import threading
import time
from time import sleep
from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from ..helpers import (
    RowKeyLockRegistry,
    create_chunk,
    make_cg_with_row_key_lock_registry,
    to_label,
)
from ...graph import exceptions
from ...graph.locks import L2ChunkLock, _l2_chunk_lock_row_key
from ...graph.lineage import get_future_root_ids
from ...ingest.create.parent_layer import add_parent_chunk


class TestGraphLocks:
    @pytest.mark.timeout(30)
    def test_lock_unlock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try unlock (opid = 1)
        (4) Try lock (opid = 2)
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_root(root_id=root_id, operation_id=operation_id_1)

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    @pytest.mark.timeout(30)
    def test_lock_expiration(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """
        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds())

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
            max_tries=10,
            waittime_s=0.5,
        )[0]

    @pytest.mark.timeout(30)
    def test_lock_renew(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.renew_locks(root_ids=[root_id], operation_id=operation_id_1)

    @pytest.mark.timeout(30)
    def test_lock_merge_lock_old_id(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Merge (includes lock opid 1)
        (2) Try lock opid 2 --> should be successful and return new root id
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        new_root_ids = cg.add_edges(
            "Chuck Norris",
            [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            affinities=1.0,
        ).new_root_ids

        assert new_root_ids is not None

        operation_id_2 = cg.id_client.create_operation_id()
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        success, new_root_id = cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
            max_tries=10,
            waittime_s=0.5,
        )

        assert success
        assert new_root_ids[0] == new_root_id

    @pytest.mark.timeout(30)
    def test_indefinite_lock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try indefinite lock (opid = 1), get indefinite lock
        (2) Try normal lock (opid = 2), doesn't get the normal lock
        (3) Try unlock indefinite lock (opid = 1), should unlock indefinite lock
        (4) Try lock (opid = 2), should get the normal lock
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots_indefinitely(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_indefinitely_locked_root(
            root_id=root_id, operation_id=operation_id_1
        )

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    @pytest.mark.timeout(30)
    def test_indefinite_lock_with_normal_lock_expiration(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try normal lock (opid = 1), get normal lock
        (2) Try indefinite lock (opid = 1), get indefinite lock
        (3) Wait until normal lock expires
        (4) Try normal lock (opid = 2), doesn't get the normal lock
        (5) Try unlock indefinite lock (opid = 1), should unlock indefinite lock
        (6) Try lock (opid = 2), should get the normal lock
        """

        # 1. TODO renew lock test when getting indefinite lock
        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.lock_roots_indefinitely(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds())

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_indefinitely_locked_root(
            root_id=root_id, operation_id=operation_id_1
        )

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]


# =====================================================================
# Pure unit tests (no BigTable emulator needed)
# =====================================================================
from unittest.mock import MagicMock, patch
from collections import defaultdict
import networkx as nx

from ...graph.locks import RootLock, IndefiniteRootLock
from ...graph.exceptions import LockingError


def _make_mock_cg():
    """Create a mock ChunkedGraph object with the methods needed by locks."""
    cg = MagicMock()
    cg.id_client.create_operation_id.return_value = np.uint64(42)
    cg.client.lock_roots.return_value = (True, [np.uint64(100)])
    cg.client.unlock_root.return_value = None
    cg.client.renew_locks.return_value = True
    cg.client.lock_roots_indefinitely.return_value = (
        True,
        [np.uint64(100)],
        [],
    )
    cg.client.unlock_indefinitely_locked_root.return_value = None
    cg.get_node_timestamps.return_value = [MagicMock()]
    return cg


class TestRootLockPrivilegedMode:
    def test_rootlock_privileged_mode(self):
        """privileged_mode=True should skip locking entirely and return self."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        op_id = np.uint64(999)

        lock = RootLock(cg, root_ids, operation_id=op_id, privileged_mode=True)
        result = lock.__enter__()

        assert result is lock
        assert lock.lock_acquired is False
        cg.client.lock_roots.assert_not_called()

    def test_rootlock_privileged_mode_exit_no_unlock(self):
        """When privileged and lock was never acquired, __exit__ should not unlock."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        op_id = np.uint64(999)

        lock = RootLock(cg, root_ids, operation_id=op_id, privileged_mode=True)
        lock.__enter__()
        lock.__exit__(None, None, None)

        cg.client.unlock_root.assert_not_called()


class TestRootLockCreatesOperationId:
    def test_rootlock_creates_operation_id(self):
        """When operation_id is None, __enter__ should create one via cg.id_client."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])

        mock_graph = nx.DiGraph()
        mock_graph.add_node(np.uint64(100))

        with patch("pychunkedgraph.graph.locks.lineage_graph", return_value=mock_graph):
            lock = RootLock(cg, root_ids, operation_id=None)
            lock.__enter__()

        cg.id_client.create_operation_id.assert_called_once()
        assert lock.operation_id == np.uint64(42)


class TestRootLockAcquired:
    def test_rootlock_lock_acquired(self):
        """When lock_roots returns (True, [...]), lock_acquired should be True."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        locked = [np.uint64(100), np.uint64(101)]
        cg.client.lock_roots.return_value = (True, locked)

        mock_graph = nx.DiGraph()
        mock_graph.add_node(np.uint64(100))

        with patch("pychunkedgraph.graph.locks.lineage_graph", return_value=mock_graph):
            lock = RootLock(cg, root_ids, operation_id=np.uint64(10))
            result = lock.__enter__()

        assert lock.lock_acquired is True
        assert lock.locked_root_ids == locked
        assert result is lock


class TestRootLockFailed:
    def test_rootlock_lock_failed(self):
        """When lock_roots returns (False, []), should raise LockingError."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        cg.client.lock_roots.return_value = (False, [])

        mock_graph = nx.DiGraph()
        mock_graph.add_node(np.uint64(100))

        with patch("pychunkedgraph.graph.locks.lineage_graph", return_value=mock_graph):
            lock = RootLock(cg, root_ids, operation_id=np.uint64(10))
            with pytest.raises(LockingError, match="Could not acquire root lock"):
                lock.__enter__()


class TestRootLockExitUnlocks:
    def test_rootlock_exit_unlocks(self):
        """When lock_acquired=True, __exit__ should call unlock_root for each locked_root_id."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        locked = [np.uint64(100), np.uint64(101)]
        cg.client.lock_roots.return_value = (True, locked)

        mock_graph = nx.DiGraph()
        mock_graph.add_node(np.uint64(100))

        with patch("pychunkedgraph.graph.locks.lineage_graph", return_value=mock_graph):
            lock = RootLock(cg, root_ids, operation_id=np.uint64(10))
            lock.__enter__()

        lock.__exit__(None, None, None)

        assert cg.client.unlock_root.call_count == 2
        actual_calls = cg.client.unlock_root.call_args_list
        called_root_ids = {c[0][0] for c in actual_calls}
        assert called_root_ids == {np.uint64(100), np.uint64(101)}
        for c in actual_calls:
            assert c[0][1] == np.uint64(10)

    def test_rootlock_exit_no_unlock_when_not_acquired(self):
        """When lock_acquired=False, __exit__ should not call unlock_root."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])

        lock = RootLock(cg, root_ids, operation_id=np.uint64(10))
        lock.__exit__(None, None, None)

        cg.client.unlock_root.assert_not_called()

    def test_rootlock_exit_handles_unlock_exception(self):
        """When unlock_root raises, __exit__ should log warning and not re-raise."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        locked = [np.uint64(100)]
        cg.client.lock_roots.return_value = (True, locked)
        cg.client.unlock_root.side_effect = RuntimeError("unlock failed")

        mock_graph = nx.DiGraph()
        mock_graph.add_node(np.uint64(100))

        with patch("pychunkedgraph.graph.locks.lineage_graph", return_value=mock_graph):
            lock = RootLock(cg, root_ids, operation_id=np.uint64(10))
            lock.__enter__()

        # Should not raise even though unlock_root raises
        lock.__exit__(None, None, None)


class TestIndefiniteRootLockPrivilegedMode:
    def test_indefiniterootlock_privileged_mode(self):
        """privileged_mode=True should skip locking and return self."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        op_id = np.uint64(999)

        lock = IndefiniteRootLock(cg, op_id, root_ids, privileged_mode=True)
        result = lock.__enter__()

        assert result is lock
        assert lock.acquired is False
        cg.client.renew_locks.assert_not_called()
        cg.client.lock_roots_indefinitely.assert_not_called()


class TestIndefiniteRootLockRenewFails:
    def test_indefiniterootlock_renew_fails(self):
        """When renew_locks returns False, should raise LockingError."""
        cg = _make_mock_cg()
        cg.client.renew_locks.return_value = False
        root_ids = np.array([np.uint64(100)])
        op_id = np.uint64(10)

        lock = IndefiniteRootLock(
            cg, op_id, root_ids, future_root_ids_d=defaultdict(list)
        )
        with pytest.raises(LockingError, match="Could not renew locks"):
            lock.__enter__()


class TestIndefiniteRootLockSuccess:
    def test_indefiniterootlock_lock_success(self):
        """When lock_roots_indefinitely returns (True, [...], []), acquired should be True."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        locked = [np.uint64(100)]
        cg.client.lock_roots_indefinitely.return_value = (True, locked, [])

        lock = IndefiniteRootLock(
            cg,
            np.uint64(10),
            root_ids,
            future_root_ids_d=defaultdict(list),
        )
        result = lock.__enter__()

        assert lock.acquired is True
        assert result is lock
        assert list(lock.root_ids) == locked


class TestIndefiniteRootLockFail:
    def test_indefiniterootlock_lock_fail(self):
        """When lock_roots_indefinitely returns (False, [], [...]), should raise LockingError."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        failed = [np.uint64(100)]
        cg.client.lock_roots_indefinitely.return_value = (False, [], failed)

        lock = IndefiniteRootLock(
            cg,
            np.uint64(10),
            root_ids,
            future_root_ids_d=defaultdict(list),
        )
        with pytest.raises(LockingError, match="have been locked indefinitely"):
            lock.__enter__()


class TestIndefiniteRootLockExitUnlocks:
    def test_indefiniterootlock_exit_unlocks(self):
        """When acquired=True, __exit__ should call unlock_indefinitely_locked_root."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100), np.uint64(101)])
        cg.client.lock_roots_indefinitely.return_value = (
            True,
            [np.uint64(100), np.uint64(101)],
            [],
        )

        lock = IndefiniteRootLock(
            cg,
            np.uint64(10),
            root_ids,
            future_root_ids_d=defaultdict(list),
        )
        lock.__enter__()
        lock.__exit__(None, None, None)

        assert cg.client.unlock_indefinitely_locked_root.call_count == 2
        actual_calls = cg.client.unlock_indefinitely_locked_root.call_args_list
        called_root_ids = {c[0][0] for c in actual_calls}
        assert called_root_ids == {np.uint64(100), np.uint64(101)}
        for c in actual_calls:
            assert c[0][1] == np.uint64(10)

    def test_indefiniterootlock_exit_no_unlock_when_not_acquired(self):
        """When acquired=False, __exit__ should not unlock."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        lock = IndefiniteRootLock(cg, np.uint64(10), root_ids)
        lock.__exit__(None, None, None)
        cg.client.unlock_indefinitely_locked_root.assert_not_called()

    def test_indefiniterootlock_exit_handles_exception(self):
        """When unlock_indefinitely_locked_root raises, should not re-raise."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        cg.client.lock_roots_indefinitely.return_value = (
            True,
            [np.uint64(100)],
            [],
        )
        cg.client.unlock_indefinitely_locked_root.side_effect = RuntimeError("fail")

        lock = IndefiniteRootLock(
            cg,
            np.uint64(10),
            root_ids,
            future_root_ids_d=defaultdict(list),
        )
        lock.__enter__()
        # Should not raise
        lock.__exit__(None, None, None)


class TestIndefiniteRootLockComputesFutureRootIds:
    def test_indefiniterootlock_computes_future_root_ids(self):
        """When future_root_ids_d is None, should compute from lineage_graph."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        cg.client.lock_roots_indefinitely.return_value = (
            True,
            [np.uint64(100)],
            [],
        )

        mock_lgraph = nx.DiGraph()
        mock_lgraph.add_edge(np.uint64(100), np.uint64(200))
        mock_lgraph.add_edge(np.uint64(100), np.uint64(300))

        with patch(
            "pychunkedgraph.graph.locks.lineage_graph", return_value=mock_lgraph
        ):
            lock = IndefiniteRootLock(
                cg,
                np.uint64(10),
                root_ids,
                future_root_ids_d=None,
            )
            lock.__enter__()

        assert lock.future_root_ids_d is not None
        descendants = lock.future_root_ids_d[np.uint64(100)]
        assert set(descendants) == {np.uint64(200), np.uint64(300)}


class TestRootLockContextManager:
    def test_rootlock_as_context_manager(self):
        """Test using RootLock with the `with` statement."""
        cg = _make_mock_cg()
        root_ids = np.array([np.uint64(100)])
        locked = [np.uint64(100)]
        cg.client.lock_roots.return_value = (True, locked)

        mock_graph = nx.DiGraph()
        mock_graph.add_node(np.uint64(100))

        with patch("pychunkedgraph.graph.locks.lineage_graph", return_value=mock_graph):
            with RootLock(cg, root_ids, operation_id=np.uint64(10)) as lock:
                assert lock.lock_acquired is True

        cg.client.unlock_root.assert_called_once()


class TestL2ChunkLockRowKey:
    def test_length(self):
        assert len(_l2_chunk_lock_row_key(0)) == 10

    def test_deterministic(self):
        assert _l2_chunk_lock_row_key(0xDEADBEEF) == _l2_chunk_lock_row_key(0xDEADBEEF)

    def test_distinct_chunks_distinct_keys(self):
        assert _l2_chunk_lock_row_key(42) != _l2_chunk_lock_row_key(43)

    def test_hash_prefix_scatters(self):
        """Adjacent chunk IDs should not cluster in one first-byte prefix —
        that's the whole point of the hash prefix."""
        prefixes = {_l2_chunk_lock_row_key(i)[0] for i in range(256)}
        # blake2b over 8 bytes of changing input distributes uniformly.
        assert len(prefixes) > 128


class TestL2ChunkLock:
    def test_acquire_and_release(self):
        registry = RowKeyLockRegistry()
        cg = make_cg_with_row_key_lock_registry(registry)
        with L2ChunkLock(cg, [np.uint64(1), np.uint64(2)], np.uint64(42)):
            assert len(registry._held) == 2
        assert registry._held == {}

    def test_non_overlapping_concurrent(self):
        """Disjoint chunk sets can coexist — no shared row keys."""
        registry = RowKeyLockRegistry()
        cg = make_cg_with_row_key_lock_registry(registry)
        l1 = L2ChunkLock(cg, [np.uint64(1)], np.uint64(1))
        l2 = L2ChunkLock(cg, [np.uint64(5)], np.uint64(2))
        l1.__enter__()
        l2.__enter__()
        assert len(registry._held) == 2
        l1.__exit__(None, None, None)
        l2.__exit__(None, None, None)
        assert registry._held == {}

    def test_overlapping_contends(self, monkeypatch):
        """Two overlapping acquisitions serialize: second blocks until first releases."""
        monkeypatch.setattr(L2ChunkLock, "_ACQUIRE_BACKOFF_BASE_SEC", 0.05)

        registry = RowKeyLockRegistry()
        cg = make_cg_with_row_key_lock_registry(registry)

        l1 = L2ChunkLock(cg, [np.uint64(7)], np.uint64(1))
        l1.__enter__()

        second_entered = threading.Event()
        second_failed = threading.Event()

        def second():
            lock = L2ChunkLock(cg, [np.uint64(7)], np.uint64(2))
            try:
                lock.__enter__()
                second_entered.set()
                lock.__exit__(None, None, None)
            except exceptions.LockingError:
                second_failed.set()

        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.2)
        assert not second_entered.is_set()
        l1.__exit__(None, None, None)
        t.join(timeout=2.0)
        assert second_entered.is_set()
        assert not second_failed.is_set()
        assert registry._held == {}

    def test_partial_acquire_released_on_failure(self, monkeypatch):
        """If any chunk in the set fails to lock, prior ones are released."""
        monkeypatch.setattr(L2ChunkLock, "_MAX_ACQUIRE_ATTEMPTS", 2)
        monkeypatch.setattr(L2ChunkLock, "_ACQUIRE_BACKOFF_BASE_SEC", 0.01)

        registry = RowKeyLockRegistry()
        registry.lock_by_row_key(_l2_chunk_lock_row_key(np.uint64(2)), np.uint64(99))

        cg = make_cg_with_row_key_lock_registry(registry)
        lock = L2ChunkLock(cg, [np.uint64(1), np.uint64(2)], np.uint64(1))
        with pytest.raises(exceptions.LockingError):
            lock.__enter__()
        # Only chunk 2 remains held, by the pre-existing holder.
        assert len(registry._held) == 1
        assert next(iter(registry._held)) == _l2_chunk_lock_row_key(np.uint64(2))
