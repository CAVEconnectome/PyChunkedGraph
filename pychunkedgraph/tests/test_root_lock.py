from unittest.mock import DEFAULT

import numpy as np
import pytest

import pychunkedgraph.backend.chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.backend.root_lock import RootLock

G_UINT64 = np.uint64(2 ** 63)


def big_uint64():
    """Return incremental uint64 values larger than a signed int64"""
    global G_UINT64
    if G_UINT64 == np.uint64(2 ** 64 - 1):
        G_UINT64 = np.uint64(2 ** 63)
    G_UINT64 = G_UINT64 + np.uint64(1)
    return G_UINT64


class RootLockTracker:
    def __init__(self):
        self.active_locks = dict()

    def add_locks(self, root_ids, operation_id, **kwargs):
        if operation_id not in self.active_locks:
            self.active_locks[operation_id] = set(root_ids)
        else:
            self.active_locks[operation_id].update(root_ids)
        return DEFAULT

    def remove_lock(self, root_id, operation_id, **kwargs):
        if operation_id in self.active_locks:
            self.active_locks[operation_id].discard(root_id)
        return DEFAULT


@pytest.fixture()
def root_lock_tracker():
    return RootLockTracker()


def test_successful_lock_acquisition(mocker, root_lock_tracker):
    """Ensure that root locks got released after successful
        root lock acquisition + *successful* graph operation"""
    fake_operation_id = big_uint64()
    fake_locked_root_ids = np.array((big_uint64(), big_uint64()))

    cg = mocker.MagicMock()
    cg.get_unique_operation_id = mocker.MagicMock(return_value=fake_operation_id)
    cg.lock_root_loop = mocker.MagicMock(
        return_value=(True, fake_locked_root_ids), side_effect=root_lock_tracker.add_locks
    )
    cg.unlock_root = mocker.MagicMock(return_value=True, side_effect=root_lock_tracker.remove_lock)

    with RootLock(cg, fake_locked_root_ids):
        assert fake_operation_id in root_lock_tracker.active_locks
        assert not root_lock_tracker.active_locks[fake_operation_id].difference(
            fake_locked_root_ids
        )

    assert not root_lock_tracker.active_locks[fake_operation_id]


def test_failed_lock_acquisition(mocker):
    """Ensure that LockingError is raised when lock acquisition failed"""
    fake_operation_id = big_uint64()
    fake_locked_root_ids = np.array((big_uint64(), big_uint64()))

    cg = mocker.MagicMock()
    cg.get_unique_operation_id = mocker.MagicMock(return_value=fake_operation_id)
    cg.lock_root_loop = mocker.MagicMock(
        return_value=(False, fake_locked_root_ids), side_effect=None
    )

    with pytest.raises(cg_exceptions.LockingError):
        with RootLock(cg, fake_locked_root_ids):
            pass


def test_failed_graph_operation(mocker, root_lock_tracker):
    """Ensure that root locks got released after successful
        root lock acquisition + *unsuccessful* graph operation"""
    fake_operation_id = big_uint64()
    fake_locked_root_ids = np.array((big_uint64(), big_uint64()))

    cg = mocker.MagicMock()
    cg.get_unique_operation_id = mocker.MagicMock(return_value=fake_operation_id)
    cg.lock_root_loop = mocker.MagicMock(
        return_value=(True, fake_locked_root_ids), side_effect=root_lock_tracker.add_locks
    )
    cg.unlock_root = mocker.MagicMock(return_value=True, side_effect=root_lock_tracker.remove_lock)

    with pytest.raises(cg_exceptions.PreconditionError):
        with RootLock(cg, fake_locked_root_ids):
            raise cg_exceptions.PreconditionError("Something went wrong")

    assert not root_lock_tracker.active_locks[fake_operation_id]
