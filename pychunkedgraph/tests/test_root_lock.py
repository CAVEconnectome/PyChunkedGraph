"""Integration tests for RootLock using real graph operations through the BigTable emulator.

Tests lock acquisition, release, and behavior on operation failure.
"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from .helpers import create_chunk, to_label
from ..graph import exceptions
from ..graph.locks import RootLock
from ..ingest.create.parent_layer import add_parent_chunk


class TestRootLock:
    @pytest.fixture()
    def simple_graph(self, gen_graph):
        """Build a 2-chunk graph with a single edge, return (cg, root_id)."""
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        return cg, root_id

    @pytest.mark.timeout(30)
    def test_successful_lock_and_release(self, simple_graph):
        """Lock acquired successfully inside context, released after exit."""
        cg, root_id = simple_graph

        with RootLock(cg, np.array([root_id])) as lock:
            assert lock.lock_acquired
            assert len(lock.locked_root_ids) > 0

        # After exiting the context, the lock should be released.
        # Verify by acquiring the same lock again — if it wasn't released, this would fail.
        with RootLock(cg, np.array([root_id])) as lock2:
            assert lock2.lock_acquired

    @pytest.mark.timeout(30)
    def test_lock_released_on_exception(self, simple_graph):
        """Lock should be released even when an exception occurs inside the context."""
        cg, root_id = simple_graph

        with pytest.raises(exceptions.PreconditionError):
            with RootLock(cg, np.array([root_id])) as lock:
                assert lock.lock_acquired
                raise exceptions.PreconditionError("Simulated failure")

        # Lock should still be released — acquiring again should succeed
        with RootLock(cg, np.array([root_id])) as lock2:
            assert lock2.lock_acquired

    @pytest.mark.timeout(30)
    def test_operation_with_lock_succeeds(self, simple_graph):
        """A real graph operation (split) should succeed while holding the lock."""
        cg, root_id = simple_graph

        # Use the high-level API which acquires locks internally
        result = cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )
        assert len(result.new_root_ids) == 2

        # After operation, locks should be released — verify we can re-acquire
        new_root = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        with RootLock(cg, np.array([new_root])) as lock:
            assert lock.lock_acquired
