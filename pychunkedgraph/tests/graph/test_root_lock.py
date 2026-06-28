"""Integration tests for RootLock using real graph operations through the BigTable emulator.

Tests lock acquisition, release, and behavior on operation failure.
"""


import numpy as np
import pytest

from ..helpers import SV, build_graph
from ...graph import exceptions
from ...graph.locks import RootLock


class TestRootLock:
    @pytest.fixture()
    def simple_graph(self, gen_graph):
        """Build a 2-chunk graph with a single edge, return (cg, root_id)."""
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", 0.5)],
        )
        root_id = cg.get_root(sv["a0"])
        return cg, sv, root_id

    @pytest.mark.timeout(30)
    def test_successful_lock_and_release(self, simple_graph):
        """Lock acquired successfully inside context, released after exit."""
        cg, sv, root_id = simple_graph

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
        cg, sv, root_id = simple_graph

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
        cg, sv, root_id = simple_graph

        # Use the high-level API which acquires locks internally
        result = cg.remove_edges(
            "test_user",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )
        assert len(result.new_root_ids) == 2

        # After operation, locks should be released — verify we can re-acquire
        new_root = cg.get_root(sv["a0"])
        with RootLock(cg, np.array([new_root])) as lock:
            assert lock.lock_acquired
