"""A fork reads its source as of the fork point, and so does a fork of that fork."""

from datetime import UTC, datetime

import numpy as np
from kvdbclient.safe_time import SETTLE

from ...graph.operation import MAX_EDIT_AGE


def test_a_fork_reads_its_source_as_of_the_fork_point(split_pair, fork_of):
    cg, ids, before = split_pair
    fork = fork_of(cg, "test_fork", before)
    roots = fork.get_roots(ids)
    assert roots[0] == roots[1]
    assert np.array_equal(roots, cg.get_roots(ids, time_stamp=before))
    assert not np.array_equal(roots, cg.get_roots(ids))


def test_a_fork_of_a_fork_reads_the_source_at_the_first_fork_point(split_pair, fork_of):
    cg, ids, before = split_pair
    first = fork_of(cg, "test_fork", before)
    second = fork_of(first, "test_fork2", datetime.now(UTC))
    assert np.array_equal(second.get_roots(ids), cg.get_roots(ids, time_stamp=before))


def test_a_past_request_forks_at_exactly_that_time(split_pair, fork_of):
    cg, _, before = split_pair
    marker = fork_of(cg, "test_fork", before).client.marker
    assert marker.tf == marker.requested_at


def test_a_recent_request_steps_back_past_the_settle_window(split_pair, fork_of):
    cg, ids, _ = split_pair
    requested = datetime.now(UTC).replace(microsecond=0)
    fork = fork_of(cg, "test_fork", requested)
    now = datetime.now(UTC)
    assert now - MAX_EDIT_AGE < fork.client.marker.tf <= now - SETTLE
    assert fork.client.marker.requested_at == requested
    # The split ran seconds ago, after the stepped-back fork point.
    roots = fork.get_roots(ids)
    assert roots[0] == roots[1]


def test_the_admins_wait_choice_reaches_the_safe_time_search(
    split_pair, fork_of, monkeypatch
):
    cg, _, before = split_pair
    waits = []

    def searched(client, requested, longest_edit, wait=False):
        waits.append(wait)
        return requested

    monkeypatch.setattr("pychunkedgraph.fork.create.safe_time", searched)
    fork_of(cg, "test_fork", before, wait=True)
    assert waits == [True]
