"""A fork reads its source as of the fork point, and so does a fork of that fork."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from kvdbclient.safe_time import SETTLE

from ...fork.create import create_fork
from ...graph import BackendClientInfo
from ...graph.operation import MAX_EDIT_AGE
from ..helpers import SV, build_graph, fake_timestamp


def _forked(source, name, at, request, **options):
    """`create_fork` on a test graph, given the test mocks its stored meta lacks."""
    info = BackendClientInfo(TYPE="bigtable", CONFIG=source.client._config)
    graph = create_fork(source.graph_id, name, at, client_info=info, **options)
    request.addfinalizer(graph.client._admin_table.delete)
    for attr in ("_ws_cv", "_ws_info_d", "ws_ts_scale"):
        setattr(graph.meta, attr, getattr(source.meta, attr))
    graph.meta.layer_count = source.meta.layer_count
    graph.meta.layer_chunk_bounds = source.meta.layer_chunk_bounds
    return graph


@pytest.fixture()
def split_pair(gen_graph):
    """Two joined supervoxels split now; returns (graph, ids, an earlier time)."""
    cg, sv = build_graph(
        gen_graph,
        n_layers=3,
        supervoxels={"a": SV(), "b": SV(x=1)},
        edges=[("a", "b", 0.5)],
    )
    if not cg.client.opens_forks:
        pytest.skip("this backend's client opens no forks")
    # The graph is built at fake_timestamp(), ten days back; the split runs now.
    before = fake_timestamp() + timedelta(days=1)
    cg.remove_edges("test_user", source_ids=sv["a"], sink_ids=sv["b"], mincut=False)
    return cg, np.array([sv["a"], sv["b"]], dtype=np.uint64), before


def test_a_fork_reads_its_source_as_of_the_fork_point(split_pair, request):
    cg, ids, before = split_pair
    fork = _forked(cg, "test_fork", before, request)
    roots = fork.get_roots(ids)
    assert roots[0] == roots[1]
    assert np.array_equal(roots, cg.get_roots(ids, time_stamp=before))
    assert not np.array_equal(roots, cg.get_roots(ids))


def test_a_fork_of_a_fork_reads_the_source_at_the_first_fork_point(split_pair, request):
    cg, ids, before = split_pair
    first = _forked(cg, "test_fork", before, request)
    second = _forked(first, "test_fork2", datetime.now(UTC), request)
    assert np.array_equal(second.get_roots(ids), cg.get_roots(ids, time_stamp=before))


def test_a_past_request_forks_at_exactly_that_time(split_pair, request):
    cg, _, before = split_pair
    marker = _forked(cg, "test_fork", before, request).client.marker
    assert marker.tf == marker.requested_at


def test_a_recent_request_steps_back_past_the_settle_window(split_pair, request):
    cg, ids, _ = split_pair
    requested = datetime.now(UTC).replace(microsecond=0)
    fork = _forked(cg, "test_fork", requested, request)
    now = datetime.now(UTC)
    assert now - MAX_EDIT_AGE < fork.client.marker.tf <= now - SETTLE
    assert fork.client.marker.requested_at == requested
    # The split ran seconds ago, after the stepped-back fork point.
    roots = fork.get_roots(ids)
    assert roots[0] == roots[1]


def test_the_admins_wait_choice_reaches_the_safe_time_search(
    split_pair, request, monkeypatch
):
    cg, _, before = split_pair
    waits = []

    def searched(client, requested, longest_edit, wait=False):
        waits.append(wait)
        return requested

    monkeypatch.setattr("pychunkedgraph.fork.create.safe_time", searched)
    _forked(cg, "test_fork", before, request, wait=True)
    assert waits == [True]
