"""Fork test graphs, and forks of them carrying the mocks their stored meta lacks."""

from datetime import timedelta

import numpy as np
import pytest

from ...fork.create import create_fork
from ...graph import BackendClientInfo
from ..helpers import SV, build_graph, fake_timestamp


def _pair(gen_graph, edges):
    """Two supervoxels in neighbouring chunks; returns (graph, their ids)."""
    cg, sv = build_graph(
        gen_graph,
        n_layers=3,
        supervoxels={"a": SV(), "b": SV(x=1)},
        edges=edges,
    )
    if not cg.client.opens_forks:
        pytest.skip("this backend's client opens no forks")
    return cg, np.array([sv["a"], sv["b"]], dtype=np.uint64)


@pytest.fixture()
def joined(gen_graph):
    """Two supervoxels one edge joins into one root."""
    return _pair(gen_graph, [("a", "b", 0.5)])


@pytest.fixture()
def apart(gen_graph):
    """Two supervoxels with no edge, so two roots."""
    return _pair(gen_graph, [])


@pytest.fixture()
def split_pair(joined):
    """The joined pair, split now; returns (graph, ids, an earlier time)."""
    cg, ids = joined
    # The graph is built at fake_timestamp(), ten days back; the split runs now.
    before = fake_timestamp() + timedelta(days=1)
    cg.remove_edges("test_user", source_ids=ids[0], sink_ids=ids[1], mincut=False)
    return cg, ids, before


@pytest.fixture()
def fork_of(request):
    """Forks a test graph at a time; each fork's table is deleted after the test."""

    def forked(source, name, at, **options):
        info = BackendClientInfo(TYPE="bigtable", CONFIG=source.client._config)
        graph = create_fork(source.graph_id, name, at, client_info=info, **options)
        request.addfinalizer(graph.client._admin_table.delete)
        graph.mock_edges = source.mock_edges
        for attr in ("_ws_cv", "_ws_info_d", "ws_ts_scale"):
            setattr(graph.meta, attr, getattr(source.meta, attr))
        graph.meta.layer_count = source.meta.layer_count
        graph.meta.layer_chunk_bounds = source.meta.layer_chunk_bounds
        return graph

    return forked
