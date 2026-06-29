from contextlib import ExitStack
from datetime import timedelta
from functools import partial

import numpy as np
import pytest

from kvdbclient_testing import backends as _backend_harnesses

from ..graph.chunkedgraph import ChunkedGraph
from ..graph.edges import Edges
from ..ingest.utils import bootstrap
from .helpers import (
    CloudVolumeMock,
    TensorStoreMock,
    get_layer_chunk_bounds,
    mock_ws_info,
)

# Skip the old monolithic test file if it still exists (e.g., during branch transitions)
collect_ignore = ["test_uncategorized.py"]

# Backends are discovered from KVDbClient; a new backend there is tested with no change here.
_HARNESSES = {h.name: h for h in _backend_harnesses()}


@pytest.fixture(scope="session", autouse=True)
def _backend_servers():
    """Start every available backend's local instance once for the session."""
    with ExitStack() as stack:
        handles = {}
        for name, harness in _HARNESSES.items():
            if harness.available():
                handles[name] = stack.enter_context(harness.server())
        yield handles


@pytest.fixture(scope="function", params=list(_HARNESSES))
def gen_graph(request, _backend_servers):
    name = request.param
    if name not in _backend_servers:
        pytest.skip(f"backend {name!r} unavailable in this environment")
    harness = _HARNESSES[name]
    handle = _backend_servers[name]

    def _cgraph(request, n_layers=10, atomic_chunk_bounds: np.ndarray = np.array([])):
        config = {
            "data_source": {
                "EDGES": "gs://chunked-graph/minnie65_0/edges",
                "COMPONENTS": "gs://chunked-graph/minnie65_0/components",
                "WATERSHED": "gs://microns-seunglab/minnie65/ws_minnie65_0",
            },
            "graph_config": {
                "CHUNK_SIZE": [512, 512, 64],
                "FANOUT": 2,
                "SPATIAL_BITS": 10,
                "ID_PREFIX": "",
                "ROOT_LOCK_EXPIRY": timedelta(seconds=1),
            },
            "backend_client": harness.backend_client(handle),
            "ingest_config": {},
        }

        meta, _, client_info, _ = bootstrap("test", config=config)
        graph = ChunkedGraph(graph_id="test", meta=meta, client_info=client_info)
        graph.mock_edges = Edges([], [])
        graph.meta._ws_cv = CloudVolumeMock()
        graph.meta._ws_info_d = mock_ws_info()
        graph.meta.ws_ts_scale = lambda mip=0: TensorStoreMock()
        graph.meta.layer_count = n_layers
        graph.meta.layer_chunk_bounds = get_layer_chunk_bounds(
            n_layers, atomic_chunk_bounds=atomic_chunk_bounds
        )

        graph.create()

        def fin():
            harness.delete_table(graph)

        request.addfinalizer(fin)
        return graph

    return partial(_cgraph, request)


@pytest.fixture(scope="session")
def sv_data():
    test_data_dir = "pychunkedgraph/tests/data"
    edges_file = f"{test_data_dir}/sv_edges.npy"
    sv_edges = np.load(edges_file)

    source_file = f"{test_data_dir}/sv_sources.npy"
    sv_sources = np.load(source_file)

    sinks_file = f"{test_data_dir}/sv_sinks.npy"
    sv_sinks = np.load(sinks_file)

    affinity_file = f"{test_data_dir}/sv_affinity.npy"
    sv_affinity = np.load(affinity_file)

    area_file = f"{test_data_dir}/sv_area.npy"
    sv_area = np.load(area_file)
    yield (sv_edges, sv_sources, sv_sinks, sv_affinity, sv_area)
