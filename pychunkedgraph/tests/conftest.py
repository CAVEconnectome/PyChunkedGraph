import atexit
import os
import signal
import subprocess
from functools import partial
from datetime import timedelta

import pytest

# Skip the old monolithic test file if it still exists (e.g., during branch transitions)
collect_ignore = ["test_uncategorized.py"]
import numpy as np
from google.auth import credentials
from google.cloud import bigtable

from ..ingest.utils import bootstrap
from ..graph.edges import Edges
from ..graph.chunkedgraph import ChunkedGraph
from ..ingest.create.parent_layer import add_parent_chunk

from .helpers import (
    CloudVolumeMock,
    create_chunk,
    to_label,
    get_layer_chunk_bounds,
)

_emulator_proc = None
_emulator_cleaned = False


def _cleanup_emulator():
    global _emulator_cleaned
    if _emulator_cleaned or _emulator_proc is None:
        return
    _emulator_cleaned = True
    try:
        pgid = os.getpgid(_emulator_proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            _emulator_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            _emulator_proc.wait(timeout=5)
    except (ProcessLookupError, OSError, ChildProcessError):
        pass
    # Hard kill cbtemulator in case it survived the process group signal
    subprocess.run(["pkill", "-9", "cbtemulator"], stderr=subprocess.DEVNULL)


def setup_emulator_env():
    bt_env_init = subprocess.run(
        ["gcloud", "beta", "emulators", "bigtable", "env-init"], stdout=subprocess.PIPE
    )
    os.environ["BIGTABLE_EMULATOR_HOST"] = (
        bt_env_init.stdout.decode("utf-8").strip().split("=")[-1]
    )

    c = bigtable.Client(
        project="IGNORE_ENVIRONMENT_PROJECT",
        credentials=credentials.AnonymousCredentials(),
        admin=True,
    )
    t = c.instance("emulated_instance").table("emulated_table")

    try:
        t.create()
        return True
    except Exception as err:
        print("Bigtable Emulator not yet ready: %s" % err)
        return False


@pytest.fixture(scope="session", autouse=True)
def bigtable_emulator(request):
    global _emulator_proc, _emulator_cleaned
    from time import sleep

    _emulator_cleaned = False

    # Kill any leftover emulator processes from previous runs
    subprocess.run(["pkill", "-9", "cbtemulator"], stderr=subprocess.DEVNULL)

    # Start Emulator
    _emulator_proc = subprocess.Popen(
        [
            "gcloud",
            "beta",
            "emulators",
            "bigtable",
            "start",
            "--host-port=localhost:8539",
        ],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
    )

    # Register atexit handler as safety net for abnormal exits
    atexit.register(_cleanup_emulator)

    # Wait for Emulator to start up
    print("Waiting for BigTables Emulator to start up...", end="")
    retries = 5
    while retries > 0:
        if setup_emulator_env() is True:
            break
        else:
            retries -= 1
            sleep(5)

    if retries == 0:
        print(
            "\nCouldn't start Bigtable Emulator. Make sure it is installed correctly."
        )
        _cleanup_emulator()
        exit(1)

    request.addfinalizer(_cleanup_emulator)


@pytest.fixture(scope="function")
def gen_graph(request):
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
                "ROOT_LOCK_EXPIRY": timedelta(seconds=5),
            },
            "backend_client": {
                "TYPE": "bigtable",
                "CONFIG": {
                    "ADMIN": True,
                    "READ_ONLY": False,
                    "PROJECT": "IGNORE_ENVIRONMENT_PROJECT",
                    "INSTANCE": "emulated_instance",
                    "CREDENTIALS": credentials.AnonymousCredentials(),
                    "MAX_ROW_KEY_COUNT": 1000,
                },
            },
            "ingest_config": {},
        }

        meta, _, client_info = bootstrap("test", config=config)
        graph = ChunkedGraph(graph_id="test", meta=meta, client_info=client_info)
        graph.mock_edges = Edges([], [])
        graph.meta._ws_cv = CloudVolumeMock()
        graph.meta.layer_count = n_layers
        graph.meta.layer_chunk_bounds = get_layer_chunk_bounds(
            n_layers, atomic_chunk_bounds=atomic_chunk_bounds
        )

        graph.create()

        # setup Chunked Graph - Finalizer
        def fin():
            graph.client._table.delete()

        request.addfinalizer(fin)
        return graph

    return partial(_cgraph, request)


@pytest.fixture(scope="function")
def gen_graph_with_edges(request, tmp_path):
    """Like gen_graph but with real edge/component I/O via local filesystem (file:// protocol)."""

    def _cgraph(request, n_layers=10, atomic_chunk_bounds: np.ndarray = np.array([])):
        edges_dir = f"file://{tmp_path}/edges"
        components_dir = f"file://{tmp_path}/components"
        config = {
            "data_source": {
                "EDGES": edges_dir,
                "COMPONENTS": components_dir,
                "WATERSHED": "gs://microns-seunglab/minnie65/ws_minnie65_0",
            },
            "graph_config": {
                "CHUNK_SIZE": [512, 512, 64],
                "FANOUT": 2,
                "SPATIAL_BITS": 10,
                "ID_PREFIX": "",
                "ROOT_LOCK_EXPIRY": timedelta(seconds=5),
            },
            "backend_client": {
                "TYPE": "bigtable",
                "CONFIG": {
                    "ADMIN": True,
                    "READ_ONLY": False,
                    "PROJECT": "IGNORE_ENVIRONMENT_PROJECT",
                    "INSTANCE": "emulated_instance",
                    "CREDENTIALS": credentials.AnonymousCredentials(),
                    "MAX_ROW_KEY_COUNT": 1000,
                },
            },
            "ingest_config": {},
        }

        meta, _, client_info = bootstrap("test", config=config)
        graph = ChunkedGraph(graph_id="test", meta=meta, client_info=client_info)
        # No mock_edges - use real I/O via file:// protocol
        graph.meta._ws_cv = CloudVolumeMock()
        graph.meta.layer_count = n_layers
        graph.meta.layer_chunk_bounds = get_layer_chunk_bounds(
            n_layers, atomic_chunk_bounds=atomic_chunk_bounds
        )

        graph.create()

        def fin():
            graph.client._table.delete()

        request.addfinalizer(fin)
        return graph

    return partial(_cgraph, request)


@pytest.fixture(scope="function")
def gen_graph_simplequerytest(request, gen_graph):
    """
    ┌─────┬─────┬─────┐
    │  A¹ │  B¹ │  C¹ │
    │  1  │ 3━2━┿━━4  │
    │     │     │     │
    └─────┴─────┴─────┘
    """
    from math import inf

    graph = gen_graph(n_layers=4)

    # Chunk A
    create_chunk(graph, vertices=[to_label(graph, 1, 0, 0, 0, 0)], edges=[])

    # Chunk B
    create_chunk(
        graph,
        vertices=[to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 1)],
        edges=[
            (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 1), 0.5),
            (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
        ],
    )

    # Chunk C
    create_chunk(
        graph,
        vertices=[to_label(graph, 1, 2, 0, 0, 0)],
        edges=[(to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf)],
    )

    add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
    add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
    add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

    return graph


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
