import os
import subprocess
from datetime import timedelta
from functools import partial
from math import inf
from signal import SIGTERM
from time import sleep

import numpy as np
import pytest
from google.auth import credentials
from google.cloud import bigtable

from pychunkedgraph.backend import chunkedgraph


class CloudVolumeBounds(object):
    def __init__(self, bounds=[[0, 0, 0], [0, 0, 0]]):
        self._bounds = np.array(bounds)

    @property
    def bounds(self):
        return self._bounds

    def __repr__(self):
        return self.bounds

    def to_list(self):
        return list(np.array(self.bounds).flatten())


class CloudVolumeMock(object):
    def __init__(self):
        self.resolution = np.array([1, 1, 1], dtype=int)
        self.bounds = CloudVolumeBounds()


def setup_emulator_env():
    bt_env_init = subprocess.run(
        ["gcloud", "beta", "emulators", "bigtable", "env-init"], stdout=subprocess.PIPE
    )
    os.environ["BIGTABLE_EMULATOR_HOST"] = bt_env_init.stdout.decode("utf-8").strip().split("=")[-1]

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
    # Start Emulator
    bigtable_emulator = subprocess.Popen(
        ["gcloud", "beta", "emulators", "bigtable", "start"],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
    )

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
        print("\nCouldn't start Bigtable Emulator. Make sure it is installed correctly.")
        exit(1)

    # Setup Emulator-Finalizer
    def fin():
        os.killpg(os.getpgid(bigtable_emulator.pid), SIGTERM)
        bigtable_emulator.wait()

    request.addfinalizer(fin)


@pytest.fixture(scope="function")
def lock_expired_timedelta_override(request):
    # HACK: For the duration of the test, set global LOCK_EXPIRED_TIME_DELTA
    # to 1 second (otherwise test would have to run for several minutes)

    original_timedelta = chunkedgraph.LOCK_EXPIRED_TIME_DELTA

    chunkedgraph.LOCK_EXPIRED_TIME_DELTA = timedelta(seconds=1)

    # Ensure that we restore the original value, even if the test fails.
    def fin():
        chunkedgraph.LOCK_EXPIRED_TIME_DELTA = original_timedelta

    request.addfinalizer(fin)
    return chunkedgraph.LOCK_EXPIRED_TIME_DELTA


@pytest.fixture(scope="function")
def gen_graph(request):
    def _cgraph(request, fan_out=2, n_layers=10):
        # setup Chunked Graph
        dataset_info = {"data_dir": ""}

        graph = chunkedgraph.ChunkedGraph(
            request.function.__name__,
            project_id="IGNORE_ENVIRONMENT_PROJECT",
            credentials=credentials.AnonymousCredentials(),
            instance_id="emulated_instance",
            dataset_info=dataset_info,
            chunk_size=np.array([512, 512, 64], dtype=np.uint64),
            is_new=True,
            fan_out=np.uint64(fan_out),
            n_layers=np.uint64(n_layers),
        )

        graph._cv = CloudVolumeMock()

        # setup Chunked Graph - Finalizer
        def fin():
            graph.table.delete()

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

    graph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), n_threads=1)
    graph.add_layer(3, np.array([[2, 0, 0]]), n_threads=1)
    graph.add_layer(4, np.array([[0, 0, 0], [1, 0, 0]]), n_threads=1)

    return graph


def create_chunk(cgraph, vertices=None, edges=None, timestamp=None):
    """
    Helper function to add vertices and edges to the chunkedgraph - no safety checks!
    """
    if not vertices:
        vertices = []

    if not edges:
        edges = []

    vertices = np.unique(np.array(vertices, dtype=np.uint64))
    edges = [(np.uint64(v1), np.uint64(v2), np.float32(aff)) for v1, v2, aff in edges]

    isolated_node_ids = [
        x
        for x in vertices
        if (x not in [edges[i][0] for i in range(len(edges))])
        and (x not in [edges[i][1] for i in range(len(edges))])
    ]

    edge_ids = {
        "in_connected": np.array([], dtype=np.uint64).reshape(0, 2),
        "in_disconnected": np.array([], dtype=np.uint64).reshape(0, 2),
        "cross": np.array([], dtype=np.uint64).reshape(0, 2),
        "between_connected": np.array([], dtype=np.uint64).reshape(0, 2),
        "between_disconnected": np.array([], dtype=np.uint64).reshape(0, 2),
    }
    edge_affs = {
        "in_connected": np.array([], dtype=np.float32),
        "in_disconnected": np.array([], dtype=np.float32),
        "between_connected": np.array([], dtype=np.float32),
        "between_disconnected": np.array([], dtype=np.float32),
    }

    for e in edges:
        if cgraph.test_if_nodes_are_in_same_chunk(e[0:2]):
            this_edge = np.array([e[0], e[1]], dtype=np.uint64).reshape(-1, 2)
            edge_ids["in_connected"] = np.concatenate([edge_ids["in_connected"], this_edge])
            edge_affs["in_connected"] = np.concatenate([edge_affs["in_connected"], [e[2]]])

    if len(edge_ids["in_connected"]) > 0:
        chunk_id = cgraph.get_chunk_id(edge_ids["in_connected"][0][0])
    elif len(vertices) > 0:
        chunk_id = cgraph.get_chunk_id(vertices[0])
    else:
        chunk_id = None

    for e in edges:
        if not cgraph.test_if_nodes_are_in_same_chunk(e[0:2]):
            # Ensure proper order
            if chunk_id is not None:
                if cgraph.get_chunk_id(e[0]) != chunk_id:
                    e = [e[1], e[0], e[2]]
            this_edge = np.array([e[0], e[1]], dtype=np.uint64).reshape(-1, 2)

            if np.isinf(e[2]):
                edge_ids["cross"] = np.concatenate([edge_ids["cross"], this_edge])
            else:
                edge_ids["between_connected"] = np.concatenate(
                    [edge_ids["between_connected"], this_edge]
                )
                edge_affs["between_connected"] = np.concatenate(
                    [edge_affs["between_connected"], [e[2]]]
                )

    isolated_node_ids = np.array(isolated_node_ids, dtype=np.uint64)

    cgraph.logger.debug(edge_ids)
    cgraph.logger.debug(edge_affs)

    # Use affinities as areas
    cgraph.add_atomic_edges_in_chunks(
        edge_ids, edge_affs, edge_affs, isolated_node_ids, time_stamp=timestamp
    )


def to_label(cgraph, l, x, y, z, segment_id):
    return cgraph.get_node_id(np.uint64(segment_id), layer=l, x=x, y=y, z=z)
