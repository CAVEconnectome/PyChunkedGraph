import os
import json
import subprocess
from math import inf
from time import sleep
from signal import SIGTERM
from functools import reduce
from functools import partial
from datetime import timedelta

import pytest
import numpy as np
from google.auth import credentials
from google.cloud import bigtable

import botocore
import boto3

from ..ingest.utils import bootstrap
from ..ingest.create.atomic_layer import add_atomic_edges
from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.utils import basetypes
from ..graph.chunkedgraph import ChunkedGraph
from ..ingest.create.abstract_layers import add_layer

from ..graph.client import (
    DEFAULT_BACKEND_TYPE,
    GCP_BIGTABLE_BACKEND_TYPE,
    AMAZON_DYNAMODB_BACKEND_TYPE,
)

AMAZON_LOCAL_DYNAMODB_URL = "http://localhost:8000/"


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


def setup_bigtable_emulator_env():
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


def delete_amazon_dynamodb_tables(client):
    ret = client.list_tables(Limit=100)
    tables = ret.get("TableNames")
    for table in tables:
        client.delete_table(TableName=table)
    for table in tables:
        waiter = client.get_waiter('table_not_exists')
        waiter.wait(TableName=table, WaiterConfig={'Delay': 1, 'MaxAttempts': 500})
        print(f"Deleted {table}")


def setup_amazon_dynamodb_env():
    # check if local instance is running
    boto3_conf_ = botocore.config.Config(
        retries={"max_attempts": 10, "mode": "standard"}
    )
    client = boto3.client("dynamodb", config=boto3_conf_, endpoint_url=AMAZON_LOCAL_DYNAMODB_URL)
    try:
        delete_amazon_dynamodb_tables(client)
    except Exception as e:
        print(f"Failed to list tables: {repr(e)}")
        return False
    return True


@pytest.fixture(scope="session", autouse=True)
def bigtable_emulator(request):
    # Start Emulator
    bigtable_emulator = subprocess.Popen(
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
    
    # Wait for Emulator to start up
    print("Waiting for BigTables Emulator to start up...", end="")
    retries = 5
    while retries > 0:
        if setup_bigtable_emulator_env() is True:
            break
        else:
            retries -= 1
            sleep(5)
    
    if retries == 0:
        print(
            "\nCouldn't start Bigtable Emulator. Make sure it is installed correctly."
        )
        exit(1)
    
    # Setup Emulator-Finalizer
    def fin():
        os.killpg(os.getpgid(bigtable_emulator.pid), SIGTERM)
        bigtable_emulator.wait()
    
    request.addfinalizer(fin)


@pytest.fixture(scope="session", autouse=True)
def amazon_dynamodb_emulator(request):
    # Start Local Instance
    amazon_dynamodb_emulator = subprocess.Popen(
        [
            "docker-compose",
            "--file",
            os.path.join(os.path.dirname(__file__), "amazon-dynamodb-local.yaml"),
            "up",
            "-d",
        ],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
    )
    amazon_dynamodb_emulator.wait()
    
    # Wait for docker container to start up
    print("Waiting for Amazon DynamoDB local instance to start up...", end="")
    retries = 5
    while retries > 0:
        if setup_amazon_dynamodb_env() is True:
            break
        else:
            retries -= 1
            sleep(5)
    
    if retries == 0:
        print(
            "\nCouldn't start Amazon DynamoDB local instance in docker. Make sure docker is installed and running correctly."
        )
        exit(1)
    
    # Amazon DynamoDB local instance Finalizer
    def fin():
        res = subprocess.run(
            ["docker", "ps", "--filter", "name=dynamodb-local", "--format", "{{json . }}"], stdout=subprocess.PIPE
        )
        output_s = res.stdout.decode().strip()
        if output_s:
            output_j = json.loads(output_s)
            container_id = output_j.get("ID")
            if container_id:
                subprocess.run(["docker", "kill", container_id])
                subprocess.run(["docker", "container", "rm", container_id])
    
    request.addfinalizer(fin)


PARAMS = [
    {
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
        }
    },
    {
        "backend_client": {
            "TYPE": "amazon.dynamodb",
            "CONFIG": {
                "ADMIN": True,
                "READ_ONLY": False,
                "END_POINT": AMAZON_LOCAL_DYNAMODB_URL,
                "TABLE_PREFIX": "",
            },
        }
    }
]


@pytest.fixture(scope="function", params=PARAMS)
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
                         "ROOT_LOCK_EXPIRY": timedelta(seconds=5)
                     },
                     "ingest_config": {},
                 } | request.param
        
        meta, _, client_info = bootstrap("test", config=config)
        graph = ChunkedGraph(graph_id="test", meta=meta,
                             client_info=client_info)
        graph.mock_edges = Edges([], [])
        graph.meta._ws_cv = CloudVolumeMock()
        graph.meta.layer_count = n_layers
        graph.meta.layer_chunk_bounds = get_layer_chunk_bounds(
            n_layers, atomic_chunk_bounds=atomic_chunk_bounds
        )
        
        graph.create()
        
        # setup Chunked Graph - Finalizer
        def fin():
            backend_type = config["backend_client"].get("TYPE", DEFAULT_BACKEND_TYPE)
            if backend_type == GCP_BIGTABLE_BACKEND_TYPE:
                graph.client._table.delete()
            elif backend_type == AMAZON_DYNAMODB_BACKEND_TYPE:
                boto3_conf_ = botocore.config.Config(
                    retries={"max_attempts": 10, "mode": "standard"}
                )
                client = boto3.client("dynamodb", config=boto3_conf_, endpoint_url=AMAZON_LOCAL_DYNAMODB_URL)
                delete_amazon_dynamodb_tables(client)
        
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
        vertices=[to_label(graph, 1, 1, 0, 0, 0),
                  to_label(graph, 1, 1, 0, 0, 1)],
        edges=[
            (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 1), 0.5),
            (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
        ],
    )
    
    # Chunk C
    create_chunk(
        graph,
        vertices=[to_label(graph, 1, 2, 0, 0, 0)],
        edges=[(to_label(graph, 1, 2, 0, 0, 0),
                to_label(graph, 1, 1, 0, 0, 0), inf)],
    )
    
    add_layer(graph, 3, [0, 0, 0], n_threads=1)
    add_layer(graph, 3, [1, 0, 0], n_threads=1)
    add_layer(graph, 4, [0, 0, 0], n_threads=1)
    
    return graph


def create_chunk(cg, vertices=None, edges=None, timestamp=None):
    """
    Helper function to add vertices and edges to the chunkedgraph - no safety checks!
    """
    edges = edges if edges else []
    vertices = vertices if vertices else []
    vertices = np.unique(np.array(vertices, dtype=np.uint64))
    edges = [(np.uint64(v1), np.uint64(v2), np.float32(aff))
             for v1, v2, aff in edges]
    isolated_ids = [
        x
        for x in vertices
        if (x not in [edges[i][0] for i in range(len(edges))])
           and (x not in [edges[i][1] for i in range(len(edges))])
    ]
    
    chunk_edges_active = {}
    for edge_type in EDGE_TYPES:
        chunk_edges_active[edge_type] = Edges([], [])
    
    for e in edges:
        if cg.get_chunk_id(e[0]) == cg.get_chunk_id(e[1]):
            sv1s = np.array([e[0]], dtype=basetypes.NODE_ID)
            sv2s = np.array([e[1]], dtype=basetypes.NODE_ID)
            affs = np.array([e[2]], dtype=basetypes.EDGE_AFFINITY)
            chunk_edges_active[EDGE_TYPES.in_chunk] += Edges(
                sv1s, sv2s, affinities=affs
            )
    
    chunk_id = None
    if len(chunk_edges_active[EDGE_TYPES.in_chunk]):
        chunk_id = cg.get_chunk_id(
            chunk_edges_active[EDGE_TYPES.in_chunk].node_ids1[0])
    elif len(vertices):
        chunk_id = cg.get_chunk_id(vertices[0])
    
    for e in edges:
        if not cg.get_chunk_id(e[0]) == cg.get_chunk_id(e[1]):
            # Ensure proper order
            if chunk_id is not None:
                if not chunk_id == cg.get_chunk_id(e[0]):
                    e = [e[1], e[0], e[2]]
            sv1s = np.array([e[0]], dtype=basetypes.NODE_ID)
            sv2s = np.array([e[1]], dtype=basetypes.NODE_ID)
            affs = np.array([e[2]], dtype=basetypes.EDGE_AFFINITY)
            if np.isinf(e[2]):
                chunk_edges_active[EDGE_TYPES.cross_chunk] += Edges(
                    sv1s, sv2s, affinities=affs
                )
            else:
                chunk_edges_active[EDGE_TYPES.between_chunk] += Edges(
                    sv1s, sv2s, affinities=affs
                )
    
    all_edges = reduce(lambda x, y: x + y, chunk_edges_active.values())
    cg.mock_edges += all_edges
    
    isolated_ids = np.array(isolated_ids, dtype=np.uint64)
    add_atomic_edges(
        cg,
        cg.get_chunk_coordinates(chunk_id),
        chunk_edges_active,
        isolated=isolated_ids,
    )


def to_label(cg, l, x, y, z, segment_id):
    return cg.get_node_id(np.uint64(segment_id), layer=l, x=x, y=y, z=z)


def get_layer_chunk_bounds(
        n_layers: int, atomic_chunk_bounds: np.ndarray = np.array([])
) -> dict:
    if atomic_chunk_bounds.size == 0:
        limit = 2 ** (n_layers - 2)
        atomic_chunk_bounds = np.array([limit, limit, limit])
    layer_bounds_d = {}
    for layer in range(2, n_layers):
        layer_bounds = atomic_chunk_bounds / (2 ** (layer - 2))
        layer_bounds_d[layer] = np.ceil(layer_bounds).astype(int)
    return layer_bounds_d


@pytest.fixture(scope='session')
def sv_data():
    test_data_dir = 'pychunkedgraph/tests/data'
    edges_file = f'{test_data_dir}/sv_edges.npy'
    sv_edges = np.load(edges_file)
    
    source_file = f'{test_data_dir}/sv_sources.npy'
    sv_sources = np.load(source_file)
    
    sinks_file = f'{test_data_dir}/sv_sinks.npy'
    sv_sinks = np.load(sinks_file)
    
    affinity_file = f'{test_data_dir}/sv_affinity.npy'
    sv_affinity = np.load(affinity_file)
    
    area_file = f'{test_data_dir}/sv_area.npy'
    sv_area = np.load(area_file)
    yield (sv_edges, sv_sources, sv_sinks, sv_affinity, sv_area)
