import sys
import os
import subprocess
import pytest
import numpy as np
from functools import partial
import collections
from grpc._channel import _Rendezvous
from google.cloud import bigtable
from google.auth import credentials
from math import inf
from datetime import datetime, timedelta
from time import sleep
from signal import SIGTERM
from warnings import warn

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from pychunkedgraph.backend import chunkedgraph  # noqa
from pychunkedgraph.backend.utils import serializers, column_keys  # noqa
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions  # noqa
from pychunkedgraph.creator import graph_tests  # noqa


def setup_emulator_env():
    bt_env_init = subprocess.run(
        ["gcloud", "beta", "emulators", "bigtable", "env-init"], stdout=subprocess.PIPE)
    os.environ["BIGTABLE_EMULATOR_HOST"] = \
        bt_env_init.stdout.decode("utf-8").strip().split('=')[-1]

    c = bigtable.Client(
        project='IGNORE_ENVIRONMENT_PROJECT',
        credentials=credentials.AnonymousCredentials(),
        admin=True)
    t = c.instance("emulated_instance").table("emulated_table")

    try:
        t.create()
        return True
    except Exception as err:
        print('Bigtable Emulator not yet ready: %s' % err)
        return False


@pytest.fixture(scope='session', autouse=True)
def bigtable_emulator(request):
    # Start Emulator
    bigtable_emulator = subprocess.Popen(
        ["gcloud", "beta", "emulators", "bigtable", "start"], preexec_fn=os.setsid,
        stdout=subprocess.PIPE)

    # Wait for Emulator to start up
    print("Waiting for BigTables Emulator to start up...", end='')
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


@pytest.fixture(scope='function')
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


@pytest.fixture(scope='function')
def gen_graph(request):
    def _cgraph(request, fan_out=2, n_layers=10):
        # setup Chunked Graph
        dataset_info = {
            "data_dir": ""
        }

        graph = chunkedgraph.ChunkedGraph(
            request.function.__name__,
            project_id='IGNORE_ENVIRONMENT_PROJECT',
            credentials=credentials.AnonymousCredentials(),
            instance_id="emulated_instance", dataset_info=dataset_info,
            chunk_size=np.array([512, 512, 64], dtype=np.uint64),
            is_new=True, fan_out=np.uint64(fan_out),
            n_layers=np.uint64(n_layers))

        # setup Chunked Graph - Finalizer
        def fin():
            graph.table.delete()

        request.addfinalizer(fin)
        return graph

    return partial(_cgraph, request)


@pytest.fixture(scope='function')
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
    create_chunk(graph,
                 vertices=[to_label(graph, 1, 0, 0, 0, 0)],
                 edges=[])

    # Chunk B
    create_chunk(graph,
                 vertices=[to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 1)],
                 edges=[(to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 1), 0.5),
                        (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf)])

    # Chunk C
    create_chunk(graph,
                 vertices=[to_label(graph, 1, 2, 0, 0, 0)],
                 edges=[(to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf)])

    graph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))
    graph.add_layer(3, np.array([[2, 0, 0]]))
    graph.add_layer(4, np.array([[0, 0, 0], [1, 0, 0]]))

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

    isolated_node_ids = [x for x in vertices if (x not in [edges[i][0] for i in range(len(edges))]) and
                                                (x not in [edges[i][1] for i in range(len(edges))])]

    edge_ids = {"in_connected": np.array([], dtype=np.uint64).reshape(0, 2),
                "in_disconnected": np.array([], dtype=np.uint64).reshape(0, 2),
                "cross": np.array([], dtype=np.uint64).reshape(0, 2),
                "between_connected": np.array([], dtype=np.uint64).reshape(0, 2),
                "between_disconnected": np.array([], dtype=np.uint64).reshape(0, 2)}
    edge_affs = {"in_connected": np.array([], dtype=np.float32),
                 "in_disconnected": np.array([], dtype=np.float32),
                 "between_connected": np.array([], dtype=np.float32),
                 "between_disconnected": np.array([], dtype=np.float32)}

    for e in edges:
        if cgraph.test_if_nodes_are_in_same_chunk(e[0:2]):
            this_edge = np.array([e[0], e[1]], dtype=np.uint64).reshape(-1, 2)
            edge_ids["in_connected"] = \
                np.concatenate([edge_ids["in_connected"], this_edge])
            edge_affs["in_connected"] = \
                np.concatenate([edge_affs["in_connected"], [e[2]]])

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
                edge_ids["cross"] = \
                    np.concatenate([edge_ids["cross"], this_edge])
            else:
                edge_ids["between_connected"] = \
                    np.concatenate([edge_ids["between_connected"],
                                    this_edge])
                edge_affs["between_connected"] = \
                    np.concatenate([edge_affs["between_connected"], [e[2]]])

    isolated_node_ids = np.array(isolated_node_ids, dtype=np.uint64)

    cgraph.logger.debug(edge_ids)
    cgraph.logger.debug(edge_affs)

    # Use affinities as areas
    cgraph.add_atomic_edges_in_chunks(edge_ids, edge_affs, edge_affs,
                                      isolated_node_ids,
                                      time_stamp=timestamp)


def to_label(cgraph, l, x, y, z, segment_id):
    return cgraph.get_node_id(np.uint64(segment_id), layer=l, x=x, y=y, z=z)


class TestGraphNodeConversion:
    @pytest.mark.timeout(30)
    def test_compute_bitmasks(self):
        pass

    @pytest.mark.timeout(30)
    def test_node_conversion(self, gen_graph):
        cgraph = gen_graph(n_layers=10)

        node_id = cgraph.get_node_id(np.uint64(4), layer=2, x=3, y=1, z=0)
        assert cgraph.get_chunk_layer(node_id) == 2
        assert np.all(cgraph.get_chunk_coordinates(node_id) == np.array([3, 1, 0]))

        chunk_id = cgraph.get_chunk_id(layer=2, x=3, y=1, z=0)
        assert cgraph.get_chunk_layer(chunk_id) == 2
        assert np.all(cgraph.get_chunk_coordinates(chunk_id) == np.array([3, 1, 0]))

        assert cgraph.get_chunk_id(node_id=node_id) == chunk_id
        assert cgraph.get_node_id(np.uint64(4), chunk_id=chunk_id) == node_id

    @pytest.mark.timeout(30)
    def test_node_id_adjacency(self, gen_graph):
        cgraph = gen_graph(n_layers=10)

        assert cgraph.get_node_id(np.uint64(0), layer=2, x=3, y=1, z=0) + np.uint64(1) == \
            cgraph.get_node_id(np.uint64(1), layer=2, x=3, y=1, z=0)

        assert cgraph.get_node_id(np.uint64(2**53 - 2), layer=10, x=0, y=0, z=0) + np.uint64(1) == \
            cgraph.get_node_id(np.uint64(2**53 - 1), layer=10, x=0, y=0, z=0)

    @pytest.mark.timeout(30)
    def test_serialize_node_id(self, gen_graph):
        cgraph = gen_graph(n_layers=10)

        assert serializers.serialize_uint64(cgraph.get_node_id(np.uint64(0), layer=2, x=3, y=1, z=0)) < \
               serializers.serialize_uint64(cgraph.get_node_id(np.uint64(1), layer=2, x=3, y=1, z=0))

        assert serializers.serialize_uint64(cgraph.get_node_id(np.uint64(2 ** 53 - 2), layer=10, x=0, y=0, z=0)) < \
               serializers.serialize_uint64(cgraph.get_node_id(np.uint64(2 ** 53 - 1), layer=10, x=0, y=0, z=0))

    @pytest.mark.timeout(30)
    def test_deserialize_node_id(self):
        pass

    @pytest.mark.timeout(30)
    def test_serialization_roundtrip(self):
        pass

    @pytest.mark.timeout(30)
    def test_serialize_valid_label_id(self):
        label = np.uint64(0x01FF031234556789)
        assert serializers.deserialize_uint64(
            serializers.serialize_uint64(label)) == label


class TestGraphBuild:
    @pytest.mark.timeout(30)
    def test_build_single_node(self, gen_graph):
        """
        Create graph with single RG node 1 in chunk A
        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Add Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)])

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the RG-to-CG mapping:
        # assert chunkedgraph.serialize_uint64(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 0, 0, 0, 0)

        # Check for the Level 1 CG supervoxel:
        # to_label(cgraph, 1, 0, 0, 0, 0)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 0)) in res.rows
        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 0, 0, 0, 0))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 0
        assert len(atomic_affinities) == 0
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 0, 0, 0, 1)

        # Check for the one Level 2 node that should have been created.
        # to_label(cgraph, 2, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1))].cells[cgraph.family_id]
        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 2, 0, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        for aces in atomic_cross_edge_dict.values():
            assert len(aces) == 0

        assert len(children) == 1 and children[0] == to_label(cgraph, 1, 0, 0, 0, 0)

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 1 + 1 + 1 + 1

    @pytest.mark.timeout(30)
    def test_build_single_edge(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        ┌─────┐
        │  A¹ │
        │ 1━2 │
        │     │
        └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Add Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5)])

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        # assert chunkedgraph.serialize_uint64(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 0, 0, 0, 0)

        # assert chunkedgraph.serialize_uint64(2) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(2)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 0, 0, 0, 1)

        # Check for the two original Level 1 CG supervoxels
        # to_label(cgraph, 1, 0, 0, 0, 0)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 0)) in res.rows
        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 0, 0, 0, 0))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 1 and atomic_partners[0] == to_label(cgraph, 1, 0, 0, 0, 1)
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == 0.5
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 0, 0, 0, 1)

        # to_label(cgraph, 1, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 1)) in res.rows
        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 0, 0, 0, 1))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 1 and atomic_partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == 0.5
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 0, 0, 0, 1)

        # Check for the one Level 2 node that should have been created.
        assert serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1))].cells[cgraph.family_id]

        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 2, 0, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        for aces in atomic_cross_edge_dict.values():
            assert len(aces) == 0

        assert len(children) == 2 and to_label(cgraph, 1, 0, 0, 0, 0) in children and to_label(cgraph, 1, 0, 0, 0, 1) in children

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 2 + 1 + 1 + 1

    @pytest.mark.timeout(30)
    def test_build_single_across_edge(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┌─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        # assert chunkedgraph.serialize_uint64(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 0, 0, 0, 0)

        # assert chunkedgraph.serialize_uint64(2) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(2)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 1, 0, 0, 0)

        # Check for the two original Level 1 CG supervoxels
        # to_label(cgraph, 1, 0, 0, 0, 0)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 0)) in res.rows

        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 0, 0, 0, 0))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]

        cgraph.logger.debug(atomic_node_info.keys())

        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 1 and atomic_partners[0] == to_label(cgraph, 1, 1, 0, 0, 0)
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == inf
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 0, 0, 0, 1)

        # to_label(cgraph, 1, 1, 0, 0, 0)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 1, 0, 0, 0)) in res.rows

        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 1, 0, 0, 0))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 1 and atomic_partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == inf
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 1, 0, 0, 1)

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # to_label(cgraph, 2, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1))].cells[cgraph.family_id]

        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 2, 0, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array([to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0)], dtype=np.uint64)
        assert len(atomic_cross_edge_dict[2]) == 1
        assert test_ace in atomic_cross_edge_dict[2]
        assert len(children) == 1 and to_label(cgraph, 1, 0, 0, 0, 0) in children

        # to_label(cgraph, 2, 1, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 2, 1, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 2, 1, 0, 0, 1))].cells[cgraph.family_id]

        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 2, 1, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array([to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0)], dtype=np.uint64)
        assert len(atomic_cross_edge_dict[2]) == 1
        assert test_ace in atomic_cross_edge_dict[2]
        assert len(children) == 1 and to_label(cgraph, 1, 1, 0, 0, 0) in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # to_label(cgraph, 3, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 3, 0, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 3, 0, 0, 0, 1))].cells[cgraph.family_id]
        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 3, 0, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)


        for aces in atomic_cross_edge_dict.values():
            assert len(aces) == 0
        assert len(children) == 2 and to_label(cgraph, 2, 0, 0, 0, 1) in children and to_label(cgraph, 2, 1, 0, 0, 1) in children

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 2 + 2 + 1 + 3 + 1

    @pytest.mark.timeout(30)
    def test_build_single_edge_and_single_across_edge(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2━1━┿━━3  │
        │     │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the three RG-to-CG mappings:
        # assert chunkedgraph.serialize_uint64(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 0, 0, 0, 0)

        # assert chunkedgraph.serialize_uint64(2) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(2)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 0, 0, 0, 1)

        # assert chunkedgraph.serialize_uint64(3) in res.rows
        # row = res.rows[chunkedgraph.serialize_uint64(3)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == to_label(cgraph, 1, 1, 0, 0, 0)

        # Check for the three original Level 1 CG supervoxels
        # to_label(cgraph, 1, 0, 0, 0, 0)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 0)) in res.rows
        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 0, 0, 0, 0))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 2 and to_label(cgraph, 1, 0, 0, 0, 1) in atomic_partners and to_label(cgraph, 1, 1, 0, 0, 0) in atomic_partners
        assert len(atomic_affinities) == 2
        if atomic_partners[0] == to_label(cgraph, 1, 0, 0, 0, 1):
            assert atomic_affinities[0] == 0.5 and atomic_affinities[1] == inf
        else:
            assert atomic_affinities[0] == inf and atomic_affinities[1] == 0.5
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 0, 0, 0, 1)

        # to_label(cgraph, 1, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 1)) in res.rows
        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 0, 0, 0, 1))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 1 and atomic_partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == 0.5
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 0, 0, 0, 1)

        # to_label(cgraph, 1, 1, 0, 0, 0)
        assert serializers.serialize_uint64(to_label(cgraph, 1, 1, 0, 0, 0)) in res.rows
        atomic_node_info = cgraph.get_atomic_node_info(to_label(cgraph, 1, 1, 0, 0, 0))
        atomic_affinities = atomic_node_info[column_keys.Connectivity.Affinity]
        atomic_partners = atomic_node_info[column_keys.Connectivity.Partner]
        parents = atomic_node_info[column_keys.Hierarchy.Parent]

        assert len(atomic_partners) == 1 and atomic_partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == inf
        assert len(parents) == 1 and parents[0] == to_label(cgraph, 2, 1, 0, 0, 1)

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # to_label(cgraph, 2, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 2, 0, 0, 0, 1))].cells[cgraph.family_id]
        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 2, 0, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array([to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0)], dtype=np.uint64)
        assert len(atomic_cross_edge_dict[2]) == 1
        assert test_ace in atomic_cross_edge_dict[2]
        assert len(children) == 2 and to_label(cgraph, 1, 0, 0, 0, 0) in children and to_label(cgraph, 1, 0, 0, 0, 1) in children

        # to_label(cgraph, 2, 1, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 2, 1, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 2, 1, 0, 0, 1))].cells[cgraph.family_id]
        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 2, 1, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        test_ace = np.array([to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0)], dtype=np.uint64)
        assert len(atomic_cross_edge_dict[2]) == 1
        assert test_ace in atomic_cross_edge_dict[2]
        assert len(children) == 1 and to_label(cgraph, 1, 1, 0, 0, 0) in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # to_label(cgraph, 3, 0, 0, 0, 1)
        assert serializers.serialize_uint64(to_label(cgraph, 3, 0, 0, 0, 1)) in res.rows
        row = res.rows[serializers.serialize_uint64(to_label(cgraph, 3, 0, 0, 0, 1))].cells[cgraph.family_id]
        atomic_cross_edge_dict = cgraph.get_atomic_cross_edge_dict(to_label(cgraph, 3, 0, 0, 0, 1))
        column = column_keys.Hierarchy.Child
        children = column.deserialize(row[column.key][0].value)

        for ace in atomic_cross_edge_dict.values():
            assert len(ace) == 0
        assert len(children) == 2 and to_label(cgraph, 2, 0, 0, 0, 1) in children and to_label(cgraph, 2, 1, 0, 0, 1) in children

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 3 + 2 + 1 + 3 + 1

    @pytest.mark.timeout(30)
    def test_build_big_graph(self, gen_graph):
        """
        Create graph with RG nodes 1 and 2 in opposite corners of the largest possible dataset
        ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │
        │     │     │     │
        └─────┘     └─────┘
        """

        cgraph = gen_graph(n_layers=10)

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[])

        # Preparation: Build Chunk Z
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 255, 255, 255, 0)],
                     edges=[])

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(3, np.array([[0xFF, 0xFF, 0xFF]]))
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(4, np.array([[0x7F, 0x7F, 0x7F]]))
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(5, np.array([[0x3F, 0x3F, 0x3F]]))
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(6, np.array([[0x1F, 0x1F, 0x1F]]))
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(7, np.array([[0x0F, 0x0F, 0x0F]]))
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(8, np.array([[0x07, 0x07, 0x07]]))
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(9, np.array([[0x03, 0x03, 0x03]]))
        cgraph.add_layer(10, np.array([[0x00, 0x00, 0x00], [0x01, 0x01, 0x01]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # cgraph.logger.debug(len(res.rows))
        # for row_key in res.rows.keys():
        #     cgraph.logger.debug(row_key)
        #     cgraph.logger.debug(cgraph.get_chunk_layer(chunkedgraph.deserialize_uint64(row_key)))
        #     cgraph.logger.debug(cgraph.get_chunk_coordinates(chunkedgraph.deserialize_uint64(row_key)))

        assert serializers.serialize_uint64(to_label(cgraph, 1, 0, 0, 0, 0)) in res.rows
        assert serializers.serialize_uint64(to_label(cgraph, 1, 255, 255, 255, 0)) in res.rows
        assert serializers.serialize_uint64(to_label(cgraph, 10, 0, 0, 0, 1)) in res.rows
        assert serializers.serialize_uint64(to_label(cgraph, 10, 0, 0, 0, 2)) in res.rows

    @pytest.mark.timeout(30)
    def test_double_chunk_creation(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=4)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 1),
                               to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)
        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0, 0, 0]]),
                         time_stamp=fake_timestamp)

        assert len(cgraph.range_read_chunk(layer=3, x=0, y=0, z=0)) == 6
        assert len(cgraph.range_read_chunk(layer=4, x=0, y=0, z=0)) == 3

        assert cgraph.get_chunk_layer(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))) == 4
        assert cgraph.get_chunk_layer(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 2))) == 4
        assert cgraph.get_chunk_layer(cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1))) == 4

        lvl_3_child_ids = [cgraph.get_segment_id(cgraph.read_node_id_row(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)), column_keys.Hierarchy.Child)[0].value),
                           cgraph.get_segment_id(cgraph.read_node_id_row(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 2)), column_keys.Hierarchy.Child)[0].value),
                           cgraph.get_segment_id(cgraph.read_node_id_row(cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1)), column_keys.Hierarchy.Child)[0].value)]

        assert 4 in lvl_3_child_ids
        assert 5 in lvl_3_child_ids
        assert 6 in lvl_3_child_ids


class TestGraphSimpleQueries:
    """
    ┌─────┬─────┬─────┐        L X Y Z S     L X Y Z S     L X Y Z S     L X Y Z S
    │  A¹ │  B¹ │  C¹ │     1: 1 0 0 0 0 ─── 2 0 0 0 1 ─── 3 0 0 0 1 ─── 4 0 0 0 1
    │  1  │ 3━2━┿━━4  │     2: 1 1 0 0 0 ─┬─ 2 1 0 0 1 ─── 3 0 0 0 2 ─┬─ 4 0 0 0 2
    │     │     │     │     3: 1 1 0 0 1 ─┘                           │
    └─────┴─────┴─────┘     4: 1 2 0 0 0 ─── 2 2 0 0 1 ─── 3 1 0 0 1 ─┘
    """
    @pytest.mark.timeout(30)
    def test_get_parent_and_children(self, gen_graph_simplequerytest):
        cgraph = gen_graph_simplequerytest

        children10000 = cgraph.get_children(to_label(cgraph, 1, 0, 0, 0, 0))
        children11000 = cgraph.get_children(to_label(cgraph, 1, 1, 0, 0, 0))
        children11001 = cgraph.get_children(to_label(cgraph, 1, 1, 0, 0, 1))
        children12000 = cgraph.get_children(to_label(cgraph, 1, 2, 0, 0, 0))

        parent10000 = cgraph.get_parent(to_label(cgraph, 1, 0, 0, 0, 0), get_only_relevant_parent=True, time_stamp=None)
        parent11000 = cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 0), get_only_relevant_parent=True, time_stamp=None)
        parent11001 = cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)
        parent12000 = cgraph.get_parent(to_label(cgraph, 1, 2, 0, 0, 0), get_only_relevant_parent=True, time_stamp=None)

        children20001 = cgraph.get_children(to_label(cgraph, 2, 0, 0, 0, 1))
        children21001 = cgraph.get_children(to_label(cgraph, 2, 1, 0, 0, 1))
        children22001 = cgraph.get_children(to_label(cgraph, 2, 2, 0, 0, 1))

        parent20001 = cgraph.get_parent(to_label(cgraph, 2, 0, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)
        parent21001 = cgraph.get_parent(to_label(cgraph, 2, 1, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)
        parent22001 = cgraph.get_parent(to_label(cgraph, 2, 2, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)

        children30001 = cgraph.get_children(to_label(cgraph, 3, 0, 0, 0, 1))
        children30002 = cgraph.get_children(to_label(cgraph, 3, 0, 0, 0, 2))
        children31001 = cgraph.get_children(to_label(cgraph, 3, 1, 0, 0, 1))

        parent30001 = cgraph.get_parent(to_label(cgraph, 3, 0, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)
        parent30002 = cgraph.get_parent(to_label(cgraph, 3, 0, 0, 0, 2), get_only_relevant_parent=True, time_stamp=None)
        parent31001 = cgraph.get_parent(to_label(cgraph, 3, 1, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)

        children40001 = cgraph.get_children(to_label(cgraph, 4, 0, 0, 0, 1))
        children40002 = cgraph.get_children(to_label(cgraph, 4, 0, 0, 0, 2))

        parent40001 = cgraph.get_parent(to_label(cgraph, 4, 0, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)
        parent40002 = cgraph.get_parent(to_label(cgraph, 4, 0, 0, 0, 2), get_only_relevant_parent=True, time_stamp=None)

        # (non-existing) Children of L1
        assert np.array_equal(children10000, []) is True
        assert np.array_equal(children11000, []) is True
        assert np.array_equal(children11001, []) is True
        assert np.array_equal(children12000, []) is True

        # Parent of L1
        assert parent10000 == to_label(cgraph, 2, 0, 0, 0, 1)
        assert parent11000 == to_label(cgraph, 2, 1, 0, 0, 1)
        assert parent11001 == to_label(cgraph, 2, 1, 0, 0, 1)
        assert parent12000 == to_label(cgraph, 2, 2, 0, 0, 1)

        # Children of L2
        assert len(children20001) == 1 and to_label(cgraph, 1, 0, 0, 0, 0) in children20001
        assert len(children21001) == 2 and to_label(cgraph, 1, 1, 0, 0, 0) in children21001 and to_label(cgraph, 1, 1, 0, 0, 1) in children21001
        assert len(children22001) == 1 and to_label(cgraph, 1, 2, 0, 0, 0) in children22001

        # Parent of L2
        assert parent20001 == to_label(cgraph, 3, 0, 0, 0, 1) and parent21001 == to_label(cgraph, 3, 0, 0, 0, 2) or \
            parent20001 == to_label(cgraph, 3, 0, 0, 0, 2) and parent21001 == to_label(cgraph, 3, 0, 0, 0, 1)
        assert parent22001 == to_label(cgraph, 3, 1, 0, 0, 1)

        # Children of L3
        assert len(children30001) == 1 and len(children30002) == 1 and len(children31001) == 1
        assert to_label(cgraph, 2, 0, 0, 0, 1) in children30001 and to_label(cgraph, 2, 1, 0, 0, 1) in children30002 or \
            to_label(cgraph, 2, 0, 0, 0, 1) in children30002 and to_label(cgraph, 2, 1, 0, 0, 1) in children30001
        assert to_label(cgraph, 2, 2, 0, 0, 1) in children31001

        # Parent of L3
        assert parent30001 == parent31001 or parent30002 == parent31001
        assert (parent30001 == to_label(cgraph, 4, 0, 0, 0, 1) and parent30002 == to_label(cgraph, 4, 0, 0, 0, 2)) or \
               (parent30001 == to_label(cgraph, 4, 0, 0, 0, 2) and parent30002 == to_label(cgraph, 4, 0, 0, 0, 1))

        # Children of L4
        if len(children40001) == 1:
            assert parent20001 in children40001
            assert len(children40002) == 2 and parent21001 in children40002 and parent22001 in children40002
        elif len(children40001) == 2:
            assert parent21001 in children40001 and parent22001 in children40001
            assert len(children40002) == 1 and parent20001 in children40002

        # (non-existing) Parent of L4
        assert parent40001 is None
        assert parent40002 is None

        # # Children of (non-existing) L5
        # with pytest.raises(IndexError):
        #     cgraph.get_children(to_label(cgraph, 5, 0, 0, 0, 1))

        # # Parent of (non-existing) L5
        # with pytest.raises(IndexError):
        #     cgraph.get_parent(to_label(cgraph, 5, 0, 0, 0, 1), get_only_relevant_parent=True, time_stamp=None)

        children2_separate = cgraph.get_children([to_label(cgraph, 2, 0, 0, 0, 1),
                                                  to_label(cgraph, 2, 1, 0, 0, 1),
                                                  to_label(cgraph, 2, 2, 0, 0, 1)])
        assert len(children2_separate) == 3
        assert to_label(cgraph, 2, 0, 0, 0, 1) in children2_separate and \
               np.all(np.isin(children2_separate[to_label(cgraph, 2, 0, 0, 0, 1)], children20001))
        assert to_label(cgraph, 2, 1, 0, 0, 1) in children2_separate and \
               np.all(np.isin(children2_separate[to_label(cgraph, 2, 1, 0, 0, 1)], children21001))
        assert to_label(cgraph, 2, 2, 0, 0, 1) in children2_separate and \
               np.all(np.isin(children2_separate[to_label(cgraph, 2, 2, 0, 0, 1)], children22001))

        children2_combined = cgraph.get_children([to_label(cgraph, 2, 0, 0, 0, 1),
                                                  to_label(cgraph, 2, 1, 0, 0, 1),
                                                  to_label(cgraph, 2, 2, 0, 0, 1)], flatten=True)
        assert len(children2_combined) == 4 and \
                np.all(np.isin(children20001, children2_combined)) and \
                np.all(np.isin(children21001, children2_combined)) and \
                np.all(np.isin(children22001, children2_combined))

    @pytest.mark.timeout(30)
    def test_get_root(self, gen_graph_simplequerytest):
        cgraph = gen_graph_simplequerytest

        root10000 = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0),
                                    time_stamp=None)

        root11000 = cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0),
                                    time_stamp=None)

        root11001 = cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1),
                                    time_stamp=None)

        root12000 = cgraph.get_root(to_label(cgraph, 1, 2, 0, 0, 0),
                                    time_stamp=None)

        with pytest.raises(Exception) as e:
            cgraph.get_root(0)

        assert (root10000 == to_label(cgraph, 4, 0, 0, 0, 1) and
                root11000 == root11001 == root12000 == to_label(
                    cgraph, 4, 0, 0, 0, 2)) or \
               (root10000 == to_label(cgraph, 4, 0, 0, 0, 2) and
                root11000 == root11001 == root12000 == to_label(
                    cgraph, 4, 0, 0, 0, 1))

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes(self, gen_graph_simplequerytest):
        cgraph = gen_graph_simplequerytest
        root1 = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))
        root2 = cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))

        lvl1_nodes_1 = cgraph.get_subgraph_nodes(root1)
        lvl1_nodes_2 = cgraph.get_subgraph_nodes(root2)
        assert len(lvl1_nodes_1) == 1
        assert len(lvl1_nodes_2) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in lvl1_nodes_1
        assert to_label(cgraph, 1, 1, 0, 0, 0) in lvl1_nodes_2
        assert to_label(cgraph, 1, 1, 0, 0, 1) in lvl1_nodes_2
        assert to_label(cgraph, 1, 2, 0, 0, 0) in lvl1_nodes_2

        lvl2_nodes_1 = cgraph.get_subgraph_nodes(root1, return_layers=[2])
        lvl2_nodes_2 = cgraph.get_subgraph_nodes(root2, return_layers=[2])
        assert len(lvl2_nodes_1) == 1
        assert len(lvl2_nodes_2) == 2
        assert to_label(cgraph, 2, 0, 0, 0, 1) in lvl2_nodes_1
        assert to_label(cgraph, 2, 1, 0, 0, 1) in lvl2_nodes_2
        assert to_label(cgraph, 2, 2, 0, 0, 1) in lvl2_nodes_2

        lvl3_nodes_1 = cgraph.get_subgraph_nodes(root1, return_layers=[3])
        lvl3_nodes_2 = cgraph.get_subgraph_nodes(root2, return_layers=[3])
        assert len(lvl3_nodes_1) == 1
        assert len(lvl3_nodes_2) == 2
        assert to_label(cgraph, 3, 0, 0, 0, 1) in lvl3_nodes_1
        assert to_label(cgraph, 3, 0, 0, 0, 2) in lvl3_nodes_2
        assert to_label(cgraph, 3, 1, 0, 0, 1) in lvl3_nodes_2

        lvl4_node = cgraph.get_subgraph_nodes(root1, return_layers=[4])
        assert len(lvl4_node) == 1
        assert root1 in lvl4_node

        layers = cgraph.get_subgraph_nodes(root2, return_layers=[1, 4])
        assert len(layers) == 2 and 1 in layers and 4 in layers
        assert len(layers[4]) == 1 and root2 in layers[4]
        assert len(layers[1]) == 3
        assert to_label(cgraph, 1, 1, 0, 0, 0) in layers[1]
        assert to_label(cgraph, 1, 1, 0, 0, 1) in layers[1]
        assert to_label(cgraph, 1, 2, 0, 0, 0) in layers[1]

        lvl2_nodes = cgraph.get_subgraph_nodes(root2, return_layers=[2],
                                               bounding_box=[[1, 0, 0], [2, 1, 1]],
                                               bb_is_coordinate=False)
        assert len(lvl2_nodes) == 1
        assert to_label(cgraph, 2, 1, 0, 0, 1) in lvl2_nodes

        lvl2_parent = cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 0))
        lvl1_nodes = cgraph.get_subgraph_nodes(lvl2_parent)
        assert len(lvl1_nodes) == 2
        assert to_label(cgraph, 1, 1, 0, 0, 0) in lvl1_nodes
        assert to_label(cgraph, 1, 1, 0, 0, 1) in lvl1_nodes

    @pytest.mark.timeout(30)
    def test_get_subgraph_edges(self, gen_graph_simplequerytest):
        cgraph = gen_graph_simplequerytest
        root1 = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))
        root2 = cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))

        edges, affinities, areas = cgraph.get_subgraph_edges(root1)
        assert len(edges) == 0 and len(affinities) == 0 and len(areas) == 0

        edges, affinities, areas = cgraph.get_subgraph_edges(root2)

        assert [to_label(cgraph, 1, 1, 0, 0, 0),
                to_label(cgraph, 1, 1, 0, 0, 1)] in edges or \
               [to_label(cgraph, 1, 1, 0, 0, 1),
                to_label(cgraph, 1, 1, 0, 0, 0)] in edges

        assert [to_label(cgraph, 1, 1, 0, 0, 0),
                to_label(cgraph, 1, 2, 0, 0, 0)] in edges or \
               [to_label(cgraph, 1, 2, 0, 0, 0),
                to_label(cgraph, 1, 1, 0, 0, 0)] in edges

        # assert len(edges) == 2 and len(affinities) == 2 and len(areas) == 2

        lvl2_parent = cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 0))
        edges, affinities, areas = cgraph.get_subgraph_edges(lvl2_parent)
        assert [to_label(cgraph, 1, 1, 0, 0, 0),
                to_label(cgraph, 1, 1, 0, 0, 1)] in edges or \
               [to_label(cgraph, 1, 1, 0, 0, 1),
                to_label(cgraph, 1, 1, 0, 0, 0)] in edges

        assert [to_label(cgraph, 1, 1, 0, 0, 0),
                to_label(cgraph, 1, 2, 0, 0, 0)] in edges or \
               [to_label(cgraph, 1, 2, 0, 0, 0),
                to_label(cgraph, 1, 1, 0, 0, 0)] in edges

        assert len(edges) == 2

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes_bb(self, gen_graph_simplequerytest):
        cgraph = gen_graph_simplequerytest

        bb = np.array([[1, 0, 0], [2, 1, 1]], dtype=np.int)
        bb_coord = bb * cgraph.chunk_size

        childs_1 = cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1)), bounding_box=bb)
        childs_2 = cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1)), bounding_box=bb_coord, bb_is_coordinate=True)

        assert np.all(~(np.sort(childs_1) - np.sort(childs_2)))

    @pytest.mark.timeout(30)
    def test_get_atomic_partners(self, gen_graph_simplequerytest):
        cgraph = gen_graph_simplequerytest


class TestGraphMerge:
    @pytest.mark.timeout(30)
    def test_merge_pair_same_chunk(self, gen_graph):
        """
        Add edge between existing RG supervoxels 1 and 2 (same chunk)
        Expected: Same (new) parent for RG 1 and 2 on Layer two
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Merge
        new_root_ids = cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 0)], affinities=0.3)

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cgraph.get_parent(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_id
        assert cgraph.get_parent(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_id
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert partners[0] == to_label(cgraph, 1, 0, 0, 0, 1) and affinities[0] == np.float32(0.3)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert partners[0] == to_label(cgraph, 1, 0, 0, 0, 0) and affinities[0] == np.float32(0.3)
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_id))
        assert len(leaves) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves

    @pytest.mark.timeout(30)
    def test_merge_pair_neighboring_chunks(self, gen_graph):
        """
        Add edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        # Merge
        new_root_ids = cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0)], affinities=0.3)

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == new_root_id
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert partners[0] == to_label(cgraph, 1, 1, 0, 0, 0) and affinities[0] == np.float32(0.3)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 1, 0, 0, 0))
        assert partners[0] == to_label(cgraph, 1, 0, 0, 0, 0) and affinities[0] == np.float32(0.3)
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_id))
        assert len(leaves) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 1, 0, 0, 0) in leaves

    @pytest.mark.timeout(30)
    def test_merge_pair_disconnected_chunks(self, gen_graph):
        """
        Add edge between existing RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │     │     │     │      │     │     │     │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        cgraph = gen_graph(n_layers=9)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk Z
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 127, 127, 127, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(3, np.array([[0x7F, 0x7F, 0x7F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0x3F, 0x3F, 0x3F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(5, np.array([[0x1F, 0x1F, 0x1F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(6, np.array([[0x0F, 0x0F, 0x0F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(7, np.array([[0x07, 0x07, 0x07]]), time_stamp=fake_timestamp)
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(8, np.array([[0x03, 0x03, 0x03]]), time_stamp=fake_timestamp)
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00], [0x01, 0x01, 0x01]]), time_stamp=fake_timestamp)

        # Merge
        new_root_ids = cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 127, 127, 127, 0), to_label(cgraph, 1, 0, 0, 0, 0)], affinities=0.3)

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 127, 127, 127, 0)) == new_root_id
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert partners[0] == to_label(cgraph, 1, 127, 127, 127, 0) and affinities[0] == np.float32(0.3)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 127, 127, 127, 0))
        assert partners[0] == to_label(cgraph, 1, 0, 0, 0, 0) and affinities[0] == np.float32(0.3)
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_id))
        assert len(leaves) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 127, 127, 127, 0) in leaves

    @pytest.mark.timeout(30)
    def test_merge_pair_already_connected(self, gen_graph):
        """
        Add edge between already connected RG supervoxels 1 and 2 (same chunk).
        Expected: No change, i.e. same parent (to_label(cgraph, 2, 0, 0, 0, 1)), affinity (0.5) and timestamp as before
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5)],
                     timestamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Merge
        cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 0)])
        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        # Check
        if res_old.rows != res_new.rows:
            warn("Rows were modified when merging a pair of already connected supervoxels. "
                 "While probably not an error, it is an unnecessary operation.")

    @pytest.mark.timeout(30)
    def test_merge_triple_chain_to_full_circle_same_chunk(self, gen_graph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 2), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 2), 0.5)],
                     timestamp=fake_timestamp)

        # Merge
        new_root_ids = cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 0)], affinities=0.3)

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 2)) == new_root_id
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        assert to_label(cgraph, 1, 0, 0, 0, 2) in partners
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 0, 0, 0, 2) in partners
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 2))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_id))
        assert len(leaves) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 2) in leaves

    @pytest.mark.timeout(30)
    def test_merge_triple_chain_to_full_circle_neighboring_chunks(self, gen_graph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 1, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), inf)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        # Merge
        new_root_ids = cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0)], affinities=1.0)

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == new_root_id
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        assert to_label(cgraph, 1, 1, 0, 0, 0) in partners
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 1, 0, 0, 0) in partners
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 1, 0, 0, 0))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_id))
        assert len(leaves) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves
        assert to_label(cgraph, 1, 1, 0, 0, 0) in leaves

        cross_edge_dict_layers = graph_tests.root_cross_edge_test(new_root_id, cg=cgraph) # dict: layer -> cross_edge_dict
        n_cross_edges_layer = collections.defaultdict(list)

        for child_layer in cross_edge_dict_layers.keys():
            for layer in cross_edge_dict_layers[child_layer].keys():
                n_cross_edges_layer[layer].append(len(cross_edge_dict_layers[child_layer][layer]))

        for layer in n_cross_edges_layer.keys():
            assert len(np.unique(n_cross_edges_layer[layer])) == 1

    @pytest.mark.timeout(30)
    def test_merge_triple_chain_to_full_circle_disconnected_chunks(self, gen_graph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        cgraph = gen_graph(n_layers=9)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 127, 127, 127, 0), inf)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 127, 127, 127, 0)],
                     edges=[(to_label(cgraph, 1, 127, 127, 127, 0), to_label(cgraph, 1, 0, 0, 0, 1), inf)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(3, np.array([[0x7F, 0x7F, 0x7F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0x3F, 0x3F, 0x3F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(5, np.array([[0x1F, 0x1F, 0x1F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(6, np.array([[0x0F, 0x0F, 0x0F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(7, np.array([[0x07, 0x07, 0x07]]), time_stamp=fake_timestamp)
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(8, np.array([[0x03, 0x03, 0x03]]), time_stamp=fake_timestamp)
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00], [0x01, 0x01, 0x01]]), time_stamp=fake_timestamp)

        # Merge
        new_root_ids = cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 127, 127, 127, 0), to_label(cgraph, 1, 0, 0, 0, 0)], affinities=1.0)

        assert len(new_root_ids) == 1
        new_root_id = new_root_ids[0]

        # Check
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_id
        assert cgraph.get_root(to_label(cgraph, 1, 127, 127, 127, 0)) == new_root_id
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        assert to_label(cgraph, 1, 127, 127, 127, 0) in partners
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 127, 127, 127, 0) in partners
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 127, 127, 127, 0))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_id))
        assert len(leaves) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves
        assert to_label(cgraph, 1, 127, 127, 127, 0) in leaves

        cross_edge_dict_layers = graph_tests.root_cross_edge_test(new_root_id, cg=cgraph) # dict: layer -> cross_edge_dict
        n_cross_edges_layer = collections.defaultdict(list)

        for child_layer in cross_edge_dict_layers.keys():
            for layer in cross_edge_dict_layers[child_layer].keys():
                n_cross_edges_layer[layer].append(len(cross_edge_dict_layers[child_layer][layer]))

        for layer in n_cross_edges_layer.keys():
            assert len(np.unique(n_cross_edges_layer[layer])) == 1

    @pytest.mark.timeout(30)
    def test_merge_same_node(self, gen_graph):
        """
        Try to add loop edge between RG supervoxel 1 and itself
        ┌─────┐
        │  A¹ │
        │  1  │  =>  Reject
        │     │
        └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Merge
        assert cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0)]) is None

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_merge_pair_abstract_nodes(self, gen_graph):
        """
        Try to add edge between RG supervoxel 1 and abstract node "2"
                    ┌─────┐
                    │  B² │
                    │ "2" │
                    │     │
                    └─────┘
        ┌─────┐              =>  Reject
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Merge
        assert cgraph.add_edges("Jane Doe", [to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 2, 1, 0, 0, 1)]) is None

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_diagonal_connections(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2 1━┿━━3  │
        │  /  │     │
        ┌─────┬─────┐
        │  |  │     │
        │  4━━┿━━5  │
        │  C¹ │  D¹ │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0),
                               to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0),
                             to_label(cgraph, 1, 1, 0, 0, 0), inf),
                            (to_label(cgraph, 1, 0, 0, 0, 0),
                             to_label(cgraph, 1, 0, 1, 0, 0), inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0),
                             to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        # Chunk C
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 1, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 1, 0, 0),
                             to_label(cgraph, 1, 1, 1, 0, 0), inf),
                            (to_label(cgraph, 1, 0, 1, 0, 0),
                             to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        # Chunk D
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 1, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 1, 0, 0),
                             to_label(cgraph, 1, 0, 1, 0, 0), inf)])

        cgraph.add_layer(3,
                         np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]))

        rr = cgraph.range_read_chunk(
            chunk_id=cgraph.get_chunk_id(layer=3, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())

        assert len(root_ids_t0) == 2

        child_ids = []
        for root_id in root_ids_t0:
            cgraph.logger.debug(("root_id", root_id))
            child_ids.extend(cgraph.get_subgraph_nodes(root_id))

        new_roots = cgraph.add_edges("Jane Doe",
                                     [to_label(cgraph, 1, 0, 0, 0, 0),
                                      to_label(cgraph, 1, 0, 0, 0, 1)],
                                     affinities=[.5])

        root_ids = []
        for child_id in child_ids:
            root_ids.append(cgraph.get_root(child_id))

        assert len(np.unique(root_ids)) == 1

        root_id = root_ids[0]
        assert root_id == new_roots[0]

        cross_edge_dict_layers = graph_tests.root_cross_edge_test(root_id,
                                                                  cg=cgraph)  # dict: layer -> cross_edge_dict
        n_cross_edges_layer = collections.defaultdict(list)

        for child_layer in cross_edge_dict_layers.keys():
            for layer in cross_edge_dict_layers[child_layer].keys():
                n_cross_edges_layer[layer].append(
                    len(cross_edge_dict_layers[child_layer][layer]))

        for layer in n_cross_edges_layer.keys():
            assert len(np.unique(n_cross_edges_layer[layer])) == 1

    @pytest.mark.timeout(30)
    def test_cross_edges(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─...─┬────────┬─────┐      ┌─...─┬────────┬─────┐
        |     │     A¹ │  B¹ │      |     │     A¹ │  B¹ │
        |     │  4  1━━┿━━5  │  =>  |     │  4━━1━━┿━━5  │
        |     │   /    │  |  │      |     │   /    │     │
        |     │  3  2━━┿━━6  │      |     │  3  2━━┿━━6  │
        └─...─┴────────┴─────┘      └─...─┴────────┴─────┘
        """

        cgraph = gen_graph(n_layers=6)

        chunk_offset = 6

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, chunk_offset, 0, 0, 0), to_label(cgraph, 1, chunk_offset, 0, 0, 1),
                               to_label(cgraph, 1, chunk_offset, 0, 0, 2), to_label(cgraph, 1, chunk_offset, 0, 0, 3)],
                     edges=[(to_label(cgraph, 1, chunk_offset, 0, 0, 0), to_label(cgraph, 1, chunk_offset+1, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset, 0, 0, 1), to_label(cgraph, 1, chunk_offset+1, 0, 0, 1), inf),
                            (to_label(cgraph, 1, chunk_offset, 0, 0, 0), to_label(cgraph, 1, chunk_offset, 0, 0, 2), .5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, chunk_offset+1, 0, 0, 0), to_label(cgraph, 1, chunk_offset+1, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, chunk_offset+1, 0, 0, 0), to_label(cgraph, 1, chunk_offset, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset+1, 0, 0, 1), to_label(cgraph, 1, chunk_offset, 0, 0, 1), inf),
                            (to_label(cgraph, 1, chunk_offset+1, 0, 0, 0), to_label(cgraph, 1, chunk_offset+2, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset+1, 0, 0, 1), to_label(cgraph, 1, chunk_offset+2, 0, 0, 1), inf)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk C
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, chunk_offset+2, 0, 0, 0), to_label(cgraph, 1, chunk_offset+2, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, chunk_offset+2, 0, 0, 0), to_label(cgraph, 1, chunk_offset+1, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset+2, 0, 0, 1), to_label(cgraph, 1, chunk_offset+1, 0, 0, 1), inf),
                            (to_label(cgraph, 1, chunk_offset+2, 0, 0, 0), to_label(cgraph, 1, chunk_offset+3, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset+2, 0, 0, 0), to_label(cgraph, 1, chunk_offset+2, 0, 0, 1), .5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk D
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, chunk_offset+3, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, chunk_offset+3, 0, 0, 0), to_label(cgraph, 1, chunk_offset+2, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset+3, 0, 0, 0), to_label(cgraph, 1, chunk_offset+4, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk E
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, chunk_offset+4, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, chunk_offset+4, 0, 0, 0), to_label(cgraph, 1, chunk_offset+3, 0, 0, 0), inf),
                            (to_label(cgraph, 1, chunk_offset+4, 0, 0, 0), to_label(cgraph, 1, chunk_offset+5, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk F
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, chunk_offset+5, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, chunk_offset+5, 0, 0, 0), to_label(cgraph, 1, chunk_offset+4, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)


        for i_layer in range(3, 7):
            for i_chunk in range(0, 2 ** (7 - i_layer), 2):
                cgraph.add_layer(i_layer, np.array([[i_chunk, 0, 0], [i_chunk+1, 0, 0]]), time_stamp=fake_timestamp)


        new_roots = cgraph.add_edges("Jane Doe",
                                     [to_label(cgraph, 1, chunk_offset, 0, 0, 0),
                                      to_label(cgraph, 1, chunk_offset, 0, 0, 3)],
                                     affinities=.9)

        assert len(new_roots) == 1
        root_id = new_roots[0]

        cross_edge_dict_layers = graph_tests.root_cross_edge_test(root_id, cg=cgraph)  # dict: layer -> cross_edge_dict
        n_cross_edges_layer = collections.defaultdict(list)

        for child_layer in cross_edge_dict_layers.keys():
            for layer in cross_edge_dict_layers[child_layer].keys():
                n_cross_edges_layer[layer].append(
                    len(cross_edge_dict_layers[child_layer][layer]))

        for layer in n_cross_edges_layer.keys():
            cgraph.logger.debug("LAYER %d" % layer)
            assert len(np.unique(n_cross_edges_layer[layer])) == 1


class TestGraphSplit:
    @pytest.mark.timeout(30)
    def test_split_pair_same_chunk(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (same chunk)
        Expected: Different (new) parents for RG 1 and 2 on Layer two
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1 2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5)],
                     timestamp=fake_timestamp)

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        # Check New State
        assert len(new_root_ids) == 2
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) != cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 0
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 0
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 0, 0, 0, 1) in leaves

        # Check Old State still accessible
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp) == \
            cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1), time_stamp=fake_timestamp)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 1)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1), time_stamp=fake_timestamp)
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)))
        assert len(leaves) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves

        # assert len(cgraph.get_latest_roots()) == 2
        # assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    def test_split_nonexisting_edge(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (same chunk)
        Expected: Different (new) parents for RG 1 and 2 on Layer two
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │   | │      │   | │
        │   3 │      │   3 │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 2), to_label(cgraph, 1, 0, 0, 0, 1), 0.5)],
                     timestamp=fake_timestamp)

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 2), mincut=False)

        assert len(new_root_ids) == 1
        assert len(cgraph.get_atomic_node_partners(to_label(cgraph, 1, 0, 0, 0, 0))) == 1


    @pytest.mark.timeout(30)
    def test_split_pair_neighboring_chunks(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1━━┿━━2  │  =>  │  1  │  2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), 1.0)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), 1.0)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        # Check New State
        assert len(new_root_ids) == 2
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) != cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 0
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 1, 0, 0, 0))
        assert len(partners) == 0
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 1, 0, 0, 0) in leaves

        # Check Old State still accessible
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp) == \
            cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0), time_stamp=fake_timestamp)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 1, 0, 0, 0)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 1, 0, 0, 0), time_stamp=fake_timestamp)
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)))
        assert len(leaves) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 1, 0, 0, 0) in leaves

        assert len(cgraph.get_latest_roots()) == 2
        assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    @pytest.mark.timeout(30)
    def test_split_verify_cross_chunk_edges(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┬─────┐      ┌─────┬─────┬─────┐
        |     │  A¹ │  B¹ │      |     │  A¹ │  B¹ │
        |     │  1━━┿━━3  │  =>  |     │  1━━┿━━3  │
        |     │  |  │     │      |     │     │     │
        |     │  2  │     │      |     │  2  │     │
        └─────┴─────┴─────┘      └─────┴─────┴─────┘
        """

        cgraph = gen_graph(n_layers=4)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 2, 0, 0, 0), inf),
                            (to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 1), .5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 2, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 2, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)
        cgraph.add_layer(3, np.array([[2, 0, 0], [3, 0, 0]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1))
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == cgraph.get_root(to_label(cgraph, 1, 2, 0, 0, 0))

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 1), mincut=False)

        svs2 = cgraph.get_subgraph_nodes(new_root_ids[0])
        svs1 = cgraph.get_subgraph_nodes(new_root_ids[1])
        len_set = {1, 2}
        assert len(svs1) in len_set
        len_set.remove(len(svs1))
        assert len(svs2) in len_set


        # Check New State
        assert len(new_root_ids) == 2
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) != cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1))
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == cgraph.get_root(to_label(cgraph, 1, 2, 0, 0, 0))

        cc_dict = cgraph.get_atomic_cross_edge_dict(cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 0)))
        assert len(cc_dict[3]) == 1
        assert cc_dict[3][0][0] == to_label(cgraph, 1, 1, 0, 0, 0)
        assert cc_dict[3][0][1] == to_label(cgraph, 1, 2, 0, 0, 0)

        assert len(cgraph.get_latest_roots()) == 2
        assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    @pytest.mark.timeout(30)
    def test_split_verify_loop(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬────────┬─────┐      ┌─────┬────────┬─────┐
        |     │     A¹ │  B¹ │      |     │     A¹ │  B¹ │
        |     │  4━━1━━┿━━5  │  =>  |     │  4  1━━┿━━5  │
        |     │   /    │  |  │      |     │        │  |  │
        |     │  3  2━━┿━━6  │      |     │  3  2━━┿━━6  │
        └─────┴────────┴─────┘      └─────┴────────┴─────┘
        """

        cgraph = gen_graph(n_layers=4)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 1),
                               to_label(cgraph, 1, 1, 0, 0, 2), to_label(cgraph, 1, 1, 0, 0, 3)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 2, 0, 0, 0), inf),
                            (to_label(cgraph, 1, 1, 0, 0, 1), to_label(cgraph, 1, 2, 0, 0, 1), inf),
                            (to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 2), .5),
                            (to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 3), .5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 2, 0, 0, 0), to_label(cgraph, 1, 2, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 2, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), inf),
                            (to_label(cgraph, 1, 2, 0, 0, 1), to_label(cgraph, 1, 1, 0, 0, 1), inf),
                            (to_label(cgraph, 1, 2, 0, 0, 1), to_label(cgraph, 1, 2, 0, 0, 0), .5)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)
        cgraph.add_layer(3, np.array([[2, 0, 0], [3, 0, 0]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 1))
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == cgraph.get_root(to_label(cgraph, 1, 2, 0, 0, 0))

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 2), mincut=False)

        assert len(new_root_ids) == 2

        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 3), mincut=False)

        assert len(new_root_ids) == 2

        cc_dict = cgraph.get_atomic_cross_edge_dict(cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 0)))
        assert len(cc_dict[3]) == 1
        cc_dict = cgraph.get_atomic_cross_edge_dict(cgraph.get_parent(to_label(cgraph, 1, 1, 0, 0, 0)))
        assert len(cc_dict[3]) == 1

        assert len(cgraph.get_latest_roots()) == 3
        assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    @pytest.mark.timeout(30)
    def test_split_pair_disconnected_chunks(self, gen_graph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1━━┿━━━━━┿━━2  │  =>  │  1  │     │  2  │
        │     │     │     │      │     │     │     │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        cgraph = gen_graph(n_layers=9)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 127, 127, 127, 0), 1.0)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk Z
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 127, 127, 127, 0)],
                     edges=[(to_label(cgraph, 1, 127, 127, 127, 0), to_label(cgraph, 1, 0, 0, 0, 0), 1.0)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(3, np.array([[0x7F, 0x7F, 0x7F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(4, np.array([[0x3F, 0x3F, 0x3F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(5, np.array([[0x1F, 0x1F, 0x1F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(6, np.array([[0x0F, 0x0F, 0x0F]]), time_stamp=fake_timestamp)
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(7, np.array([[0x07, 0x07, 0x07]]), time_stamp=fake_timestamp)
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]), time_stamp=fake_timestamp)
        cgraph.add_layer(8, np.array([[0x03, 0x03, 0x03]]), time_stamp=fake_timestamp)
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00], [0x01, 0x01, 0x01]]), time_stamp=fake_timestamp)

        # Split
        new_roots = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 127, 127, 127, 0), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        # Check New State
        assert len(new_roots) == 2
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) != cgraph.get_root(to_label(cgraph, 1, 127, 127, 127, 0))
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 0
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 127, 127, 127, 0))
        assert len(partners) == 0
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 127, 127, 127, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 127, 127, 127, 0) in leaves

        # Check Old State still accessible
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp) == \
            cgraph.get_root(to_label(cgraph, 1, 127, 127, 127, 0), time_stamp=fake_timestamp)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 127, 127, 127, 0)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 127, 127, 127, 0), time_stamp=fake_timestamp)
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 0)
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)))
        assert len(leaves) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 127, 127, 127, 0) in leaves

    @pytest.mark.timeout(30)
    def test_split_pair_already_disconnected(self, gen_graph):
        """
        Try to remove edge between already disconnected RG supervoxels 1 and 2 (same chunk).
        Expected: No change, no error
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1 2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Split
        with pytest.raises(cg_exceptions.PreconditionError):
            cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        # Check
        if res_old.rows != res_new.rows:
            warn("Rows were modified when splitting a pair of already disconnected supervoxels. "
                 "While probably not an error, it is an unnecessary operation.")

    @pytest.mark.timeout(30)
    def test_split_full_circle_to_triple_chain_same_chunk(self, gen_graph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1 2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 2), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 2), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.3)],
                     timestamp=fake_timestamp)

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        # Check New State
        assert len(new_root_ids) == 1
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_ids[0]
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_ids[0]
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 2)) == new_root_ids[0]
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 2)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 2)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 2))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 0, 0, 0, 1) in partners
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_ids[0]))
        assert len(leaves) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 2) in leaves

        # Check Old State still accessible
        old_root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)
        assert new_root_ids[0] != old_root_id

        # assert len(cgraph.get_latest_roots()) == 1
        # assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    @pytest.mark.timeout(30)
    def test_split_full_circle_to_triple_chain_neighboring_chunks(self, gen_graph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1━━┿━━2  │  =>  │  1  │  2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, 1, 0, 0, 0), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), 0.3)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), 0.3)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        # Check New State
        assert len(new_root_ids) == 1
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_ids[0]
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_ids[0]
        assert cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0)) == new_root_ids[0]
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 1)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 1, 0, 0, 0))
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 1)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, 1, 0, 0, 0) in partners
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_ids[0]))
        assert len(leaves) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves
        assert to_label(cgraph, 1, 1, 0, 0, 0) in leaves

        # Check Old State still accessible
        old_root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)
        assert new_root_ids[0] != old_root_id

        assert len(cgraph.get_latest_roots()) == 1
        assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    @pytest.mark.timeout(30)
    def test_split_full_circle_to_triple_chain_disconnected_chunks(self, gen_graph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1━━┿━━━━━┿━━2  │  =>  │  1  │     │  2  │
        │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        cgraph = gen_graph(n_layers=9)

        loc = 2

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 1), to_label(cgraph, 1, loc, loc, loc, 0), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, loc, loc, loc, 0), 0.3)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk Z
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, loc, loc, loc, 0)],
                     edges=[(to_label(cgraph, 1, loc, loc, loc, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, loc, loc, loc, 0), to_label(cgraph, 1, 0, 0, 0, 0), 0.3)],
                     timestamp=fake_timestamp)

        for i_layer in range(3, 10):
            if loc // 2**(i_layer - 3) == 1:
                cgraph.add_layer(i_layer, np.array([[0, 0, 0], [1, 1, 1]]), time_stamp=fake_timestamp)
            elif loc // 2**(i_layer - 3) == 0:
                cgraph.add_layer(i_layer, np.array([[0, 0, 0]]), time_stamp=fake_timestamp)
            else:
                cgraph.add_layer(i_layer, np.array([[0, 0, 0]]), time_stamp=fake_timestamp)
                cgraph.add_layer(i_layer, np.array([[loc // 2**(i_layer - 3), loc // 2**(i_layer - 3), loc // 2**(i_layer - 3)]]), time_stamp=fake_timestamp)

        assert cgraph.get_root(to_label(cgraph, 1, loc, loc, loc, 0)) == \
               cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == \
               cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))

        # Split
        new_root_ids = cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, loc, loc, loc, 0), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        # Check New State
        assert len(new_root_ids) == 1
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == new_root_ids[0]
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1)) == new_root_ids[0]
        assert cgraph.get_root(to_label(cgraph, 1, loc, loc, loc, 0)) == new_root_ids[0]
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 1)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, loc, loc, loc, 0))
        assert len(partners) == 1 and partners[0] == to_label(cgraph, 1, 0, 0, 0, 1)
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 1))
        assert len(partners) == 2
        assert to_label(cgraph, 1, 0, 0, 0, 0) in partners
        assert to_label(cgraph, 1, loc, loc, loc, 0) in partners
        leaves = np.unique(cgraph.get_subgraph_nodes(new_root_ids[0]))
        assert len(leaves) == 3
        assert to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        assert to_label(cgraph, 1, 0, 0, 0, 1) in leaves
        assert to_label(cgraph, 1, loc, loc, loc, 0) in leaves

        # Check Old State still accessible
        old_root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0), time_stamp=fake_timestamp)
        assert new_root_ids[0] != old_root_id

        assert len(cgraph.get_latest_roots()) == 1
        assert len(cgraph.get_latest_roots(fake_timestamp)) == 1

    @pytest.mark.timeout(30)
    def test_split_same_node(self, gen_graph):
        """
        Try to remove (non-existing) edge between RG supervoxel 1 and itself
        ┌─────┐
        │  A¹ │
        │  1  │  =>  Reject
        │     │
        └─────┘
        """

        cgraph = gen_graph(n_layers=2)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Split
        with pytest.raises(cg_exceptions.PreconditionError):
            cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_split_pair_abstract_nodes(self, gen_graph):
        """
        Try to remove (non-existing) edge between RG supervoxel 1 and abstract node "2"
                    ┌─────┐
                    │  B² │
                    │ "2" │
                    │     │
                    └─────┘
        ┌─────┐              =>  Reject
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Split
        with pytest.raises(cg_exceptions.PreconditionError):
            cgraph.remove_edges("Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 2, 1, 0, 0, 1), mincut=False)

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_diagonal_connections(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2━1━┿━━3  │
        │  /  │     │
        ┌─────┬─────┐
        │  |  │     │
        │  4━━┿━━5  │
        │  C¹ │  D¹ │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), inf),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 1, 0, 0), inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        # Chunk C
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 1, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 1, 0, 0), to_label(cgraph, 1, 1, 1, 0, 0), inf),
                            (to_label(cgraph, 1, 0, 1, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        # Chunk D
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 1, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 1, 0, 0), to_label(cgraph, 1, 0, 1, 0, 0), inf)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]))

        rr = cgraph.range_read_chunk(chunk_id=cgraph.get_chunk_id(layer=3, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())

        assert len(root_ids_t0) == 1

        child_ids = []
        for root_id in root_ids_t0:
            cgraph.logger.debug(("root_id", root_id))
            child_ids.extend(cgraph.get_subgraph_nodes(root_id))



        new_roots = cgraph.remove_edges("Jane Doe",
                                        to_label(cgraph, 1, 0, 0, 0, 0),
                                        to_label(cgraph, 1, 0, 0, 0, 1),
                                        mincut=False)

        assert len(new_roots) == 2
        assert cgraph.get_root(to_label(cgraph, 1, 1, 1, 0, 0)) == \
               cgraph.get_root(to_label(cgraph, 1, 0, 1, 0, 0))
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) == \
               cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))

    @pytest.mark.timeout(30)
    def test_shatter(self, gen_graph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2━1━┿━━3  │
        │  /  │     │
        ┌─────┬─────┐
        │  |  │     │
        │  4━━┿━━5  │
        │  C¹ │  D¹ │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Chunk A
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 1), 0.5),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), inf),
                            (to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 0, 1, 0, 0), inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        # Chunk C
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 1, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 1, 0, 0), to_label(cgraph, 1, 1, 1, 0, 0), .1),
                            (to_label(cgraph, 1, 0, 1, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), inf)])

        # Chunk D
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 1, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 1, 0, 0), to_label(cgraph, 1, 0, 1, 0, 0), .1)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]))

        new_root_ids = cgraph.shatter_nodes("Jane Doe", atomic_node_ids=[to_label(cgraph, 1, 0, 0, 0, 0)])

        cgraph.logger.debug(new_root_ids)

        assert len(new_root_ids) == 3



class TestGraphMergeSplit:
    @pytest.mark.timeout(30)
    def test_multiple_cuts_and_splits(self, gen_graph_simplequerytest):
        """
        ┌─────┬─────┬─────┐        L X Y Z S     L X Y Z S     L X Y Z S     L X Y Z S
        │  A¹ │  B¹ │  C¹ │     1: 1 0 0 0 0 ─── 2 0 0 0 1 ─── 3 0 0 0 1 ─── 4 0 0 0 1
        │  1  │ 3━2━┿━━4  │     2: 1 1 0 0 0 ─┬─ 2 1 0 0 1 ─── 3 0 0 0 2 ─┬─ 4 0 0 0 2
        │     │     │     │     3: 1 1 0 0 1 ─┘                           │
        └─────┴─────┴─────┘     4: 1 2 0 0 0 ─── 2 2 0 0 1 ─── 3 1 0 0 1 ─┘
        """
        cgraph = gen_graph_simplequerytest

        rr = cgraph.range_read_chunk(chunk_id=cgraph.get_chunk_id(layer=4, x=0, y=0, z=0))
        root_ids_t0 = list(rr.keys())
        child_ids = []
        for root_id in root_ids_t0:
            cgraph.logger.debug(f"root_id {root_id}")
            child_ids.extend(cgraph.get_subgraph_nodes(root_id))

        for i in range(10):
            cgraph.logger.debug(f"\n\nITERATION {i}/10")

            cgraph.logger.debug("\n\nMERGE 1 & 3\n\n")
            new_roots = cgraph.add_edges("Jane Doe",
                                         [to_label(cgraph, 1, 0, 0, 0, 0),
                                          to_label(cgraph, 1, 1, 0, 0, 1)],
                                         affinities=.9)
            assert len(new_roots) == 1
            assert len(cgraph.get_subgraph_nodes(new_roots[0])) == 4

            root_ids = []
            for child_id in child_ids:
                root_ids.append(cgraph.get_root(child_id))
                cgraph.logger.debug((child_id, cgraph.get_chunk_coordinates(child_id), root_ids[-1]))

                parent_id = cgraph.get_parent(child_id)
                cgraph.logger.debug((parent_id, cgraph.read_cross_chunk_edges(parent_id)))

            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 1


            # ------------------------------------------------------------------


            cgraph.logger.debug("\n\nSPLIT 2 & 3\n\n")

            new_roots = cgraph.remove_edges("John Doe", to_label(cgraph, 1, 1, 0, 0, 0),
                                            to_label(cgraph, 1, 1, 0, 0, 1), mincut=False)

            assert len(np.unique(new_roots)) == 2

            for root in new_roots:
                cgraph.logger.debug(("SUBGRAPH", cgraph.get_subgraph_nodes(root)))

            cgraph.logger.debug("test children")
            root_ids = []
            for child_id in child_ids:
                root_ids.append(cgraph.get_root(child_id))
                cgraph.logger.debug((child_id, cgraph.get_chunk_coordinates(child_id), cgraph.get_segment_id(child_id), root_ids[-1]))
                cgraph.logger.debug((cgraph.get_atomic_node_info(child_id)))

            cgraph.logger.debug("test root")
            u_root_ids = np.unique(root_ids)
            these_child_ids = []
            for root_id in u_root_ids:
                these_child_ids.extend(cgraph.get_subgraph_nodes(root_id, verbose=False))
                cgraph.logger.debug((root_id, cgraph.get_subgraph_nodes(root_id, verbose=False)))

            assert len(these_child_ids) == 4
            assert len(u_root_ids) == 2

            # ------------------------------------------------------------------

            cgraph.logger.debug("\n\nSPLIT 1 & 3\n\n")
            new_roots = cgraph.remove_edges("Jane Doe",
                                            to_label(cgraph, 1, 0, 0, 0, 0),
                                            to_label(cgraph, 1, 1, 0, 0, 1),
                                            mincut=False)
            assert len(new_roots) == 2

            root_ids = []
            for child_id in child_ids:
                root_ids.append(cgraph.get_root(child_id))
                cgraph.logger.debug((child_id, cgraph.get_chunk_coordinates(child_id), root_ids[-1]))

                parent_id = cgraph.get_parent(child_id)
                cgraph.logger.debug((parent_id, cgraph.read_cross_chunk_edges(parent_id)))

            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 3


            # ------------------------------------------------------------------

            cgraph.logger.debug("\n\nMERGE 2 & 3\n\n")
            new_roots = cgraph.add_edges("Jane Doe",
                                         [to_label(cgraph, 1, 1, 0, 0, 0),
                                          to_label(cgraph, 1, 1, 0, 0, 1)],
                                         affinities=.9)
            assert len(new_roots) == 1

            root_ids = []
            for child_id in child_ids:
                root_ids.append(cgraph.get_root(child_id))
                cgraph.logger.debug((child_id, cgraph.get_chunk_coordinates(child_id), root_ids[-1]))

                parent_id = cgraph.get_parent(child_id)
                cgraph.logger.debug((parent_id, cgraph.read_cross_chunk_edges(parent_id)))

            u_root_ids = np.unique(root_ids)
            assert len(u_root_ids) == 2


            for root_id in root_ids:
                cross_edge_dict_layers = graph_tests.root_cross_edge_test(root_id, cg=cgraph) # dict: layer -> cross_edge_dict
                n_cross_edges_layer = collections.defaultdict(list)

                for child_layer in cross_edge_dict_layers.keys():
                    for layer in cross_edge_dict_layers[child_layer].keys():
                        n_cross_edges_layer[layer].append(len(cross_edge_dict_layers[child_layer][layer]))

                for layer in n_cross_edges_layer.keys():
                    assert len(np.unique(n_cross_edges_layer[layer])) == 1


class TestGraphMinCut:
    # TODO: Ideally, those tests should focus only on mincut retrieving the correct edges.
    #       The edge removal part should be tested exhaustively in TestGraphSplit
    @pytest.mark.timeout(30)
    def test_cut_regular_link(self, gen_graph):
        """
        Regular link between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), 0.5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), 0.5)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        # Mincut
        new_root_ids = cgraph.remove_edges(
                "Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0),
                [0, 0, 0], [2*cgraph.chunk_size[0], 2*cgraph.chunk_size[1], cgraph.chunk_size[2]],
                mincut=True)

        # Check New State
        assert len(new_root_ids) == 2
        assert cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0)) != cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 0, 0, 0, 0))
        assert len(partners) == 0
        partners, affinities, areas = cgraph.get_atomic_partners(to_label(cgraph, 1, 1, 0, 0, 0))
        assert len(partners) == 0
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 0, 0, 0, 0) in leaves
        leaves = np.unique(cgraph.get_subgraph_nodes(cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))))
        assert len(leaves) == 1 and to_label(cgraph, 1, 1, 0, 0, 0) in leaves

    @pytest.mark.timeout(30)
    def test_cut_no_link(self, gen_graph):
        """
        No connection between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  2  │
        │     │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Mincut
        with pytest.raises(cg_exceptions.PreconditionError):
            cgraph.remove_edges(
                "Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0),
                [0, 0, 0], [2*cgraph.chunk_size[0], 2*cgraph.chunk_size[1], cgraph.chunk_size[2]],
                mincut=True)

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_cut_old_link(self, gen_graph):
        """
        Link between 1 and 2 got removed previously (aff = 0.0)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1┅┅╎┅┅2  │
        │     │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0), 0.5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), 0.5)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]), time_stamp=fake_timestamp)
        cgraph.remove_edges("John Doe", to_label(cgraph, 1, 1, 0, 0, 0), to_label(cgraph, 1, 0, 0, 0, 0), mincut=False)

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Mincut
        with pytest.raises(cg_exceptions.PreconditionError):
            cgraph.remove_edges(
                "Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0), to_label(cgraph, 1, 1, 0, 0, 0),
                [0, 0, 0], [2*cgraph.chunk_size[0], 2*cgraph.chunk_size[1], cgraph.chunk_size[2]],
                mincut=True)

        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_cut_indivisible_link(self, gen_graph):
        """
        Sink: 1, Source: 2
        Link between 1 and 2 is set to `inf` and must not be cut.
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1══╪══2  │
        │     │     │
        └─────┴─────┘
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0),
                             to_label(cgraph, 1, 1, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0),
                             to_label(cgraph, 1, 0, 0, 0, 0), inf)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)

        original_parents_1 = cgraph.get_all_parents(
            to_label(cgraph, 1, 0, 0, 0, 0))
        original_parents_2 = cgraph.get_all_parents(
            to_label(cgraph, 1, 1, 0, 0, 0))

        # Mincut
        assert cgraph.remove_edges(
            "Jane Doe", to_label(cgraph, 1, 0, 0, 0, 0),
            to_label(cgraph, 1, 1, 0, 0, 0),
            [0, 0, 0], [2 * cgraph.chunk_size[0], 2 * cgraph.chunk_size[1],
                        cgraph.chunk_size[2]],
            mincut=True) is None

        new_parents_1 = cgraph.get_all_parents(to_label(cgraph, 1, 0, 0, 0, 0))
        new_parents_2 = cgraph.get_all_parents(to_label(cgraph, 1, 1, 0, 0, 0))

        assert np.all(np.array(original_parents_1) == np.array(new_parents_1))
        assert np.all(np.array(original_parents_2) == np.array(new_parents_2))


class TestGraphMultiCut:
    @pytest.mark.timeout(30)
    def test_cut_multi_tree(self, gen_graph):
        pass


class TestGraphHistory:
    """ These test inadvertantly also test merge and split operations """
    @pytest.mark.timeout(30)
    def test_cut_merge_history(self, gen_graph):
        """
        Regular link between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        (1) Split 1 and 2
        (2) Merge 1 and 2
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 0, 0, 0, 0),
                             to_label(cgraph, 1, 1, 0, 0, 0), 0.5)],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 0)],
                     edges=[(to_label(cgraph, 1, 1, 0, 0, 0),
                             to_label(cgraph, 1, 0, 0, 0, 0), 0.5)],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)

        first_root = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 0))
        assert first_root == cgraph.get_root(to_label(cgraph, 1, 1, 0, 0, 0))
        timestamp_before_split = datetime.utcnow()
        split_roots = cgraph.remove_edges("Jane Doe",
                                          to_label(cgraph, 1, 0, 0, 0, 0),
                                          to_label(cgraph, 1, 1, 0, 0, 0),
                                          mincut=False)

        assert len(split_roots) == 2
        timestamp_after_split = datetime.utcnow() 
        merge_roots = cgraph.add_edges("Jane Doe",
                                      [to_label(cgraph, 1, 0, 0, 0, 0),
                                       to_label(cgraph, 1, 1, 0, 0, 0)],
                                      affinities=.4)
        assert len(merge_roots) == 1
        merge_root = merge_roots[0]
        timestamp_after_merge = datetime.utcnow()

        assert len(cgraph.get_root_id_history(first_root,
                                              time_stamp_past=datetime.min,
                                              time_stamp_future=datetime.max)) == 4
        assert len(cgraph.get_root_id_history(split_roots[0],
                                              time_stamp_past=datetime.min,
                                              time_stamp_future=datetime.max)) == 3
        assert len(cgraph.get_root_id_history(split_roots[1],
                                              time_stamp_past=datetime.min,
                                              time_stamp_future=datetime.max)) == 3
        assert len(cgraph.get_root_id_history(merge_root,
                                              time_stamp_past=datetime.min,
                                              time_stamp_future=datetime.max)) == 4

        new_roots, old_roots = cgraph.get_delta_roots(timestamp_before_split,
                                                      timestamp_after_split)
        assert(len(old_roots)==1)
        assert(old_roots[0]==first_root)
        assert(len(new_roots)==2)
        assert(np.all(np.isin(new_roots, split_roots)))

        new_roots2, old_roots2 = cgraph.get_delta_roots(timestamp_after_split,
                                                        timestamp_after_merge)
        assert(len(new_roots2)==1)
        assert(new_roots2[0]==merge_root)
        assert(len(old_roots2)==2)
        assert(np.all(np.isin(old_roots2, split_roots)))
        
        new_roots3, old_roots3 = cgraph.get_delta_roots(timestamp_before_split,
                                                        timestamp_after_merge)
        assert(len(new_roots3)==1)
        assert(new_roots3[0]==merge_root)
        assert(len(old_roots3)==1)
        assert(old_roots3[0]==first_root)


class TestGraphLocks:
    @pytest.mark.timeout(30)
    def test_lock_unlock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try unlock (opid = 1)
        (4) Try lock (opid = 2)
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 1),
                               to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)

        operation_id_1 = cgraph.get_unique_operation_id()
        root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))
        assert cgraph.lock_root_loop(root_ids=[root_id],
                                     operation_id=operation_id_1)[0]

        operation_id_2 = cgraph.get_unique_operation_id()
        assert not cgraph.lock_root_loop(root_ids=[root_id],
                                         operation_id=operation_id_2)[0]

        assert cgraph.unlock_root(root_id=root_id,
                                  operation_id=operation_id_1)

        assert cgraph.lock_root_loop(root_ids=[root_id],
                                     operation_id=operation_id_2)[0]

    @pytest.mark.timeout(30)
    def test_lock_expiration(self, gen_graph, lock_expired_timedelta_override):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 1),
                               to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)

        operation_id_1 = cgraph.get_unique_operation_id()
        root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))
        assert cgraph.lock_root_loop(root_ids=[root_id],
                                     operation_id=operation_id_1)[0]

        operation_id_2 = cgraph.get_unique_operation_id()
        assert not cgraph.lock_root_loop(root_ids=[root_id],
                                         operation_id=operation_id_2)[0]

        assert cgraph.lock_root_loop(root_ids=[root_id],
                                     operation_id=operation_id_2,
                                     max_tries=10, waittime_s=.5)[0]

    @pytest.mark.timeout(30)
    def test_lock_renew(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 1),
                               to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)

        operation_id_1 = cgraph.get_unique_operation_id()
        root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))
        assert cgraph.lock_root_loop(root_ids=[root_id],
                                     operation_id=operation_id_1)[0]

        assert cgraph.check_and_renew_root_locks(root_ids=[root_id],
                                                 operation_id=operation_id_1)

    @pytest.mark.timeout(30)
    def test_lock_merge_lock_old_id(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Merge (includes lock opid 1)
        (2) Try lock opid 2 --> should be successful and return new root id
        """

        cgraph = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.utcnow() - timedelta(days=10)
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 0, 0, 0, 1),
                               to_label(cgraph, 1, 0, 0, 0, 2)],
                     edges=[],
                     timestamp=fake_timestamp)

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[to_label(cgraph, 1, 1, 0, 0, 1)],
                     edges=[],
                     timestamp=fake_timestamp)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]),
                         time_stamp=fake_timestamp)

        root_id = cgraph.get_root(to_label(cgraph, 1, 0, 0, 0, 1))

        new_root_ids = cgraph.add_edges("Chuck Norris", [to_label(cgraph, 1, 0, 0, 0, 1),
                                                       to_label(cgraph, 1, 0, 0, 0, 2)], affinities=1.)

        assert new_root_ids is not None

        operation_id_2 = cgraph.get_unique_operation_id()
        success, new_root_id = cgraph.lock_root_loop(root_ids=[root_id],
                                                      operation_id=operation_id_2,
                                                      max_tries=10, waittime_s=.5)

        cgraph.logger.debug(new_root_id)
        assert success
        assert new_root_ids[0] == new_root_id
