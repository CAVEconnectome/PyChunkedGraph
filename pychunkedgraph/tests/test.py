import sys
import os
import subprocess
import grpc
import pytest
import numpy as np
from google.cloud import bigtable, exceptions
from math import inf
from time import sleep
from signal import SIGTERM
from warnings import warn

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from backend import chunkedgraph # noqa


@pytest.fixture(scope='session', autouse=True)
def bigtable_emulator(request):
    # setup Emulator
    bigtables_emulator = subprocess.Popen(["gcloud", "beta", "emulators", "bigtable", "start"], preexec_fn=os.setsid, stdout=subprocess.PIPE)

    bt_env_init = subprocess.run(["gcloud", "beta", "emulators", "bigtable",  "env-init"], stdout=subprocess.PIPE)
    os.environ["BIGTABLE_EMULATOR_HOST"] = bt_env_init.stdout.decode("utf-8").strip().split('=')[-1]

    print("Waiting for BigTables Emulator to start up...", end='')
    c = bigtable.Client(project='', admin=True)
    retries = 5
    while retries > 0:
        try:
            c.list_instances()
        except exceptions._Rendezvous as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:  # Good error - means emulator is up!
                print(" Ready!")
                break
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                sleep(1)
            retries -= 1
            print(".", end='')
    if retries == 0:
        print("\nCouldn't start Bigtable Emulator. Make sure it is setup correctly.")
        exit(1)

    # setup Emulator-Finalizer
    def fin():
        os.killpg(os.getpgid(bigtables_emulator.pid), SIGTERM)
        bigtables_emulator.wait()

    request.addfinalizer(fin)


@pytest.fixture(scope='function')
def cgraph(request):
    # setup Chunked Graph
    graph = chunkedgraph.ChunkedGraph(project_id='emulated', instance_id="chunkedgraph", table_id=request.function.__name__)
    graph.table.create()

    column_family = graph.table.column_family(graph.family_id)
    column_family.create()

    # setup Chunked Graph - Finalizer
    def fin():
        graph.table.delete()

    request.addfinalizer(fin)

    return graph


def create_chunk(cgraph, vertices=None, edges=None):
    """
    Helper function to add vertices and edges to the chunkedgraph - no safety checks!
    """
    if not vertices:
        vertices = []

    if not edges:
        edges = []

    vertices = np.unique(np.array(vertices, dtype=np.uint64))
    edges = [(np.uint64(v1), np.uint64(v2), np.float32(aff)) for v1, v2, aff in edges]
    edge_ids = []
    cross_edge_ids = []
    edge_affs = []
    cross_edge_affs = []
    isolated_node_ids = [x for x in vertices if (x not in [edges[i][0] for i in range(len(edges))]) and
                                                (x not in [edges[i][1] for i in range(len(edges))])]

    for e in edges:
        if chunkedgraph.test_if_nodes_are_in_same_chunk(e[0:2]):
            edge_ids.append([e[0], e[1]])
            #  edge_ids.append([e[1], e[0]])
            edge_affs.append(e[2])
            #  edge_affs.append(e[2])
        else:
            cross_edge_ids.append([e[0], e[1]])
            cross_edge_affs.append(e[2])

    edge_ids = np.array(edge_ids, dtype=np.uint64).reshape(-1, 2)
    edge_affs = np.array(edge_affs, dtype=np.float32).reshape(-1, 1)
    cross_edge_ids = np.array(cross_edge_ids, dtype=np.uint64).reshape(-1, 2)
    cross_edge_affs = np.array(cross_edge_affs, dtype=np.float32).reshape(-1, 1)
    isolated_node_ids = np.array(isolated_node_ids, dtype=np.uint64)
    rg2cg = dict(list(enumerate(vertices, 1)))
    cg2rg = {v: k for k, v in rg2cg.items()}

    cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                      edge_affs, cross_edge_affs,
                                      isolated_node_ids, cg2rg, rg2cg)


class TestCoreUtils:
    def test_serialize_valid_node_id(self):
        assert chunkedgraph.serialize_node_id(0x01102001000000FF) == b'00076596382332354815'


class TestGraphBuild:
    def test_build_single_node(self, cgraph):
        """
        Create graph with single RG node 1 in chunk A
        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        # Add Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000])

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the RG-to-CG mapping:
        # assert chunkedgraph.serialize_node_id(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        # Check for the Level 1 CG supervoxel:
        # 0x0100000000000000
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0100000000000000)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 0
        assert len(atomic_affinities) == 0
        assert len(parents) == 1 and parents[0] == 0x0200000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 1

        # Check for the one Level 2 node that should have been created.
        # 0x0200000000000000
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0200000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 0
        assert len(children) == 1 and children[0] == 0x0100000000000000

        # Make sure there are not any more entries in the table
        # assert len(res.rows) == 3
        assert len(res.rows) == 2

    def test_build_single_edge(self, cgraph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        ┌─────┐
        │  A¹ │
        │ 1━2 │
        │     │
        └─────┘
        """

        # Add Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001],
                     edges=[(0x0100000000000000, 0x0100000000000001, 0.5)])

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        # assert chunkedgraph.serialize_node_id(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        # assert chunkedgraph.serialize_node_id(2) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(2)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000001

        # Check for the two original Level 1 CG supervoxels
        # 0x0100000000000000
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0100000000000000)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 1 and atomic_partners[0] == 0x0100000000000001
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == 0.5
        assert len(parents) == 1 and parents[0] == 0x0200000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 1

        # 0x0100000000000001
        assert chunkedgraph.serialize_node_id(0x0100000000000001) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0100000000000001)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 1 and atomic_partners[0] == 0x0100000000000000
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == 0.5
        assert len(parents) == 1 and parents[0] == 0x0200000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 2

        # Check for the one Level 2 node that should have been created.
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0200000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 0
        assert len(children) == 2 and 0x0100000000000000 in children and 0x0100000000000001 in children

        # Make sure there are not any more entries in the table
        # assert len(res.rows) == 5
        assert len(res.rows) == 3

    def test_build_single_across_edge(self, cgraph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┌─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """

        # Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000],
                     edges=[(0x0100000000000000, 0x0101000000000000, inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[0x0101000000000000],
                     edges=[(0x0101000000000000, 0x0100000000000000, inf)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        # assert chunkedgraph.serialize_node_id(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        # assert chunkedgraph.serialize_node_id(2) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(2)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0101000000000000

        # Check for the two original Level 1 CG supervoxels
        # 0x0100000000000000
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0100000000000000)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 1 and atomic_partners[0] == 0x0101000000000000
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == inf
        assert len(parents) == 1 and parents[0] == 0x0200000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 1

        # 0x0101000000000000
        assert chunkedgraph.serialize_node_id(0x0101000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0101000000000000)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 1 and atomic_partners[0] == 0x0100000000000000
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == inf
        assert len(parents) == 1 and parents[0] == 0x0201000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 2

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # 0x0200000000000000
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0200000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 1 and [0x0100000000000000, 0x0101000000000000] in atomic_cross_edges
        assert len(children) == 1 and 0x0100000000000000 in children

        # 0x0201000000000000
        assert chunkedgraph.serialize_node_id(0x0201000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0201000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 1 and [0x0101000000000000, 0x0100000000000000] in atomic_cross_edges
        assert len(children) == 1 and 0x0101000000000000 in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # 0x0300000000000000
        assert chunkedgraph.serialize_node_id(0x0300000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0300000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 0
        assert len(children) == 2 and 0x0200000000000000 in children and 0x0201000000000000 in children

        # Make sure there are not any more entries in the table
        # assert len(res.rows) == 7
        assert len(res.rows) == 5

    def test_build_single_edge_and_single_across_edge(self, cgraph):
        """
        Create graph with edge between RG supervoxels 1 and 2 (same chunk)
        and edge between RG supervoxels 1 and 3 (neighboring chunks)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │ 2━1━┿━━3  │
        │     │     │
        └─────┴─────┘
        """

        # Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001],
                     edges=[(0x0100000000000000, 0x0100000000000001, 0.5),
                            (0x0100000000000000, 0x0101000000000000, inf)])

        # Chunk B
        create_chunk(cgraph,
                     vertices=[0x0101000000000000],
                     edges=[(0x0101000000000000, 0x0100000000000000, inf)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the three RG-to-CG mappings:
        # assert chunkedgraph.serialize_node_id(1) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        # assert chunkedgraph.serialize_node_id(2) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(2)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000001

        # assert chunkedgraph.serialize_node_id(3) in res.rows
        # row = res.rows[chunkedgraph.serialize_node_id(3)].cells[cgraph.family_id]
        # assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0101000000000000

        # Check for the three original Level 1 CG supervoxels
        # 0x0100000000000000
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0100000000000000)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 2 and 0x0100000000000001 in atomic_partners and 0x0101000000000000 in atomic_partners
        assert len(atomic_affinities) == 2
        if atomic_partners[0] == 0x0100000000000001:
            assert atomic_affinities[0] == 0.5 and atomic_affinities[1] == inf
        else:
            assert atomic_affinities[0] == inf and atomic_affinities[1] == 0.5
        assert len(parents) == 1 and parents[0] == 0x0200000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 1

        # 0x0100000000000001
        assert chunkedgraph.serialize_node_id(0x0100000000000001) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0100000000000001)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 1 and atomic_partners[0] == 0x0100000000000000
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == 0.5
        assert len(parents) == 1 and parents[0] == 0x0200000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 2

        # 0x0101000000000000
        assert chunkedgraph.serialize_node_id(0x0101000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0101000000000000)].cells[cgraph.family_id]
        atomic_affinities = np.frombuffer(row[b'atomic_affinities'][0].value, np.float32)
        atomic_partners = np.frombuffer(row[b'atomic_partners'][0].value, np.uint64)
        parents = np.frombuffer(row[b'parents'][0].value, np.uint64)
        rg_id = np.frombuffer(row[b'rg_id'][0].value, np.uint64)

        assert len(atomic_partners) == 1 and atomic_partners[0] == 0x0100000000000000
        assert len(atomic_affinities) == 1 and atomic_affinities[0] == inf
        assert len(parents) == 1 and parents[0] == 0x0201000000000000
        assert len(rg_id) == 1  # and rg_id[0] == 3

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        # 0x0200000000000000
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0200000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 1 and [0x0100000000000000, 0x0101000000000000] in atomic_cross_edges
        assert len(children) == 2 and 0x0100000000000000 in children and 0x0100000000000001 in children

        # 0x0201000000000000
        assert chunkedgraph.serialize_node_id(0x0201000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0201000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 1 and [0x0101000000000000, 0x0100000000000000] in atomic_cross_edges
        assert len(children) == 1 and 0x0101000000000000 in children

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        # 0x0300000000000000
        assert chunkedgraph.serialize_node_id(0x0300000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0300000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 0
        assert len(children) == 2 and 0x0200000000000000 in children and 0x0201000000000000 in children

        # Make sure there are not any more entries in the table
        # assert len(res.rows) == 9
        assert len(res.rows) == 6

    def test_build_big_graph(self, cgraph):
        """
        Create graph with RG nodes 1 and 2 in opposite corners of the largest possible dataset
        ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │
        │     │     │     │
        └─────┘     └─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000],
                     edges=[])

        # Preparation: Build Chunk Z
        create_chunk(cgraph,
                     vertices=[0x01FFFFFF00000000],
                     edges=[])

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(3, np.array([[0xFF, 0xFF, 0xFF]]))
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(4, np.array([[0xFE, 0xFE, 0xFE], [0xFF, 0xFF, 0xFF]]))
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(5, np.array([[0xFC, 0xFC, 0xFC], [0xFE, 0xFE, 0xFE]]))
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(6, np.array([[0xF8, 0xF8, 0xF8], [0xFC, 0xFC, 0xFC]]))
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(7, np.array([[0xF0, 0xF0, 0xF0], [0xF8, 0xF8, 0xF8]]))
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(8, np.array([[0xE0, 0xE0, 0xE0], [0xF0, 0xF0, 0xF0]]))
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(9, np.array([[0xC0, 0xC0, 0xC0], [0xE0, 0xE0, 0xE0]]))
        cgraph.add_layer(10, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(10, np.array([[0x80, 0x80, 0x80], [0xC0, 0xC0, 0xC0]]))
        cgraph.add_layer(11, np.array([[0x00, 0x00, 0x00], [0x80, 0x80, 0x80]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x01FFFFFF00000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0B00000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0B00000000000001) in res.rows


class TestGraphQueries:
    def test_get_parent(self, cgraph):
        pass

    def test_get_root(self, cgraph):
        pass

    def test_get_children(self, cgraph):
        pass

    def test_get_atomic_partners(self, cgraph):
        pass

    def test_get_cg_id_from_rg_id(self, cgraph):
        pass

    def test_get_rg_id_from_cg_id(self, cgraph):
        pass


class TestGraphMerge:
    def test_merge_pair_same_chunk(self, cgraph):
        """
        Add edge between existing RG supervoxels 1 and 2 (same chunk)
        Expected: Same (new) parent for RG 1 and 2 on Layer two
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001],
                     edges=[])

        # Merge
        new_root_id = cgraph.add_edge([0x0100000000000001, 0x0100000000000000], affinity=0.3, is_cg_id=True)
        res = cgraph.table.read_rows()
        res.consume_all()

        # Check
        assert cgraph.get_parent(0x0100000000000000) == new_root_id
        assert cgraph.get_parent(0x0100000000000001) == new_root_id
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000000)
        assert partners[0] == 0x0100000000000001 and affinities[0] == np.float32(0.3)
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000001)
        assert partners[0] == 0x0100000000000000 and affinities[0] == np.float32(0.3)
        children = cgraph.get_children(new_root_id)
        assert 0x0100000000000000 in children
        assert 0x0100000000000001 in children

    def test_merge_pair_neighboring_chunks(self, cgraph):
        """
        Add edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000],
                     edges=[])

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[0x0101000000000000],
                     edges=[])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        # Merge
        new_root_id = cgraph.add_edge([0x0101000000000000, 0x0100000000000000], affinity=0.3, is_cg_id=True)
        res = cgraph.table.read_rows()
        res.consume_all()

        # Check
        assert cgraph.get_root(0x0100000000000000) == new_root_id
        assert cgraph.get_root(0x0101000000000000) == new_root_id
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000000)
        assert partners[0] == 0x0101000000000000 and affinities[0] == np.float32(0.3)
        partners, affinities = cgraph.get_atomic_partners(0x0101000000000000)
        assert partners[0] == 0x0100000000000000 and affinities[0] == np.float32(0.3)
        # children = cgraph.get_children(new_root_id)
        # assert 0x0100000000000000 in children
        # assert 0x0101000000000000 in children

    def test_merge_pair_disconnected_chunks(self, cgraph):
        """
        Add edge between existing RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │     │     │     │      │     │     │     │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000],
                     edges=[])

        # Preparation: Build Chunk Z
        create_chunk(cgraph,
                     vertices=[0x017F7F7F00000000],
                     edges=[])

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(3, np.array([[0x7F, 0x7F, 0x7F]]))
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(4, np.array([[0x7E, 0x7E, 0x7E], [0x7F, 0x7F, 0x7F]]))
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(5, np.array([[0x7C, 0x7C, 0x7C], [0x7E, 0x7E, 0x7E]]))
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(6, np.array([[0x78, 0x78, 0x78], [0x7C, 0x7C, 0x7C]]))
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(7, np.array([[0x70, 0x70, 0x70], [0x78, 0x78, 0x78]]))
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(8, np.array([[0x60, 0x60, 0x60], [0x70, 0x70, 0x70]]))
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(9, np.array([[0x40, 0x40, 0x40], [0x60, 0x60, 0x60]]))
        cgraph.add_layer(10, np.array([[0x00, 0x00, 0x00], [0x40, 0x40, 0x40]]))

        # Merge
        new_root_id = cgraph.add_edge([0x017F7F7F00000000, 0x0100000000000000], affinity=0.3, is_cg_id=True)
        res = cgraph.table.read_rows()
        res.consume_all()

        # Check
        assert cgraph.get_root(0x0100000000000000) == new_root_id
        assert cgraph.get_root(0x017F7F7F00000000) == new_root_id
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000000)
        assert partners[0] == 0x017F7F7F00000000 and affinities[0] == np.float32(0.3)
        partners, affinities = cgraph.get_atomic_partners(0x017F7F7F00000000)
        assert partners[0] == 0x0100000000000000 and affinities[0] == np.float32(0.3)
        # children = cgraph.get_children(new_root_id)
        # assert 0x0100000000000000 in children
        # assert 0x017F7F7F00000000 in children

    def test_merge_pair_already_connected(self, cgraph):
        """
        Add edge between already connected RG supervoxels 1 and 2 (same chunk).
        Expected: No change, i.e. same parent (0x0200000000000000), affinity (0.5) and timestamp as before
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001],
                     edges=[(0x0100000000000000, 0x0100000000000001, 0.5)])

        res_old = cgraph.table.read_rows()
        res_old.consume_all()

        # Merge
        cgraph.add_edge([0x0100000000000001, 0x0100000000000000], is_cg_id=True)
        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        # Check
        if res_old != res_new:
            warn("Rows were modified when merging a pair of already connected supervoxels. While not an error, this is an unnecessary operation.")
        # assert res_old == res_new

    def test_merge_triple_chain_to_full_circle_same_chunk(self, cgraph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001, 0x0100000000000002],
                     edges=[(0x0100000000000000, 0x0100000000000002, 0.5),
                            (0x0100000000000001, 0x0100000000000002, 0.5)])

        # Merge
        new_root_id = cgraph.add_edge([0x0100000000000001, 0x0100000000000000], affinity=0.3, is_cg_id=True)
        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        # Check
        assert cgraph.get_root(0x0100000000000000) == new_root_id
        assert cgraph.get_root(0x0100000000000001) == new_root_id
        assert cgraph.get_root(0x0100000000000002) == new_root_id
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000000)
        assert 0x0100000000000001 in partners
        assert 0x0100000000000002 in partners
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000001)
        assert 0x0100000000000000 in partners
        assert 0x0100000000000002 in partners
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000002)
        assert 0x0100000000000000 in partners
        assert 0x0100000000000001 in partners
        children = cgraph.get_children(new_root_id)
        assert 0x0100000000000000 in children
        assert 0x0100000000000001 in children
        assert 0x0100000000000002 in children

    def test_merge_triple_chain_to_full_circle_neighboring_chunks(self, cgraph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001],
                     edges=[(0x0100000000000000, 0x0100000000000001, 0.5),
                            (0x0100000000000001, 0x0101000000000000, inf)])

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[0x0101000000000000],
                     edges=[(0x0101000000000000, 0x0100000000000001, inf)])

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        # Merge
        new_root_id = cgraph.add_edge([0x0101000000000000, 0x0100000000000000], affinity=1.0, is_cg_id=True)
        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        # Check
        assert cgraph.get_root(0x0100000000000000) == new_root_id
        assert cgraph.get_root(0x0100000000000001) == new_root_id
        assert cgraph.get_root(0x0101000000000000) == new_root_id
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000000)
        assert 0x0100000000000001 in partners
        assert 0x0101000000000000 in partners
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000001)
        assert 0x0100000000000000 in partners
        assert 0x0101000000000000 in partners
        partners, affinities = cgraph.get_atomic_partners(0x0101000000000000)
        assert 0x0100000000000000 in partners
        assert 0x0100000000000001 in partners
        # children = cgraph.get_children(new_root_id)
        # assert 0x0100000000000000 in children
        # assert 0x0100000000000001 in children
        # assert 0x0100000000000002 in children

    def test_merge_triple_chain_to_full_circle_disconnected_chunks(self, cgraph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        # Preparation: Build Chunk A
        create_chunk(cgraph,
                     vertices=[0x0100000000000000, 0x0100000000000001],
                     edges=[(0x0100000000000000, 0x0100000000000001, 0.5),
                            (0x0100000000000001, 0x017F7F7F00000000, inf)])

        # Preparation: Build Chunk B
        create_chunk(cgraph,
                     vertices=[0x017F7F7F00000000],
                     edges=[(0x017F7F7F00000000, 0x0100000000000001, inf)])

        cgraph.add_layer(3, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(3, np.array([[0x7F, 0x7F, 0x7F]]))
        cgraph.add_layer(4, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(4, np.array([[0x7E, 0x7E, 0x7E], [0x7F, 0x7F, 0x7F]]))
        cgraph.add_layer(5, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(5, np.array([[0x7C, 0x7C, 0x7C], [0x7E, 0x7E, 0x7E]]))
        cgraph.add_layer(6, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(6, np.array([[0x78, 0x78, 0x78], [0x7C, 0x7C, 0x7C]]))
        cgraph.add_layer(7, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(7, np.array([[0x70, 0x70, 0x70], [0x78, 0x78, 0x78]]))
        cgraph.add_layer(8, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(8, np.array([[0x60, 0x60, 0x60], [0x70, 0x70, 0x70]]))
        cgraph.add_layer(9, np.array([[0x00, 0x00, 0x00]]))
        cgraph.add_layer(9, np.array([[0x40, 0x40, 0x40], [0x60, 0x60, 0x60]]))
        cgraph.add_layer(10, np.array([[0x00, 0x00, 0x00], [0x40, 0x40, 0x40]]))

        # Merge
        new_root_id = cgraph.add_edge([0x017F7F7F00000000, 0x0100000000000000], affinity=1.0, is_cg_id=True)
        res_new = cgraph.table.read_rows()
        res_new.consume_all()

        # Check
        assert cgraph.get_root(0x0100000000000000) == new_root_id
        assert cgraph.get_root(0x0100000000000001) == new_root_id
        assert cgraph.get_root(0x017F7F7F00000000) == new_root_id
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000000)
        assert 0x0100000000000001 in partners
        assert 0x017F7F7F00000000 in partners
        partners, affinities = cgraph.get_atomic_partners(0x0100000000000001)
        assert 0x0100000000000000 in partners
        assert 0x017F7F7F00000000 in partners
        partners, affinities = cgraph.get_atomic_partners(0x017F7F7F00000000)
        assert 0x0100000000000000 in partners
        assert 0x0100000000000001 in partners
        # children = cgraph.get_children(new_root_id)
        # assert 0x0100000000000000 in children
        # assert 0x0100000000000001 in children
        # assert 0x0100000000000002 in children

    def test_merge_same_node(self, cgraph):  # This should fail
        """
        Try to add loop edge between RG supervoxel 1 and itself
        ┌─────┐
        │  A¹ │
        │  1  │  =>  Reject
        │     │
        └─────┘
        """
        pass

    def test_merge_pair_abstract_nodes(self, cgraph):  # This should fail
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
        pass


class TestGraphSplit:
    def test_split_pair_same_chunk(self, cgraph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (same chunk)
        Expected: Different (new) parents for RG 1 and 2 on Layer two
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1 2 │
        │     │      │     │
        └─────┘      └─────┘
        """
        pass

    def test_split_pair_neighboring_chunks(self, cgraph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1━━┿━━2  │  =>  │  1  │  2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """
        pass

    def test_split_pair_disconnected_chunks(self, cgraph):
        """
        Remove edge between existing RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1━━┿━━━━━┿━━2  │  =>  │  1  │     │  2  │
        │     │     │     │      │     │     │     │
        └─────┘     └─────┘      └─────┘     └─────┘
        """
        pass

    def test_split_pair_already_disconnected(self, cgraph):
        """
        Try to remove edge between already disconnected RG supervoxels 1 and 2 (same chunk).
        Expected: No change, no error
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1 2 │
        │     │      │     │
        └─────┘      └─────┘
        """
        pass

    def test_split_full_circle_to_triple_chain_same_chunk(self, cgraph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1 2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """
        pass

    def test_split_full_circle_to_triple_chain_neighboring_chunks(self, cgraph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1━━┿━━2  │  =>  │  1  │  2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """
        pass

    def test_split_full_circle_to_triple_chain_disconnected_chunks(self, cgraph):
        """
        Remove direct edge between RG supervoxels 1 and 2, but leave indirect connection (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1━━┿━━━━━┿━━2  │  =>  │  1  │     │  2  │
        │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
        └─────┘     └─────┘      └─────┘     └─────┘
        """
        pass

    def test_split_same_node(self, cgraph):  # This should fail
        """
        Try to remove (non-existing) edge between RG supervoxel 1 and itself
        ┌─────┐
        │  A¹ │
        │  1  │  =>  Reject
        │     │
        └─────┘
        """
        pass

    def test_split_pair_abstract_nodes(self, cgraph):  # This should fail
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
        pass


class TestGraphMinCut:
    def test_cut_low_affinity_chain(self, cgraph):
        pass

    def test_cut_zero_affinity_chain(self, cgraph):
        pass

    def test_cut_inf_affinity_chain(self, cgraph):
        pass

    def test_cut_no_path(self, cgraph):
        pass


class TestGraphMultiCut:
    def test_cut_multi_tree(self, cgraph):
        pass
