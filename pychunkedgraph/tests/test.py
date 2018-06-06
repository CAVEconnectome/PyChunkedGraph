import sys
import os
import subprocess
import pytest
import numpy as np
from math import inf
from time import sleep
from signal import SIGTERM

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from backend import chunkedgraph # noqa


@pytest.fixture(scope='session', autouse=True)
def bigtable_emulator(request):
    # setup Emulator
    bigtables_emulator = subprocess.Popen(["gcloud", "beta", "emulators", "bigtable", "start"], preexec_fn=os.setsid, stdout=subprocess.PIPE)

    print("Waiting for BigTables Emulator to start up...")
    sleep(5)  # Wait for Emulator to start

    bt_env_init = subprocess.run(["gcloud", "beta", "emulators", "bigtable",  "env-init"], stdout=subprocess.PIPE)
    os.environ["BIGTABLE_EMULATOR_HOST"] = bt_env_init.stdout.decode("utf-8").strip().split('=')[-1]

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

    return graph


class TestCoreUtils:
    def test_serialize_valid_node_id(self):
        assert chunkedgraph.serialize_node_id(0x01102001000000FF) == b'00076596382332354815'


class TestBuildGraph:
    def test_build_single_node(self, cgraph):
        """
        Add single RG node 1 to chunk A
        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """

        # Add Chunk A
        cg2rg = {0x0100000000000000: 1}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.empty((0, 2), dtype=np.uint64)
        cross_edge_affs = np.empty((0, 1), dtype=np.float32)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the RG-to-CG mapping:
        assert chunkedgraph.serialize_node_id(1) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

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
        assert len(rg_id) == 1 and rg_id[0] == 1

        # Check for the one Level 2 node that should have been created.
        # 0x0200000000000000
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0200000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 0
        assert len(children) == 1 and children[0] == 0x0100000000000000

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 3

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
        cg2rg = {0x0100000000000000: 1, 0x0100000000000001: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.array([[0x0100000000000000, 0x0100000000000001]], dtype=np.uint64)
        edge_affs = np.array([0.5], dtype=np.float32, ndmin=2)

        cross_edge_ids = np.empty((0, 2), dtype=np.uint64)
        cross_edge_affs = np.empty((0, 1), dtype=np.float32)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        assert chunkedgraph.serialize_node_id(1) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        assert chunkedgraph.serialize_node_id(2) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(2)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000001

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
        assert len(rg_id) == 1 and rg_id[0] == 1

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
        assert len(rg_id) == 1 and rg_id[0] == 2

        # Check for the one Level 2 node that should have been created.
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(0x0200000000000000)].cells[cgraph.family_id]
        atomic_cross_edges = np.frombuffer(row[b'atomic_cross_edges'][0].value, np.uint64).reshape(-1, 2)
        children = np.frombuffer(row[b'children'][0].value, np.uint64)

        assert len(atomic_cross_edges) == 0
        assert len(children) == 2 and 0x0100000000000000 in children and 0x0100000000000001 in children

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 5

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
        cg2rg = {0x0100000000000000: 1, 0x0101000000000000: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.array([[0x0100000000000000, 0x0101000000000000]], dtype=np.uint64)
        cross_edge_affs = np.array([inf], dtype=np.float32, ndmin=2)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        # Chunk B
        cg2rg = {0x0100000000000000: 1, 0x0101000000000000: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.array([[0x0101000000000000, 0x0100000000000000]], dtype=np.uint64)
        cross_edge_affs = np.array([inf], dtype=np.float32, ndmin=2)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        assert chunkedgraph.serialize_node_id(1) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        assert chunkedgraph.serialize_node_id(2) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(2)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0101000000000000

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
        assert len(rg_id) == 1 and rg_id[0] == 1

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
        assert len(rg_id) == 1 and rg_id[0] == 2

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
        assert len(res.rows) == 7

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
        edge_ids = np.array([[0x0100000000000000, 0x0100000000000001]], dtype=np.uint64)
        edge_affs = np.array([0.5], dtype=np.float32, ndmin=2)

        cross_edge_ids = np.array([[0x0100000000000000, 0x0101000000000000]], dtype=np.uint64)
        cross_edge_affs = np.array([inf], dtype=np.float32, ndmin=2)

        cg2rg = {0x0100000000000000: 1, 0x0100000000000001: 2, 0x0101000000000000: 3}
        rg2cg = {v: k for k, v in cg2rg.items()}

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        # Chunk B
        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.array([[0x0101000000000000, 0x0100000000000000]], dtype=np.uint64)
        cross_edge_affs = np.array([inf], dtype=np.float32, ndmin=2)

        cg2rg = {0x0100000000000000: 1, 0x0101000000000000: 3}
        rg2cg = {v: k for k, v in cg2rg.items()}

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the three RG-to-CG mappings:
        assert chunkedgraph.serialize_node_id(1) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(1)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000000

        assert chunkedgraph.serialize_node_id(2) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(2)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0100000000000001

        assert chunkedgraph.serialize_node_id(3) in res.rows
        row = res.rows[chunkedgraph.serialize_node_id(3)].cells[cgraph.family_id]
        assert np.frombuffer(row[b'cg_id'][0].value, np.uint64)[0] == 0x0101000000000000

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
        assert len(rg_id) == 1 and rg_id[0] == 1

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
        assert len(rg_id) == 1 and rg_id[0] == 2

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
        assert len(rg_id) == 1 and rg_id[0] == 3

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
        assert len(res.rows) == 9


class TestGraphMerge:
    def test_merge_pair_same_chunk(self, cgraph):
        """
        Add edge between existing RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        pass

    def test_merge_pair_neighboring_chunks(self, cgraph):
        """
        Add edge between existing RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │     │     │      │     │     │
        └─────┴─────┘      └─────┴─────┘
        """

        pass

    def test_merge_pair_disconnected_chunks(self, cgraph):
        """
        Add edge between existing RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │     │     │     │      │     │     │     │
        └─────┘     └─────┘      └─────┘     └─────┘
        """

        pass

    def test_merge_pair_already_connected(self, cgraph):
        """
        Add edge between already connected RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1━2 │  =>  │ 1━2 │
        │     │      │     │
        └─────┘      └─────┘
        """

        # Preparation: Build Chunk A
        cg2rg = {0x0100000000000000: 1, 0x0100000000000001: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.array([[0x0100000000000000, 0x0100000000000001]], dtype=np.uint64)
        edge_affs = np.array([0.5], dtype=np.float32, ndmin=2)

        cross_edge_ids = np.empty((0, 2), dtype=np.uint64)
        cross_edge_affs = np.empty((0, 1), dtype=np.float32)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        # Merge
        assert cgraph.add_edge([0x0100000000000001, 0x0100000000000000], is_cg_id=True) == 0x020000000

        # Check
        res = cgraph.table.read_rows()
        res.consume_all()

    def test_merge_triple_chain_to_full_circle_same_chunk(self, cgraph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (same chunk)
        ┌─────┐      ┌─────┐
        │  A¹ │      │  A¹ │
        │ 1 2 │  =>  │ 1━2 │
        │ ┗3┛ │      │ ┗3┛ │
        └─────┘      └─────┘
        """
        pass

    def test_merge_triple_chain_to_full_circle_neighboring_chunks(self, cgraph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (neighboring chunks)
        ┌─────┬─────┐      ┌─────┬─────┐
        │  A¹ │  B¹ │      │  A¹ │  B¹ │
        │  1  │  2  │  =>  │  1━━┿━━2  │
        │  ┗3━┿━━┛  │      │  ┗3━┿━━┛  │
        └─────┴─────┘      └─────┴─────┘
        """
        pass

    def test_merge_triple_chain_to_full_circle_disconnected_chunks(self, cgraph):
        """
        Add edge between indirectly connected RG supervoxels 1 and 2 (disconnected chunks)
        ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐
        │  A¹ │ ... │  Z¹ │      │  A¹ │ ... │  Z¹ │
        │  1  │     │  2  │  =>  │  1━━┿━━━━━┿━━2  │
        │  ┗3━┿━━━━━┿━━┛  │      │  ┗3━┿━━━━━┿━━┛  │
        └─────┘     └─────┘      └─────┘     └─────┘
        """
        pass

    def test_merge_pair_abstract_nodes(self, cgraph):  # This should fail
        """
        Add edge between RG supervoxel 1 and abstract node "2".
                     ┌─────┐
                     │  B² │
                     │ "2" │
                     │     │
                     └─────┘
        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        pass


class TestGraphSplit:
    def test_split_pair_same_chunk(self, cgraph):
        pass

    def test_split_pair_neighboring_chunks(self, cgraph):
        pass

    def test_split_pair_disconnected_chunks(self, cgraph):
        pass

    def test_split_pair_already_disconnected(self, cgraph):
        pass

    def test_split_full_circle_to_triple_chain_same_chunk(self, cgraph):
        pass

    def test_split_full_circle_to_triple_chain_neighboring_chunks(self, cgraph):
        pass

    def test_split_full_circle_to_triple_chain_disconnected_chunks(self, cgraph):
        pass

    def test_split_pair_abstract_nodes(self, cgraph):  # This should fail
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
