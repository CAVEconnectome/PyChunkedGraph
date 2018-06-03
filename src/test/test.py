import sys
import os
import subprocess
import pytest
import numpy as np
from math import inf
from time import sleep
from signal import SIGTERM

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from pychunkedgraph import chunkedgraph # noqa


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


class TestChunkedGraph:
    def test_serialize_valid_node_id(self):
        assert chunkedgraph.serialize_node_id(0x01102001000000FF) == b'00076596382332354815'

    def test_add_single_edge(self, cgraph):
        """
        Add edge between RG supervoxels 1 and 2 (same chunk)
        +-----+
        |  A  |
        | 1-2 |
        |     |
        +-----+
        """

        # Add Chunk A
        cg2rg = {0x0100000000000000: 1, 0x0100000000000001: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.array([[0x0100000000000000, 0x0100000000000001]])
        edge_affs = np.array([0.5], ndmin=2)

        cross_edge_ids = np.empty((0, 2), np.uint64)
        cross_edge_affs = np.empty((0, 1), np.float32)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        assert chunkedgraph.serialize_node_id(1) in res.rows
        assert chunkedgraph.serialize_node_id(2) in res.rows

        # Check for the two original Level 1 CG supervoxels
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0100000000000001) in res.rows

        # Check for the one Level 2 node that should have been created.
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 5

    def test_add_single_across_edge(self, cgraph):
        """
        Add edge between RG supervoxels 1 and 2 (neighboring chunks)
        +-----+-----+
        |  A  |  B  |
        |   1-|-2   |
        |     |     |
        +-----+-----+
        """

        # Chunk A
        cg2rg = {0x0100000000000000: 1, 0x0101000000000000: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.array([[0x0100000000000000, 0x0101000000000000]])
        cross_edge_affs = np.array([inf], ndmin=2)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        # Chunk B
        cg2rg = {0x0100000000000000: 1, 0x0101000000000000: 2}
        rg2cg = {v: k for k, v in cg2rg.items()}

        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.array([[0x0101000000000000, 0x0100000000000000]])
        cross_edge_affs = np.array([inf], ndmin=2)

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        cgraph.add_layer(3, np.array([[0, 0, 0], [1, 0, 0]]))

        res = cgraph.table.read_rows()
        res.consume_all()

        # Check for the two RG-to-CG mappings:
        assert chunkedgraph.serialize_node_id(1) in res.rows
        assert chunkedgraph.serialize_node_id(2) in res.rows

        # Check for the two original Level 1 CG supervoxels
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0101000000000000) in res.rows

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0201000000000000) in res.rows

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        assert chunkedgraph.serialize_node_id(0x0300000000000000) in res.rows

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 7

    def test_add_single_edge_and_single_across_edge(self, cgraph):
        """
        Add edge between RG supervoxels 1 and 2 (same chunk)
        Add edge between RG supervoxels 1 and 3 (neighboring chunks)
        +-----+-----+
        |  A  |  B  |
        | 2-1-|-3   |
        |     |     |
        +-----+-----+
        """

        # Chunk A
        edge_ids = np.array([[0x0100000000000000, 0x0100000000000001]])
        edge_affs = np.array([0.5], ndmin=2)

        cross_edge_ids = np.array([[0x0100000000000000, 0x0101000000000000]])
        cross_edge_affs = np.array([inf], ndmin=2)

        cg2rg = {0x0100000000000000: 1, 0x0100000000000001: 2, 0x0101000000000000: 3}
        rg2cg = {v: k for k, v in cg2rg.items()}

        cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                          edge_affs, cross_edge_affs,
                                          cg2rg, rg2cg)

        # Chunk B
        edge_ids = np.empty((0, 2), np.uint64)
        edge_affs = np.empty((0, 1), np.float32)

        cross_edge_ids = np.array([[0x0101000000000000, 0x0100000000000000]])
        cross_edge_affs = np.array([inf], ndmin=2)

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
        assert chunkedgraph.serialize_node_id(2) in res.rows
        assert chunkedgraph.serialize_node_id(3) in res.rows

        # Check for the three original Level 1 CG supervoxels
        assert chunkedgraph.serialize_node_id(0x0100000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0100000000000001) in res.rows
        assert chunkedgraph.serialize_node_id(0x0101000000000000) in res.rows

        # Check for the two Level 2 nodes that should have been created. Since Level 2 has the same
        # dimensions as Level 1, we also expect them to be in different chunks
        assert chunkedgraph.serialize_node_id(0x0200000000000000) in res.rows
        assert chunkedgraph.serialize_node_id(0x0201000000000000) in res.rows

        # Check for the one Level 3 node that should have been created. This one combines the two
        # connected components of Level 2
        assert chunkedgraph.serialize_node_id(0x0300000000000000) in res.rows

        # Make sure there are not any more entries in the table
        assert len(res.rows) == 9
