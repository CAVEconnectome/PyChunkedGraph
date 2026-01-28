"""
Tests specifically for HBase backend implementation.
These tests verify that the HBase client works correctly.

Note: These tests require HBase to be running. They will be skipped if HBase is not available.
For coverage verification, run with: BACKEND_TYPE=hbase pytest --cov=pychunkedgraph/graph/client/hbase
"""

import pytest
import numpy as np
from datetime import timedelta

try:
    import happybase
    HBASE_AVAILABLE = True
except ImportError:
    HBASE_AVAILABLE = False

from .helpers import gen_graph, create_chunk, to_label, hbase_server
from ..graph.chunkedgraph import ChunkedGraph
from ..graph import attributes


@pytest.mark.skipif(not HBASE_AVAILABLE, reason="happybase not available")
class TestHBaseBackend:
    """Test HBase backend implementation."""
    
    # hbase_server fixture is imported from helpers and will be automatically used
    # when tests request it as a parameter
    
    def test_hbase_client_creation(self, hbase_server, gen_graph):
        """Test that HBase client can be created and graph can be initialized."""
        graph = gen_graph(backend="hbase", n_layers=4)
        assert graph is not None
        assert graph.client is not None
        # Verify it's an HBase client
        from ..graph.client.hbase import HBaseClient
        assert isinstance(graph.client, HBaseClient)
    
    def test_hbase_read_write_nodes(self, hbase_server, gen_graph):
        """Test reading and writing nodes with HBase backend."""
        graph = gen_graph(backend="hbase", n_layers=4)
        
        # Create a simple chunk
        node_id = to_label(graph, 1, 0, 0, 0, 0)
        create_chunk(graph, vertices=[node_id], edges=[])
        
        # Read the node back
        nodes = graph.client.read_nodes(node_ids=[node_id])
        assert node_id in nodes
        assert len(nodes[node_id]) > 0
    
    def test_hbase_read_node_range(self, hbase_server, gen_graph):
        """Test reading a range of nodes with HBase backend."""
        graph = gen_graph(backend="hbase", n_layers=4)
        
        # Create nodes
        node1 = to_label(graph, 1, 0, 0, 0, 0)
        node2 = to_label(graph, 1, 0, 0, 0, 1)
        create_chunk(graph, vertices=[node1, node2], edges=[])
        
        # Read range
        nodes = graph.client.read_nodes(start_id=node1, end_id=node2)
        assert len(nodes) >= 2
    
    def test_hbase_non_contiguous_read(self, hbase_server, gen_graph):
        """Test reading non-contiguous row sets with HBase backend."""
        graph = gen_graph(backend="hbase", n_layers=4)
        
        # Create multiple non-contiguous nodes
        node_ids = [
            to_label(graph, 1, 0, 0, 0, 0),
            to_label(graph, 1, 1, 0, 0, 0),
            to_label(graph, 1, 2, 0, 0, 0),
        ]
        for node_id in node_ids:
            create_chunk(graph, vertices=[node_id], edges=[])
        
        # Read non-contiguous set
        nodes = graph.client.read_nodes(node_ids=node_ids)
        assert len(nodes) == len(node_ids)
        for node_id in node_ids:
            assert node_id in nodes
    
    def test_hbase_lock_operations(self, hbase_server, gen_graph):
        """Test locking operations with HBase backend."""
        graph = gen_graph(backend="hbase", n_layers=4)
        
        # Create a root node
        root_id = to_label(graph, 4, 0, 0, 0, 0)
        create_chunk(graph, vertices=[root_id], edges=[])
        
        # Test lock
        operation_id = graph.client.create_operation_id()
        locked = graph.client.lock_root(root_id, operation_id)
        assert locked is True
        
        # Test unlock
        unlocked = graph.client.unlock_root(root_id, operation_id)
        assert unlocked is True
    
    def test_hbase_id_generation(self, hbase_server, gen_graph):
        """Test ID generation with HBase backend."""
        graph = gen_graph(backend="hbase", n_layers=4)
        
        chunk_id = graph.get_chunk_id(layer=1, x=0, y=0, z=0)
        
        # Generate node IDs
        node_ids = graph.client.create_node_ids(chunk_id, size=5)
        assert len(node_ids) == 5
        assert all(isinstance(nid, np.uint64) for nid in node_ids)
        
        # Check max node ID
        max_id = graph.client.get_max_node_id(chunk_id)
        assert max_id is not None
    
    def test_hbase_graph_meta(self, hbase_server, gen_graph):
        """Test graph metadata operations with HBase backend."""
        graph = gen_graph(backend="hbase", n_layers=4)
        
        # Read version
        version = graph.client.read_graph_version()
        assert version is not None
        
        # Read meta
        meta = graph.client.read_graph_meta()
        assert meta is not None
        assert meta.graph_config.CHUNK_SIZE == [512, 512, 64]

