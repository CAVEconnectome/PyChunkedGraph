"""Tests for pychunkedgraph.ingest.create.cross_edges"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from pychunkedgraph.ingest.create.cross_edges import (
    _find_min_layer,
    get_children_chunk_cross_edges,
    get_chunk_nodes_cross_edge_layer,
    _get_chunk_nodes_cross_edge_layer_helper,
)
from pychunkedgraph.graph.utils import basetypes

from .helpers import create_chunk, to_label
from ..ingest.create.parent_layer import add_parent_chunk


class TestFindMinLayer:
    """Pure unit tests for _find_min_layer helper."""

    def test_single_batch(self):
        """One array of node_ids and layers results in correct min layers."""
        node_layer_d = {}
        node_ids_shared = [np.array([10, 20, 30], dtype=basetypes.NODE_ID)]
        node_layers_shared = [np.array([3, 5, 4], dtype=np.uint8)]

        _find_min_layer(node_layer_d, node_ids_shared, node_layers_shared)

        assert node_layer_d[10] == 3
        assert node_layer_d[20] == 5
        assert node_layer_d[30] == 4
        assert len(node_layer_d) == 3

    def test_multiple_batches_min_wins(self):
        """Two batches with the same node_id but different layers; smallest layer wins."""
        node_layer_d = {}
        node_ids_shared = [
            np.array([10, 20], dtype=basetypes.NODE_ID),
            np.array([20, 30], dtype=basetypes.NODE_ID),
        ]
        node_layers_shared = [
            np.array([5, 7], dtype=np.uint8),
            np.array([3, 4], dtype=np.uint8),
        ]

        _find_min_layer(node_layer_d, node_ids_shared, node_layers_shared)

        assert node_layer_d[10] == 5
        # node 20 appears in both batches with layers 7 and 3; min is 3
        assert node_layer_d[20] == 3
        assert node_layer_d[30] == 4

    def test_empty_batches(self):
        """Empty arrays produce an empty dict."""
        node_layer_d = {}
        node_ids_shared = [np.array([], dtype=basetypes.NODE_ID)]
        node_layers_shared = [np.array([], dtype=np.uint8)]

        _find_min_layer(node_layer_d, node_ids_shared, node_layers_shared)

        assert len(node_layer_d) == 0


class TestGetChildrenChunkCrossEdges:
    """Integration tests for get_children_chunk_cross_edges using gen_graph."""

    def test_no_cross_edges(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )

        result = get_children_chunk_cross_edges(graph, 3, [0, 0, 0], use_threads=False)
        # Should return empty or no cross edges
        assert len(result) == 0 or result.size == 0

    def test_with_cross_edges(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )

        result = get_children_chunk_cross_edges(graph, 3, [0, 0, 0], use_threads=False)
        assert len(result) > 0

    @pytest.mark.timeout(30)
    def test_no_atomic_chunks_returns_empty(self, gen_graph):
        """When the chunk coordinate is out of bounds, get_touching_atomic_chunks
        returns empty and the function returns early with an empty list."""
        cg = gen_graph(n_layers=3, atomic_chunk_bounds=np.array([1, 1, 1]))
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)

        # chunk_coord [1,0,0] is out of bounds for atomic_chunk_bounds=[1,1,1]
        # so get_touching_atomic_chunks returns empty, triggering early return
        result = get_children_chunk_cross_edges(
            cg, layer=3, chunk_coord=[1, 0, 0], use_threads=False
        )
        assert len(result) == 0

    @pytest.mark.timeout(30)
    def test_basic_cross_edges(self, gen_graph):
        """A 4-layer graph with cross-chunk connected SVs returns cross edges
        when called with use_threads=False."""
        cg = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        # Chunk A (0,0,0): sv 0 connected cross-chunk to chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )

        # Chunk B (1,0,0): sv 0 connected cross-chunk to chunk A
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )

        # Build parent layer so L3 nodes exist
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)

        # Layer 3, chunk [0,0,0] should have cross edges connecting children chunks
        result = get_children_chunk_cross_edges(
            cg, layer=3, chunk_coord=[0, 0, 0], use_threads=False
        )
        result = np.array(result)
        assert result.size > 0
        assert result.ndim == 2
        assert result.shape[1] == 2


class TestGetChildrenChunkCrossEdgesAdditional:
    """Additional tests for get_children_chunk_cross_edges (serial path)."""

    @pytest.mark.timeout(30)
    def test_multiple_cross_edges(self, gen_graph):
        """Multiple SVs with cross-chunk edges should all be found."""
        cg = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        # Chunk A: two SVs, each cross-chunk connected
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1), 0.5),
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 1, 0, 0, 1), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 1)],
            edges=[
                (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf),
                (to_label(cg, 1, 1, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 1), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)

        result = get_children_chunk_cross_edges(
            cg, layer=3, chunk_coord=[0, 0, 0], use_threads=False
        )
        result = np.array(result)
        assert result.size > 0
        assert result.ndim == 2

    @pytest.mark.timeout(30)
    def test_cross_edges_layer4(self, gen_graph):
        """Cross edges that span L3 chunk boundaries should appear at layer 4.
        The SVs must be on the touching face between L3 children:
        L4 [0,0,0] has L3 children [0,0,0] (x=0,1) and [1,0,0] (x=2,3).
        Touching face is at L2 x=1 and x=2."""
        cg = gen_graph(n_layers=5)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        # SV at L1 [1,0,0] - on the right boundary of L3 [0,0,0]
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        # SV at L1 [2,0,0] - on the left boundary of L3 [1,0,0]
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 2, 0, 0, 0)],
            edges=[
                (to_label(cg, 1, 2, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(cg, 3, [1, 0, 0], n_threads=1)

        # At layer 4, chunk [0,0,0] should find cross edges at the L3 boundary
        result = get_children_chunk_cross_edges(
            cg, layer=4, chunk_coord=[0, 0, 0], use_threads=False
        )
        result = np.array(result)
        assert result.size > 0
        assert result.ndim == 2


class TestGetChunkNodesCrossEdgeLayer:
    """Tests for get_chunk_nodes_cross_edge_layer (lines 112-147)."""

    @pytest.mark.timeout(60)
    def test_no_threads_with_cross_edges(self, gen_graph):
        """use_threads=False should return dict mapping node_id to layer.
        Cross edge between [0,0,0] and [2,0,0] has layer 3.
        get_bounding_atomic_chunks(meta, 3, [0,0,0]) returns L2 boundary
        chunks of L3 [0,0,0], which includes L2 at x=0 with AtomicCrossChunkEdge[3].
        """
        cg = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        # SV at L1 [0,0,0] with cross edge to [2,0,0] (layer-3 cross edge)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 2, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        # SV at L1 [2,0,0]
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 2, 0, 0, 0)],
            edges=[
                (to_label(cg, 1, 2, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(cg, 3, [1, 0, 0], n_threads=1)

        result = get_chunk_nodes_cross_edge_layer(
            cg, layer=3, chunk_coord=[0, 0, 0], use_threads=False
        )
        assert isinstance(result, dict)
        assert len(result) > 0
        for node_id, layer in result.items():
            assert layer >= 3

    @pytest.mark.timeout(60)
    def test_no_threads_empty_chunk(self, gen_graph):
        """use_threads=False with out-of-bounds chunk should return empty dict."""
        cg = gen_graph(n_layers=3, atomic_chunk_bounds=np.array([1, 1, 1]))
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)

        # Out of bounds chunk coord
        result = get_chunk_nodes_cross_edge_layer(
            cg, layer=3, chunk_coord=[1, 0, 0], use_threads=False
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    @pytest.mark.timeout(60)
    def test_no_cross_edges_returns_empty(self, gen_graph):
        """When chunks have no cross edges at the relevant layers, result is empty."""
        cg = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], n_threads=1)

        result = get_chunk_nodes_cross_edge_layer(
            cg, layer=3, chunk_coord=[0, 0, 0], use_threads=False
        )
        assert isinstance(result, dict)
        assert len(result) == 0


class TestFindMinLayerExtended:
    """Additional tests for _find_min_layer with edge cases."""

    def test_single_node_multiple_batches(self):
        """Same node_id across multiple batches; lowest layer wins."""
        node_layer_d = {}
        node_ids_shared = [
            np.array([100], dtype=basetypes.NODE_ID),
            np.array([100], dtype=basetypes.NODE_ID),
            np.array([100], dtype=basetypes.NODE_ID),
        ]
        node_layers_shared = [
            np.array([8], dtype=np.uint8),
            np.array([3], dtype=np.uint8),
            np.array([5], dtype=np.uint8),
        ]

        _find_min_layer(node_layer_d, node_ids_shared, node_layers_shared)
        assert node_layer_d[100] == 3

    def test_no_overlap(self):
        """All unique node_ids across batches should just pass through."""
        node_layer_d = {}
        node_ids_shared = [
            np.array([1, 2], dtype=basetypes.NODE_ID),
            np.array([3, 4], dtype=basetypes.NODE_ID),
        ]
        node_layers_shared = [
            np.array([5, 6], dtype=np.uint8),
            np.array([7, 8], dtype=np.uint8),
        ]

        _find_min_layer(node_layer_d, node_ids_shared, node_layers_shared)
        assert node_layer_d[1] == 5
        assert node_layer_d[2] == 6
        assert node_layer_d[3] == 7
        assert node_layer_d[4] == 8
