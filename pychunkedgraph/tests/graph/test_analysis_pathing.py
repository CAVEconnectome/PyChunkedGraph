"""Tests for pychunkedgraph.graph.analysis.pathing"""

from datetime import datetime, timedelta, UTC
from math import inf
from unittest.mock import MagicMock

import numpy as np
import pytest

from pychunkedgraph.graph.analysis.pathing import (
    get_first_shared_parent,
    get_children_at_layer,
    get_lvl2_edge_list,
    find_l2_shortest_path,
    compute_rough_coordinate_path,
)

from ..helpers import create_chunk, to_label
from ...ingest.create.parent_layer import add_parent_chunk


class TestGetFirstSharedParent:
    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
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
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    def test_same_root(self, gen_graph):
        graph = self._build_graph(gen_graph)
        sv0 = to_label(graph, 1, 0, 0, 0, 0)
        sv1 = to_label(graph, 1, 1, 0, 0, 0)
        parent = get_first_shared_parent(graph, sv0, sv1)
        assert parent is not None
        # The shared parent should be an ancestor of both SVs
        root = graph.get_root(sv0)
        # Verify the shared parent is on the path to root
        assert graph.get_root(parent) == root

    def test_different_roots_returns_none(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        # Create two disconnected chunks
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        sv0 = to_label(graph, 1, 0, 0, 0, 0)
        sv1 = to_label(graph, 1, 1, 0, 0, 0)
        parent = get_first_shared_parent(graph, sv0, sv1)
        assert parent is None


class TestGetChildrenAtLayer:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        children = get_children_at_layer(graph, root, 2)
        assert len(children) > 0
        for child in children:
            assert graph.get_chunk_layer(child) == 2

    def test_allow_lower_layers(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        children = get_children_at_layer(graph, root, 2, allow_lower_layers=True)
        assert len(children) > 0


class TestGetLvl2EdgeList:
    def _build_3chunk_graph(self, gen_graph):
        """Build a graph with 3 chunks A(0,0,0), B(1,0,0), C(2,0,0) connected by cross-chunk edges.

        A:sv0 -- B:sv0 -- C:sv0
        """
        graph = gen_graph(n_layers=4)

        # Chunk A: sv0 connected to B:sv0 via cross-chunk edge
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )

        # Chunk B: sv0 connected to A:sv0 and C:sv0 via cross-chunk edges
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
            ],
        )

        # Chunk C: sv0 connected to B:sv0 via cross-chunk edge
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 2, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )

        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        return graph

    def test_basic(self, gen_graph):
        """get_lvl2_edge_list should return edges between L2 IDs for a connected root."""
        graph = self._build_3chunk_graph(gen_graph)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        edges = get_lvl2_edge_list(graph, root)

        # There should be at least 2 edges: A_l2--B_l2 and B_l2--C_l2
        assert edges.shape[0] >= 2
        assert edges.shape[1] == 2

        # All edge IDs should be L2 nodes (layer 2)
        for edge in edges:
            for node_id in edge:
                assert graph.get_chunk_layer(node_id) == 2

    def test_single_chunk_no_cross_edges(self, gen_graph):
        """A single isolated chunk should produce no L2 edges."""
        graph = gen_graph(n_layers=4)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        edges = get_lvl2_edge_list(graph, root)

        assert edges.shape[0] == 0


class TestFindL2ShortestPath:
    def _build_3chunk_graph(self, gen_graph):
        """Build a graph with 3 chunks A(0,0,0), B(1,0,0), C(2,0,0) connected linearly.

        A:sv0 -- B:sv0 -- C:sv0
        """
        graph = gen_graph(n_layers=4)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
            ],
        )

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 2, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )

        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        return graph

    def test_path_between_endpoints(self, gen_graph):
        """find_l2_shortest_path should return a path from source to target L2 IDs."""
        graph = self._build_3chunk_graph(gen_graph)

        # Get L2 parents of the supervoxels
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_c = to_label(graph, 1, 2, 0, 0, 0)
        l2_a = graph.get_parent(sv_a)
        l2_c = graph.get_parent(sv_c)

        path = find_l2_shortest_path(graph, l2_a, l2_c)

        assert path is not None
        assert len(path) == 3  # A_l2 -> B_l2 -> C_l2
        # Path should start at source and end at target
        assert path[0] == l2_a
        assert path[-1] == l2_c
        # All nodes in path should be layer 2
        for node_id in path:
            assert graph.get_chunk_layer(node_id) == 2

    def test_adjacent_l2_ids(self, gen_graph):
        """find_l2_shortest_path between directly connected L2 IDs should return length 2 path."""
        graph = self._build_3chunk_graph(gen_graph)

        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        l2_a = graph.get_parent(sv_a)
        l2_b = graph.get_parent(sv_b)

        path = find_l2_shortest_path(graph, l2_a, l2_b)

        assert path is not None
        assert len(path) == 2
        assert path[0] == l2_a
        assert path[-1] == l2_b

    def test_disconnected_returns_none(self, gen_graph):
        """find_l2_shortest_path should return None when L2 IDs belong to different roots."""
        graph = gen_graph(n_layers=4)

        # Create two disconnected chunks
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[],
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        l2_a = graph.get_parent(sv_a)
        l2_b = graph.get_parent(sv_b)

        path = find_l2_shortest_path(graph, l2_a, l2_b)
        assert path is None


class TestGetChildrenAtLayerEdgeCases:
    """Test get_children_at_layer with various edge cases."""

    def test_children_at_layer_2_with_multiple_svs(self, gen_graph):
        """Query children at layer 2 when root has multiple SVs in same chunk."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
                to_label(graph, 1, 0, 0, 0, 2),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 1), to_label(graph, 1, 0, 0, 0, 2), 0.5),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        children = get_children_at_layer(graph, root, 2)
        assert len(children) > 0
        for child in children:
            assert graph.get_chunk_layer(child) == 2

    def test_children_at_intermediate_layer(self, gen_graph):
        """Query children at layer 3 from root at layer 4."""
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
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        children = get_children_at_layer(graph, root, 3)
        assert len(children) > 0
        for child in children:
            assert graph.get_chunk_layer(child) == 3

    def test_children_allow_lower_layers_with_cross_chunk(self, gen_graph):
        """Query with allow_lower_layers=True should include layer<=target."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        # Ask for layer 3 with allow_lower_layers=True
        children = get_children_at_layer(graph, root, 3, allow_lower_layers=True)
        assert len(children) > 0
        for child in children:
            assert graph.get_chunk_layer(child) <= 3

    def test_children_at_layer_from_l2_node(self, gen_graph):
        """Querying children at layer 2 from a layer 2 node should return the node itself
        or its layer-2 children (which is itself)."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        sv = to_label(graph, 1, 0, 0, 0, 0)
        l2 = graph.get_parent(sv)
        # From l2, get children at layer 2 (with allow_lower=True since
        # the children of an L2 node are SVs at layer 1)
        children = get_children_at_layer(graph, l2, 2, allow_lower_layers=True)
        assert len(children) > 0


class TestGetLvl2EdgeListWithBbox:
    """Test get_lvl2_edge_list with a bounding box parameter."""

    def _build_3chunk_graph(self, gen_graph):
        """Build a graph with 3 chunks connected linearly."""
        graph = gen_graph(n_layers=4)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
            ],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 2, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )

        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        return graph

    def test_lvl2_edge_list_with_bbox(self, gen_graph):
        """get_lvl2_edge_list with a bbox should return edges within the bbox."""
        graph = self._build_3chunk_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))

        # Use a large bbox that encompasses everything
        bbox = np.array([[0, 0, 0], [2048, 2048, 256]])
        edges = get_lvl2_edge_list(graph, root, bbox=bbox)

        # Should have edges
        assert edges.shape[1] == 2
        # All IDs should be L2 nodes
        for edge in edges:
            for node_id in edge:
                assert graph.get_chunk_layer(node_id) == 2


class TestFindL2ShortestPathEdgeCases:
    """Test find_l2_shortest_path with additional edge cases."""

    def test_path_through_chain(self, gen_graph):
        """find_l2_shortest_path through a 4-chunk chain should return correct length."""
        graph = gen_graph(n_layers=4)

        # Build a 4-chunk chain: A(0,0,0)--B(1,0,0)--C(2,0,0)--D(3,0,0)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
            ],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 2, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
                (to_label(graph, 1, 2, 0, 0, 0), to_label(graph, 1, 3, 0, 0, 0), inf),
            ],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 3, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 3, 0, 0, 0), to_label(graph, 1, 2, 0, 0, 0), inf),
            ],
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_d = to_label(graph, 1, 3, 0, 0, 0)
        l2_a = graph.get_parent(sv_a)
        l2_d = graph.get_parent(sv_d)

        path = find_l2_shortest_path(graph, l2_a, l2_d)
        assert path is not None
        assert len(path) == 4  # A_l2 -> B_l2 -> C_l2 -> D_l2
        assert path[0] == l2_a
        assert path[-1] == l2_d


class TestComputeRoughCoordinatePath:
    """Test compute_rough_coordinate_path returns proper coordinates."""

    def test_basic_coordinate_path(self, gen_graph):
        """compute_rough_coordinate_path should return a list of float32 3D coordinates."""
        graph = gen_graph(n_layers=4)

        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        l2_a = graph.get_parent(sv_a)
        l2_b = graph.get_parent(sv_b)

        path = find_l2_shortest_path(graph, l2_a, l2_b)
        assert path is not None

        # Mock cv methods that CloudVolumeMock doesn't have
        mock_cv = MagicMock()
        mock_cv.mip_voxel_offset = MagicMock(return_value=np.array([0, 0, 0]))
        mock_cv.mip_resolution = MagicMock(return_value=np.array([1, 1, 1]))
        graph.meta._ws_cv = mock_cv

        coordinate_path = compute_rough_coordinate_path(graph, path)
        assert len(coordinate_path) == len(path)
        for coord in coordinate_path:
            assert isinstance(coord, np.ndarray)
            assert coord.dtype == np.float32
            assert len(coord) == 3
