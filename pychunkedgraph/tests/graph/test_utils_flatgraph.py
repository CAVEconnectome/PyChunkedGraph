"""Tests for pychunkedgraph.graph.utils.flatgraph"""

import numpy as np

from pychunkedgraph.graph.utils.flatgraph import (
    build_gt_graph,
    connected_components,
    remap_ids_from_graph,
    neighboring_edges,
    harmonic_mean_paths,
    remove_overlapping_edges,
    check_connectedness,
    adjust_affinities,
    flatten_edge_list,
    team_paths_all_to_all,
)


class TestBuildGtGraph:
    def test_directed(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=True)
        assert graph.is_directed()
        assert graph.num_vertices() == 3
        assert graph.num_edges() == 2
        assert cap is None

    def test_undirected(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        assert not graph.is_directed()
        assert graph.num_vertices() == 3

    def test_with_weights(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        weights = np.array([0.5, 0.9])
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, weights=weights)
        assert cap is not None

    def test_make_directed(self):
        edges = np.array([[0, 1]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, make_directed=True)
        assert graph.is_directed()
        # make_directed doubles edges (forward + reverse)
        assert graph.num_edges() == 2

    def test_unique_ids_remapping(self):
        # Non-contiguous node IDs
        edges = np.array([[100, 200], [200, 300]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges)
        np.testing.assert_array_equal(unique_ids, [100, 200, 300])


class TestConnectedComponents:
    def test_two_components(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.uint64)
        graph, _, _, _ = build_gt_graph(edges, is_directed=False)
        ccs = connected_components(graph)
        assert len(ccs) == 2

    def test_single_component(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        graph, _, _, _ = build_gt_graph(edges, is_directed=False)
        ccs = connected_components(graph)
        assert len(ccs) == 1


class TestRemapIdsFromGraph:
    def test_basic(self):
        unique_ids = np.array([100, 200, 300], dtype=np.uint64)
        graph_ids = np.array([0, 2])
        result = remap_ids_from_graph(graph_ids, unique_ids)
        np.testing.assert_array_equal(result, [100, 300])


class TestNeighboringEdges:
    def test_basic(self):
        """Build graph 0-1-2 (undirected), neighboring_edges(graph, 1) returns neighbors of vertex 1."""
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        add_v, add_e, weights = neighboring_edges(graph, 1)
        # Should return one list of vertices and one list of edges
        assert len(add_v) == 1
        assert len(add_e) == 1
        # Vertex 1 has two neighbors (0 and 2) in undirected graph
        neighbor_ids = sorted([int(v) for v in add_v[0]])
        assert len(neighbor_ids) == 2
        assert 0 in neighbor_ids
        assert 2 in neighbor_ids
        # Should return edges corresponding to those neighbors
        assert len(add_e[0]) == 2
        # Weights is always [1]
        assert weights == [1]

    def test_isolated_vertex(self):
        """A vertex with no out-neighbors returns empty lists."""
        # Build a directed graph: 0->1. Vertex 1 has no out-neighbors.
        edges = np.array([[0, 1]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=True)
        add_v, add_e, weights = neighboring_edges(graph, 1)
        assert len(add_v) == 1
        assert len(add_v[0]) == 0
        assert len(add_e) == 1
        assert len(add_e[0]) == 0


class TestHarmonicMeanPaths:
    def test_two_values(self):
        """harmonic_mean_paths([4, 16]) should return geometric mean = 8.0"""
        result = harmonic_mean_paths([4, 16])
        assert result == 8.0

    def test_single_value(self):
        """harmonic_mean_paths([9]) should return 9.0"""
        result = harmonic_mean_paths([9])
        assert result == 9.0


class TestRemoveOverlappingEdges:
    def test_no_overlap(self):
        """Two path sets with no shared vertices return the same edges, do_check=False."""
        # Build two separate graphs: 0-1 and 2-3
        edges = np.array([[0, 1], [2, 3]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        # Paths for "team s": vertex 0 and vertex 1 with edge 0-1
        v0 = graph.vertex(0)
        v1 = graph.vertex(1)
        e01 = graph.edge(0, 1)
        paths_v_s = [[v0, v1]]
        paths_e_s = [[e01]]

        # Paths for "team y": vertex 2 and vertex 3 with edge 2-3
        v2 = graph.vertex(2)
        v3 = graph.vertex(3)
        e23 = graph.edge(2, 3)
        paths_v_y = [[v2, v3]]
        paths_e_y = [[e23]]

        out_s, out_y, do_check = remove_overlapping_edges(
            paths_v_s, paths_e_s, paths_v_y, paths_e_y
        )
        # No overlap, so do_check is False
        assert do_check is False
        # Original edges returned unchanged
        assert out_s == paths_e_s
        assert out_y == paths_e_y

    def test_with_overlap(self):
        """Paths sharing some vertices cause overlapping edges to be removed, do_check=True."""
        # Build a linear graph: 0-1-2-3 (undirected)
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        # Team s path: 0-1-2 (shares vertex 1 and 2)
        v0 = graph.vertex(0)
        v1 = graph.vertex(1)
        v2 = graph.vertex(2)
        v3 = graph.vertex(3)
        e01 = graph.edge(0, 1)
        e12 = graph.edge(1, 2)
        e23 = graph.edge(2, 3)

        paths_v_s = [[v0, v1, v2]]
        paths_e_s = [[e01, e12]]

        # Team y path: 1-2-3 (shares vertex 1 and 2)
        paths_v_y = [[v1, v2, v3]]
        paths_e_y = [[e12, e23]]

        out_s, out_y, do_check = remove_overlapping_edges(
            paths_v_s, paths_e_s, paths_v_y, paths_e_y
        )
        assert do_check is True
        # Overlapping vertices are 1 and 2
        # Edges touching vertices 1 or 2 should be removed
        # All edges in both paths touch vertex 1 or 2, so both should be empty
        assert len(out_s[0]) == 0
        assert len(out_y[0]) == 0


class TestCheckConnectedness:
    def test_connected(self):
        """A connected set of edges returns True."""
        # Build a connected graph: 0-1-2
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        v0 = graph.vertex(0)
        v1 = graph.vertex(1)
        v2 = graph.vertex(2)
        e01 = graph.edge(0, 1)
        e12 = graph.edge(1, 2)

        vertices = [[v0, v1, v2]]
        edge_list = [[e01, e12]]

        assert check_connectedness(vertices, edge_list, expected_number=1) is True

    def test_disconnected(self):
        """A disconnected set returns False (more than expected_number components)."""
        # Build a graph with two disconnected components: 0-1 and 2-3
        edges = np.array([[0, 1], [2, 3]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        v0 = graph.vertex(0)
        v1 = graph.vertex(1)
        v2 = graph.vertex(2)
        v3 = graph.vertex(3)
        e01 = graph.edge(0, 1)
        e23 = graph.edge(2, 3)

        # Include all vertices but edges that form two components
        vertices = [[v0, v1, v2, v3]]
        edge_list = [[e01, e23]]

        # Expecting 1 component but there are 2, so should return False
        assert check_connectedness(vertices, edge_list, expected_number=1) is False


class TestAdjustAffinities:
    def test_basic(self):
        """Build a graph with known capacities, adjust a subset, verify capacities changed."""
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)
        weights = np.array([0.5, 0.8])
        graph, cap, g_edges, unique_ids = build_gt_graph(
            edges, weights=weights, make_directed=True
        )
        assert cap is not None

        # Get the edge 0->1 and adjust its affinity
        e01 = graph.edge(0, 1)
        original_cap_01 = cap[e01]
        assert original_cap_01 == 0.5

        paths_e = [[e01]]
        new_cap = adjust_affinities(graph, cap, paths_e, value=999.0)

        # The original capacity should be unchanged (adjust_affinities copies)
        assert cap[e01] == 0.5
        # The new capacity for the adjusted edge should be 999.0
        assert new_cap[e01] == 999.0
        # The reverse edge should also be adjusted
        e10 = graph.edge(1, 0)
        assert new_cap[e10] == 999.0
        # Edge 1->2 should be unchanged
        e12 = graph.edge(1, 2)
        assert new_cap[e12] == 0.8


class TestFlattenEdgeList:
    def test_basic(self):
        """Flatten a list of graph-tool edges to unique vertex indices."""
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint64)
        graph, cap, g_edges, unique_ids = build_gt_graph(edges, is_directed=False)
        e01 = graph.edge(0, 1)
        e12 = graph.edge(1, 2)
        e23 = graph.edge(2, 3)

        paths_e = [[e01, e12], [e23]]
        result = flatten_edge_list(paths_e)
        # Should contain unique vertex indices from all edges
        assert isinstance(result, np.ndarray)
        assert set(result.tolist()) == {0, 1, 2, 3}
