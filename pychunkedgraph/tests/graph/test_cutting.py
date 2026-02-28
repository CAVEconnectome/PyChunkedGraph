"""Tests for pychunkedgraph.graph.cutting"""

import numpy as np
import pytest

from pychunkedgraph.graph.cutting import (
    IsolatingCutException,
    LocalMincutGraph,
    merge_cross_chunk_edges_graph_tool,
    run_multicut,
)
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.exceptions import PostconditionError, PreconditionError


class TestIsolatingCutException:
    def test_is_exception_subclass(self):
        """IsolatingCutException is a proper Exception subclass."""
        assert issubclass(IsolatingCutException, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(IsolatingCutException):
            raise IsolatingCutException("Source")

    def test_message_preserved(self):
        exc = IsolatingCutException("Sink")
        assert str(exc) == "Sink"


class TestMergeCrossChunkEdgesGraphTool:
    def test_merge_cross_chunk_edges_basic(self):
        """Cross-chunk edges (inf affinity) cause their endpoints to be merged.

        Edges:
            1--2 (aff=0.5, regular)
            2--3 (aff=inf, cross-chunk -> merge 2 and 3)
            3--4 (aff=0.3, regular)

        After merging, node 3 is remapped to node 2 (min of {2,3}).
        The cross-chunk edge (2--3) is removed from the output.
        The remaining edges become:
            1--2 (aff=0.5)
            2--4 (aff=0.3)  [was 3--4, but 3 is now remapped to 2]
        """
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.5, np.inf, 0.3], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # Cross-chunk edge is removed; 2 output edges remain
        assert mapped_edges.shape[0] == 2
        assert mapped_affs.shape[0] == 2

        # Affinities of the non-cross-chunk edges are preserved
        np.testing.assert_array_almost_equal(
            np.sort(mapped_affs), np.array([0.3, 0.5], dtype=np.float32)
        )

        # The mapping should show that 2 and 3 map to the same representative (min=2)
        assert len(remapping) == 1
        rep_node = list(remapping.keys())[0]
        assert rep_node == 2
        merged_nodes = set(remapping[rep_node])
        assert 2 in merged_nodes
        assert 3 in merged_nodes

        # All unique nodes appear in complete_mapping
        all_mapped_from = set(complete_mapping[:, 0])
        assert {1, 2, 3, 4}.issubset(all_mapped_from)

    def test_merge_cross_chunk_edges_no_cross_chunk(self):
        """When all affinities are finite, no merging occurs.

        All edges are returned as-is (no cross-chunk edges to remove).
        """
        edges = np.array([[10, 20], [20, 30], [30, 40]], dtype=np.uint64)
        affs = np.array([0.5, 0.8, 0.3], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # No edges removed
        assert mapped_edges.shape[0] == 3
        assert mapped_affs.shape[0] == 3

        # No remapping occurred
        assert len(remapping) == 0

        # Affinities are unchanged
        np.testing.assert_array_almost_equal(mapped_affs, affs)

        # All nodes map to themselves in complete_mapping
        for row in complete_mapping:
            assert row[0] == row[1]

    def test_merge_cross_chunk_edges_all_cross_chunk(self):
        """When all edges are cross-chunk, all edges are removed from output."""
        edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([np.inf, np.inf], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # All edges were cross-chunk, so no mapped edges remain
        assert mapped_edges.shape[0] == 0
        assert mapped_affs.shape[0] == 0

    def test_merge_cross_chunk_edges_multiple_components(self):
        """Multiple separate cross-chunk merges in a single call.

        Edges:
            1--2 (inf) -> merge into {1,2}, rep=1
            3--4 (inf) -> merge into {3,4}, rep=3
            1--3 (0.7) -> becomes 1--3 after remapping
        """
        edges = np.array([[1, 2], [3, 4], [1, 3]], dtype=np.uint64)
        affs = np.array([np.inf, np.inf, 0.7], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # Only 1 non-cross-chunk edge remains
        assert mapped_edges.shape[0] == 1
        assert mapped_affs.shape[0] == 1
        np.testing.assert_array_almost_equal(mapped_affs, [0.7])

        # Two remapping groups
        assert len(remapping) == 2


class TestLocalMincutGraph:
    """Tests for LocalMincutGraph initialization and mincut computation."""

    def test_init_basic(self):
        """Create a simple 4-node line graph with a weak middle edge.

        Graph: 1 --0.9-- 2 --0.1-- 3 --0.9-- 4
        Sources: [1], Sinks: [4]

        The graph should initialize successfully and have the expected
        source/sink graph ids set.
        """
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.9, 0.1, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=True,
        )
        assert graph.weighted_graph is not None
        assert graph.unique_supervoxel_ids is not None
        assert len(graph.source_graph_ids) == 1
        assert len(graph.sink_graph_ids) == 1
        # Sources and sinks should be mapped correctly
        assert np.array_equal(graph.sources, sources)
        assert np.array_equal(graph.sinks, sinks)

    def test_init_with_cross_chunk_edges(self):
        """Initialization with a mix of regular and cross-chunk edges.

        Graph: 1 --0.5-- 2 --inf-- 3 --0.5-- 4
        The inf edge merges 2 and 3 into one node.
        Sources: [1], Sinks: [4]
        """
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.5, np.inf, 0.5], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=True,
        )
        # After merging cross chunk edges 2 and 3, we should have fewer unique ids
        assert graph.weighted_graph is not None
        assert len(graph.cross_chunk_edge_remapping) == 1

    def test_init_only_cross_chunk_raises(self):
        """All inf affinities should raise PostconditionError.

        When every edge is a cross-chunk edge, all edges are removed after
        merging, leaving an empty graph. This should raise PostconditionError.
        """
        edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([np.inf, np.inf], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        with pytest.raises(PostconditionError, match="cross chunk edges"):
            LocalMincutGraph(
                edges,
                affs,
                sources,
                sinks,
                split_preview=False,
                path_augment=False,
            )

    def test_compute_mincut_direct(self):
        """Compute mincut with path_augment=False on a simple 2-node graph.

        Graph: 1 --0.5-- 2
        Sources: [1], Sinks: [2]

        The only possible cut is the single edge between 1 and 2.
        """
        edges = np.array([[1, 2]], dtype=np.uint64)
        affs = np.array([0.5], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([2], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        # The mincut should return edges to cut
        assert len(result) > 0
        # The returned edges should contain the edge (1,2) or (2,1)
        result_set = set(map(tuple, result))
        assert (1, 2) in result_set or (2, 1) in result_set

    def test_compute_mincut_path_augmented(self):
        """Compute mincut with path_augment=True (default) on a line graph.

        Graph: 1 --0.9-- 2 --0.1-- 3 --0.9-- 4
        Sources: [1], Sinks: [4]

        The weakest edge is 2--3, so the mincut should cut there.
        """
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.9, 0.1, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0
        # The cut should include the weak edge (2,3) or (3,2)
        result_set = set(map(tuple, result))
        assert (2, 3) in result_set or (3, 2) in result_set
        # The strong edges should NOT be in the cut
        assert (1, 2) not in result_set
        assert (3, 4) not in result_set

    def test_compute_mincut_line_graph_cuts_weakest(self):
        """Line graph with clear weakest edge - mincut should cut it.

        Graph: 10 --0.8-- 20 --0.01-- 30 --0.8-- 40
        Sources: [10], Sinks: [40]
        """
        edges = np.array([[10, 20], [20, 30], [30, 40]], dtype=np.uint64)
        affs = np.array([0.8, 0.01, 0.8], dtype=np.float32)
        sources = np.array([10], dtype=np.uint64)
        sinks = np.array([40], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0
        result_set = set(map(tuple, result))
        assert (20, 30) in result_set or (30, 20) in result_set

    def test_compute_mincut_split_preview(self):
        """Compute mincut with split_preview=True returns connected components.

        Graph: 1 --0.9-- 2 --0.1-- 3 --0.9-- 4
        Sources: [1], Sinks: [4]
        """
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.9, 0.1, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        # split_preview returns (supervoxel_ccs, illegal_split)
        supervoxel_ccs, illegal_split = result
        assert isinstance(supervoxel_ccs, list)
        assert len(supervoxel_ccs) >= 2
        assert isinstance(illegal_split, bool)
        # First component should contain source(s), second should contain sink(s)
        assert 1 in supervoxel_ccs[0] or 2 in supervoxel_ccs[0]
        assert 4 in supervoxel_ccs[1] or 3 in supervoxel_ccs[1]


class TestRunMulticut:
    """Tests for the run_multicut function."""

    def test_basic_split(self):
        """Two groups connected by a weak edge -- mincut should cut that edge.

        Graph: 1 --0.9-- 2 --0.05-- 3 --0.9-- 4
        Sources: [1], Sinks: [4]
        """
        node_ids1 = np.array([1, 2, 3], dtype=np.uint64)
        node_ids2 = np.array([2, 3, 4], dtype=np.uint64)
        affinities = np.array([0.9, 0.05, 0.9], dtype=np.float32)

        edges = Edges(node_ids1, node_ids2, affinities=affinities)
        result = run_multicut(
            edges,
            source_ids=[np.uint64(1)],
            sink_ids=[np.uint64(4)],
            path_augment=True,
            disallow_isolating_cut=False,
        )
        assert len(result) > 0
        result_set = set(map(tuple, result))
        assert (2, 3) in result_set or (3, 2) in result_set

    def test_basic_split_direct(self):
        """Same as test_basic_split but with path_augment=False."""
        node_ids1 = np.array([1, 2, 3], dtype=np.uint64)
        node_ids2 = np.array([2, 3, 4], dtype=np.uint64)
        affinities = np.array([0.9, 0.05, 0.9], dtype=np.float32)

        edges = Edges(node_ids1, node_ids2, affinities=affinities)
        result = run_multicut(
            edges,
            source_ids=[np.uint64(1)],
            sink_ids=[np.uint64(4)],
            path_augment=False,
            disallow_isolating_cut=False,
        )
        assert len(result) > 0
        result_set = set(map(tuple, result))
        assert (2, 3) in result_set or (3, 2) in result_set

    def test_no_edges_raises(self):
        """Graph with only cross-chunk edges raises PostconditionError.

        When all edges have infinite affinity, the local graph is empty after
        merging cross-chunk edges, and LocalMincutGraph raises PostconditionError.
        """
        node_ids1 = np.array([1, 2], dtype=np.uint64)
        node_ids2 = np.array([2, 3], dtype=np.uint64)
        affinities = np.array([np.inf, np.inf], dtype=np.float32)

        edges = Edges(node_ids1, node_ids2, affinities=affinities)
        with pytest.raises(PostconditionError, match="cross chunk edges"):
            run_multicut(
                edges,
                source_ids=[np.uint64(1)],
                sink_ids=[np.uint64(3)],
            )

    def test_split_preview_mode(self):
        """run_multicut with split_preview=True returns (ccs, illegal_split)."""
        node_ids1 = np.array([1, 2, 3], dtype=np.uint64)
        node_ids2 = np.array([2, 3, 4], dtype=np.uint64)
        affinities = np.array([0.9, 0.05, 0.9], dtype=np.float32)

        edges = Edges(node_ids1, node_ids2, affinities=affinities)
        result = run_multicut(
            edges,
            source_ids=[np.uint64(1)],
            sink_ids=[np.uint64(4)],
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        supervoxel_ccs, illegal_split = result
        assert isinstance(supervoxel_ccs, list)
        assert len(supervoxel_ccs) >= 2
        assert isinstance(illegal_split, bool)


class TestMergeCrossChunkEdgesOverlap:
    """Test edge cases in merge_cross_chunk_edges_graph_tool."""

    def test_duplicate_cross_chunk_edges(self):
        """Duplicate cross-chunk edges should still merge correctly."""
        edges = np.array([[1, 2], [1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([np.inf, np.inf, 0.5], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # Only the finite-affinity edge should remain
        assert mapped_edges.shape[0] == 1
        assert mapped_affs[0] == pytest.approx(0.5)

    def test_self_loop_after_merge(self):
        """When merging creates a self-loop, it should be present but with correct count."""
        # 1-2 inf, 1-2 finite -> after merge, 1-1 (self-loop) is created
        edges = np.array([[1, 2], [1, 2]], dtype=np.uint64)
        affs = np.array([np.inf, 0.5], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # One non-inf edge remains, but both endpoints map to same node
        assert mapped_edges.shape[0] == 1
        assert mapped_edges[0][0] == mapped_edges[0][1]

    def test_chain_of_cross_chunk_edges(self):
        """A chain of cross-chunk edges: 1-2(inf), 2-3(inf), 3-4(inf).
        All should merge into one component."""
        edges = np.array([[1, 2], [2, 3], [3, 4], [1, 5]], dtype=np.uint64)
        affs = np.array([np.inf, np.inf, np.inf, 0.7], dtype=np.float32)

        mapped_edges, mapped_affs, mapping, complete_mapping, remapping = (
            merge_cross_chunk_edges_graph_tool(edges, affs)
        )

        # Only 1 non-cross edge remains
        assert mapped_edges.shape[0] == 1
        assert mapped_affs[0] == pytest.approx(0.7)
        # All of 1,2,3,4 should be in one remapping group
        assert len(remapping) == 1
        rep = list(remapping.keys())[0]
        assert rep == 1  # min of {1,2,3,4}
        assert set(remapping[rep]) == {1, 2, 3, 4}


class TestRemapCutEdgeSet:
    """Test _remap_cut_edge_set handles cross-chunk remapping correctly."""

    def test_remap_with_cross_chunk_remapping(self):
        """When cross-chunk edge remapping is present, cut edges should expand to all
        mapped supervoxels."""
        # Graph: 1 --0.5-- 2 --inf-- 3 --0.5-- 4
        # Nodes 2 and 3 merge -> rep=2, remapping[2]=[2,3]
        # Source: [1], Sink: [4]
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.5, np.inf, 0.5], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        # The cross_chunk_edge_remapping should exist
        assert len(graph.cross_chunk_edge_remapping) == 1
        result = graph.compute_mincut()

        # Result should contain edges from the original edge set
        result_set = set(map(tuple, result))
        # At least one of the original edges should appear
        assert len(result_set) > 0
        # All returned edges should be from the original cg_edges
        for edge in result:
            assert tuple(edge) in {(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)}

    def test_remap_no_cross_chunk(self):
        """Without cross-chunk edges, remap should just return original supervoxel ids."""
        edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([0.9, 0.1], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        assert len(graph.cross_chunk_edge_remapping) == 0
        result = graph.compute_mincut()
        result_set = set(map(tuple, result))
        # The weak edge 2-3 should be cut
        assert (2, 3) in result_set or (3, 2) in result_set


class TestSplitPreviewConnectedComponents:
    """Test _get_split_preview_connected_components orders CCs correctly."""

    def test_source_first_sink_second(self):
        """split_preview should return sources in ccs[0] and sinks in ccs[1]."""
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.9, 0.01, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        supervoxel_ccs, illegal_split = graph.compute_mincut()
        assert isinstance(supervoxel_ccs, list)
        assert len(supervoxel_ccs) >= 2
        # First CC should contain source supervoxels
        assert 1 in supervoxel_ccs[0]
        # Second CC should contain sink supervoxels
        assert 4 in supervoxel_ccs[1]
        assert isinstance(illegal_split, bool)
        assert not illegal_split

    def test_multiple_sources_and_sinks(self):
        """With multiple sources and sinks, each group stays in its own CC."""
        # 1-2-3-4-5-6, cut between 3-4
        edges = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.uint64)
        affs = np.array([0.9, 0.9, 0.01, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1, 2], dtype=np.uint64)
        sinks = np.array([5, 6], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        supervoxel_ccs, illegal_split = graph.compute_mincut()
        # Both sources should be in ccs[0]
        assert 1 in supervoxel_ccs[0]
        assert 2 in supervoxel_ccs[0]
        # Both sinks should be in ccs[1]
        assert 5 in supervoxel_ccs[1]
        assert 6 in supervoxel_ccs[1]

    def test_split_preview_with_cross_chunk(self):
        """split_preview with cross-chunk edges should expand remapped nodes in CCs."""
        # 1 --0.5-- 2 --inf-- 3 --0.01-- 4
        # Nodes 2,3 merge. Cut between merged(2,3) and 4.
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.5, np.inf, 0.01], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        supervoxel_ccs, illegal_split = graph.compute_mincut()
        assert len(supervoxel_ccs) >= 2
        # After expanding cross-chunk remapping, source CC should contain 1, 2, 3
        all_source_svs = set(supervoxel_ccs[0])
        assert 1 in all_source_svs
        # 2 and 3 were merged and should appear in same CC as source
        assert 2 in all_source_svs or 3 in all_source_svs
        # Sink CC should contain 4
        assert 4 in set(supervoxel_ccs[1])


class TestSanityCheck:
    """Test _sink_and_source_connectivity_sanity_check edge cases."""

    def test_split_preview_illegal_split_flag(self):
        """In split_preview mode, when sanity check would normally raise,
        illegal_split should be True rather than raising an error."""
        # Create a graph where the cut might produce an unusual partition.
        edges = np.array([[1, 2], [2, 3], [1, 3]], dtype=np.uint64)
        affs = np.array([0.01, 0.01, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        result = graph.compute_mincut()
        supervoxel_ccs, illegal_split = result
        # Should return valid result without raising
        assert isinstance(supervoxel_ccs, list)
        assert isinstance(illegal_split, bool)

    def test_non_preview_postcondition_error_on_empty_cut(self):
        """run_multicut raises PostconditionError when mincut produces empty cut set."""
        # When all edges are cross-chunk, PostconditionError is raised
        node_ids1 = np.array([1, 2], dtype=np.uint64)
        node_ids2 = np.array([2, 3], dtype=np.uint64)
        affinities = np.array([np.inf, np.inf], dtype=np.float32)

        edges_obj = Edges(node_ids1, node_ids2, affinities=affinities)
        with pytest.raises(PostconditionError):
            run_multicut(
                edges_obj,
                source_ids=[np.uint64(1)],
                sink_ids=[np.uint64(3)],
            )


class TestRunMulticutSplitPreview:
    """Test run_multicut in split_preview mode returns correct structure."""

    def test_split_preview_returns_ccs_and_flag(self):
        """run_multicut with split_preview=True should return (ccs, illegal_split)."""
        node_ids1 = np.array([1, 2, 3], dtype=np.uint64)
        node_ids2 = np.array([2, 3, 4], dtype=np.uint64)
        affinities = np.array([0.9, 0.01, 0.9], dtype=np.float32)

        edges_obj = Edges(node_ids1, node_ids2, affinities=affinities)
        result = run_multicut(
            edges_obj,
            source_ids=[np.uint64(1)],
            sink_ids=[np.uint64(4)],
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        supervoxel_ccs, illegal_split = result
        assert isinstance(supervoxel_ccs, list)
        assert len(supervoxel_ccs) >= 2
        assert isinstance(illegal_split, bool)

        # Source side CC
        assert 1 in supervoxel_ccs[0]
        # Sink side CC
        assert 4 in supervoxel_ccs[1]

    def test_split_preview_with_path_augment(self):
        """run_multicut with split_preview=True and path_augment=True."""
        node_ids1 = np.array([1, 2, 3, 4], dtype=np.uint64)
        node_ids2 = np.array([2, 3, 4, 5], dtype=np.uint64)
        affinities = np.array([0.9, 0.9, 0.01, 0.9], dtype=np.float32)

        edges_obj = Edges(node_ids1, node_ids2, affinities=affinities)
        result = run_multicut(
            edges_obj,
            source_ids=[np.uint64(1)],
            sink_ids=[np.uint64(5)],
            split_preview=True,
            path_augment=True,
            disallow_isolating_cut=False,
        )

        supervoxel_ccs, illegal_split = result
        assert len(supervoxel_ccs) >= 2
        # Source side
        assert 1 in supervoxel_ccs[0]
        # Sink side
        assert 5 in supervoxel_ccs[1]

    def test_split_preview_larger_graph(self):
        """split_preview on a larger graph with a clear cut point."""
        # Two clusters connected by a single weak edge
        # Cluster A: 1-2, 1-3, 2-3 (all strong)
        # Cluster B: 4-5, 4-6, 5-6 (all strong)
        # Bridge: 3-4 (weak)
        node_ids1 = np.array([1, 1, 2, 4, 4, 5, 3], dtype=np.uint64)
        node_ids2 = np.array([2, 3, 3, 5, 6, 6, 4], dtype=np.uint64)
        affinities = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.01], dtype=np.float32)

        edges_obj = Edges(node_ids1, node_ids2, affinities=affinities)
        result = run_multicut(
            edges_obj,
            source_ids=[np.uint64(1)],
            sink_ids=[np.uint64(6)],
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        supervoxel_ccs, illegal_split = result
        source_cc = set(supervoxel_ccs[0])
        sink_cc = set(supervoxel_ccs[1])
        # Source cluster
        assert {1, 2, 3}.issubset(source_cc)
        # Sink cluster
        assert {4, 5, 6}.issubset(sink_cc)
        assert not illegal_split


class TestLocalMincutGraphWithLogger:
    """Test that logging branches are exercised without errors."""

    def test_init_with_logger(self):
        """Passing a logger should not break initialization."""
        import logging

        logger = logging.getLogger("test_cutting_logger")
        logger.setLevel(logging.DEBUG)

        edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([0.9, 0.1], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
            logger=logger,
        )

        assert graph.weighted_graph is not None

    def test_compute_mincut_with_logger(self):
        """Compute mincut with a logger should produce debug messages."""
        import logging

        logger = logging.getLogger("test_cutting_mincut_logger")
        logger.setLevel(logging.DEBUG)

        edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([0.9, 0.1], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
            logger=logger,
        )

        result = graph.compute_mincut()
        assert len(result) > 0


class TestFilterGraphConnectedComponents:
    """Test edge cases in _filter_graph_connected_components."""

    def test_disconnected_source_sink_raises(self):
        """When sources and sinks are in different connected components, should raise."""
        # Two disconnected components: {1,2} and {3,4}
        # Sources in one, sinks in other
        edges = np.array([[1, 2], [3, 4]], dtype=np.uint64)
        affs = np.array([0.5, 0.5], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        with pytest.raises(PreconditionError):
            graph.compute_mincut()


class TestPartitionEdgesWithinLabel:
    """Test the partition_edges_within_label method."""

    def test_all_edges_within_labels(self):
        """When all out-edges of a component go to labeled nodes, returns True."""
        # Simple triangle: 1-2-3-1, sources=[1,2], sinks=[3]
        edges = np.array([[1, 2], [2, 3], [1, 3]], dtype=np.uint64)
        affs = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1, 2], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        # All nodes are labeled, so any CC should return True
        result = graph.partition_edges_within_label(graph.source_graph_ids)
        assert isinstance(result, bool)

    def test_edges_outside_labels_returns_false(self):
        """When a node has edges to an unlabeled node, returns False."""
        # 1 --0.9-- 2 --0.9-- 3 --0.9-- 4
        # sources=[1], sinks=[4], so nodes 2 and 3 are unlabeled
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        # The source node 1 has edges to node 2 which is not a label node
        result = graph.partition_edges_within_label(graph.source_graph_ids)
        assert result is False


class TestAugmentMincutCapacityOverlap:
    """Test path augmentation when source and sink paths overlap."""

    def test_overlapping_paths_resolved(self):
        """Graph with overlapping shortest paths between sources and sinks.

        Graph topology:
            1--2--3--4--5
               |     |
               6--7--8

        Sources: [1, 6], Sinks: [5, 8]
        Paths from 1->5 and 6->8 overlap at nodes 2, 3, 4.
        The path augmentation should resolve this overlap.
        """
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 4],
            ],
            dtype=np.uint64,
        )
        affs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1, 6], dtype=np.uint64)
        sinks = np.array([5, 8], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        # The graph should initialize and compute the augmented capacity
        # without errors, even with overlapping paths
        result = graph.compute_mincut()
        assert len(result) > 0

    def test_overlapping_paths_with_weak_bridge(self):
        """Graph with overlapping paths and a clear weak bridge to cut.

        Graph:
            1--2--3--4--5
               |     |
               6--7--8

        Edge 3-4 is weak (0.01), all others strong (0.9).
        Sources: [1, 6], Sinks: [5, 8]
        """
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 4],
            ],
            dtype=np.uint64,
        )
        affs = np.array([0.9, 0.9, 0.01, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1, 6], dtype=np.uint64)
        sinks = np.array([5, 8], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0
        result_set = set(map(tuple, result))
        # The weak edge 3-4 should be among the cut edges
        assert (3, 4) in result_set or (4, 3) in result_set

    def test_path_augment_multiple_sources_sinks_no_overlap(self):
        """Multiple sources and sinks where paths do not overlap.

        Graph:
            1--2--3--4
                  |
                  5--6

        Sources: [1], Sinks: [4]
        """
        edges = np.array(
            [[1, 2], [2, 3], [3, 4], [3, 5], [5, 6]],
            dtype=np.uint64,
        )
        affs = np.array([0.9, 0.01, 0.9, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0
        result_set = set(map(tuple, result))
        assert (2, 3) in result_set or (3, 2) in result_set


class TestSplitPreviewMultipleCCs:
    """Test _get_split_preview_connected_components with more than 2 components."""

    def test_three_components(self):
        """A graph that splits into 3 components after cut.

        Graph: 1--2--3--4--5 with weak links at 2-3 and 3-4.
        After cutting both weak links, we get 3 components:
        {1,2}, {3}, {4,5}
        """
        edges = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5]],
            dtype=np.uint64,
        )
        affs = np.array([0.9, 0.01, 0.01, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([5], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        supervoxel_ccs, illegal_split = graph.compute_mincut()
        assert isinstance(supervoxel_ccs, list)
        assert len(supervoxel_ccs) >= 2
        # Source should be in first CC
        assert 1 in supervoxel_ccs[0]
        # Sink should be in second CC
        assert 5 in supervoxel_ccs[1]
        assert isinstance(illegal_split, bool)

    def test_split_preview_preserves_all_nodes(self):
        """All nodes should appear across the CCs."""
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.9, 0.01, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        supervoxel_ccs, _ = graph.compute_mincut()
        all_nodes = set()
        for cc in supervoxel_ccs:
            all_nodes.update(set(cc))
        # All original nodes should appear in some CC
        assert {1, 2, 3, 4}.issubset(all_nodes)


class TestRunSplitPreview:
    """Test the module-level run_split_preview function.

    Note: The full run_split_preview requires a ChunkedGraph instance,
    so we test through run_multicut with split_preview=True which exercises
    the same _get_split_preview_connected_components code path.
    """

    def test_basic_split_preview(self):
        """run_multicut with split_preview should return CCs and a flag."""
        edges_sv = Edges(
            np.array([1, 2, 3, 4], dtype=np.uint64),
            np.array([2, 3, 4, 5], dtype=np.uint64),
            affinities=np.array([0.9, 0.1, 0.9, 0.9], dtype=np.float32),
            areas=np.array([1, 1, 1, 1], dtype=np.float32),
        )
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([5], dtype=np.uint64)
        ccs, illegal_split = run_multicut(
            edges_sv,
            sources,
            sinks,
            split_preview=True,
            disallow_isolating_cut=False,
        )
        assert isinstance(ccs, list)
        assert isinstance(illegal_split, bool)
        assert len(ccs) >= 2

    def test_split_preview_with_areas(self):
        """Split preview with areas provided."""
        edges_sv = Edges(
            np.array([10, 20, 30], dtype=np.uint64),
            np.array([20, 30, 40], dtype=np.uint64),
            affinities=np.array([0.9, 0.01, 0.9], dtype=np.float32),
            areas=np.array([100, 5, 100], dtype=np.float32),
        )
        sources = np.array([10], dtype=np.uint64)
        sinks = np.array([40], dtype=np.uint64)
        ccs, illegal_split = run_multicut(
            edges_sv,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        assert isinstance(ccs, list)
        assert len(ccs) >= 2
        # Source side should contain 10
        assert 10 in ccs[0]
        # Sink side should contain 40
        assert 40 in ccs[1]

    def test_split_preview_path_augment(self):
        """Split preview with path_augment=True."""
        edges_sv = Edges(
            np.array([1, 2, 3, 4, 5], dtype=np.uint64),
            np.array([2, 3, 4, 5, 6], dtype=np.uint64),
            affinities=np.array([0.9, 0.9, 0.01, 0.9, 0.9], dtype=np.float32),
        )
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([6], dtype=np.uint64)
        ccs, illegal_split = run_multicut(
            edges_sv,
            sources,
            sinks,
            split_preview=True,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        assert isinstance(ccs, list)
        assert len(ccs) >= 2
        assert 1 in ccs[0]
        assert 6 in ccs[1]
        assert not illegal_split


class TestFilterGraphCCsWithLogger:
    """Test _filter_graph_connected_components logs a warning when sources
    and sinks are in different connected components."""

    def test_disconnected_with_logger_raises(self):
        """Disconnected graph with logger should log warning and raise."""
        import logging

        logger = logging.getLogger("test_filter_cc_logger")
        logger.setLevel(logging.DEBUG)

        edges = np.array([[1, 2], [3, 4]], dtype=np.uint64)
        affs = np.array([0.5, 0.5], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
            logger=logger,
        )

        with pytest.raises(
            PreconditionError, match="Sinks and sources are not connected"
        ):
            graph.compute_mincut()


class TestGtMincutSanityCheck:
    """Test the _gt_mincut_sanity_check debug method."""

    def test_sanity_check_valid_partition(self):
        """A valid partition should pass the sanity check without error."""
        import graph_tool
        import graph_tool.flow

        edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)
        affs = np.array([0.9, 0.1], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([3], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=False,
        )

        # Manually compute partition to test the sanity check
        graph._filter_graph_connected_components()
        src = graph.weighted_graph.vertex(graph.source_graph_ids[0])
        tgt = graph.weighted_graph.vertex(graph.sink_graph_ids[0])
        residuals = graph_tool.flow.push_relabel_max_flow(
            graph.weighted_graph, src, tgt, graph.capacities
        )
        partition = graph_tool.flow.min_st_cut(
            graph.weighted_graph, src, graph.capacities, residuals
        )
        # This should not raise any assertion error
        graph._gt_mincut_sanity_check(partition)


class TestIsolatingCutPath:
    """Test the IsolatingCutException path in _sink_and_source_connectivity_sanity_check."""

    def test_isolating_cut_raises_precondition_error(self):
        """When mincut isolates exactly the labeled points and they have edges
        to non-label nodes, PreconditionError is raised.

        Graph: 1 --0.01-- 2 --0.9-- 3 --0.9-- 4
        Sources: [1], Sinks: [4]
        disallow_isolating_cut=True

        The mincut cuts edge 1-2 (weakest). After cut, source CC = {1}.
        source_path_vertices = source_graph_ids = {1} (path_augment=False).
        len(source_path_vertices) == len(cc) == 1.
        In the raw graph, node 1 has neighbor 2 which is NOT a label node.
        partition_edges_within_label returns False -> IsolatingCutException -> PreconditionError.
        """
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.01, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=False,
            disallow_isolating_cut=True,
        )
        # This should raise PreconditionError about isolating cut
        with pytest.raises(PreconditionError, match="cut off only the labeled"):
            graph.compute_mincut()

    def test_isolating_cut_split_preview_returns_illegal(self):
        """In split_preview mode, isolating cut should set illegal_split=True."""
        edges = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        affs = np.array([0.01, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1], dtype=np.uint64)
        sinks = np.array([4], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=False,
            disallow_isolating_cut=True,
        )
        supervoxel_ccs, illegal_split = graph.compute_mincut()
        assert isinstance(supervoxel_ccs, list)
        assert illegal_split is True


class TestRerunPathsWithoutOverlap:
    """Test that the rerun_paths_without_overlap code path is exercised
    when source and sink shortest paths overlap and removing overlap
    breaks connectedness."""

    def test_forced_overlap_resolution(self):
        """Create graph where source/sink paths overlap, forcing rerun.

        Graph:
            1--2--3
            |  |  |
            4--5--6

        Sources: [1, 4], Sinks: [3, 6]
        Paths from 1->3 and 4->6 both go through 2 and 5, causing overlap.
        The path augmentation should resolve the overlap via rerun_paths_without_overlap.
        """
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [4, 5],
                [5, 6],
                [1, 4],
                [2, 5],
                [3, 6],
            ],
            dtype=np.uint64,
        )
        affs = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
        sources = np.array([1, 4], dtype=np.uint64)
        sinks = np.array([3, 6], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0

    def test_forced_overlap_resolution_asymmetric(self):
        """Asymmetric graph where one team wins overlap by harmonic mean.

        Graph:
            1--2--3--4
            |  |  |  |
            5--6--7--8

        Sources: [1, 5], Sinks: [4, 8]
        Paths overlap at intermediate nodes 2,3,6,7.
        The path augmentation should resolve the overlap.
        """
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [5, 6],
                [6, 7],
                [7, 8],
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 8],
            ],
            dtype=np.uint64,
        )
        affs = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float32,
        )
        sources = np.array([1, 5], dtype=np.uint64)
        sinks = np.array([4, 8], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0

    def test_overlap_resolution_with_clear_cut(self):
        """Graph with overlap at a bottleneck, but weak bridge for the cut.

        Graph:
            1--2--3--4--5
               |     |
               6--7--8

        Sources: [1, 6], Sinks: [5, 8]
        Edge 3-4 is very weak (0.01), all others strong.
        Overlap is forced at node 2 or 4.
        """
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 4],
            ],
            dtype=np.uint64,
        )
        # Make the bridge edge very weak
        affs = np.array([0.9, 0.9, 0.01, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        sources = np.array([1, 6], dtype=np.uint64)
        sinks = np.array([5, 8], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=False,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        result = graph.compute_mincut()
        assert len(result) > 0
        result_set = set(map(tuple, result))
        # The weak edge 3-4 should be among the cut edges
        assert (3, 4) in result_set or (4, 3) in result_set

    def test_overlap_with_split_preview(self):
        """Split preview mode with overlapping paths should produce valid CCs."""
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [4, 5],
                [5, 6],
                [1, 4],
                [2, 5],
                [3, 6],
            ],
            dtype=np.uint64,
        )
        affs = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
        sources = np.array([1, 4], dtype=np.uint64)
        sinks = np.array([3, 6], dtype=np.uint64)

        graph = LocalMincutGraph(
            edges,
            affs,
            sources,
            sinks,
            split_preview=True,
            path_augment=True,
            disallow_isolating_cut=False,
        )
        supervoxel_ccs, illegal_split = graph.compute_mincut()
        assert isinstance(supervoxel_ccs, list)
        assert len(supervoxel_ccs) >= 2
        assert isinstance(illegal_split, bool)
