"""Integration tests for stale edge detection and resolution.

Tests get_stale_nodes() and get_new_nodes() from stale.py using real graph
operations through the BigTable emulator.
"""


from math import inf

import numpy as np
import pytest

from ..helpers import SV, build_graph
from ...graph.edges.stale import get_stale_nodes, get_new_nodes


class TestStaleEdges:
    @pytest.mark.timeout(30)
    def test_stale_nodes_detected_after_split(self, gen_graph):
        """
        After a split, the old L2 parent IDs become stale.
        get_stale_nodes should identify them.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", inf)],
        )

        # Get old parents before edit
        old_root = cg.get_root(sv["a0"])

        # Split
        cg.remove_edges(
            "test_user",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )

        # The old root should now be stale
        stale = get_stale_nodes(cg, [old_root])
        assert old_root in stale

    @pytest.mark.timeout(30)
    def test_no_stale_nodes_for_current_ids(self, gen_graph):
        """
        Current (post-edit) node IDs should not be flagged as stale.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", inf)],
        )

        # Split
        cg.remove_edges(
            "test_user",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )

        # Current roots should not be stale
        new_root_1 = cg.get_root(sv["a0"])
        new_root_2 = cg.get_root(sv["b"])
        stale = get_stale_nodes(cg, [new_root_1, new_root_2])
        assert new_root_1 not in stale
        assert new_root_2 not in stale

    @pytest.mark.timeout(30)
    def test_get_new_nodes_resolves_to_correct_layer(self, gen_graph):
        """
        get_new_nodes should follow the parent chain from a supervoxel
        to the correct layer and return the current node at that layer.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", inf)],
        )

        # Get L2 parent of SV 1 before edit
        sv1 = sv["a0"]
        old_l2_parent = cg.get_parent(sv1)

        # Split
        cg.remove_edges(
            "test_user",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )

        # get_new_nodes should resolve SV to its current L2 parent
        new_l2 = get_new_nodes(cg, np.array([sv1], dtype=np.uint64), layer=2)
        current_l2_parent = cg.get_parent(sv1)
        assert new_l2[0] == current_l2_parent

    @pytest.mark.timeout(30)
    def test_no_stale_nodes_in_unaffected_region(self, gen_graph):
        """
        Nodes not involved in an edit should not be flagged as stale.

        ┌─────┬─────┬─────┐
        │  A¹ │  B¹ │  C¹ │
        │  1━━┿━━2  │  3  │
        │     │     │     │
        └─────┴─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "b": SV(x=1), "c": SV(x=2)},
            edges=[("a0", "b", inf)],
        )

        # Get the isolated node's root before edit
        isolated_root = cg.get_root(sv["c"])

        # Split nodes 1 and 2
        cg.remove_edges(
            "test_user",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            mincut=False,
        )

        # The isolated root should not be stale — it was unaffected
        stale = get_stale_nodes(cg, [isolated_root])
        assert isolated_root not in stale

    @pytest.mark.timeout(30)
    def test_get_new_nodes_returns_self_for_non_stale(self, gen_graph):
        """
        For freshly created nodes with no edits, get_new_nodes should return
        the nodes themselves (identity mapping).

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV()},
        )

        sv0 = sv["a0"]
        l2_parent = cg.get_parent(sv0)

        # get_new_nodes at layer 2 should return the same L2 parent
        result = get_new_nodes(cg, np.array([sv0], dtype=np.uint64), layer=2)
        assert result[0] == l2_parent

    @pytest.mark.timeout(30)
    def test_get_stale_nodes_empty_for_fresh_graph(self, gen_graph):
        """
        In a freshly built graph with no edits, no nodes should be stale.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", inf)],
        )

        root = cg.get_root(sv["a0"])
        l2_0 = cg.get_parent(sv["a0"])
        l2_1 = cg.get_parent(sv["b"])

        # No edits have been performed, so all nodes should be non-stale
        stale = get_stale_nodes(cg, [root, l2_0, l2_1])
        assert len(stale) == 0

    @pytest.mark.timeout(30)
    def test_get_new_nodes_multiple_svs(self, gen_graph):
        """
        get_new_nodes with multiple supervoxels should return an array
        of the same length, each mapped to its current L2 parent.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV(), "b": SV(x=1)},
            edges=[("a0", "b", inf)],
        )

        sv1 = sv["a0"]
        sv2 = sv["b"]
        svs = np.array([sv1, sv2], dtype=np.uint64)

        result = get_new_nodes(cg, svs, layer=2)
        assert result.shape == (2,)
        # Each SV should map to its L2 parent
        assert result[0] == cg.get_parent(sv1)
        assert result[1] == cg.get_parent(sv2)

    @pytest.mark.timeout(30)
    def test_get_new_nodes_with_duplicate_svs(self, gen_graph):
        """
        get_new_nodes should handle duplicate SVs correctly,
        returning the same result for duplicate inputs.

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=3,
            supervoxels={"a0": SV()},
        )

        sv0 = sv["a0"]
        svs = np.array([sv0, sv0, sv0], dtype=np.uint64)

        result = get_new_nodes(cg, svs, layer=2)
        assert result.shape == (3,)
        # All should map to the same L2 parent
        expected = cg.get_parent(sv0)
        assert np.all(result == expected)

    @pytest.mark.timeout(30)
    def test_get_stale_nodes_with_l2_ids_after_merge(self, gen_graph):
        """
        After a merge, the old L2 IDs should become stale.

        ┌─────┐
        │  A¹ │
        │ 1 2 │  (isolated, then merged)
        │     │
        └─────┘
        """
        atomic_chunk_bounds = np.array([1, 1, 1])
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=atomic_chunk_bounds,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
        )

        sv0 = sv["a0"]
        sv1 = sv["a1"]

        # Get L2 parents before merge (each SV has its own L2 parent)
        old_l2_0 = cg.get_parent(sv0)
        old_l2_1 = cg.get_parent(sv1)

        # Merge
        cg.add_edges(
            "test_user",
            [sv0, sv1],
            affinities=[0.3],
        )

        # Old L2 parents should now be stale
        stale = get_stale_nodes(cg, [old_l2_0, old_l2_1])
        assert old_l2_0 in stale or old_l2_1 in stale

    @pytest.mark.timeout(30)
    def test_get_stale_nodes_returns_numpy_array(self, gen_graph):
        """
        get_stale_nodes should always return a numpy ndarray, even when
        no nodes are stale.

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        atomic_chunk_bounds = np.array([1, 1, 1])
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=atomic_chunk_bounds,
            supervoxels={"a0": SV()},
        )

        sv0 = sv["a0"]

        root = cg.get_root(sv0)
        stale = get_stale_nodes(cg, [root])
        assert isinstance(stale, np.ndarray)

    @pytest.mark.timeout(30)
    def test_get_new_nodes_at_root_layer(self, gen_graph):
        """
        get_new_nodes called with layer=root_layer should return the root node.

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV()},
        )

        sv0 = sv["a0"]

        root = cg.get_root(sv0)
        root_layer = cg.get_chunk_layer(root)

        result = get_new_nodes(cg, np.array([sv0], dtype=np.uint64), layer=root_layer)
        assert result.shape == (1,)
        assert result[0] == root
