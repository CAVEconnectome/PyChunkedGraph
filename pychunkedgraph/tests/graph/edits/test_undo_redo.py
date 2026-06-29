"""Integration tests for undo/redo operations through the full graph.

Tests that undo and redo correctly restore graph state using real graph
operations through the BigTable emulator.
"""


import numpy as np
import pytest


class TestUndoRedo:
    @pytest.mark.timeout(30)
    def test_undo_split_restores_merged_root(self, connected_pair):
        """Split two nodes, undo — nodes should share a common root again."""
        cg, sv = connected_pair
        sv1 = sv["a"]
        sv2 = sv["b"]

        # Initially, both SVs share a root
        assert cg.get_root(sv1) == cg.get_root(sv2)

        # Split
        split_result = cg.remove_edges(
            "test_user", source_ids=sv1, sink_ids=sv2, mincut=False
        )
        assert len(split_result.new_root_ids) == 2
        assert cg.get_root(sv1) != cg.get_root(sv2)

        # Undo the split
        cg.undo_operation("test_user", split_result.operation_id)

        # After undo, both SVs should share a root again
        assert cg.get_root(sv1) == cg.get_root(sv2)

    @pytest.mark.timeout(30)
    def test_redo_restores_operation_result(self, connected_pair):
        """Split, undo, redo the original split — state should match the post-split state."""
        cg, sv = connected_pair
        sv1 = sv["a"]
        sv2 = sv["b"]

        # Split
        split_result = cg.remove_edges(
            "test_user", source_ids=sv1, sink_ids=sv2, mincut=False
        )
        assert cg.get_root(sv1) != cg.get_root(sv2)

        # Undo (merges back)
        cg.undo_operation("test_user", split_result.operation_id)
        assert cg.get_root(sv1) == cg.get_root(sv2)

        # Redo the original split operation (re-applies the split)
        cg.redo_operation("test_user", split_result.operation_id)

        # After redo, nodes should be split again
        assert cg.get_root(sv1) != cg.get_root(sv2)

    @pytest.mark.timeout(30)
    def test_undo_preserves_subgraph_leaves(self, connected_pair):
        """After undo, subgraph leaves should match the pre-operation state."""
        cg, sv = connected_pair
        sv1 = sv["a"]
        sv2 = sv["b"]

        # Get initial leaf set
        initial_root = cg.get_root(sv1)
        initial_leaves = set(
            np.unique(cg.get_subgraph([initial_root], leaves_only=True))
        )
        assert sv1 in initial_leaves
        assert sv2 in initial_leaves

        # Split
        split_result = cg.remove_edges(
            "test_user", source_ids=sv1, sink_ids=sv2, mincut=False
        )

        # Undo
        cg.undo_operation("test_user", split_result.operation_id)

        # After undo, the root's subgraph should contain both SVs again
        restored_root = cg.get_root(sv1)
        restored_leaves = set(
            np.unique(cg.get_subgraph([restored_root], leaves_only=True))
        )
        assert sv1 in restored_leaves
        assert sv2 in restored_leaves
