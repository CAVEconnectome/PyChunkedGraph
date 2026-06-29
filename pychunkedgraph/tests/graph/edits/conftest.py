"""Shared fixtures for the edit-operation tests."""

import pytest

from ...helpers import SV, build_graph


@pytest.fixture()
def connected_pair(gen_graph):
    """Two supervoxels joined by an edge across neighboring chunks (3 layers).

    ┌─────┬─────┐
    │  A¹ │  B¹ │
    │  a━━┿━━b  │
    └─────┴─────┘
    """
    return build_graph(
        gen_graph,
        n_layers=3,
        supervoxels={"a": SV(), "b": SV(x=1)},
        edges=[("a", "b", 0.5)],
    )


@pytest.fixture()
def split_then_merge_ops(connected_pair):
    """`connected_pair` split apart then merged back; returns (cg, merge_op_id, split_op_id)."""
    cg, sv = connected_pair
    split_result = cg.remove_edges(
        "test_user", source_ids=sv["a"], sink_ids=sv["b"], mincut=False
    )
    merge_result = cg.add_edges(
        "test_user",
        atomic_edges=[[sv["a"], sv["b"]]],
        source_coords=[0, 0, 0],
        sink_coords=[0, 0, 0],
    )
    return cg, merge_result.operation_id, split_result.operation_id


@pytest.fixture()
def split_then_undo(connected_pair):
    """`connected_pair` split, then the split undone; returns (cg, split_op_id, undo_result)."""
    cg, sv = connected_pair
    split_result = cg.remove_edges(
        "test_user", source_ids=sv["a"], sink_ids=sv["b"], mincut=False
    )
    undo_result = cg.undo_operation("test_user", split_result.operation_id)
    return cg, split_result.operation_id, undo_result
