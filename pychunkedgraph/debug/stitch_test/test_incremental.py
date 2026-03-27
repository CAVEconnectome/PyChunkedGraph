"""
Tests for incremental sibling processing.

Run: pytest test_incremental.py -v
"""

import numpy as np
import pytest

from pychunkedgraph.graph import basetypes

from .incremental import IncrementalState

NODE_ID = basetypes.NODE_ID


def _node(layer: int, seg: int) -> int:
    return (layer << 56) | seg


class FakeCtx:
    def __init__(self, cx_cache: dict, sibling_ids: set, new_node_ids: set, old_to_new: dict):
        self.cx_cache = cx_cache
        self.sibling_ids = sibling_ids
        self.new_node_ids = new_node_ids
        self.old_to_new = old_to_new


class TestIncrementalState:

    def test_empty_state_all_affected(self) -> None:
        state = IncrementalState()
        siblings = np.array([10, 20, 30], dtype=NODE_ID)
        affected, unchanged, cached = state.partition_siblings(
            siblings, old_hierarchy={}, l2ids=np.array([], dtype=NODE_ID),
        )
        assert len(affected) == 3
        assert len(unchanged) == 0
        assert len(cached) == 0

    def test_no_changes_all_unchanged(self) -> None:
        state = IncrementalState()
        state.previous_siblings = {10, 20, 30}
        state.previous_cx = {10: {2: "edges_10"}, 20: {2: "edges_20"}, 30: {2: "edges_30"}}
        state.previous_old_to_new = {}
        state.previous_new_node_ids = set()

        siblings = np.array([10, 20, 30], dtype=NODE_ID)
        affected, unchanged, cached = state.partition_siblings(
            siblings, old_hierarchy={}, l2ids=np.array([], dtype=NODE_ID),
        )
        assert len(affected) == 0
        assert len(unchanged) == 3
        assert len(cached) == 3

    def test_new_siblings_are_affected(self) -> None:
        state = IncrementalState()
        state.previous_siblings = {10, 20}
        state.previous_cx = {10: {2: "e10"}, 20: {2: "e20"}}
        state.previous_old_to_new = {}
        state.previous_new_node_ids = set()

        siblings = np.array([10, 20, 30, 40], dtype=NODE_ID)
        affected, unchanged, cached = state.partition_siblings(
            siblings, old_hierarchy={}, l2ids=np.array([], dtype=NODE_ID),
        )
        assert set(int(x) for x in affected) == {30, 40}
        assert set(int(x) for x in unchanged) == {10, 20}

    def test_siblings_under_replaced_parent_are_affected(self) -> None:
        old_l3 = _node(3, 100)
        new_l3 = _node(3, 200)

        state = IncrementalState()
        state.previous_siblings = {10, 20, 30}
        state.previous_cx = {10: {2: "e"}, 20: {2: "e"}, 30: {2: "e"}}
        state.previous_old_to_new = {old_l3: new_l3}
        state.previous_new_node_ids = {new_l3}

        old_hierarchy = {
            10: {3: old_l3},
            20: {3: old_l3},
            30: {3: _node(3, 999)},
        }

        siblings = np.array([10, 20, 30], dtype=NODE_ID)
        affected, unchanged, cached = state.partition_siblings(
            siblings, old_hierarchy=old_hierarchy,
            l2ids=np.array([], dtype=NODE_ID),
        )
        assert set(int(x) for x in affected) == {10, 20}
        assert set(int(x) for x in unchanged) == {30}
        assert 30 in cached

    def test_update_from_stitch(self) -> None:
        state = IncrementalState()
        ctx = FakeCtx(
            cx_cache={10: {2: "edges"}, 20: {3: "edges2"}},
            sibling_ids={10, 20, 30},
            new_node_ids={100, 200},
            old_to_new={50: 100, 60: 200},
        )
        state.update_from_stitch(ctx)
        assert state.previous_cx == {10: {2: "edges"}, 20: {3: "edges2"}}
        assert state.previous_siblings == {10, 20, 30}
        assert state.previous_new_node_ids == {100, 200}
        assert state.previous_old_to_new == {50: 100, 60: 200}

    def test_find_affected_parents_empty(self) -> None:
        state = IncrementalState()
        assert state.find_affected_parents({}) == set()

    def test_find_affected_parents_with_replacements(self) -> None:
        old_l3 = _node(3, 100)
        new_l3 = _node(3, 200)
        old_l4 = _node(4, 300)

        state = IncrementalState()
        state.previous_old_to_new = {old_l3: new_l3}
        state.previous_new_node_ids = {new_l3}

        old_hierarchy = {
            10: {3: old_l3, 4: old_l4},
            20: {3: _node(3, 999)},
        }
        affected = state.find_affected_parents(old_hierarchy)
        assert old_l3 in affected
        assert old_l4 not in affected

    def test_cached_cx_only_for_unchanged(self) -> None:
        state = IncrementalState()
        state.previous_siblings = {10, 20, 30}
        state.previous_cx = {10: {2: "e10"}, 30: {2: "e30"}}
        state.previous_old_to_new = {}
        state.previous_new_node_ids = set()

        siblings = np.array([10, 20, 30], dtype=NODE_ID)
        _, _, cached = state.partition_siblings(
            siblings, old_hierarchy={}, l2ids=np.array([], dtype=NODE_ID),
        )
        assert 10 in cached
        assert 30 in cached
        assert 20 not in cached  # was sibling but had no cx_cache entry

    def test_removed_siblings_not_in_partition(self) -> None:
        state = IncrementalState()
        state.previous_siblings = {10, 20, 30, 40}
        state.previous_cx = {10: {}, 20: {}, 30: {}, 40: {}}
        state.previous_old_to_new = {}
        state.previous_new_node_ids = set()

        siblings = np.array([10, 20], dtype=NODE_ID)
        affected, unchanged, cached = state.partition_siblings(
            siblings, old_hierarchy={}, l2ids=np.array([], dtype=NODE_ID),
        )
        assert set(int(x) for x in unchanged) == {10, 20}
        assert 30 not in cached
        assert 40 not in cached

    def test_multi_wave_evolution(self) -> None:
        state = IncrementalState()

        # Wave 0: all new
        ctx0 = FakeCtx(
            cx_cache={1: {2: "w0"}, 2: {2: "w0"}, 3: {2: "w0"}},
            sibling_ids={1, 2, 3},
            new_node_ids={100},
            old_to_new={50: 100},
        )
        state.update_from_stitch(ctx0)

        # Wave 1: node 50 was replaced by 100. Siblings under 50's parent are affected.
        old_hier = {1: {3: 50}, 2: {3: 50}, 3: {3: 999}}
        siblings = np.array([1, 2, 3, 4], dtype=NODE_ID)  # 4 is new
        affected, unchanged, cached = state.partition_siblings(
            siblings, old_hierarchy=old_hier,
            l2ids=np.array([], dtype=NODE_ID),
        )
        assert 4 in set(int(x) for x in affected)  # new
        assert 1 in set(int(x) for x in affected)  # under replaced parent
        assert 2 in set(int(x) for x in affected)  # under replaced parent
        assert 3 in set(int(x) for x in unchanged)  # under unaffected parent
