"""
Incremental sibling processing across waves.

Tracks which nodes changed per wave, caches resolved CX for unchanged siblings.
Only re-processes siblings whose parent hierarchy was modified.
"""

from dataclasses import dataclass, field

import numpy as np

from pychunkedgraph.graph import basetypes


@dataclass
class IncrementalState:
    """Persists across waves. Tracks what changed and caches sibling CX."""

    previous_cx: dict = field(default_factory=dict)
    previous_siblings: set = field(default_factory=set)
    previous_new_node_ids: set = field(default_factory=set)
    previous_old_to_new: dict = field(default_factory=dict)

    def update_from_stitch(self, ctx) -> None:
        """Call after each wave's stitch completes. Saves state for next wave."""
        self.previous_cx = dict(ctx.cx_cache)
        self.previous_siblings = set(ctx.sibling_ids)
        self.previous_new_node_ids = set(ctx.new_node_ids)
        self.previous_old_to_new = dict(ctx.old_to_new)

    def find_affected_parents(self, old_hierarchy: dict) -> set:
        """Find old parent IDs that contain any node replaced in the prior wave."""
        if not self.previous_old_to_new:
            return set()
        replaced = set(self.previous_old_to_new.keys()) | set(self.previous_old_to_new.values())
        replaced.update(self.previous_new_node_ids)
        affected = set()
        for chain in old_hierarchy.values():
            for parent in chain.values():
                if int(parent) in replaced:
                    affected.add(int(parent))
        return affected

    def partition_siblings(
        self,
        all_siblings: np.ndarray,
        old_hierarchy: dict,
        l2ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Split siblings into affected (need re-processing) and unchanged (reuse cache).

        Returns: (affected_siblings, unchanged_siblings, cached_cx_for_unchanged)
        """
        if not self.previous_siblings:
            return all_siblings, np.array([], dtype=basetypes.NODE_ID), {}

        affected_parents = self.find_affected_parents(old_hierarchy)
        if not affected_parents:
            known_sibs = set(int(x) for x in all_siblings) & self.previous_siblings
            new_sibs = set(int(x) for x in all_siblings) - self.previous_siblings
            affected = np.array(list(new_sibs), dtype=basetypes.NODE_ID)
            unchanged = np.array(list(known_sibs), dtype=basetypes.NODE_ID)
            cached_cx = {
                nid: self.previous_cx[nid]
                for nid in known_sibs
                if nid in self.previous_cx
            }
            return affected, unchanged, cached_cx

        our_l2_set = set(int(x) for x in l2ids)
        affected_set = set()
        unchanged_set = set()

        for sib in all_siblings:
            sib_int = int(sib)
            if sib_int not in self.previous_siblings:
                affected_set.add(sib_int)
                continue
            chain = old_hierarchy.get(sib_int, {})
            if any(int(p) in affected_parents for p in chain.values()):
                affected_set.add(sib_int)
            else:
                unchanged_set.add(sib_int)

        cached_cx = {
            nid: self.previous_cx[nid]
            for nid in unchanged_set
            if nid in self.previous_cx
        }

        return (
            np.array(list(affected_set), dtype=basetypes.NODE_ID),
            np.array(list(unchanged_set), dtype=basetypes.NODE_ID),
            cached_cx,
        )
