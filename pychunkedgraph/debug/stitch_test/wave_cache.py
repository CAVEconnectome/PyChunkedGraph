"""
Two-layer cache for multiwave stitch. See CACHE_DESIGN.md.

Invariants:
  - No BigTable row read more than once (has_batch gates all reads)
  - New nodes never read from BigTable (created data in _*_local)
  - No BigTable row written more than once (dirty check skips unchanged siblings)

Layers:
  _*_local: reads from BigTable + created node data (flush_created)
  ro_*: preloaded from parent via fork COW (immutable)
  ChainMap(_*_local, ro_*): unified lookup

Pure storage + query. No BigTable I/O.
"""

from collections import ChainMap, defaultdict
from dataclasses import dataclass, field

import numpy as np

from pychunkedgraph.graph import basetypes


@dataclass
class SiblingEntry:
    raw_cx_edges: dict
    resolver_entries: dict  # {sv_int: {2: l2_int}}
    written_cx: dict = field(default_factory=dict)  # CX written to BigTable, for dirty check


class WaveCache:

    def __init__(self, preloaded: tuple = None, incremental: tuple = None) -> None:
        ro_p = preloaded[0] if preloaded else {}
        ro_c = preloaded[1] if preloaded else {}
        ro_a = preloaded[2] if preloaded else {}

        self._parents_local: dict = {}
        self._children_local: dict = {}
        self._acx_local: dict = {}

        self.parents = ChainMap(self._parents_local, ro_p)
        self.children = ChainMap(self._children_local, ro_c)
        self.acx = ChainMap(self._acx_local, ro_a)

        if incremental:
            self._siblings, self.accumulated_replacements, self._new_node_ids = incremental
        else:
            self._siblings: dict = {}
            self.accumulated_replacements: dict = {}
            self._new_node_ids: set = set()

        self._init_stitch_state()

    def _init_stitch_state(self) -> None:
        self.resolver: dict = {}
        self.raw_cx_edges: dict = {}
        self.cx_cache: dict = {}
        self.children_cache: dict = {}
        self.parents_cache: dict = {}
        self.l2_atomic_cx: dict = {}
        self.new_ids_d: dict = defaultdict(list)
        self.new_node_ids: set = set()
        self.sibling_ids: set = set()
        self.dirty_siblings: set = set()
        self.old_to_new: dict = {}
        self.sv_to_l2: dict = {}
        self.old_hierarchy: dict = {}
        self.l2ids: np.ndarray = np.array([], dtype=basetypes.NODE_ID)
        self.l2_edges: np.ndarray = np.array([], dtype=basetypes.NODE_ID)
        self.l2_cx_edges: dict = defaultdict(lambda: defaultdict(list))
        self.siblings_d: dict = defaultdict(list)
        self.children_d: dict = {}
        self.atomic_cx_stitch: dict = {}

    def begin_stitch(self) -> None:
        self._init_stitch_state()

    def flush_created(self) -> None:
        self._parents_local.update(self.parents_cache)
        self._children_local.update(self.children_cache)
        self._acx_local.update(self.l2_atomic_cx)

    def compute_dirty_siblings(self) -> None:
        affected_ids = set(self.old_to_new.keys()) | set(int(x) for x in self.new_ids_d.get(2, []))
        self.dirty_siblings = set()
        for sib in self.sibling_ids:
            sib_int = int(sib)
            if sib_int not in self._siblings:
                self.dirty_siblings.add(sib_int)
                continue
            raw = self.raw_cx_edges.get(sib_int, {})
            dirty = False
            for layer_edges in raw.values():
                if len(layer_edges) == 0:
                    continue
                for sv in layer_edges[:, 1]:
                    identity = self.resolver.get(int(sv), {}).get(2, int(sv))
                    if identity in affected_ids:
                        dirty = True
                        break
                if dirty:
                    break
            if dirty:
                self.dirty_siblings.add(sib_int)

    def save_wave_state(
        self, old_to_new: dict, new_node_ids: set,
        sibling_ids: set, raw_cx_edges: dict, children,
    ) -> None:
        self.flush_created()
        self.accumulated_replacements = dict(old_to_new)
        self._new_node_ids = set(new_node_ids)
        for sib_id in sibling_ids:
            sib_int = int(sib_id)
            ch = children.get(sib_int, children.get(np.uint64(sib_int), np.array([], dtype=basetypes.NODE_ID)))
            prior = self._siblings.get(sib_int)
            prior_cx = prior.written_cx if prior else {}
            self._siblings[sib_int] = SiblingEntry(
                raw_cx_edges=raw_cx_edges.get(sib_int, {}),
                resolver_entries={int(sv): {2: sib_int} for sv in ch},
                written_cx=self.cx_cache.get(sib_int, prior_cx),
            )

    def inc_snapshot_from(
        self, old_to_new: dict, new_node_ids: set,
        sibling_ids: set, raw_cx_edges: dict, children,
    ) -> dict:
        sibling_data = {}
        for sid in sibling_ids:
            sid_int = int(sid)
            ch = children.get(sid_int, children.get(np.uint64(sid_int), np.array([], dtype=basetypes.NODE_ID)))
            prior = self._siblings.get(sid_int)
            prior_cx = prior.written_cx if prior else {}
            sibling_data[sid_int] = {
                "children": ch,
                "raw_cx": raw_cx_edges.get(sid_int, {}),
                "written_cx": self.cx_cache.get(sid_int, prior_cx),
            }
        return {
            "old_to_new": dict(old_to_new),
            "new_node_ids": set(new_node_ids),
            "sibling_data": sibling_data,
        }

    # -- Stable layer: batch put from BigTable reads --

    def put_parent(self, node_id: int, parent: int) -> None:
        self._parents_local[node_id] = parent

    def put_children(self, node_id: int, children: np.ndarray) -> None:
        self._children_local[node_id] = children

    def put_acx(self, node_id: int, acx_d: dict) -> None:
        self._acx_local[node_id] = acx_d

    def has(self, node_id: int) -> bool:
        return node_id in self.parents

    def has_batch(self, node_ids: np.ndarray) -> np.ndarray:
        return np.array([int(n) in self.parents for n in node_ids])

    # -- Merge operations (after pool waves) --

    def merge_reader(self, snapshot: tuple) -> None:
        rp, rc, ra = snapshot
        self._parents_local.update(rp)
        self._children_local.update(rc)
        self._acx_local.update(ra)

    def merge_inc(self, inc_snap: dict) -> None:
        self.accumulated_replacements.update(inc_snap["old_to_new"])
        self._new_node_ids.update(inc_snap["new_node_ids"])
        for sib_int, sdata in inc_snap["sibling_data"].items():
            self._siblings[sib_int] = SiblingEntry(
                raw_cx_edges=sdata["raw_cx"],
                resolver_entries={int(sv): {2: sib_int} for sv in sdata["children"]},
                written_cx=sdata.get("written_cx", {}),
            )

    # -- Snapshot for pool workers --

    def preloaded(self) -> tuple:
        return (self.parents, self.children, self.acx)

    def incremental_state(self) -> tuple:
        return (self._siblings, self.accumulated_replacements, self._new_node_ids)

    def local_snapshot(self) -> tuple:
        return (self._parents_local, self._children_local, self._acx_local)

    # -- Incremental sibling support --

    def split_known_siblings(self, all_siblings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._siblings:
            return np.array([], dtype=basetypes.NODE_ID), all_siblings
        known, unknown = [], []
        for sib in all_siblings:
            (known if int(sib) in self._siblings else unknown).append(sib)
        return (
            np.array(known, dtype=basetypes.NODE_ID),
            np.array(unknown, dtype=basetypes.NODE_ID),
        )

    def get_sibling(self, sib_int: int) -> SiblingEntry | None:
        return self._siblings.get(sib_int)

    def stats(self) -> str:
        return (
            f"{len(self.parents)}p {len(self.children)}c "
            f"{len(self.acx)}a {len(self._siblings)}s"
        )
