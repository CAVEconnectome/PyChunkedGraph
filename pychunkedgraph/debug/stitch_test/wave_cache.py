"""
Cache for multiwave stitch. See CACHE_DESIGN.md.

Uses RowCache (row_cache.py) for BigTable-mirroring row/column storage.
Each node is a CacheRow with parent/children/acx/cx columns.

Invariants:
  - No row read more than once (has gates reads)
  - Created rows never read from BigTable (put during stitch)
  - No unnecessary writes (dirty_siblings gates write skip)

Pure storage + query. No BigTable I/O.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pychunkedgraph.graph import basetypes

from .row_cache import CacheRow, RowCache


@dataclass
class SiblingEntry:
    unresolved_acx: dict
    resolver_entries: dict  # {sv_int: {2: l2_int}}
    partner_ids: dict = field(default_factory=dict)  # {sv: resolved_l2} for complete dirty detection


class WaveCache:

    def __init__(self, preloaded=None, incremental: tuple = None) -> None:
        if isinstance(preloaded, dict):
            self._rows = RowCache(preloaded=preloaded)
        elif preloaded is not None:
            self._rows = RowCache(preloaded=_tuple_to_rows(preloaded))
        else:
            self._rows = RowCache()

        if incremental:
            self._siblings, self.accumulated_replacements, self._new_node_ids = incremental
        else:
            self._siblings: dict = {}
            self.accumulated_replacements: dict = {}
            self._new_node_ids: set = set()

        self._init_stitch_state()

    def _init_stitch_state(self) -> None:
        self.resolver: dict = {}
        self.unresolved_acx: dict = {}
        self.new_ids_d: dict = defaultdict(list)
        self.new_node_ids: set = set()
        self.sibling_ids: set = set()
        self.counterpart_ids: set = set()
        self.dirty_siblings: set = set()
        self.old_to_new: dict = {}
        self.old_hierarchy: dict = {}
        self.new_to_old: dict = {}
        self.l2ids: np.ndarray = np.array([], dtype=basetypes.NODE_ID)
        self.l2_edges: np.ndarray = np.array([], dtype=basetypes.NODE_ID)
        self.l2_cx_edges: dict = defaultdict(lambda: defaultdict(list))
        self.siblings_d: dict = defaultdict(list)

    def begin_stitch(self) -> None:
        self._rows.clear_local()
        self._init_stitch_state()

    def compute_dirty_siblings(self) -> None:
        affected_ids = set(self.old_to_new.keys()) | set(int(x) for x in self.new_ids_d.get(2, []))
        self.dirty_siblings = set()

        # Build vectorized resolver lookup: sv → L2 identity
        resolver_svs = np.array(list(self.resolver.keys()), dtype=np.int64) if self.resolver else np.array([], dtype=np.int64)
        resolver_ids = np.array([self.resolver[k].get(2, k) for k in resolver_svs], dtype=np.int64) if len(resolver_svs) > 0 else np.array([], dtype=np.int64)
        affected_arr = np.array(list(affected_ids), dtype=np.int64) if affected_ids else np.array([], dtype=np.int64)

        if len(resolver_svs) > 0:
            sort_idx = np.argsort(resolver_svs)
            resolver_svs_sorted = resolver_svs[sort_idx]
            resolver_ids_sorted = resolver_ids[sort_idx]
        else:
            resolver_svs_sorted = resolver_svs
            resolver_ids_sorted = resolver_ids

        affected_sorted = np.sort(affected_arr) if len(affected_arr) > 0 else affected_arr

        for sib in self.sibling_ids:
            sib_int = int(sib)
            if sib_int not in self._siblings:
                self.dirty_siblings.add(sib_int)
                continue
            raw = self.unresolved_acx.get(sib_int, {})
            all_svs = []
            for layer_edges in raw.values():
                if len(layer_edges) > 0:
                    all_svs.append(layer_edges[:, 1])
            if not all_svs:
                continue
            partner_svs = np.concatenate(all_svs).astype(np.int64)
            if len(resolver_svs_sorted) > 0:
                idx = np.searchsorted(resolver_svs_sorted, partner_svs)
                idx = np.clip(idx, 0, len(resolver_svs_sorted) - 1)
                found = resolver_svs_sorted[idx] == partner_svs
                identities = np.where(found, resolver_ids_sorted[idx], partner_svs)
            else:
                identities = partner_svs
            if len(affected_sorted) > 0:
                aff_idx = np.searchsorted(affected_sorted, identities)
                aff_idx = np.clip(aff_idx, 0, len(affected_sorted) - 1)
                if np.any(affected_sorted[aff_idx] == identities):
                    self.dirty_siblings.add(sib_int)
                    continue
            # Close the 0.4% gap: check if resolver entries changed from prior wave
            entry = self._siblings[sib_int]
            if entry.partner_ids:
                for sv, old_id in entry.partner_ids.items():
                    new_id = self.resolver.get(int(sv), {}).get(2, int(sv))
                    if new_id != old_id:
                        self.dirty_siblings.add(sib_int)
                        break

    def save_wave_state(
        self, old_to_new: dict, new_node_ids: set,
        sibling_ids: set, unresolved_acx: dict,
    ) -> None:
        self._rows.promote_local()
        self.accumulated_replacements = dict(old_to_new)
        self._new_node_ids = set(new_node_ids)
        for sib_id in sibling_ids:
            sib_int = int(sib_id)
            ch = self.children.get(sib_int, np.array([], dtype=basetypes.NODE_ID))
            cx = self.cx.get(sib_int, {})
            raw = unresolved_acx.get(sib_int, {})
            pids = {}
            for layer_edges in raw.values():
                if len(layer_edges) > 0:
                    for sv in layer_edges[:, 1]:
                        sv_int = int(sv)
                        pids[sv_int] = self.resolver.get(sv_int, {}).get(2, sv_int)
            self._siblings[sib_int] = SiblingEntry(
                unresolved_acx=raw,
                resolver_entries={int(sv): {2: sib_int} for sv in ch},
                partner_ids=pids,
            )

    def inc_snapshot_from(
        self, old_to_new: dict, new_node_ids: set,
        sibling_ids: set, unresolved_acx: dict,
    ) -> dict:
        sibling_data = {}
        for sid in sibling_ids:
            sid_int = int(sid)
            ch = self.children.get(sid_int, np.array([], dtype=basetypes.NODE_ID))
            cx = self.cx.get(sid_int, {})
            raw = unresolved_acx.get(sid_int, {})
            pids = {}
            for layer_edges in raw.values():
                if len(layer_edges) > 0:
                    for sv in layer_edges[:, 1]:
                        sv_int = int(sv)
                        pids[sv_int] = self.resolver.get(sv_int, {}).get(2, sv_int)
            sibling_data[sid_int] = {
                "children": ch,
                "raw_cx": raw,
                "partner_ids": pids,
            }
        return {
            "old_to_new": dict(old_to_new),
            "new_node_ids": set(new_node_ids),
            "sibling_data": sibling_data,
        }

    # -- Row access --

    def put_parent(self, node_id: int, parent: int) -> None:
        self._rows.set_parent(node_id, parent)

    def put_children(self, node_id: int, children: np.ndarray) -> None:
        self._rows.set_children(node_id, children)

    def put_acx(self, node_id: int, acx_d: dict) -> None:
        self._rows.set_acx(node_id, acx_d)

    def put_cx(self, node_id: int, cx_d: dict) -> None:
        self._rows.set_cx(node_id, cx_d)

    def set_cx_layer(self, node_id: int, layer: int, edges: np.ndarray) -> None:
        self._rows.set_cx_layer(node_id, layer, edges)

    def has(self, node_id: int) -> bool:
        return self._rows.has(node_id)

    def has_batch(self, node_ids: np.ndarray) -> np.ndarray:
        return self._rows.has_batch(node_ids)

    @property
    def parents(self):
        return _ColView(self._rows, 'parent')

    @property
    def children(self):
        return _ColView(self._rows, 'children')

    @property
    def acx(self):
        return _ColView(self._rows, 'acx')

    @property
    def cx(self):
        return _ColView(self._rows, 'cx')

    # -- Merge --

    def merge_reader(self, snapshot) -> None:
        if isinstance(snapshot, dict):
            self._rows.merge_local(snapshot)
        else:
            rp, rc, ra = snapshot
            for nid, p in rp.items():
                self._rows.set_parent(int(nid), p)
            for nid, ch in rc.items():
                self._rows.set_children(int(nid), ch)
            for nid, acx_d in ra.items():
                self._rows.set_acx(int(nid), acx_d)

    def merge_inc(self, inc_snap: dict) -> None:
        self.accumulated_replacements.update(inc_snap["old_to_new"])
        self._new_node_ids.update(inc_snap["new_node_ids"])
        for sib_int, sdata in inc_snap["sibling_data"].items():
            self._siblings[sib_int] = SiblingEntry(
                unresolved_acx=sdata["raw_cx"],
                resolver_entries={int(sv): {2: sib_int} for sv in sdata["children"]},
                partner_ids=sdata.get("partner_ids", {}),
            )

    # -- Snapshot --

    def preloaded(self):
        return self._rows.preloaded_data()

    def incremental_state(self) -> tuple:
        return (self._siblings, self.accumulated_replacements, self._new_node_ids)

    def local_snapshot(self):
        return self._rows.local_data()

    # -- Siblings --

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
        return f"{self._rows.stats()} {len(self._siblings)}s"


class _ColView:
    """Dict-like view over a single column in RowCache."""

    def __init__(self, rows: RowCache, col: str) -> None:
        self._rows = rows
        self._col = col

    def __getitem__(self, nid):
        row = self._rows.get(int(nid))
        if row is None:
            raise KeyError(nid)
        v = getattr(row, self._col)
        if v is None:
            raise KeyError(nid)
        return v

    def __contains__(self, nid) -> bool:
        row = self._rows.get(int(nid))
        return row is not None and getattr(row, self._col) is not None

    def get(self, nid, default=None):
        row = self._rows.get(int(nid))
        if row is None:
            return default
        v = getattr(row, self._col)
        return v if v is not None else default

    def __len__(self) -> int:
        count = 0
        for nid in set(self._rows._local) | set(self._rows._preloaded):
            row = self._rows.get(nid)
            if row and getattr(row, self._col) is not None:
                count += 1
        return count


def _tuple_to_rows(preloaded: tuple) -> dict:
    """Convert old-style (parents, children, acx) tuple to row dict."""
    ro_p, ro_c, ro_a = preloaded
    rows = {}
    all_ids = set()
    for d in [ro_p, ro_c, ro_a]:
        if hasattr(d, 'keys'):
            all_ids.update(d.keys())
    for nid in all_ids:
        row = CacheRow()
        nid_int = int(nid)
        if nid in ro_p:
            row.parent = ro_p[nid]
        if nid in ro_c:
            row.children = ro_c[nid]
        if nid in ro_a:
            row.acx = ro_a[nid]
        rows[nid_int] = row
    return rows
