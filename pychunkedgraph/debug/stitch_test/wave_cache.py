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
    partner_ids: dict = field(default_factory=dict)


class WaveCache:

    def __init__(self, read_fn, preloaded=None, incremental: tuple = None) -> None:
        self._read_fn = read_fn
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

        # Collect all partner SVs across all siblings, batch get_parents once
        all_partner_svs = set()
        sib_partner_map = {}
        for sib in self.sibling_ids:
            sib_int = int(sib)
            if sib_int not in self._siblings:
                self.dirty_siblings.add(sib_int)
                continue
            raw = self.unresolved_acx.get(sib_int, {})
            svs = []
            for layer_edges in raw.values():
                if len(layer_edges) > 0:
                    svs.append(layer_edges[:, 1])
            if svs:
                partner_svs = np.concatenate(svs).astype(np.int64)
                sib_partner_map[sib_int] = partner_svs
                all_partner_svs.update(int(sv) for sv in partner_svs)

        # Batch resolve all partner SVs → L2 via cache
        if all_partner_svs:
            sv_arr = np.array(list(all_partner_svs), dtype=basetypes.NODE_ID)
            parents = self.get_parents(sv_arr)
            sv_to_l2 = {int(sv): int(p) for sv, p in zip(sv_arr, parents)}
        else:
            sv_to_l2 = {}

        affected_sorted = np.sort(np.array(list(affected_ids), dtype=np.int64)) if affected_ids else np.array([], dtype=np.int64)

        for sib_int, partner_svs in sib_partner_map.items():
            identities = np.array([sv_to_l2.get(int(sv), int(sv)) for sv in partner_svs], dtype=np.int64)
            if len(affected_sorted) > 0:
                aff_idx = np.searchsorted(affected_sorted, identities)
                aff_idx = np.clip(aff_idx, 0, len(affected_sorted) - 1)
                if np.any(affected_sorted[aff_idx] == identities):
                    self.dirty_siblings.add(sib_int)
                    continue
            entry = self._siblings[sib_int]
            if entry.partner_ids:
                for sv, old_id in entry.partner_ids.items():
                    new_id = sv_to_l2.get(int(sv), int(sv))
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
        pids_map = self._build_partner_ids(sibling_ids, unresolved_acx)
        for sib_int in (int(s) for s in sibling_ids):
            self._siblings[sib_int] = SiblingEntry(
                unresolved_acx=unresolved_acx.get(sib_int, {}),
                partner_ids=pids_map.get(sib_int, {}),
            )

    def inc_snapshot_from(
        self, old_to_new: dict, new_node_ids: set,
        sibling_ids: set, unresolved_acx: dict,
    ) -> dict:
        pids_map = self._build_partner_ids(sibling_ids, unresolved_acx)
        sibling_data = {}
        for sid_int in (int(s) for s in sibling_ids):
            ch = self._rows.get_children(sid_int)
            ch = ch if ch is not None else np.array([], dtype=basetypes.NODE_ID)
            sibling_data[sid_int] = {
                "children": ch,
                "raw_cx": unresolved_acx.get(sid_int, {}),
                "partner_ids": pids_map.get(sid_int, {}),
            }
        return {
            "old_to_new": dict(old_to_new),
            "new_node_ids": set(new_node_ids),
            "sibling_data": sibling_data,
        }

    def _build_partner_ids(self, sibling_ids: set, unresolved_acx: dict) -> dict:
        """Batch resolve all partner SVs → L2 for dirty detection."""
        all_svs = set()
        sib_svs = {}
        for sib_int in (int(s) for s in sibling_ids):
            raw = unresolved_acx.get(sib_int, {})
            svs = []
            for layer_edges in raw.values():
                if len(layer_edges) > 0:
                    svs.extend(int(sv) for sv in layer_edges[:, 1])
            sib_svs[sib_int] = svs
            all_svs.update(svs)
        if not all_svs:
            return {}
        sv_arr = np.array(list(all_svs), dtype=basetypes.NODE_ID)
        parents = self.get_parents(sv_arr)
        sv_to_l2 = {int(sv): int(p) for sv, p in zip(sv_arr, parents)}
        return {
            sib_int: {sv: sv_to_l2.get(sv, sv) for sv in svs}
            for sib_int, svs in sib_svs.items() if svs
        }

    # -- Read function for cache misses --

    def _ensure_read(self, node_ids: np.ndarray) -> None:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        if len(node_ids) == 0:
            return
        uncached = node_ids[~self._rows.has_batch(node_ids)]
        if len(uncached) > 0:
            self._read_fn(uncached)

    # -- Batch read access (ensure cached, then return) --

    def get_parents(self, node_ids: np.ndarray) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_read(node_ids)
        return np.array(
            [self._rows.get_parent(int(n)) or 0 for n in node_ids],
            dtype=basetypes.NODE_ID,
        )

    def get_children_batch(self, node_ids: np.ndarray) -> dict:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_read(node_ids)
        empty = np.array([], dtype=basetypes.NODE_ID)
        result = {}
        for n in node_ids:
            v = self._rows.get_children(int(n))
            result[int(n)] = v if v is not None else empty
        return result

    def get_acx_batch(self, node_ids: np.ndarray) -> dict:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_read(node_ids)
        result = {}
        for n in node_ids:
            v = self._rows.get_acx(int(n))
            result[int(n)] = v if v is not None else {}
        return result

    def get_cx_batch(self, node_ids: np.ndarray) -> dict:
        node_ids = np.asarray(node_ids, dtype=basetypes.NODE_ID)
        self._ensure_read(node_ids)
        result = {}
        for n in node_ids:
            v = self._rows.get_cx(int(n))
            result[int(n)] = v if v is not None else {}
        return result

    # -- Single-node write access (unchanged) --

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
