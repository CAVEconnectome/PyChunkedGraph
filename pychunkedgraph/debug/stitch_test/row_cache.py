"""
Row-based cache mirroring BigTable structure. See CACHE_DESIGN.md.

Each cached node is a CacheRow with columns matching BigTable:
  parent   — Hierarchy.Parent (int)
  children — Hierarchy.Child (np.ndarray)
  acx      — AtomicCrossChunkEdge per layer (dict, L2 only)
  cx       — CrossChunkEdge per layer (dict, resolved during stitch)

Two-layer lookup: _local (reads + creates) → _preloaded (fork COW).
No ChainMap — 7x faster has_batch.

Invariants:
  - No row read from BigTable more than once (has gates reads)
  - Created rows never read from BigTable (put during stitch)
  - No unnecessary writes (dirty_siblings gates write skip)
"""

import numpy as np

from pychunkedgraph.graph import basetypes


class CacheRow:
    __slots__ = ('parent', 'children', 'acx', 'cx')

    def __init__(self) -> None:
        self.parent = None
        self.children = None
        self.acx = None
        self.cx = None


class RowCache:

    def __init__(self, preloaded: dict = None) -> None:
        self._local: dict[int, CacheRow] = {}
        self._preloaded: dict[int, CacheRow] = preloaded or {}

    def clear_local(self) -> None:
        self._local.clear()

    def promote_local(self) -> None:
        self._preloaded.update(self._local)
        self._local.clear()

    def get(self, node_id: int) -> CacheRow | None:
        return self._local.get(node_id) or self._preloaded.get(node_id)

    def has(self, node_id: int) -> bool:
        return node_id in self._local or node_id in self._preloaded

    def has_batch(self, node_ids: np.ndarray) -> np.ndarray:
        local = self._local
        pre = self._preloaded
        return np.array([int(n) in local or int(n) in pre for n in node_ids])

    def put(self, node_id: int) -> CacheRow:
        row = self._local.get(node_id)
        if row is None:
            pre = self._preloaded.get(node_id)
            if pre is not None:
                row = CacheRow()
                row.parent = pre.parent
                row.children = pre.children
                row.acx = pre.acx
                row.cx = pre.cx
            else:
                row = CacheRow()
            self._local[node_id] = row
        return row

    def get_parent(self, node_id: int) -> int | None:
        row = self.get(node_id)
        return row.parent if row else None

    def get_children(self, node_id: int) -> np.ndarray | None:
        row = self.get(node_id)
        return row.children if row else None

    def get_acx(self, node_id: int) -> dict | None:
        row = self.get(node_id)
        return row.acx if row else None

    def get_cx(self, node_id: int) -> dict | None:
        row = self.get(node_id)
        return row.cx if row else None

    def set_parent(self, node_id: int, parent: int) -> None:
        self.put(node_id).parent = parent

    def set_children(self, node_id: int, children: np.ndarray) -> None:
        self.put(node_id).children = children

    def set_acx(self, node_id: int, acx_d: dict) -> None:
        self.put(node_id).acx = acx_d

    def set_cx(self, node_id: int, cx_d: dict) -> None:
        self.put(node_id).cx = cx_d

    def set_cx_layer(self, node_id: int, layer: int, edges: np.ndarray) -> None:
        row = self.put(node_id)
        if row.cx is None:
            row.cx = {}
        row.cx[layer] = edges

    def clear_cx(self, node_id: int) -> None:
        row = self._local.get(node_id)
        if row:
            row.cx = None

    def local_data(self) -> dict[int, CacheRow]:
        return self._local

    def preloaded_data(self) -> dict[int, CacheRow]:
        return {**self._preloaded, **self._local}

    def stats(self) -> str:
        return f"cache: {_fmt(len(self._local))} local, {_fmt(len(self._preloaded))} preloaded"

    def merge_local(self, other_local: dict[int, CacheRow]) -> None:
        for nid, row in other_local.items():
            existing = self._local.get(nid)
            if existing is None:
                self._local[nid] = row
            else:
                if row.parent is not None:
                    existing.parent = row.parent
                if row.children is not None:
                    existing.children = row.children
                if row.acx is not None:
                    existing.acx = row.acx
                if row.cx is not None:
                    existing.cx = row.cx


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)
