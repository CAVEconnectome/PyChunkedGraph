"""Where a dynamic mesh lives, across a table's own dynamic dir and the dirs it inherits.

A node's creation time hints at its dir; a probe checks the hint, and a miss searches the
other dirs that can hold the node, own dir first, so a table finds every mesh it wrote
whatever the hint says, and never another node's under a reused id.
"""

import posixpath
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from cloudfiles import CloudFiles

from ..meshgen_utils import get_mesh_name
from ..mesh_meta import MeshMeta
from ...graph.basetypes import NODE_ID

#: What ``search_dynamic`` asks of a dir: each name it holds, mapped to a truthy value.
Probe = Callable[[str, List[str]], Dict[str, object]]


class Found(NamedTuple):
    """A dynamic mesh as found: its dir, its file name, and what the probe read."""

    path: str
    name: str
    value: object


def dynamic_groups(cg, new_ids) -> Dict[str, np.ndarray]:
    """``new_ids`` grouped by the dynamic dir their creation time hints at; node
    timestamps are read only for a copy, the one table with inherited dirs."""
    mm = MeshMeta(cg)
    new_ids = np.asarray(new_ids, dtype=NODE_ID)
    created = cg.get_node_timestamps(new_ids) if mm.inherited and new_ids.size else None
    return mm.by_dynamic_dir(new_ids, created)


@lru_cache(maxsize=None)
def dynamic_store(path: str) -> CloudFiles:
    """One store of a dynamic dir, made once per process and shared by every reader."""
    return CloudFiles(path)


def probe_exists(path: str, names: List[str]) -> Dict[str, bool]:
    """Whether ``path`` holds each named mesh."""
    return dynamic_store(path).exists(names)


def probe_read(path: str, names: List[str]) -> Dict[str, Optional[bytes]]:
    """Each named mesh's content in ``path``, ``None`` where it holds none."""
    contents = {}
    for res in dynamic_store(path).get(names):
        if res["error"]:
            raise res["error"]
        contents[res["path"]] = res["content"]
    return contents


def search_dynamic(cg, new_ids, probe: Probe) -> Dict[int, Found]:
    """Each new node's dynamic mesh, probed first in its creation-time dir, then on a
    miss in each of that hint's ``fallback_dirs``, each dir probed once, in
    ``dynamic_stack`` order; a node in none is absent.
    """
    mm = MeshMeta(cg)
    names = {int(i): get_mesh_name(cg, i) for i in new_ids}
    hint, found = {}, {}

    def run(path: str, ids: List[int]) -> None:
        got = probe(path, [names[i] for i in ids])
        found.update(
            {i: Found(path, names[i], got[names[i]]) for i in ids if got.get(names[i])}
        )

    for path, ids in dynamic_groups(cg, list(names)).items():
        hint.update({int(i): path for i in ids})
        run(path, [int(i) for i in ids])
    fallbacks = {path: mm.fallback_dirs(path) for path in set(hint.values())}
    for path in mm.dynamic_stack:
        rest = [i for i in names if i not in found and path in fallbacks[hint[i]]]
        if rest:
            run(path, rest)
    return found


def dynamic_row(own: str, path: str, name: str) -> str:
    """A dynamic mesh's row: its file name, under its dir unless that dir is ``own``."""
    return name if path == own else posixpath.join(path, name)


def row_location(own: str, row: str) -> Tuple[str, str]:
    """A dynamic row's dir and file name, the inverse of ``dynamic_row``."""
    path, name = posixpath.split(row)
    return path or own, name


def hinted_rows(cg, new_ids) -> List[str]:
    """Unprobed rows for ``new_ids``, in order, each under its creation-time hint dir."""
    own = MeshMeta(cg).dynamic_path
    hint = {
        int(i): path for path, ids in dynamic_groups(cg, new_ids).items() for i in ids
    }
    return [dynamic_row(own, hint[int(i)], get_mesh_name(cg, i)) for i in new_ids]


def dynamic_buckets(cg, rows: Sequence[str]) -> Dict[str, List[str]]:
    """v2 dynamic rows grouped by the dir holding each, read off the row: no timestamp."""
    own = MeshMeta(cg).dynamic_path
    buckets: Dict[str, List[str]] = {}
    for row in rows:
        path, name = row_location(own, row)
        buckets.setdefault(path, []).append(name)
    return buckets
