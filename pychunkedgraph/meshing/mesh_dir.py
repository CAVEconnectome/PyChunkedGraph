"""Single source of truth for graphene mesh locations.

Pure-stdlib so it is a cheap top-level import and never pulls cloudvolume.
Depends only on accessors identical across the pcgv2 and pcgv3 lines:
``custom_data["mesh"]`` (``dir`` / ``dynamic_mesh_dir`` / ``path``) and
``data_source.WATERSHED``.
"""

import posixpath
from typing import Dict, Optional, Tuple

_DEFAULT_DIR = "graphene_meshes"


def resolve(custom_data: Dict, watershed: str) -> str:
    """Absolute mesh dir: pinned ``custom_data['mesh']['path']`` when set, else
    ``<watershed>/<dir>`` (dir defaults ``graphene_meshes``)."""
    mesh = custom_data.get("mesh", {})
    path = mesh.get("path")
    if path:
        return path.rstrip("/")
    return posixpath.join(watershed, mesh.get("dir", _DEFAULT_DIR))


def served_abs(custom_data: Dict, watershed: str) -> Optional[str]:
    """Value for the served ``mesh_dir_abs`` /info key; ``None`` unless a path is
    pinned, so legacy /info stays byte-identical for un-migrated datasets."""
    if not custom_data.get("mesh", {}).get("path"):
        return None
    return resolve(custom_data, watershed)


def dynamic(custom_data: Dict, watershed: str, subdir: str) -> str:
    """Absolute dynamic-mesh dir ``<resolve>/<subdir>``. ``subdir`` is the dir the
    caller already computes, so the path is unchanged for un-migrated datasets."""
    return posixpath.join(resolve(custom_data, watershed), subdir)


def anchor_split(custom_data: Dict, watershed: str) -> Tuple[str, str]:
    """``(data_dir, mesh)`` such that an unpatched cloudvolume composing
    ``join(data_dir, mesh)`` lands at ``resolve``."""
    return posixpath.split(resolve(custom_data, watershed))


def pin_mesh_path(cg, path: Optional[str] = None) -> str:
    """Freeze the absolute mesh dir into ``custom_data['mesh']['path']`` and
    persist; defaults to today's resolved location."""
    meta = cg.meta
    mesh = dict(meta.custom_data.get("mesh", {}))
    mesh["path"] = (path or resolve(meta.custom_data, meta.data_source.WATERSHED)).rstrip("/")
    meta.custom_data["mesh"] = mesh
    cg.update_meta(meta, overwrite=True)
    return mesh["path"]


def set_watershed(cg, watershed: str) -> None:
    """Repoint the watershed/supervoxel source and persist. Re-instantiate the
    ChunkedGraph afterward — in-memory ws caches are not invalidated here."""
    meta = cg.meta
    meta._data_source = meta._data_source._replace(WATERSHED=watershed)
    cg.update_meta(meta, overwrite=True)
