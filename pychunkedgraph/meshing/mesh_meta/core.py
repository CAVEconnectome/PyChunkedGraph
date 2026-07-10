"""MeshMeta — mesh dir locations and per-dataset params for a ChunkedGraph.

``custom_data["mesh"]`` is the single source of truth. See README.md.
"""

import posixpath

_DEFAULT_MESH_DIR = "graphene_meshes"


def _resolve(root: str, value: str) -> str:
    """Absolute ``value`` (a cloud URL) as-is, else joined under ``root``."""
    return value.rstrip("/") if "://" in value else posixpath.join(root, value)


class MeshMeta:
    """Mesh metadata for a ChunkedGraph, read from ``custom_data["mesh"]``."""

    def __init__(self, cg):
        self._mesh = cg.meta.custom_data.get("mesh", {})
        self._watershed = cg.meta.data_source.WATERSHED

    @property
    def dir(self) -> str:
        """Sharded mesh dir name under the watershed."""
        return self._mesh.get("dir", _DEFAULT_MESH_DIR)

    @property
    def _root(self) -> str:
        return posixpath.join(self._watershed, self.dir)

    @property
    def initial_path(self) -> str:
        """Absolute dir of the shared initial (sharded) meshes."""
        return _resolve(self._root, self._mesh.get("initial_mesh_dir", "initial"))

    @property
    def dynamic_path(self) -> str:
        """Absolute dir of the per-graph dynamic (unsharded) meshes."""
        return _resolve(self._root, self._mesh.get("dynamic_mesh_dir", "dynamic"))

    @property
    def needs_v2(self) -> bool:
        """True when meshes live outside the watershed dir (old clients can't reach)."""
        root = self._root
        return not (
            self.initial_path.startswith(root)
            and self.dynamic_path.startswith(root)
        )

    @property
    def reader_anchor(self):
        """``(data_dir, mesh)`` so a CloudVolume that appends ``initial/`` lands at
        ``initial_path`` — points the shard reader at a migrated bucket."""
        return posixpath.split(posixpath.dirname(self.initial_path))

    @property
    def max_layer(self) -> int:
        return self._mesh.get("max_layer", 2)

    @property
    def mip(self) -> int:
        return self._mesh["mip"]

    @property
    def max_error(self):
        return self._mesh["max_error"]

    @property
    def initial_ts(self):
        return self._mesh["initial_ts"]
