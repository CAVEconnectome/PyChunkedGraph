"""MeshMeta — mesh dir locations and per-dataset params for a ChunkedGraph.

``custom_data["mesh"]`` is the single source of truth. See README.md.
"""

import posixpath

#: DEPRECATED shim: client mesh keys older setups wrote into the watershed info only. A key
#: the table's mesh block lacks is copied from there at read time; drop this once every
#: table's block carries them.
_WATERSHED_FALLBACK_KEYS = (
    "uniform_draco_grid_size",
    "max_meshed_layer",
    "draco_grid_sizes",
)


def _resolve(root: str, value: str) -> str:
    """Absolute ``value`` (a cloud URL) as-is, else joined under ``root``."""
    return value.rstrip("/") if "://" in value else posixpath.join(root, value)


class MeshMeta:
    """Mesh metadata for a ChunkedGraph, read from ``custom_data["mesh"]``."""

    def __init__(self, cg):
        self._mesh = cg.meta.custom_data.get("mesh", {})
        self._watershed = cg.meta.data_source.WATERSHED
        self._meta = cg.meta

    @property
    def dir(self) -> str:
        """Sharded mesh dir name under the watershed; setup writes it, so a table without
        one has no meshes and this raises ``KeyError``."""
        return self._mesh["dir"]

    @property
    def _root(self) -> str:
        return posixpath.join(self._watershed, self.dir)

    @property
    def initial_path(self) -> str:
        """Absolute dir of the shared initial (sharded) meshes."""
        return _resolve(self._root, self._mesh.get("initial_mesh_dir", "initial"))

    @property
    def dynamic_mesh_dir(self) -> str:
        """The dynamic dir as configured: a name under ``dir``, or a cloud URL."""
        return self._mesh.get("dynamic_mesh_dir", "dynamic")

    @property
    def dynamic_path(self) -> str:
        """Absolute dir of the per-graph dynamic (unsharded) meshes."""
        return _resolve(self._root, self.dynamic_mesh_dir)

    @property
    def client_metadata(self) -> dict[str, object]:
        """``mesh_metadata`` for a graphene info. cloud-volume reads the dynamic dir as
        ``unsharded_mesh_dir``, and a v2 client reads the two absolute paths."""
        ws_mesh = self._meta.ws_info.get("mesh_metadata", {})
        fallback = {
            key: ws_mesh[key]
            for key in _WATERSHED_FALLBACK_KEYS
            if key in ws_mesh and key not in self._mesh
        }
        own = {
            key: self._mesh[key]
            for key in _WATERSHED_FALLBACK_KEYS
            if key in self._mesh
        }
        return {
            **fallback,
            **own,
            "unsharded_mesh_dir": self.dynamic_mesh_dir,
            "initial_mesh_path": self.initial_path,
            "dynamic_mesh_path": self.dynamic_path,
        }

    @property
    def needs_v2(self) -> bool:
        """True when meshes live outside the watershed dir (old clients can't reach)."""
        root = self._root
        return not (
            self.initial_path.startswith(root) and self.dynamic_path.startswith(root)
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
