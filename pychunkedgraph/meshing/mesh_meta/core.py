"""MeshMeta — mesh dir locations and per-dataset params for a ChunkedGraph.

``custom_data["mesh"]`` is the single source of truth. See README.md.
"""

import posixpath
from typing import NamedTuple

import numpy as np

#: DEPRECATED shim: client mesh keys older setups wrote into the watershed info only. A key
#: the table's mesh block lacks is copied from there at read time; drop this once every
#: table's block carries them.
_WATERSHED_FALLBACK_KEYS = (
    "uniform_draco_grid_size",
    "max_meshed_layer",
    "draco_grid_sizes",
)


def is_absolute(value: str) -> bool:
    """Whether mesh dir is cloud URL used as-is, rather than name under root."""
    return "://" in value


def _resolve(root: str, value: str) -> str:
    """Absolute ``value`` (a cloud URL) as-is, else joined under ``root``."""
    return value.rstrip("/") if is_absolute(value) else posixpath.join(root, value)


#: The block key a copy's rewrite fills: its sources' dynamic dirs, oldest first.
INHERITED = "inherited_dynamic"


class Layer(NamedTuple):
    """A source's dynamic dir, holding the meshes of nodes made before ``until``."""

    path: str
    #: Unix seconds: the copy's first open, before which the source made the node.
    until: int


class MeshMeta:
    """Mesh metadata of a ChunkedGraph or its meta alone, from ``custom_data["mesh"]``."""

    def __init__(self, cg):
        meta = getattr(cg, "meta", cg)
        self._mesh = meta.custom_data.get("mesh", {})
        self._watershed = meta.data_source.WATERSHED
        self._meta = meta

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
    def inherited(self) -> tuple[Layer, ...]:
        """Dynamic dirs inherited as a copy, oldest first; none for any other table."""
        return tuple(
            Layer(str(one["path"]), int(one["until"]))
            for one in self._mesh.get(INHERITED, ())
        )

    @property
    def dynamic_stack(self) -> tuple[str, ...]:
        """Every dynamic dir a node's mesh may be in, in the order a miss searches them:
        the table's own first, then inherited dirs newest first."""
        return (self.dynamic_path, *(one.path for one in reversed(self.inherited)))

    def by_dynamic_dir(self, node_ids, created) -> dict[str, np.ndarray]:
        """New node ids grouped by the dynamic dir their creation time hints at, each dir
        once: the first inherited dir a node's ``created`` (datetime64) predates, else
        ``dynamic_path``. ``created`` must hold one time per id; this is not checked,
        since every caller reads the times of nodes that exist."""
        node_ids = np.asarray(node_ids)
        if not node_ids.size:
            return {}
        layers = self.inherited
        if not layers:
            return {self.dynamic_path: node_ids}
        untils = np.array([np.datetime64(one.until, "s") for one in layers])
        at = np.searchsorted(
            untils, np.asarray(created, dtype="datetime64[s]"), side="right"
        )
        paths = [one.path for one in layers] + [self.dynamic_path]
        return {paths[i]: node_ids[at == i] for i in np.unique(at)}

    @property
    def _grid_keys(self) -> dict[str, object]:
        """The client grid keys: the block's own, else the watershed info's (the shim)."""
        ws_mesh = self._meta.ws_info.get("mesh_metadata", {})
        return {
            key: self._mesh[key] if key in self._mesh else ws_mesh[key]
            for key in _WATERSHED_FALLBACK_KEYS
            if key in self._mesh or key in ws_mesh
        }

    @property
    def client_metadata(self) -> dict[str, object]:
        """``mesh_metadata`` for a graphene info. cloud-volume reads the dynamic dir as
        ``unsharded_mesh_dir``, and a v2 client reads the two absolute paths."""
        return {
            **self._grid_keys,
            "unsharded_mesh_dir": self.dynamic_mesh_dir,
            "initial_mesh_path": self.initial_path,
            "dynamic_mesh_path": self.dynamic_path,
        }

    @property
    def pinned_block(self) -> dict[str, object]:
        """The block with every value it takes from the watershed written in, so it reads
        the same under any watershed; a block without ``dir`` takes none."""
        if "dir" not in self._mesh:
            return dict(self._mesh)
        return {
            **self._mesh,
            **self._grid_keys,
            "initial_mesh_dir": self.initial_path,
            "dynamic_mesh_dir": self.dynamic_path,
        }

    @property
    def needs_v2(self) -> bool:
        """True when meshes live outside the watershed dir, or across inherited dynamic
        dirs: old clients reach neither."""
        root = self._root
        return bool(self.inherited) or not (
            self.initial_path.startswith(root) and self.dynamic_path.startswith(root)
        )

    @property
    def spec_dir(self) -> str:
        """Dir of the shard spec ``info``: the parent of ``initial_path``, wherever that
        lives, so a watershed elsewhere never hides it."""
        return posixpath.dirname(self.initial_path)

    @property
    def reader_anchor(self):
        """``(data_dir, mesh)`` so a CloudVolume that appends ``initial/`` lands at
        ``initial_path`` — points the shard reader at a migrated bucket."""
        return posixpath.split(self.spec_dir)

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
