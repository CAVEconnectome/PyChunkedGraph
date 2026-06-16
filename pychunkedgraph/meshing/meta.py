"""MeshConfig dataclass — single source of truth for per-CG mesh setup values.

Read from the dataset yaml under a ``mesh_config:`` block, exactly like
``OcdbtConfig`` is read from ``ocdbt_config:``. Every static field is
required — the helper that applies it (``setup_mesh_meta``) does not
substitute defaults for missing yaml entries. The only optional field
is :attr:`dynamic_mesh_dir`, which is graph-id-derived and filled in by
:meth:`with_graph_id` when omitted from the yaml. The mesh chunk_size is
derived (CG CHUNK_SIZE / per-axis downsample at ``mip``), not configured.

Example yaml block::

    mesh_config:
      dir: graphene_meshes
      mip: 0
      max_layer: 6
      max_error: 40
      minishard_bits: {2: 1, 3: 3, 4: 6, 5: 9, 6: 12}
      # dynamic_mesh_dir: my_custom_dir  # optional; default "dynamic_<graph_id>"
"""

from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional


@dataclass
class MeshConfig:
    """Per-CG mesh setup config. See module docstring for yaml schema."""

    dir: str
    mip: int
    max_layer: int
    max_error: int
    minishard_bits: Dict[int, int]
    dynamic_mesh_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "MeshConfig":
        """Build from a yaml-parsed dict.

        Unknown keys are dropped (so older yamls don't break newer code).
        ``minishard_bits`` keys are coerced to ``int`` so the yaml is
        tolerant of bare-int vs quoted-string keys.
        """
        if not d:
            raise ValueError(
                "MeshConfig.from_dict: empty config — yaml `mesh_config:` "
                "block is required"
            )
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "minishard_bits" in kwargs:
            kwargs["minishard_bits"] = {
                int(k): int(v) for k, v in kwargs["minishard_bits"].items()
            }
        return cls(**kwargs)

    def with_graph_id(self, graph_id: str) -> "MeshConfig":
        """Return a copy with ``dynamic_mesh_dir`` filled in if unset."""
        if self.dynamic_mesh_dir is not None:
            return self
        return replace(self, dynamic_mesh_dir=f"dynamic_{graph_id}")

    def to_dict(self) -> Dict:
        return asdict(self)
