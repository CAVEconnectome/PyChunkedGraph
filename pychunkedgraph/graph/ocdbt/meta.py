"""OcdbtConfig dataclass — single source of truth for per-CG OCDBT settings."""

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class OcdbtConfig:
    """Per-CG OCDBT settings, persisted in ``ChunkedGraphMeta.custom_data["ocdbt_config"]``.

    Carries both ingest-time choices (populate base? at which layer?) and
    tensorstore kvstore options (compression, inline byte cap) that must
    stay consistent for the lifetime of the OCDBT base. Built once from
    the dataset yaml's ``ocdbt_config:`` section and stored alongside the
    CG so every code path that opens an OCDBT store reads back the same
    values.
    """

    enabled: bool = False
    populate_base: bool = False
    populate_layer: int = 3
    sv_split_threshold: int = 10
    compression: Dict = field(default_factory=lambda: {"id": "zstd", "level": 12})
    # Inline chunk values into B+tree leaves so they share the leaf's zstd
    # compression context. Default tensorstore value (100 bytes) puts every
    # chunk in its own out-of-line blob with independent zstd framing →
    # ~7× bloat on GCS. 1 MiB is tensorstore's hard ceiling for this field
    # and captures every compressed_segmentation chunk we've measured.
    max_inline_value_bytes: int = 1048576

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "OcdbtConfig":
        """Build from a dict. Unknown keys are ignored so older configs don't
        break newer code, and newer fields default in when older configs are
        loaded.
        """
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def resolve(cls, *dicts: Optional[Dict]) -> "OcdbtConfig":
        """Layered merge: later dicts override earlier ones, all over defaults.

        Use to express precedence — e.g. ``resolve(yaml_dict, info_file_dict)``
        gives info-file values priority over yaml-supplied ones, with
        dataclass defaults filling anything neither side specifies.
        ``None`` and empty dicts are no-ops.
        """
        merged: Dict = {}
        for d in dicts:
            if d:
                merged.update(d)
        return cls.from_dict(merged)

    def to_dict(self) -> Dict:
        return asdict(self)

    def ts_config(self) -> Dict:
        """The subset that belongs inside a tensorstore OCDBT kvstore ``config``."""
        return {
            "compression": dict(self.compression),
            "max_inline_value_bytes": self.max_inline_value_bytes,
        }
