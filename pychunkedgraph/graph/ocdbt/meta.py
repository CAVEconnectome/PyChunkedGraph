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
    # Inline-vs-out-of-line threshold. Values ≤ this size live in the btree
    # leaf bytes; larger values get written to a d/ file and the mutation
    # carries only an IndirectDataReference. This directly determines
    # cooperator-forwarded RPC size in distributed mode: inline values are
    # carried inside the gRPC WriteRequest's `mutations` field, so a leaf's
    # batch can blow past tensorstore's hardcoded 4 MiB gRPC max-receive
    # whenever multiple inline values pile up on the same node. Verified
    # by reading btree_writer.cc StagePending in v0.1.81.
    #
    # 4 KiB keeps small metadata (info JSON ~1.5 KB, populate-marker files)
    # inline while forcing every segmentation chunk value out-of-line —
    # chunks compress to 100s of KB even for the smallest scales. With
    # chunk bytes out-of-line the WriteRequest stays tiny regardless of
    # how many keys a worker commits at once. Tradeoff vs the previous
    # 1 MiB cap: each chunk now has its own zstd-framed d/ blob instead of
    # sharing a leaf's compression context, which can cost a few percent
    # of compression ratio (much less than the originally-feared "7×
    # bloat", which only applied at the 100-byte default).
    max_inline_value_bytes: int = 4096

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
