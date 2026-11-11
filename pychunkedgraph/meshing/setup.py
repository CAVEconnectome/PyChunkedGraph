"""One-shot mesh metadata setup for a CG.

Writes every mesh-related field a new or freshly-copied graph needs
before any mesh fragment can be served:

    1. the shard spec ``info`` in ``MeshMeta.spec_dir``  (per-layer sharded mesh spec)
    2. ``cg.meta.custom_data["mesh"]``  (CG bigtable meta block, draco grid size included)

The watershed's own info is not written: the CG meta block is the only mesh source.

Idempotent: re-running overwrites the fields the config sets and keeps the rest of
the block, so ``initial_mesh_dir`` survives, and ``initial_ts`` is never reset, since
changing it would reclassify every node id and silently break served manifests.
"""

import logging
from datetime import datetime, timezone

from cloudfiles import CloudFiles

from ..graph.chunkedgraph import ChunkedGraph
from .meshgen import get_draco_encoding_settings_for_chunk
from .meshgen_utils import get_mesh_block_shape_for_mip
from .mesh_meta import MeshMeta
from .meta import MeshConfig

logger = logging.getLogger(__name__)


def derive_initial_ts(cg: ChunkedGraph) -> int:
    """Unix-seconds boundary for ``mesh.initial_ts`` (see ``segregate_node_ids``).

    ``get_earliest_timestamp`` returns the first edit, or — pre-edit — the
    ingest-completion boundary stamped during the root-layer build. ``+1`` makes the
    second-granularity threshold strictly above the last initial root (the check
    is ``<`` and ``int()`` truncates).
    """
    earliest = cg.get_earliest_timestamp()
    if earliest <= datetime.fromtimestamp(0, tz=timezone.utc):
        raise RuntimeError(
            "derive_initial_ts: no operations and no ingest earliest_ts stamped"
        )
    return int(earliest.timestamp()) + 1


def setup_mesh_meta(
    cg: ChunkedGraph,
    mesh_config: MeshConfig,
) -> dict:
    """Write every mesh.* metadata field this graph needs to serve meshes.

    Writes go to two places: the mesh dir's sharding ``info`` on GCS (step 1)
    and the CG's bigtable meta block (step 2).

    ``initial_ts`` is set once and never overwritten — if the existing
    bigtable mesh meta already has one, it is reused as-is. Otherwise
    it is derived via :func:`derive_initial_ts` and persisted.

    Returns the mesh meta dict persisted into bigtable.
    """
    cfg = mesh_config.with_graph_id(cg.graph_id)
    n_scales = len(cg.meta.ws_cv.info["scales"])
    if not 0 <= cfg.mip < n_scales:
        raise ValueError(
            f"mesh_config.mip {cfg.mip} exceeds watershed scales (available 0..{n_scales - 1})"
        )
    existing_mesh = cg.meta.custom_data.get("mesh", {})
    existing_ts = existing_mesh.get("initial_ts")
    initial_ts = int(existing_ts) if existing_ts is not None else derive_initial_ts(cg)
    if existing_ts is not None:
        logger.info("preserving existing initial_ts=%d", initial_ts)

    # 1. sharded mesh spec — same template per layer, layer-specific bits.
    layer_shard_spec = {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "murmurhash3_x86_128",
        "shard_bits": 0,
        "minishard_index_encoding": "gzip",
        "data_encoding": "raw",
    }
    sharding = {
        str(layer): {**layer_shard_spec, "minishard_bits": int(bits)}
        for layer, bits in cfg.minishard_bits.items()
        if layer <= cfg.max_layer
    }
    mesh_chunk_size = get_mesh_block_shape_for_mip(cg, 2, cfg.mip)
    mesh_spec = {
        "@type": "neuroglancer_legacy_mesh",
        "spatial_index": None,
        "mip": int(cfg.mip),
        "chunk_size": [int(x) for x in mesh_chunk_size],
        "sharding": sharding,
    }
    # uniform draco grid size, derived from layer-2 draco settings.
    draco = get_draco_encoding_settings_for_chunk(
        cg, cg.get_chunk_id(layer=2, x=0, y=0, z=0), mip=cfg.mip
    )
    grid_size = draco["quantization_range"] / (2 ** draco["quantization_bits"] - 1)

    # 2. CG-side bigtable meta block. Keys the config cannot set, such as
    # initial_mesh_dir, are kept, so a re-run never re-points a table's meshes.
    mesh_meta = {
        **existing_mesh,
        "uniform_draco_grid_size": grid_size,
        "max_layer": int(cfg.max_layer),
        "dynamic_mesh_dir": cfg.dynamic_mesh_dir,
        "mip": int(cfg.mip),
        "max_error": int(cfg.max_error),
        "dir": cfg.dir,
        "initial_ts": int(initial_ts),
    }
    cg.meta.custom_data["mesh"] = mesh_meta
    spec_dir = MeshMeta(cg).spec_dir
    CloudFiles(spec_dir).put_json("info", mesh_spec, cache_control="no-cache")
    logger.info("wrote mesh spec for layers %s in %s", sorted(sharding), spec_dir)
    cg.update_meta(cg.meta, overwrite=True)
    logger.info("wrote cg.meta.custom_data['mesh']=%r", mesh_meta)
    return mesh_meta
