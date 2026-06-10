"""One-shot mesh metadata setup for a CG.

Writes every mesh-related field a new or freshly-copied graph needs
before any mesh fragment can be served:

    1. ``cg.meta.ws_cv.info["mesh"]``  (mesh dir in the watershed cv info.json)
    2. ``cg.meta.ws_cv.mesh.meta.info``  (per-layer sharded mesh spec)
    3. ``cg.meta.ws_cv.info["mesh_metadata"]``  (uniform draco grid + dynamic dir)
    4. ``cg.meta.custom_data["mesh"]``  (CG bigtable meta block)

Idempotent: re-running overwrites the same fields with the current
inputs, with one exception — ``initial_ts`` is preserved if already
set, because changing it after the fact would reclassify every node
id and silently break served manifests.

Also the container entrypoint, run once per graph after root-layer ingest:

    python -m pychunkedgraph.pipeline.meshing.setup <graph_id>

reads ``mesh_config:`` from the mounted dataset yaml (``PCG_DATASET``).
"""

import argparse
import logging
from os import environ

import numpy as np
import yaml

from ...graph.chunkedgraph import ChunkedGraph
from ...meshing.meshgen import get_draco_encoding_settings_for_chunk
from .meta import MeshConfig

logger = logging.getLogger(__name__)

# Predetermined mount path of the dataset yaml (the chart mounts the dataset
# ConfigMap here); overridable for local/testing.
DATASET_PATH = environ.get("PCG_DATASET", "/app/datasets/dataset.yml")


def derive_initial_ts(cg: ChunkedGraph) -> int:
    """Unix-seconds timestamp of a root id sampled from the dataset center.

    ``mesh.initial_ts`` is the threshold ``segregate_node_ids`` (see
    ``meshing/manifest/utils.py``) uses to classify root ids as initial
    vs post-ingest. It must sit above the last initial-ingest commit
    and below any post-ingest commit. Picking a root id near the
    volume center and using its commit timestamp satisfies both bounds
    for any graph that completed initial ingest.

    Walks shells outward from the center of the L2 chunk grid (L1
    shares L2's coordinate grid) and returns the timestamp of the root
    of the first SV found.
    """
    hi = np.asarray(cg.meta.layer_chunk_bounds[2])
    center = hi // 2
    for r in range(int(hi.max()) + 1):
        box = (
            np.array(
                np.meshgrid(
                    np.arange(-r, r + 1),
                    np.arange(-r, r + 1),
                    np.arange(-r, r + 1),
                    indexing="ij",
                )
            )
            .reshape(3, -1)
            .T
        )
        shell = box[np.max(np.abs(box), axis=1) == r]
        coords = np.unique(np.clip(center + shell, 0, hi - 1), axis=0)
        for c in coords:
            chunk_id = cg.get_chunk_id(layer=1, x=int(c[0]), y=int(c[1]), z=int(c[2]))
            svs = list(cg.range_read_chunk(chunk_id))
            if svs:
                sv = svs[len(svs) // 2]
                root = cg.get_root(sv)
                ts = cg.get_node_timestamps(np.array([root]), return_numpy=False)[0]
                return int(ts.timestamp())
    raise RuntimeError("derive_initial_ts: no SVs found anywhere in the volume")


def setup_mesh_meta(
    cg: ChunkedGraph,
    mesh_config: MeshConfig,
) -> dict:
    """Write every mesh.* metadata field this graph needs to serve meshes.

    Writes go to two places: the watershed CloudVolume (steps 1-3, via
    ``info.json`` / ``mesh/info`` on GCS) and the CG's bigtable meta
    block (step 4).

    ``initial_ts`` is set once and never overwritten — if the existing
    bigtable mesh meta already has one, it is reused as-is. Otherwise
    it is derived via :func:`derive_initial_ts` and persisted.

    Returns the mesh meta dict persisted into bigtable.
    """
    cfg = mesh_config.with_graph_id(cg.graph_id)
    existing_mesh = cg.meta.custom_data.get("mesh", {})
    existing_ts = existing_mesh.get("initial_ts")
    initial_ts = int(existing_ts) if existing_ts is not None else derive_initial_ts(cg)
    if existing_ts is not None:
        logger.info("preserving existing initial_ts=%d", initial_ts)

    # 1. watershed CV info — mesh dir.
    cg.meta.ws_cv.info["mesh"] = cfg.dir
    cg.meta.ws_cv.commit_info()
    logger.info("wrote ws_cv.info['mesh']=%r", cfg.dir)

    # 2. sharded mesh spec — same template per layer, layer-specific bits.
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
    mesh_spec = {
        "@type": "neuroglancer_legacy_mesh",
        "spatial_index": None,
        "mip": int(cfg.mip),
        "chunk_size": list(cfg.chunk_size),
        "sharding": sharding,
    }
    cg.meta.ws_cv.mesh.meta.info = mesh_spec
    cg.meta.ws_cv.mesh.meta.commit_info()
    logger.info("wrote sharded mesh spec for layers %s", sorted(sharding.keys()))

    # 3. uniform draco grid size, derived from layer-2 draco settings.
    draco = get_draco_encoding_settings_for_chunk(
        cg, cg.get_chunk_id(layer=2, x=0, y=0, z=0), mip=cfg.mip
    )
    grid_size = draco["quantization_range"] / (2 ** draco["quantization_bits"] - 1)
    cg.meta.ws_cv.info["mesh_metadata"] = {
        "uniform_draco_grid_size": grid_size,
        "unsharded_mesh_dir": "dynamic",
    }
    cg.meta.ws_cv.commit_info()
    logger.info("wrote mesh_metadata uniform_draco_grid_size=%s", grid_size)

    # 4. CG-side bigtable meta block.
    mesh_meta = {
        "max_layer": int(cfg.max_layer),
        "dynamic_mesh_dir": cfg.dynamic_mesh_dir,
        "mip": int(cfg.mip),
        "max_error": int(cfg.max_error),
        "dir": cfg.dir,
        "initial_ts": int(initial_ts),
    }
    cg.meta.custom_data["mesh"] = mesh_meta
    cg.update_meta(cg.meta, overwrite=True)
    logger.info("wrote cg.meta.custom_data['mesh']=%r", mesh_meta)
    return mesh_meta


def main() -> None:
    parser = argparse.ArgumentParser(prog="pychunkedgraph.pipeline.meshing.setup")
    parser.add_argument("graph_id")
    args = parser.parse_args()
    with open(DATASET_PATH) as stream:
        config = yaml.safe_load(stream)
    if "mesh_config" not in config:
        raise SystemExit(
            f"{DATASET_PATH} has no `mesh_config:` block — required for mesh meta setup."
        )
    mesh_cfg = MeshConfig.from_dict(config["mesh_config"])
    cg = ChunkedGraph(graph_id=args.graph_id)
    result = setup_mesh_meta(cg, mesh_cfg)
    print(f"mesh meta written for {args.graph_id}: {result}")


if __name__ == "__main__":
    main()
