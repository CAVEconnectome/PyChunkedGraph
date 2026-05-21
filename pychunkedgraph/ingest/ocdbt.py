"""OCDBT-specific ingest helpers.

Single home for everything OCDBT-related at the ingest layer:
  * coordinator-server lifecycle (`coordinator`)
  * per-chunk populate task (`populate_chunk`), used from `create_parent_chunk`
  * shared base setup (`setup_base`), used by both ingest and upgrade CLIs
"""

from contextlib import contextmanager
from os import environ

import tensorstore as ts

from pychunkedgraph import get_logger

from ..graph.ocdbt import (
    OcdbtConfig,
    _layer_bbox,
    base_exists,
    copy_ws_bbox_multiscale,
    create_base_ocdbt,
    fork_base_manifest,
    mark_chunk_populated,
    open_base_ocdbt,
    read_populate_meta,
    write_populate_meta,
)

logger = get_logger(__name__)

_COORD_HOST_KEY = "OCDBT_COORDINATOR_HOST"
_COORD_PORT_KEY = "OCDBT_COORDINATOR_PORT"


@contextmanager
def coordinator(redis):
    """Start a ``DistributedCoordinatorServer`` and advertise its address in
    Redis so parallel populate workers route every OCDBT commit through this
    one server — no manifest-CAS races, no orphan ``d/`` files.

    The server lives as long as the ``with`` block does; on exit the Redis
    advertisement is cleared so a stale address can't outlive the server.
    Caller blocks inside the ``with`` body (e.g. ``while True: sleep(60)``)
    to keep the server reference alive across the populate phase.
    """
    server = ts.ocdbt.DistributedCoordinatorServer()
    host = environ.get("MY_POD_IP", "localhost")
    redis.set(_COORD_HOST_KEY, host)
    redis.set(_COORD_PORT_KEY, str(server.port))
    logger.note(f"OCDBT Coordinator listening at {host}:{server.port}")
    try:
        yield server
    finally:
        redis.delete(_COORD_HOST_KEY, _COORD_PORT_KEY)
        logger.note("OCDBT Coordinator advertisement cleared.")


def get_coordinator_address(redis) -> str:
    """Return the advertised ``"host:port"`` for the OCDBT coordinator.

    The address goes into the OCDBT kvstore spec's ``coordinator`` field —
    the only routing knob tensorstore actually honors (verified against
    the tensorstore binary; ``OCDBT_COORDINATOR_HOST/PORT`` env vars are
    not consulted).

    Distributed callers MUST go through this getter so the populate fails
    loudly when the coordinator isn't advertised — uncoordinated parallel
    commits race the shared manifest and leak orphan ``d/`` files, the
    exact bug this code exists to prevent.
    """
    host = redis.get(_COORD_HOST_KEY)
    port = redis.get(_COORD_PORT_KEY)
    if not host or not port:
        raise RuntimeError(
            "OCDBT coordinator address not advertised in Redis "
            f"({_COORD_HOST_KEY}/{_COORD_PORT_KEY} unset). "
            "Run `flask ingest layer N` (with N == ocdbt_populate_layer) to "
            "start the coordinator before queuing populate workers."
        )
    return f"{host.decode()}:{port.decode()}"


def populate_chunk(
    imanager, ws: str, layer: int, coords, coordinator_address: str | None = None
) -> None:
    """One LN parent-layer task's OCDBT populate.

    When ``coordinator_address`` is set, every commit routes through that
    server (mandatory for distributed workers — see ``get_coordinator_address``).
    Single-process callers (notebooks, local one-off runs) can omit it and
    write directly; safe as long as no other writer is committing concurrently.

    Copies the base-resolution bbox at every scale under one atomic
    transaction and records the per-chunk completion marker.
    """
    cfg = OcdbtConfig.from_dict(imanager.ocdbt_config)
    src_list, dst_list, resolutions = open_base_ocdbt(
        ws, cfg, coordinator_address=coordinator_address
    )
    lo, hi = _layer_bbox(imanager.cg.meta, layer, coords)
    copy_ws_bbox_multiscale(src_list, dst_list, resolutions, lo, hi)
    mark_chunk_populated(ws, layer, coords)


def setup_base(cg, ocdbt_cfg: OcdbtConfig) -> OcdbtConfig:
    """Idempotent OCDBT base + fork setup, shared by ingest and upgrade.

    Creates the base if missing; reconciles the yaml/CLI-supplied config
    with the on-disk populate_meta (info-file wins per
    ``OcdbtConfig.resolve``); persists the resolved config to
    ``cg.meta.custom_data["ocdbt_config"]``; forks the manifest for this
    CG. Returns the resolved OcdbtConfig. To wipe and start over, use
    ``gcloud storage rm -r gs://<ws>/ocdbt/`` before invoking.
    """
    ws = cg.meta.data_source.WATERSHED
    if not base_exists(ws):
        create_base_ocdbt(ws, ocdbt_cfg)
    info = read_populate_meta(ws)
    resolved = OcdbtConfig.resolve(ocdbt_cfg.to_dict(), info)
    if resolved.populate_base:
        write_populate_meta(ws, resolved.to_dict())
    cg.meta.custom_data["ocdbt_config"] = resolved.to_dict()
    cg.update_meta(cg.meta, overwrite=True)
    fork_base_manifest(ws, cg.meta.graph_id)
    return resolved
