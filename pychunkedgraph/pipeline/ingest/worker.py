"""Ingest per-chunk processor: a Bigtable lock + renew heartbeat around dispatch.

Each chunk is claimed under a per-chunk lock (skip if already done, defer if held
live by another worker), processed via the branch-aware dispatch, then marked done
only if we still hold the claim. Plugged into the generic ``pipeline.worker`` harness.
"""

import logging
import os
import threading
import time
from datetime import timedelta

from cave_pipeline.distribution import FatalChunkError, lock
from cave_pipeline.distribution.harness import run

from ...ingest import simple_tests
from .. import cg_factory, layer_bounds
from .dispatch import IngestConfig
from . import dispatch

logger = logging.getLogger(__name__)


def _token() -> int:
    return int.from_bytes(os.urandom(8), "big")


def _atomic_config(cg) -> IngestConfig:
    """Atomic-layer config from stored meta (set by setup) — never yaml/env.

    The agglomeration source lives in ``meta.custom_data["agg"]`` as
    ``{"path": str, "raw": bool}`` (everything else is in ``data_source``).
    """
    agg = cg.meta.custom_data.get("agg", {})
    raw = bool(agg.get("raw", False))
    return IngestConfig(
        AGGLOMERATION=agg.get("path"),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
    )


def _expiry(layer: int, scale: float) -> timedelta:
    """Per-layer lock TTL (L2 = 3 min, +3 min per layer) times a user scale factor.

    The renew heartbeat keeps a live worker's claim fresh, so this only bounds how
    long a *dead* worker's chunk stays locked before retry — short for cheap L2.
    """
    return timedelta(minutes=3 * (layer - 1) * scale)


def _renew_loop(table, chunk_id, token, interval, stop):
    """Keep our claim fresh while the chunk runs; exits on `stop` or if the claim is lost."""
    while not stop.wait(interval):
        if not lock.renew(table, chunk_id, token):
            logger.warning(f"chunk {chunk_id} claim renew failed")
            return


def make_processor(cg, layer, env):
    """Build the ingest per-chunk processor for this batch (binds lock + config)."""
    table = cg.client  # kvdbclient client; lock.* speak its lock_by_row_key API
    config = _atomic_config(cg) if layer == 2 else None
    expiry = _expiry(layer, float(os.environ.get("PCG_LOCK_EXPIRY_SCALE", 1)))
    opts = {
        "n_processes": env["n_processes"],
        "expiry": expiry,
        "renew": expiry.total_seconds() / 3,
        "poll": float(os.environ.get("PCG_LOCK_POLL_SEC", 10)),
        "held_max_wait": float(os.environ.get("PCG_HELD_MAX_WAIT_SEC", 120)),
    }

    def process_one(coord):
        return _process_one(table, cg, layer, coord, config, opts)

    return process_one


def _process_one(table, cg, layer, coord, config, opts) -> str:
    """Returns 'done' | 'ok' | 'transient' | 'fatal' for one chunk."""
    chunk_id = int(cg.get_chunk_id(layer=layer, x=coord[0], y=coord[1], z=coord[2]))
    token = _token()

    deadline = time.monotonic() + opts["held_max_wait"]
    while True:
        state = lock.acquire(table, chunk_id, token, opts["expiry"])
        if state == lock.DONE:
            return "done"
        if state == lock.ACQUIRED:
            break
        if time.monotonic() >= deadline:  # someone else holds it; let the batch retry
            logger.warning(
                f"chunk {layer}_{tuple(coord)} held by another worker; deferring"
            )
            return "transient"
        time.sleep(opts["poll"])

    # Renew the claim while we work so a live worker's in-progress chunk never goes
    # stale — no other worker can start a chunk still being processed. Only a dead
    # worker (heartbeat stopped) lets the claim expire, after which retry is safe.
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_renew_loop,
        args=(table, chunk_id, token, opts["renew"], stop),
        daemon=True,
    )
    heartbeat.start()
    try:
        dispatch.process_chunk(
            cg, layer, coord, config, n_processes=opts["n_processes"]
        )
        outcome = "ok"
    except FatalChunkError:
        logger.exception(f"fatal chunk {layer}_{tuple(coord)}")
        outcome = "fatal"
    except Exception:
        logger.exception(f"transient failure on chunk {layer}_{tuple(coord)}")
        outcome = "transient"
    finally:
        stop.set()
        heartbeat.join()

    if outcome != "ok":
        lock.release(table, chunk_id, token)
        return outcome
    if lock.mark_done(table, chunk_id, token):
        return "ok"
    logger.warning(f"chunk {layer}_{tuple(coord)} claim lost before done; deferring")
    return "transient"


def _verify_root(cg, layer) -> None:
    """Once the root chunk is built, run the hierarchy sanity suite.

    The chunk is already marked done, so a re-submitted root layer re-runs only this
    check, never the build. The mesh ingest boundary (``earliest_ts``) is stamped during
    the root-layer write itself, where the roots are already in memory.
    """
    if layer != cg.meta.layer_count:
        return
    simple_tests.run_all(cg)


def main() -> int:
    return run(
        make_processor,
        context_factory=cg_factory,
        bounds_fn=layer_bounds,
        finalize=_verify_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
