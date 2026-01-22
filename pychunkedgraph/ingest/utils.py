# pylint: disable=invalid-name, missing-docstring

import logging
import functools
import math, random, sys
from os import environ
from time import sleep
from typing import Any, Generator, Tuple

import numpy as np
import tensorstore as ts
from rq import Queue, Retry, Worker
from rq.worker import WorkerStatus

from . import IngestConfig
from .manager import IngestionManager
from ..graph.meta import ChunkedGraphMeta, DataSource, GraphConfig
from ..graph.client import BackendClientInfo
from ..graph.client.bigtable import BigTableConfig
from ..utils.general import chunked
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def bootstrap(
    graph_id: str,
    config: dict,
    raw: bool = False,
    test_run: bool = False,
) -> Tuple[ChunkedGraphMeta, IngestConfig, BackendClientInfo]:
    """Parse config loaded from a yaml file."""
    ingest_config = IngestConfig(
        **config.get("ingest_config", {}),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
        TEST_RUN=test_run,
    )
    client_config = BigTableConfig(**config["backend_client"]["CONFIG"])
    client_info = BackendClientInfo(config["backend_client"]["TYPE"], client_config)

    graph_config = GraphConfig(
        ID=f"{graph_id}",
        OVERWRITE=False,
        **config["graph_config"],
    )
    data_source = DataSource(**config["data_source"])

    meta = ChunkedGraphMeta(graph_config, data_source)
    return (meta, ingest_config, client_info)


def move_up(lines: int = 1):
    sys.stdout.write(f"\033[{lines}A")


def postprocess_edge_data(im, edge_dict):
    data_version = im.cg_meta.data_source.DATA_VERSION
    if data_version == 2:
        return edge_dict
    elif data_version in [3, 4]:
        new_edge_dict = {}
        for k in edge_dict:
            new_edge_dict[k] = {}
            if edge_dict[k] is None or len(edge_dict[k]) == 0:
                continue

            areas = (
                edge_dict[k]["area_x"] * im.cg_meta.resolution[0]
                + edge_dict[k]["area_y"] * im.cg_meta.resolution[1]
                + edge_dict[k]["area_z"] * im.cg_meta.resolution[2]
            )

            affs = (
                edge_dict[k]["aff_x"] * im.cg_meta.resolution[0]
                + edge_dict[k]["aff_y"] * im.cg_meta.resolution[1]
                + edge_dict[k]["aff_z"] * im.cg_meta.resolution[2]
            )

            new_edge_dict[k]["sv1"] = edge_dict[k]["sv1"]
            new_edge_dict[k]["sv2"] = edge_dict[k]["sv2"]
            new_edge_dict[k]["area"] = areas
            new_edge_dict[k]["aff"] = affs

        return new_edge_dict
    else:
        raise ValueError(f"Unknown data_version: {data_version}")


def start_ocdbt_server(imanager: IngestionManager, server: Any):
    spec = {"driver": "ocdbt", "base": f"{imanager.cg.meta.data_source.EDGES}/ocdbt"}
    spec["coordinator"] = {"address": f"localhost:{server.port}"}
    ts.KvStore.open(spec).result()
    imanager.redis.set("OCDBT_COORDINATOR_PORT", str(server.port))
    ocdbt_host = environ.get("MY_POD_IP", "localhost")
    imanager.redis.set("OCDBT_COORDINATOR_HOST", ocdbt_host)
    logging.info(f"OCDBT Coordinator address {ocdbt_host}:{server.port}")


def randomize_grid_points(X: int, Y: int, Z: int) -> Generator[int, int, int]:
    indices = np.arange(X * Y * Z)
    np.random.shuffle(indices)
    for index in indices:
        yield np.unravel_index(index, (X, Y, Z))


def get_chunks_not_done(
    imanager: IngestionManager, layer: int, coords: list, splits: int = 0
) -> list:
    """check for set membership in redis in batches"""
    coords_strs = []
    if splits > 0:
        split_coords = []
        for coord in coords:
            for split in range(splits):
                jid = "_".join(map(str, coord)) + f"_{split}"
                coords_strs.append(jid)
                split_coords.append((coord, split))
    else:
        coords_strs = ["_".join(map(str, coord)) for coord in coords]
    try:
        completed = imanager.redis.smismember(f"{layer}c", coords_strs)
    except Exception:
        return split_coords if splits > 0 else coords

    if splits > 0:
        return [coord for coord, c in zip(split_coords, completed) if not c]
    return [coord for coord, c in zip(coords, completed) if not c]


def print_completion_rate(imanager: IngestionManager, layer: int, span: int = 30):
    rate = 0.0
    while True:
        counts = []
        print(f"{rate} chunks per second.")
        for _ in range(span + 1):
            counts.append(imanager.redis.scard(f"{layer}c"))
            sleep(1)
        rate = np.diff(counts).sum() / span
        move_up()


def print_status(imanager: IngestionManager, redis, upgrade: bool = False):
    """
    Helper to print status to console.
    If `upgrade=True`, status does not include the root layer,
    since there is no need to update cross edges for root ids.
    """
    layers = range(2, imanager.cg_meta.layer_count + 1)
    if upgrade:
        layers = range(2, imanager.cg_meta.layer_count)

    def _refresh_status():
        pipeline = redis.pipeline()
        pipeline.get(r_keys.JOB_TYPE)
        worker_busy = ["-"] * len(layers)
        for layer in layers:
            pipeline.scard(f"{layer}c")
            queue = Queue(f"l{layer}", connection=redis)
            pipeline.llen(queue.key)
            pipeline.zcard(queue.failed_job_registry.key)

        results = pipeline.execute()
        job_type = "not_available"
        if results[0] is not None:
            job_type = results[0].decode()
        completed = []
        queued = []
        failed = []
        for i in range(1, len(results), 3):
            result = results[i : i + 3]
            completed.append(result[0])
            queued.append(result[1])
            failed.append(result[2])
        return job_type, completed, queued, failed, worker_busy

    job_type, completed, queued, failed, worker_busy = _refresh_status()

    layer_counts = imanager.cg_meta.layer_chunk_counts
    header = (
        f"\njob_type: \t{job_type}"
        f"\nversion: \t{imanager.cg.version}"
        f"\ngraph_id: \t{imanager.cg.graph_id}"
        f"\nchunk_size: \t{imanager.cg.meta.graph_config.CHUNK_SIZE}"
        "\n\nlayer status:"
    )
    print(header)
    while True:
        for layer, done, count in zip(layers, completed, layer_counts):
            print(
                f"{layer}\t| {done:9} / {count} \t| {math.floor((done/count)*100):6}%"
            )

        print("\n\nqueue status:")
        for layer, q, f, wb in zip(layers, queued, failed, worker_busy):
            print(f"l{layer}\t| queued: {q:<10} failed: {f:<10} busy: {wb}")

        sleep(1)
        _, completed, queued, failed, worker_busy = _refresh_status()
        move_up(lines=2 * len(layers) + 3)


def queue_layer_helper(
    parent_layer: int, imanager: IngestionManager, fn, splits: int = 0
):
    if parent_layer == imanager.cg_meta.layer_count:
        chunk_coords = [(0, 0, 0)]
    else:
        bounds = imanager.cg_meta.layer_chunk_bounds[parent_layer]
        chunk_coords = randomize_grid_points(*bounds)

    q = imanager.get_task_queue(f"l{parent_layer}")
    batch_size = int(environ.get("JOB_BATCH_SIZE", 10000))
    timeout_scale = int(environ.get("TIMEOUT_SCALE_FACTOR", 1))
    batches = chunked(chunk_coords, batch_size)
    failure_ttl = int(environ.get("FAILURE_TTL", 300))
    for batch in batches:
        _coords = get_chunks_not_done(imanager, parent_layer, batch, splits=splits)
        # buffer for optimal use of redis memory
        if len(q) > int(environ.get("QUEUE_SIZE", 100000)):
            interval = int(environ.get("QUEUE_INTERVAL", 300))
            logging.info(f"Queue full; sleeping {interval}s...")
            sleep(interval)

        job_datas = []
        retry = int(environ.get("RETRY_COUNT", 0))
        for chunk_coord in _coords:
            if splits > 0:
                coord, split = chunk_coord
                jid = chunk_id_str(parent_layer, coord) + f"_{split}"
                job_datas.append(
                    Queue.prepare_data(
                        fn,
                        args=(parent_layer, coord, split, splits),
                        result_ttl=0,
                        job_id=jid,
                        timeout=f"{timeout_scale * int(parent_layer * parent_layer)}m",
                        retry=Retry(retry) if retry > 1 else None,
                        description="",
                        failure_ttl=failure_ttl,
                    )
                )
            else:
                job_datas.append(
                    Queue.prepare_data(
                        fn,
                        args=(parent_layer, chunk_coord),
                        result_ttl=0,
                        job_id=chunk_id_str(parent_layer, chunk_coord),
                        timeout=f"{timeout_scale * int(parent_layer * parent_layer)}m",
                        retry=Retry(retry) if retry > 1 else None,
                        description="",
                        failure_ttl=failure_ttl,
                    )
                )
        q.enqueue_many(job_datas)
        logging.info(f"Queued {len(job_datas)} chunks.")


def job_type_guard(job_type: str):
    def decorator_job_type_guard(func):
        @functools.wraps(func)
        def wrapper_job_type_guard(*args, **kwargs):
            redis = get_redis_connection()
            current_type = redis.get(r_keys.JOB_TYPE)
            if current_type is not None:
                current_type = current_type.decode()
                msg = (
                    f"Currently running `{current_type}`. You're attempting to run `{job_type}`."
                    f"\nRun `[flask] {current_type} flush_redis` to clear the current job and restart."
                )
                if current_type != job_type:
                    print(f"\n*WARNING*\n{msg}")
                    exit(1)
            return func(*args, **kwargs)

        return wrapper_job_type_guard

    return decorator_job_type_guard
