# pylint: disable=invalid-name, missing-docstring

import logging
from os import environ
from typing import Any, Generator, Tuple

import numpy as np
import tensorstore as ts
from rq import Queue
from rq import Worker
from rq.worker import WorkerStatus

from . import ClusterIngestConfig
from . import IngestConfig
from .manager import IngestionManager
from ..graph.meta import ChunkedGraphMeta
from ..graph.meta import DataSource
from ..graph.meta import GraphConfig

from ..graph.client import BackendClientInfo
from ..graph.client.bigtable import BigTableConfig

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
        CLUSTER=ClusterIngestConfig(),
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


def randomize_grid_points(X: int, Y: int, Z: int) -> Generator[int, int, int]:
    indices = np.arange(X * Y * Z)
    np.random.shuffle(indices)
    for index in indices:
        yield np.unravel_index(index, (X, Y, Z))


def start_ocdbt_server(imanager: IngestionManager, server: Any):
    spec = {"driver": "ocdbt", "base": f"{imanager.cg.meta.data_source.EDGES}/ocdbt"}
    spec["coordinator"] = {"address": f"localhost:{server.port}"}
    ts.KvStore.open(spec).result()
    imanager.redis.set("OCDBT_COORDINATOR_PORT", str(server.port))
    ocdbt_host = environ.get("MY_POD_IP", "localhost")
    imanager.redis.set("OCDBT_COORDINATOR_HOST", ocdbt_host)
    logging.info(f"OCDBT Coordinator address {ocdbt_host}:{server.port}")


def print_ingest_status(imanager: IngestionManager, redis):
    """Print ingest status to console by layer."""
    layers = range(2, imanager.cg_meta.layer_count + 1)
    layer_counts = imanager.cg_meta.layer_chunk_counts

    pipeline = redis.pipeline()
    worker_busy = []
    for layer in layers:
        pipeline.scard(f"{layer}c")
        queue = Queue(f"l{layer}", connection=redis)
        pipeline.llen(queue.key)
        pipeline.zcard(queue.failed_job_registry.key)
        workers = Worker.all(queue=queue)
        worker_busy.append(sum([w.get_state() == WorkerStatus.BUSY for w in workers]))

    results = pipeline.execute()
    completed = []
    queued = []
    failed = []
    for i in range(0, len(results), 3):
        result = results[i : i + 3]
        completed.append(result[0])
        queued.append(result[1])
        failed.append(result[2])

    print(f"version: \t{imanager.cg.version}")
    print(f"graph_id: \t{imanager.cg.graph_id}")
    print(f"chunk_size: \t{imanager.cg.meta.graph_config.CHUNK_SIZE}")
    print("\nlayer status:")
    for layer, done, count in zip(layers, completed, layer_counts):
        print(f"{layer}\t: {done} / {count}")

    print("\n\nqueue status:")
    for layer, q, f, wb in zip(layers, queued, failed, worker_busy):
        print(f"l{layer}\t: queued: {q}\t\t failed: {f}\t\t busy: {wb}")


def queue_layer_helper(parent_layer: int, imanager: IngestionManager, redis, fn):
    if parent_layer == imanager.cg_meta.layer_count:
        chunk_coords = [(0, 0, 0)]
    else:
        bounds = imanager.cg_meta.layer_chunk_bounds[parent_layer]
        chunk_coords = randomize_grid_points(*bounds)

    for coords in chunk_coords:
        task_q = imanager.get_task_queue(f"l{parent_layer}")
        task_q.enqueue(
            fn,
            job_id=chunk_id_str(parent_layer, coords),
            job_timeout=f"{int(parent_layer * parent_layer)}m",
            result_ttl=0,
            args=(parent_layer, coords),
        )
