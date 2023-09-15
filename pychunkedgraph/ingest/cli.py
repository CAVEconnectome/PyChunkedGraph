# pylint: disable=invalid-name, missing-function-docstring, unspecified-encoding

"""
cli for running ingest
"""

from os import environ
from time import sleep

import click
import yaml
from flask.cli import AppGroup
from rq import Queue
from rq import Worker
from rq.worker import WorkerStatus

from .cluster import create_atomic_chunk
from .cluster import create_parent_chunk
from .cluster import enqueue_atomic_tasks
from .cluster import randomize_grid_points
from .manager import IngestionManager
from .utils import bootstrap
from .utils import chunk_id_str
from .simple_tests import run_all
from .create.abstract_layers import add_layer
from ..graph.chunkedgraph import ChunkedGraph
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys
from ..utils.general import chunked

ingest_cli = AppGroup("ingest")


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)


@ingest_cli.command("flush_redis")
def flush_redis():
    """FLush redis db."""
    redis = get_redis_connection()
    redis.flushdb()


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
@click.option("--test", is_flag=True)
@click.option("--retry", is_flag=True)
def ingest_graph(
    graph_id: str, dataset: click.Path, raw: bool, test: bool, retry: bool
):
    """
    Main ingest command.
    Takes ingest config from a yaml file and queues atomic tasks.
    """
    with open(dataset, "r") as stream:
        config = yaml.safe_load(stream)

    meta, ingest_config, client_info = bootstrap(
        graph_id,
        config=config,
        raw=raw,
        test_run=test,
    )
    cg = ChunkedGraph(meta=meta, client_info=client_info)
    if not retry:
        cg.create()
    enqueue_atomic_tasks(IngestionManager(ingest_config, meta))


@ingest_cli.command("imanager")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
def pickle_imanager(graph_id: str, dataset: click.Path, raw: bool):
    """
    Load ingest config into redis server.
    Must only be used if ingest config is lost/corrupted during ingest.
    """
    with open(dataset, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    meta, ingest_config, _ = bootstrap(graph_id, config=config, raw=raw)
    imanager = IngestionManager(ingest_config, meta)
    imanager.redis  # pylint: disable=pointless-statement


@ingest_cli.command("layer")
@click.argument("parent_layer", type=int)
def queue_layer(parent_layer):
    """
    Queue all chunk tasks at a given layer.
    Must be used when all the chunks at `parent_layer - 1` have completed.
    """
    assert parent_layer > 2, "This command is for layers 3 and above."
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    if parent_layer == imanager.cg_meta.layer_count:
        chunk_coords = [(0, 0, 0)]
    else:
        bounds = imanager.cg_meta.layer_chunk_bounds[parent_layer]
        chunk_coords = randomize_grid_points(*bounds)

    for coords in chunk_coords:
        task_q = imanager.get_task_queue(f"l{parent_layer}")
        task_q.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(parent_layer, coords),
            job_timeout=f"{int(parent_layer * parent_layer)}m",
            result_ttl=0,
            args=(parent_layer, coords),
        )


@ingest_cli.command("status")
def ingest_status():
    """Print ingest status to console by layer."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
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


@ingest_cli.command("chunk")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
def ingest_chunk(queue: str, chunk_info):
    """Manually queue chunk when a job is stuck for whatever reason."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    layer = chunk_info[0]
    coords = chunk_info[1:]
    queue = imanager.get_task_queue(queue)
    if layer == 2:
        func = create_atomic_chunk
        args = (coords,)
    else:
        func = create_parent_chunk
        args = (layer, coords)
    queue.enqueue(
        func,
        job_id=chunk_id_str(layer, coords),
        job_timeout=f"{int(layer * layer)}m",
        result_ttl=0,
        args=args,
    )


@ingest_cli.command("chunk_local")
@click.argument("graph_id", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@click.option("--n_threads", type=int, default=1)
def ingest_chunk_local(graph_id: str, chunk_info, n_threads: int):
    """Manually ingest a chunk on a local machine."""
    from .create.abstract_layers import add_layer
    from .cluster import _create_atomic_chunk

    if chunk_info[0] == 2:
        _create_atomic_chunk(chunk_info[1:])
    else:
        cg = ChunkedGraph(graph_id=graph_id)
        add_layer(cg, chunk_info[0], chunk_info[1:], n_threads=n_threads)
    cg = ChunkedGraph(graph_id=graph_id)
    add_layer(cg, chunk_info[0], chunk_info[1:], n_threads=n_threads)


@ingest_cli.command("run_tests")
@click.argument("graph_id", type=str)
def run_tests(graph_id):
    run_all(ChunkedGraph(graph_id=graph_id))
