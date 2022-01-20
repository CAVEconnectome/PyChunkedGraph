"""
cli for running ingest
"""

import yaml
import numpy as np
import click
from flask.cli import AppGroup

from .utils import chunk_id_str
from .utils import bootstrap
from .manager import IngestionManager
from .cluster import enqueue_atomic_tasks
from .cluster import create_parent_chunk
from ..graph.chunkedgraph import ChunkedGraph
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys

ingest_cli = AppGroup("ingest")


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)


@ingest_cli.command("flushdb")
def flushdb():
    """FLush redis db."""
    redis = get_redis_connection()
    redis.flushdb()


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
@click.option("--test", is_flag=True)
def ingest_graph(graph_id: str, dataset: click.Path, raw: bool, test: bool):
    """
    Main ingest command.
    Takes ingest config from a yaml file and queues atomic tasks.
    """
    with open(dataset, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    meta, ingest_config, client_info = bootstrap(
        graph_id,
        config=config,
        raw=raw,
        test_run=test,
    )
    cg = ChunkedGraph(meta=meta, client_info=client_info)
    cg.create()
    enqueue_atomic_tasks(IngestionManager(ingest_config, meta))


@ingest_cli.command("status")
def ingest_status():
    """Print ingest status to console by layer."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    layers = range(2, imanager.cg_meta.layer_count + 1)
    for layer, layer_count in zip(layers, imanager.cg_meta.layer_chunk_counts):
        completed = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {completed} / {layer_count}")


@ingest_cli.command("chunk")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
def ingest_chunk(queue: str, chunk_info):
    """Manually queue chunk when a job is stuck for whatever reason."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    layer = chunk_info[0]
    coords = np.array(chunk_info[1:], int)
    chunk_str = "_".join(map(str, coords))

    queue = imanager.get_task_queue(queue)
    queue.enqueue(
        create_parent_chunk,
        job_id=chunk_id_str(layer, coords),
        job_timeout=f"{int(layer * layer)}m",
        result_ttl=0,
        args=(
            imanager.serialized(pickled=True),
            layer,
            coords,
        ),
    )
    imanager.redis.hdel(layer, chunk_str)
    imanager.redis.hset(f"{layer}q", chunk_str, "")


@ingest_cli.command("chunk_local")
@click.argument("graph_id", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@click.option("--n_threads", type=int, default=1)
def ingest_chunk_local(graph_id: str, chunk_info, n_threads: int):
    """Manually ingest a chunk on a local machine."""
    from .initial.abstract_layers import add_layer

    cg = ChunkedGraph(graph_id=graph_id)
    add_layer(cg, chunk_info[0], chunk_info[1:], n_threads=n_threads)
