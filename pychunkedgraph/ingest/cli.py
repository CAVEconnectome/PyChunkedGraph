"""
cli for running ingest
"""

import time
from itertools import product

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
from ..graph.chunks.hierarchy import get_children_chunk_coords

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
    for layer in range(2, imanager.chunkedgraph_meta.layer_count + 1):
        layer_count = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {layer_count}")
    print(imanager.chunkedgraph_meta.layer_chunk_counts)


@ingest_cli.command("parent")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
def queue_parent(queue: str, chunk_info):
    """Manually queue parent of a child chunk."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    parent_layer = chunk_info[0] + 1
    parent_coords = (
        np.array(chunk_info[1:], int) // imanager.chunkedgraph_meta.graph_config.FANOUT
    )
    parent_chunk_str = "_".join(map(str, parent_coords))

    parents_queue = imanager.get_task_queue(queue)
    parents_queue.enqueue(
        create_parent_chunk,
        job_id=chunk_id_str(parent_layer, parent_coords),
        job_timeout=f"{int(parent_layer * parent_layer)}m",
        result_ttl=0,
        args=(
            imanager.serialized(pickled=True),
            parent_layer,
            parent_coords,
        ),
    )
    imanager.redis.hdel(parent_layer, parent_chunk_str)
    imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


@ingest_cli.command("chunk_local")
@click.argument("graph_id", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@click.option("--n_threads", type=int, default=1)
def ingest_chunk(graph_id: str, chunk_info, n_threads: int):
    """Manually ingest a chunk on a local machine."""
    from .initial.abstract_layers import add_layer

    cg = ChunkedGraph(graph_id=graph_id)

    add_layer(cg, chunk_info[0], chunk_info[1:], n_threads=n_threads)
