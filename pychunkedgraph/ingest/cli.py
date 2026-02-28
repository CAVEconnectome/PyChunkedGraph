# pylint: disable=invalid-name, missing-function-docstring, unspecified-encoding

"""
cli for running ingest
"""

import logging

import click
import yaml
from flask.cli import AppGroup

from .cluster import create_atomic_chunk, create_parent_chunk, enqueue_l2_tasks
from .manager import IngestionManager
from .utils import (
    bootstrap,
    chunk_id_str,
    print_completion_rate,
    print_status,
    queue_layer_helper,
    job_type_guard,
)
from .simple_tests import run_all
from .create.parent_layer import add_parent_chunk
from ..graph.chunkedgraph import ChunkedGraph
from ..utils.redis import get_redis_connection, keys as r_keys

group_name = "ingest"
ingest_cli = AppGroup(group_name)


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)


@ingest_cli.command("flush_redis")
@click.confirmation_option(prompt="Are you sure you want to flush redis?")
@job_type_guard(group_name)
def flush_redis():
    """FLush redis db."""
    redis = get_redis_connection()
    redis.flushdb()


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True, help="Read edges from agglomeration output.")
@click.option("--test", is_flag=True, help="Test 8 chunks at the center of dataset.")
@click.option("--retry", is_flag=True, help="Rerun without creating a new table.")
@job_type_guard(group_name)
def ingest_graph(
    graph_id: str, dataset: click.Path, raw: bool, test: bool, retry: bool
):
    """
    Main ingest command.
    Takes ingest config from a yaml file and queues atomic tasks.
    """
    redis = get_redis_connection()
    redis.set(r_keys.JOB_TYPE, group_name)
    with open(dataset, "r") as stream:
        config = yaml.safe_load(stream)

    if test:
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    meta, ingest_config, client_info = bootstrap(graph_id, config, raw, test)
    cg = ChunkedGraph(meta=meta, client_info=client_info)
    if not retry:
        cg.create()

    imanager = IngestionManager(ingest_config, meta)
    enqueue_l2_tasks(imanager, create_atomic_chunk)


@ingest_cli.command("imanager")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
@job_type_guard(group_name)
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
    imanager.redis.set(r_keys.JOB_TYPE, group_name)


@ingest_cli.command("layer")
@click.argument("parent_layer", type=int)
@job_type_guard(group_name)
def queue_layer(parent_layer):
    """
    Queue all chunk tasks at a given layer.
    Must be used when all the chunks at `parent_layer - 1` have completed.
    """
    assert parent_layer > 2, "This command is for layers 3 and above."
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    queue_layer_helper(parent_layer, imanager, create_parent_chunk)


@ingest_cli.command("status")
@job_type_guard(group_name)
def ingest_status():
    """Print ingest status to console by layer."""
    redis = get_redis_connection()
    try:
        imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
        print_status(imanager, redis)
    except TypeError as err:
        print(f"\nNo current `{group_name}` job found in redis: {err}")


@ingest_cli.command("chunk")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@job_type_guard(group_name)
def ingest_chunk(queue: str, chunk_info):
    """Manually queue chunk when a job is stuck for whatever reason."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    layer, coords = chunk_info[0], chunk_info[1:]

    func = create_parent_chunk
    args = (layer, coords)
    if layer == 2:
        func = create_atomic_chunk
        args = (coords,)
    queue = imanager.get_task_queue(queue)
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
@job_type_guard(group_name)
def ingest_chunk_local(graph_id: str, chunk_info, n_threads: int):
    """Manually ingest a chunk on a local machine."""
    layer, coords = chunk_info[0], chunk_info[1:]
    if layer == 2:
        create_atomic_chunk(coords)
    else:
        cg = ChunkedGraph(graph_id=graph_id)
        add_parent_chunk(cg, layer, coords, n_threads=n_threads)
    cg = ChunkedGraph(graph_id=graph_id)
    add_parent_chunk(cg, layer, coords, n_threads=n_threads)


@ingest_cli.command("rate")
@click.argument("layer", type=int)
@click.option("--span", default=10, help="Time span to calculate rate.")
@job_type_guard(group_name)
def rate(layer: int, span: int):
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    print_completion_rate(imanager, layer, span=span)


@ingest_cli.command("run_tests")
@click.argument("graph_id", type=str)
@job_type_guard(group_name)
def run_tests(graph_id):
    run_all(ChunkedGraph(graph_id=graph_id))
