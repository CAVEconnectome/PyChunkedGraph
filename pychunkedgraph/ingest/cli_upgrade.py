# pylint: disable=invalid-name, missing-function-docstring, unspecified-encoding

"""
cli for running upgrade
"""

import logging
from time import sleep

import click
import tensorstore as ts
from flask.cli import AppGroup
from pychunkedgraph import __version__
from pychunkedgraph.graph.meta import GraphConfig

from . import IngestConfig
from .cluster import (
    convert_to_ocdbt,
    enqueue_l2_tasks,
    upgrade_atomic_chunk,
    upgrade_parent_chunk,
)
from .manager import IngestionManager
from .utils import (
    chunk_id_str,
    print_completion_rate,
    print_status,
    queue_layer_helper,
    start_ocdbt_server,
    job_type_guard,
)
from ..graph.chunkedgraph import ChunkedGraph, ChunkedGraphMeta
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys

group_name = "upgrade"
upgrade_cli = AppGroup(group_name)


def init_upgrade_cmds(app):
    app.cli.add_command(upgrade_cli)


@upgrade_cli.command("flush_redis")
@click.confirmation_option(prompt="Are you sure you want to flush redis?")
@job_type_guard(group_name)
def flush_redis():
    """FLush redis db."""
    redis = get_redis_connection()
    redis.flushdb()


@upgrade_cli.command("graph")
@click.argument("graph_id", type=str)
@click.option("--test", is_flag=True, help="Test 8 chunks at the center of dataset.")
@click.option("--ocdbt", is_flag=True, help="Store edges using ts ocdbt kv store.")
@job_type_guard(group_name)
def upgrade_graph(graph_id: str, test: bool, ocdbt: bool):
    """
    Main upgrade command. Queues atomic tasks.
    """
    redis = get_redis_connection()
    redis.set(r_keys.JOB_TYPE, group_name)
    ingest_config = IngestConfig(TEST_RUN=test)
    cg = ChunkedGraph(graph_id=graph_id)
    cg.client.add_table_version(__version__, overwrite=True)

    if graph_id != cg.graph_id:
        gc = cg.meta.graph_config._asdict()
        gc["ID"] = graph_id
        new_meta = ChunkedGraphMeta(
            GraphConfig(**gc), cg.meta.data_source, cg.meta.custom_data
        )
        cg.update_meta(new_meta, overwrite=True)
        cg = ChunkedGraph(graph_id=graph_id)

    try:
        # create new column family for cross chunk edges
        cg.client.create_column_family("4")
    except Exception:
        ...

    imanager = IngestionManager(ingest_config, cg.meta)
    server = ts.ocdbt.DistributedCoordinatorServer()
    if ocdbt:
        start_ocdbt_server(imanager, server)

    fn = convert_to_ocdbt if ocdbt else upgrade_atomic_chunk
    enqueue_l2_tasks(imanager, fn)

    if ocdbt:
        logging.info("All tasks queued. Keep this alive for ocdbt coordinator server.")
        while True:
            sleep(60)


@upgrade_cli.command("layer")
@click.argument("parent_layer", type=int)
@click.option("--splits", default=0, help="Split chunks into multiple tasks.")
@job_type_guard(group_name)
def queue_layer(parent_layer: int, splits: int = 0):
    """
    Queue all chunk tasks at a given layer.
    Must be used when all the chunks at `parent_layer - 1` have completed.
    """
    assert parent_layer > 2, "This command is for layers 3 and above."
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    queue_layer_helper(parent_layer, imanager, upgrade_parent_chunk, splits=splits)


@upgrade_cli.command("status")
@job_type_guard(group_name)
def upgrade_status():
    """Print upgrade status to console."""
    redis = get_redis_connection()
    try:
        imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
        print_status(imanager, redis, upgrade=True)
    except TypeError as err:
        print(f"\nNo current `{group_name}` job found in redis: {err}")


@upgrade_cli.command("chunk")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@job_type_guard(group_name)
def upgrade_chunk(queue: str, chunk_info):
    """Manually queue chunk when a job is stuck for whatever reason."""
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    layer, coords = chunk_info[0], chunk_info[1:]

    func = upgrade_parent_chunk
    args = (layer, coords)
    if layer == 2:
        func = upgrade_atomic_chunk
        args = (coords,)
    queue = imanager.get_task_queue(queue)
    queue.enqueue(
        func,
        job_id=chunk_id_str(layer, coords),
        job_timeout=f"{int(layer * layer)}m",
        result_ttl=0,
        args=args,
    )


@upgrade_cli.command("rate")
@click.argument("layer", type=int)
@click.option("--span", default=10, help="Time span to calculate rate.")
@job_type_guard(group_name)
def rate(layer: int, span: int):
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    print_completion_rate(imanager, layer, span=span)
