# pylint: disable=invalid-name, missing-function-docstring, unspecified-encoding

"""
cli for running upgrade
"""

import click
from flask.cli import AppGroup

from pychunkedgraph import __version__, get_logger
from pychunkedgraph.graph.meta import GraphConfig

from . import IngestConfig
from .cluster import enqueue_l2_tasks, upgrade_atomic_chunk, upgrade_parent_chunk
from .manager import IngestionManager
from .ocdbt import setup_base
from .utils import (
    job_type_guard,
    print_completion_rate,
    print_status,
    queue_layer_helper,
    requeue_chunk,
)
from ..graph.chunkedgraph import ChunkedGraph, ChunkedGraphMeta
from ..graph.ocdbt import OcdbtConfig
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys

logger = get_logger(__name__)

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
@click.option("--ocdbt", is_flag=True, help="Enable ocdbt seg (SV splitting support).")
@click.option(
    "--sv-split-threshold",
    type=int,
    default=10,
    help="Distance threshold for SV split edge matching.",
)
@job_type_guard(group_name)
def upgrade_graph(
    graph_id: str,
    test: bool,
    ocdbt: bool,
    sv_split_threshold: int,
):
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

    if ocdbt:
        ocdbt_cfg = OcdbtConfig.from_dict(cg.meta.custom_data.get("ocdbt_config"))
        ocdbt_cfg.enabled = True
        ocdbt_cfg.sv_split_threshold = sv_split_threshold
        setup_base(cg, ocdbt_cfg)
        logger.note(f"enabled ocdbt seg with sv_split_threshold={sv_split_threshold}")
    try:
        cg.client.create_column_family("4")
    except Exception:
        ...

    imanager = IngestionManager(ingest_config, cg.meta)
    enqueue_l2_tasks(imanager, upgrade_atomic_chunk)


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
@click.option("--refresh", type=int, default=5, help="Seconds between redis polls.")
@job_type_guard(group_name)
def upgrade_status(refresh: int):
    """Print upgrade status to console."""
    redis = get_redis_connection()
    try:
        imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
        print_status(imanager, redis, upgrade=True, refresh_seconds=refresh)
    except TypeError as err:
        print(f"\nNo current `{group_name}` job found in redis: {err}")


@upgrade_cli.command("chunk")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@job_type_guard(group_name)
def upgrade_chunk(queue: str, chunk_info):
    """Manually queue chunk when a job is stuck for whatever reason."""
    requeue_chunk(queue, chunk_info, upgrade_atomic_chunk, upgrade_parent_chunk)


@upgrade_cli.command("rate")
@click.argument("layer", type=int)
@click.option("--span", default=10, help="Time span to calculate rate.")
@job_type_guard(group_name)
def rate(layer: int, span: int):
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    print_completion_rate(imanager, layer, span=span)
