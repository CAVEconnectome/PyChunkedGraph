# pylint: disable=invalid-name, missing-function-docstring, unspecified-encoding

"""
cli for running ingest
"""

import os
from time import sleep

from pychunkedgraph import configure_logging, DEBUG

import click
import yaml
from flask.cli import AppGroup

from .cluster import create_atomic_chunk, create_parent_chunk, enqueue_l2_tasks
from .manager import IngestionManager
from .ocdbt import coordinator, setup_base
from .utils import (
    bootstrap,
    job_type_guard,
    print_completion_rate,
    print_status,
    queue_layer_helper,
    requeue_chunk,
)
from .simple_tests import run_all
from .create.parent_layer import add_parent_chunk
from ..graph.chunkedgraph import ChunkedGraph
from ..graph.ocdbt import OcdbtConfig, fork_base_manifest
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
@click.argument("dataset", type=click.Path(exists=True), required=False)
@click.option("--raw", is_flag=True, help="Read edges from agglomeration output.")
@click.option("--retry", is_flag=True, help="Rerun without creating a new table.")
@click.option("--test", is_flag=True, help="Test 8 chunks at the center of dataset.")
@job_type_guard(group_name)
def ingest_graph(
    graph_id: str,
    dataset: click.Path,
    raw: bool,
    retry: bool,
    test: bool,
):
    """Main ingest command. Takes config from yaml, queues atomic tasks."""
    redis = get_redis_connection()

    if retry:
        imanager_pickle = redis.get(r_keys.INGESTION_MANAGER)
        if imanager_pickle is None:
            raise click.ClickException(
                f"--retry requires an existing `{group_name}` job in redis. "
                f"Run without --retry to start a new job."
            )
        if test:
            configure_logging(level=DEBUG)
        imanager = IngestionManager.from_pickle(imanager_pickle)
        if imanager.ocdbt_seg:
            ws = imanager.cg_meta.data_source.WATERSHED
            fork_base_manifest(ws, graph_id, wipe_existing=True)
        enqueue_l2_tasks(imanager, create_atomic_chunk)
        os._exit(0)

    if dataset is None:
        raise click.ClickException("dataset is required unless --retry is passed.")

    redis.set(r_keys.JOB_TYPE, group_name)
    with open(dataset, "r") as stream:
        config = yaml.safe_load(stream)

    if test:
        configure_logging(level=DEBUG)

    meta, ingest_config, client_info, ocdbt_config_dict = bootstrap(
        graph_id, config, raw, test
    )
    cg = ChunkedGraph(meta=meta, client_info=client_info)
    cg.create()

    ocdbt_cfg = OcdbtConfig.from_dict(ocdbt_config_dict)
    if ocdbt_cfg.enabled:
        resolved = setup_base(cg, ocdbt_cfg)
        ocdbt_config_dict = resolved.to_dict()

    imanager = IngestionManager(
        ingest_config,
        meta,
        ocdbt_config=ocdbt_config_dict,
    )
    enqueue_l2_tasks(imanager, create_atomic_chunk)
    os._exit(0)


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

    meta, ingest_config, _, ocdbt_config_dict = bootstrap(
        graph_id, config=config, raw=raw
    )
    imanager = IngestionManager(ingest_config, meta, ocdbt_config=ocdbt_config_dict)
    imanager.redis.set(r_keys.JOB_TYPE, group_name)


@ingest_cli.command("layer")
@click.argument("parent_layer", type=int)
@job_type_guard(group_name)
def queue_layer(parent_layer):
    """
    Queue all chunk tasks at a given layer.
    Must be used when all the chunks at `parent_layer - 1` have completed.

    When this layer is the OCDBT populate layer, also start a
    ``DistributedCoordinatorServer`` so every worker's commit routes through
    one process — eliminates manifest-CAS races and orphan ``d/`` files.
    Stays in the foreground until killed.
    """
    assert parent_layer > 2, "This command is for layers 3 and above."
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    if (
        imanager.ocdbt_seg
        and imanager.ocdbt_populate_base
        and parent_layer == imanager.ocdbt_populate_layer
    ):
        with coordinator(imanager.redis):
            queue_layer_helper(parent_layer, imanager, create_parent_chunk)
            while True:
                sleep(60)
    else:
        queue_layer_helper(parent_layer, imanager, create_parent_chunk)


@ingest_cli.command("status")
@click.option("--refresh", type=int, default=5, help="Seconds between redis polls.")
@job_type_guard(group_name)
def ingest_status(refresh: int):
    """Print ingest status to console by layer."""
    redis = get_redis_connection()
    try:
        imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
        print_status(imanager, redis, refresh_seconds=refresh)
    except TypeError as err:
        print(f"\nNo current `{group_name}` job found in redis: {err}")


@ingest_cli.command("chunk")
@click.argument("queue", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@job_type_guard(group_name)
def ingest_chunk(queue: str, chunk_info):
    """Manually queue chunk when a job is stuck for whatever reason."""
    requeue_chunk(queue, chunk_info, create_atomic_chunk, create_parent_chunk)


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
