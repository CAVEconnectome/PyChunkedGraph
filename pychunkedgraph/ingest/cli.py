# pylint: disable=invalid-name, missing-function-docstring, unspecified-encoding

"""
cli for running ingest
"""

import os
from functools import partial
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
    purge_layer_state,
    queue_layer_helper,
    requeue_chunk,
)
from .simple_tests import run_all
from .create.parent_layer import add_parent_chunk
from ..graph.chunkedgraph import ChunkedGraph
from ..graph.ocdbt import OcdbtConfig
from ..meshing.meta import MeshConfig
from ..meshing.setup import setup_mesh_meta
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
@click.option(
    "--retry",
    "-r",
    is_flag=True,
    help="Re-run setup against the existing table (no cg.create()).",
)
@click.option(
    "--skip-queue",
    "-s",
    is_flag=True,
    help="Set up everything but don't enqueue L2 tasks.",
)
@click.option(
    "--test",
    "-t",
    is_flag=True,
    help="Test 8 chunks at the center of dataset.",
)
@job_type_guard(group_name)
def ingest_graph(
    graph_id: str,
    dataset: click.Path,
    raw: bool,
    retry: bool,
    skip_queue: bool,
    test: bool,
):
    """Main ingest command. Takes config from yaml, queues atomic tasks.

    Purely about the bigtable graph: creates the table and enqueues L2
    tasks. OCDBT base + fork creation happens in ``ingest layer N`` when
    N matches ``ocdbt_populate_layer``; that's the single owner of the
    OCDBT lifecycle.

    ``--retry`` reuses the existing IngestionManager from redis and skips
    ``cg.create()``. Pair with ``--skip-queue`` to skip L2 enqueue too.
    """
    redis = get_redis_connection()
    if test:
        configure_logging(level=DEBUG)

    if retry:
        imanager_pickle = redis.get(r_keys.INGESTION_MANAGER)
        if imanager_pickle is None:
            raise click.ClickException(
                f"--retry requires an existing `{group_name}` job in redis. "
                f"Run without --retry to start a new job."
            )
        imanager = IngestionManager.from_pickle(imanager_pickle)
    else:
        if dataset is None:
            raise click.ClickException("dataset is required unless --retry is passed.")
        redis.set(r_keys.JOB_TYPE, group_name)
        with open(dataset, "r") as stream:
            config = yaml.safe_load(stream)
        meta, ingest_config, client_info, ocdbt_config_dict = bootstrap(
            graph_id, config, raw, test
        )
        cg = ChunkedGraph(meta=meta, client_info=client_info)
        cg.create()
        imanager = IngestionManager(
            ingest_config,
            meta,
            ocdbt_config=ocdbt_config_dict,
        )

    if not skip_queue:
        enqueue_l2_tasks(imanager, create_atomic_chunk)
    os._exit(0)


@ingest_cli.command("mesh_meta")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@job_type_guard(group_name)
def mesh_meta(graph_id: str, dataset: click.Path):
    """Set up every mesh.* metadata field for GRAPH_ID from DATASET yaml.

    Reads ``mesh_config:`` from the yaml, applies it to the graph. Run
    once per new/copied graph, after the operator has verified initial
    ingest (including the root layer) is complete — no automatic gate.
    """
    with open(dataset, "r") as stream:
        config = yaml.safe_load(stream)
    if "mesh_config" not in config:
        raise click.ClickException(
            f"{dataset} has no `mesh_config:` block — required for mesh_meta."
        )
    mesh_cfg = MeshConfig.from_dict(config["mesh_config"])
    cg = ChunkedGraph(graph_id=graph_id)
    result = setup_mesh_meta(cg, mesh_cfg)
    click.echo(f"mesh meta written for {graph_id}: {result}")


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
@click.option(
    "--queue-only",
    "-q",
    is_flag=True,
    help="Only enqueue tasks; do not start the OCDBT coordinator. "
    "Use when a coordinator is already running in another process.",
)
@click.option(
    "--ocdbt-only",
    "-o",
    is_flag=True,
    help="Workers run only OCDBT populate (skip add_parent_chunk). "
    "Requires the OCDBT populate layer.",
)
@click.option(
    "--ingest-only",
    "-i",
    is_flag=True,
    help="Workers run only add_parent_chunk (skip OCDBT populate). "
    "Use when the OCDBT base is already populated for this layer.",
)
@job_type_guard(group_name)
def queue_layer(parent_layer, queue_only, ocdbt_only, ingest_only):
    """
    Queue all chunk tasks at a given layer.
    Must be used when all the chunks at `parent_layer - 1` have completed.

    When this layer is the OCDBT populate layer, this command also owns the
    OCDBT lifecycle: idempotently creates the base + fork via ``setup_base``
    and starts a ``DistributedCoordinatorServer`` so every worker's commit
    routes through one process (eliminates manifest-CAS races and orphan
    ``d/`` files). Stays in the foreground until killed.

    Flags:
      ``--queue-only``  skips the coordinator (one is assumed running elsewhere).
      ``--ocdbt-only``  task body = OCDBT populate only.
      ``--ingest-only`` task body = add_parent_chunk only.
    """
    assert parent_layer > 2, "This command is for layers 3 and above."
    if ocdbt_only and ingest_only:
        raise click.ClickException(
            "--ocdbt-only and --ingest-only are mutually exclusive."
        )
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    is_populate_layer = imanager.is_ocdbt_populate_layer(parent_layer)
    if ocdbt_only and not is_populate_layer:
        raise click.ClickException(
            "--ocdbt-only requires running at the OCDBT populate layer."
        )

    if is_populate_layer:
        # Single owner of the OCDBT lifecycle: create base + fork if
        # missing, reconcile config with on-disk meta, then re-pickle
        # imanager so queued workers read the resolved config.
        resolved = setup_base(imanager.cg, OcdbtConfig.from_dict(imanager.ocdbt_config))
        imanager.ocdbt_config = resolved.to_dict()
        imanager.redis.set(r_keys.INGESTION_MANAGER, imanager.serialized(pickled=True))

    mode = "ocdbt" if ocdbt_only else ("ingest" if ingest_only else "full")
    task_fn = (
        partial(create_parent_chunk, mode=mode)
        if mode != "full"
        else create_parent_chunk
    )

    # Coordinator only matters when OCDBT populate will actually run.
    needs_coordinator = (
        is_populate_layer and mode in ("full", "ocdbt") and not queue_only
    )
    if needs_coordinator:
        with coordinator(imanager.redis):
            queue_layer_helper(parent_layer, imanager, task_fn)
            while True:
                sleep(60)
    else:
        queue_layer_helper(parent_layer, imanager, task_fn)


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


@ingest_cli.command("purge_layer")
@click.argument("layer", type=int)
@click.confirmation_option(prompt="Purge ALL redis state for this layer?")
@job_type_guard(group_name)
def purge_layer(layer: int):
    """Drop the per-layer RQ queue + registries + completion set so the
    layer can be re-run from a previous layer's backup."""
    purge_layer_state(get_redis_connection(), layer)
    click.echo(f"purged redis state for layer {layer}")
