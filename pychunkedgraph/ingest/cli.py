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
from .manager import IngestionManager
from .cluster import create_parent_chunk
from ..backend.chunkedgraph import ChunkedGraph
from .redis import keys as r_keys
from .redis import get_redis_connection
from ..backend.chunks.hierarchy import get_children_coords

ingest_cli = AppGroup("ingest")


@ingest_cli.command("atomic")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
@click.option("--overwrite", is_flag=True, help="Overwrite existing graph")
@click.option("--test", is_flag=True)
def ingest_graph(
    graph_id: str, dataset: click.Path, overwrite: bool, raw: bool, test: bool
):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks
    """
    from . import IngestConfig
    from . import ClusterIngestConfig
    from .cluster import enqueue_atomic_tasks
    from ..backend import BigTableConfig
    from ..backend import DataSource
    from ..backend import GraphConfig
    from ..backend import ChunkedGraphMeta

    with open(dataset, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ingest_config = IngestConfig(
        **config["ingest_config"],
        CLUSTER=ClusterIngestConfig(FLUSH_REDIS=True),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
        TEST_RUN=test,
    )

    graph_config = GraphConfig(
        graph_id=graph_id,
        chunk_size=np.array([256, 256, 512], dtype=int),
        overwrite=True,
    )

    data_source = DataSource(
        agglomeration=config["ingest_config"]["AGGLOMERATION"],
        watershed=config["data_source"]["WATERSHED"],
        edges=config["data_source"]["EDGES"],
        components=config["data_source"]["COMPONENTS"],
        data_version=config["data_source"]["DATA_VERSION"],
        use_raw_edges=raw,
        use_raw_components=raw,
    )

    meta = ChunkedGraphMeta(data_source, graph_config, BigTableConfig())
    enqueue_atomic_tasks(IngestionManager(ingest_config, meta))


@ingest_cli.command("layer")
@click.argument("parent_layer", type=int)
def queue_layer(parent_layer):
    """
    Helper command
    Queue all chunk tasks at a given layer
    Use this only when all the chunks at `parent_layer - 1` have been built.
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    if parent_layer == imanager.cg_meta.layer_count:
        chunk_coords = [(0, 0, 0)]
    else:
        bounds = imanager.cg_meta.layer_chunk_bounds[parent_layer]
        chunk_coords = list(product(*[range(r) for r in bounds]))
        np.random.shuffle(chunk_coords)

    for coords in chunk_coords:
        task_q = imanager.get_task_queue(imanager.config.CLUSTER.PARENTS_Q_NAME)
        task_q.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(parent_layer, coords),
            job_timeout=f"{int(parent_layer * parent_layer)}m",
            result_ttl=0,
            args=(
                imanager.get_serialized_info(pickled=True),
                parent_layer,
                coords,
            ),
        )


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    for layer in range(2, imanager.cg_meta.layer_count + 1):
        layer_count = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {layer_count}")
    print(imanager.cg_meta.layer_chunk_counts)


@ingest_cli.command("chunk")
@click.argument("graph_id", type=str)
@click.argument("chunk_info", nargs=4, type=int)
@click.option("--single", is_flag=True)
def ingest_chunk(graph_id: str, chunk_info, single: bool):
    """
    Helper command
    Directly ingest chunk
    """
    from .initialization.abstract_layers import add_layer

    cg = ChunkedGraph(graph_id=graph_id)

    if single:
        add_layer(cg, chunk_info[0], chunk_info[1:], n_threads=1)
    add_layer(cg, chunk_info[0], chunk_info[1:])


@ingest_cli.command("parent")
@click.argument("chunk_info", nargs=4, type=int)
def queue_parent(chunk_info):
    """
    Helper command
    Queue parent chunk of a given child chunk
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    parent_layer = chunk_info[0] + 1
    parent_coords = (
        np.array(chunk_info[1:], int) // imanager.cg_meta.graph_config.FANOUT
    )
    parent_chunk_str = "_".join(map(str, parent_coords))

    parents_queue = imanager.get_task_queue(imanager.config.CLUSTER.PARENTS_Q_NAME)
    parents_queue.enqueue(
        create_parent_chunk,
        job_id=chunk_id_str(parent_layer, parent_coords),
        job_timeout=f"{int(parent_layer * parent_layer)}m",
        result_ttl=0,
        args=(
            imanager.get_serialized_info(pickled=True),
            parent_layer,
            parent_coords,
        ),
    )
    imanager.redis.hdel(parent_layer, parent_chunk_str)
    imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


@ingest_cli.command("children")
@click.argument("chunk_info", nargs=4, type=int)
def queue_children(chunk_info):
    """
    Helper command
    Queue all children chunk tasks of a given parent chunk
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    parent_layer = chunk_info[0]
    parent_coords = np.array(chunk_info[1:], int)

    children = get_children_coords(imanager.cg_meta, parent_layer, parent_coords)
    children_layer = parent_layer - 1
    for coords in children:
        task_q = imanager.get_task_queue(imanager.config.CLUSTER.PARENTS_Q_NAME)
        task_q.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(children_layer, coords),
            job_timeout=f"{int(children_layer * children_layer)}m",
            result_ttl=0,
            args=(
                imanager.get_serialized_info(pickled=True),
                children_layer,
                coords,
            ),
        )


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
