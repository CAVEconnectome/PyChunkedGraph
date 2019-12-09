"""
cli for running ingest
"""

import time
from itertools import product

import yaml
import numpy as np
import click
from flask.cli import AppGroup

from . import ClusterIngestConfig
from . import IngestConfig
from .utils import chunk_id_str
from .utils import initialize_chunkedgraph
from .manager import IngestionManager
from .cluster import enqueue_atomic_tasks
from .cluster import create_parent_chunk
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys
from ..graph.meta import ChunkedGraphMeta
from ..graph.meta import DataSource
from ..graph.meta import GraphConfig
from ..graph.meta import BigTableConfig
from ..graph.chunks.hierarchy import get_children_chunk_coords

ingest_cli = AppGroup("ingest")


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
@click.option("--overwrite", is_flag=True, help="Overwrite existing graph")
def ingest_graph(graph_id: str, dataset: click.Path, raw: bool, overwrite: bool):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks    
    """
    with open(dataset, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ingest_config = IngestConfig(
        **config["ingest_config"], CLUSTER=ClusterIngestConfig(FLUSH_REDIS=True)
    )
    bigtable_config = BigTableConfig(**config["bigtable_config"])
    graph_config = GraphConfig(
        **config["graph_config"],
        graph_id=f"{bigtable_config.table_id_prefix}{graph_id}",
        overwrite=overwrite,
    )

    data_source = DataSource(
        **config["data_source"], use_raw_components=raw, use_raw_edges=raw
    )

    meta = ChunkedGraphMeta(data_source, graph_config)
    initialize_chunkedgraph(meta)
    imanager = IngestionManager(ingest_config, meta)
    enqueue_atomic_tasks(imanager)


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
        np.array(chunk_info[1:], int) // imanager.chunkedgraph_meta.graph_config.FANOUT
    )
    parent_chunk_str = "_".join(map(str, parent_coords))

    parents_queue = imanager.get_task_queue(imanager.config.parents_q_name)
    parents_queue.enqueue(
        create_parent_chunk,
        job_id=chunk_id_str(parent_layer, parent_coords),
        job_timeout=f"{int(10 * parent_layer)}m",
        result_ttl=0,
        args=(
            imanager.serialized(),
            parent_layer,
            parent_coords,
            get_children_chunk_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            ),
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

    children = get_children_chunk_coords(
        imanager.chunkedgraph_meta, parent_layer, parent_coords
    )
    children_layer = parent_layer - 1
    for coords in children:
        task_q = imanager.get_task_queue(imanager.config.parents_q_name)
        task_q.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(children_layer, coords),
            job_timeout=f"{int(10 * parent_layer)}m",
            result_ttl=0,
            args=(
                imanager.serialized(),
                children_layer,
                coords,
                get_children_chunk_coords(
                    imanager.chunkedgraph_meta, children_layer, coords
                ),
            ),
        )


@ingest_cli.command("layer")
@click.argument("parent_layer", type=int)
def queue_layer(parent_layer):
    """
    Helper command
    Queue all chunk tasks at a given layer
    Use this only when all the chunks at `layer - 1` have been built.
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    layer_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[parent_layer]
    chunk_coords = list(product(*[range(r) for r in layer_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    for coords in chunk_coords:
        task_q = imanager.get_task_queue(imanager.config.parents_q_name)
        task_q.enqueue(
            create_parent_chunk,
            job_id=chunk_id_str(parent_layer, coords),
            job_timeout=f"{int(10 * parent_layer)}m",
            result_ttl=0,
            args=(
                imanager.serialized(),
                parent_layer,
                coords,
                get_children_chunk_coords(
                    imanager.chunkedgraph_meta, parent_layer, coords
                ),
            ),
        )


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    for layer in range(2, imanager.chunkedgraph_meta.layer_count + 1):
        layer_count = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {layer_count}")
    print(imanager.chunkedgraph_meta.layer_chunk_counts)


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
