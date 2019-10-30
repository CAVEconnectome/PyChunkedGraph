"""
cli for running ingest
"""

import time

import numpy as np
import click
from flask.cli import AppGroup

from . import IngestConfig
from .utils import chunk_id_str
from .utils import initialize_chunkedgraph
from .manager import IngestionManager
from .cluster import enqueue_atomic_tasks
from .cluster import create_parent_chunk
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys
from ..backend import ChunkedGraphMeta
from ..backend import DataSource
from ..backend import GraphConfig
from ..backend import BigTableConfig
from ..backend.chunks.hierarchy import get_children_coords

ingest_cli = AppGroup("ingest")


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
@click.option("--raw", is_flag=True, help="Use processed data to build graph")
@click.option("--overwrite", is_flag=True, help="Overwrite existing graph")
def ingest_graph(graph_id: str, raw: bool, overwrite: bool):

    ingest_config = IngestConfig()
    bigtable_config = BigTableConfig(table_id_prefix="ak")

    graph_config = GraphConfig(
        graph_id=f"{bigtable_config.table_id_prefix}-{graph_id}",
        chunk_size=np.array([512, 512, 128], dtype=int),
        overwrite=overwrite,
    )

    data_source = DataSource(
        agglomeration="gs://ranl/scratch/pinky100_ca_com/agg",
        watershed="gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com",
        edges="gs://chunkedgraph/pinky100/edges",
        components="gs://chunkedgraph/pinky100/components",
        data_version=4,
    )

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    initialize_chunkedgraph(meta)

    imanager = IngestionManager(ingest_config, meta)
    imanager.redis.flushdb()
    enqueue_atomic_tasks(imanager)


@ingest_cli.command("queue")
@click.argument("chunk_info", nargs=4, type=int)
def queue_parent(chunk_info):

    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))

    parent_layer = chunk_info[0] + 1
    parent_coords = (
        np.array(chunk_info[1:], int) // imanager.chunkedgraph_meta.graph_config.fanout
    )
    parent_chunk_str = "_".join(map(str, parent_coords))

    parents_queue = imanager.get_task_queue(imanager.config.parents_q_name)
    parents_queue.enqueue(
        create_parent_chunk,
        job_id=chunk_id_str(parent_layer, parent_coords),
        job_timeout=f"{int(1.5 * parent_layer)}m",
        result_ttl=0,
        args=(
            imanager.serialized(),
            parent_layer,
            parent_coords,
            get_children_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            ),
        ),
    )
    imanager.redis.hdel(parent_layer, parent_chunk_str)
    imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


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
