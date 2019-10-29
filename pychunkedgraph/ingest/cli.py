"""
cli for running ingest
"""

import time

import numpy as np
import click
from flask.cli import AppGroup

from . import IngestConfig
from .utils import initialize_chunkedgraph
from .manager import IngestionManager
from .cluster import enqueue_atomic_tasks
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys
from ..backend import ChunkedGraphMeta
from ..backend import DataSource
from ..backend import GraphConfig
from ..backend import BigTableConfig

ingest_cli = AppGroup("ingest")


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
@click.option("--raw", is_flag=True, help="Use processed data to build graph")
def ingest_graph(graph_id: str, raw: bool):

    ingest_config = IngestConfig()
    bigtable_config = BigTableConfig(table_id_prefix="akhilesh")

    graph_config = GraphConfig(
        graph_id=f"{bigtable_config.table_id_prefix}-{graph_id}",
        chunk_size=np.array([256, 256, 512], dtype=int),
    )

    data_source = DataSource(
        agglomeration="gs://ranl-scratch/minnie65_0/agg",
        watershed="gs://microns-seunglab/minnie65/ws_minnie65_0",
        edges="gs://chunkedgraph/minnie65_0/edges",
        components="gs://chunkedgraph/minnie65_0/components",
        data_version=2,
    )

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    initialize_chunkedgraph(meta)

    imanager = IngestionManager(ingest_config, meta)
    imanager.redis.flushdb()
    enqueue_atomic_tasks(imanager)


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
