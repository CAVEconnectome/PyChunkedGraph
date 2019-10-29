"""
cli for running ingest
"""

import time
import pickle

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
    ingest_config = IngestConfig(build_graph=True)
    data_source = DataSource(
        agglomeration="gs://ranl/scratch/pinky100_ca_com/agg",
        watershed="gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com",
        edges="gs://akhilesh-pcg/pinky100-test/edges",
        components="gs://akhilesh-pcg/pinky100-test/components",
        data_version=4,
    )
    graph_config = GraphConfig(
        graph_id=graph_id,
        chunk_size=np.array([512, 512, 128], dtype=int),
        fanout=2,
        s_bits_atomic_layer=10,
    )
    bigtable_config = BigTableConfig()

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    imanager = IngestionManager(ingest_config, meta)
    imanager.redis.flushdb()

    if ingest_config.build_graph:
        initialize_chunkedgraph(meta)

    enqueue_atomic_tasks(imanager)


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    imanager = pickle.loads(redis.get(r_keys.INGESTION_MANAGER))
    for layer in range(2, imanager.chunkedgraph_meta.layer_count + 1):
        layer_count = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {layer_count}")
    print(imanager.chunkedgraph_meta.layer_chunk_counts)


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
