"""
cli for running ingest
"""
import sys
import time
from collections import defaultdict
from itertools import product
from typing import List
from typing import Sequence

import numpy as np
import click
from flask.cli import AppGroup

from . import IngestConfig
from .ingestion_utils import initialize_chunkedgraph
from .ingestionmanager import IngestionManager
from .ran_ingestion_v2 import enqueue_atomic_tasks
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys
from ..backend import ChunkedGraphMeta
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig

ingest_cli = AppGroup("ingest")


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
# @click.option("--agglomeration", required=True, type=str)
# @click.option("--watershed", required=True, type=str)
# @click.option("--edges", required=True, type=str)
# @click.option("--components", required=True, type=str)
@click.option("--processed", is_flag=True, help="Use processed data to build graph")
# @click.option("--data-size", required=False, nargs=3, type=int)
# @click.option("--chunk-size", required=True, nargs=3, type=int)
# @click.option("--fanout", required=False, type=int)
# @click.option("--gcp-project-id", required=False, type=str)
# @click.option("--bigtable-instance-id", required=False, type=str)
# @click.option("--interval", required=False, type=float)
def ingest_graph(
    graph_id,
    # agglomeration,
    # watershed,
    # edges,
    # components,
    processed,
    # chunk_size,
    # data_size=None,
    # fanout=2,
    # gcp_project_id=None,
    # bigtable_instance_id=None,
    # interval=90.0
):
    ingest_config = IngestConfig(build_graph=True)
    data_source = DataSource(
        agglomeration="gs://ranl-scratch/minnie65_0/agg",
        watershed="gs://microns-seunglab/minnie65/ws_minnie65_0",
        edges="gs://chunkedgraph/minnie65_0/edges",
        components="gs://chunkedgraph/minnie65_0/components",
        use_raw_edges=not processed,
        use_raw_components=not processed,
        data_version=2,
    )
    graph_config = GraphConfig(
        graph_id=graph_id,
        chunk_size=np.array([256, 256, 512], dtype=int),
        fanout=2,
        s_bits_atomic_layer=10,
    )
    bigtable_config = BigTableConfig()

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    imanager = IngestionManager(ingest_config, meta)
    imanager.redis.flushdb()

    if ingest_config.build_graph:
        initialize_chunkedgraph(
            graph_config.graph_id,
            data_source.watershed,
            graph_config.chunk_size,
            s_bits_atomic_layer=graph_config.s_bits_atomic_layer,
            edge_dir=data_source.edges,
        )

    enqueue_atomic_tasks(imanager)


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    for layer in range(2, imanager.chunkedgraph_meta.layer_count):
        layer_count = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {layer_count}")


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)


# for layer_id in range(2, 13):
#     print(layer_id)
#     child_chunk_coords = im.chunk_coords // 2 ** (layer_id - 3)
#     child_chunk_coords = child_chunk_coords.astype(np.int)
#     child_chunk_coords = np.unique(child_chunk_coords, axis=0)

#     parent_chunk_coords = child_chunk_coords // 2
#     parent_chunk_coords = parent_chunk_coords.astype(np.int)
#     parent_chunk_coords = np.unique(parent_chunk_coords, axis=0)
#     print(len(child_chunk_coords), len(parent_chunk_coords))
