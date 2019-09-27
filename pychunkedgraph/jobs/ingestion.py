"""
cli for running ingest
"""
import sys
import time
from collections import defaultdict, deque
from itertools import product
from typing import List, Union, Tuple

import numpy as np
import click
from flask.cli import AppGroup
from twisted.internet import task, reactor

from ..ingest.ran_ingestion_v2 import INGEST_CHANNEL
from ..ingest.ran_ingestion_v2 import INGEST_QUEUE
from ..ingest.ran_ingestion_v2 import ingest_into_chunkedgraph
from ..ingest.ran_ingestion_v2 import enqueue_atomic_tasks
from ..ingest.ran_ingestion_v2 import create_parent_chunk
from ..utils.redis import get_rq_queue
from ..utils.redis import get_redis_connection
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig

ingest_cli = AppGroup("ingest")

imanager = None
connection = get_redis_connection()
task_q = get_rq_queue(INGEST_QUEUE)


# 1. ingest from raw data
#    raw_ag_path, raw_ws_path, edges_path, components_path, graph_config, bigtable_config
# 2. create intermediate data
# 3. ingest from intermediate data


def _get_children_coords(layer, parent_coords) -> Tuple[bool, List[np.ndarray]]:
    global imanager
    layer_bounds = imanager.chunk_id_bounds / (2 ** (layer - 2))
    layer_bounds = np.ceil(layer_bounds).astype(np.int)
    children_coords = []
    parent_coords = np.array(parent_coords, dtype=int)
    for dcoord in product(*[range(imanager.cg.fan_out)] * 3):
        dcoord = np.array(dcoord, dtype=int)
        child_coords = parent_coords * imanager.cg.fan_out + dcoord
        check_bounds = np.less(child_coords, layer_bounds[:, 1])
        if np.all(check_bounds):
            children_coords.append(child_coords)
    return children_coords


def enqueue_parent_tasks():
    """
    from redis results get parent chunks ids
    determine child chunks for these parents
    check their existence in redis results
    if all exist, enqueue parent, delete children    
    """
    global connection
    global imanager
    zset_name = f"rq:finished:{INGEST_QUEUE}"
    results = connection.zrange(zset_name, 0, -1)
    layer_counts_d = defaultdict(int)
    parent_chunks_d = defaultdict(list)  # (layer, x, y, z) as keys
    for chunk_str in results:
        chunk_str = chunk_str.decode("utf-8")
        layer, x, y, z = map(int, chunk_str.split("_"))
        layer_counts_d[layer] += 1
        layer += 1
        x, y, z = np.array([x, y, z], int) // imanager.cg.fan_out
        parent_chunks_d[(layer, x, y, z)].append(chunk_str)

    count = 0
    for parent_chunk in parent_chunks_d:
        children_coords = _get_children_coords(parent_chunk[0] - 1, parent_chunk[1:])
        children_results = parent_chunks_d[parent_chunk]
        job_id = f"{parent_chunk[0]}_{'_'.join(map(str, parent_chunk[1:]))}"
        if len(children_coords) == len(children_results):
            task_q.enqueue(
                create_parent_chunk,
                at_front=True,
                job_id=job_id,
                job_timeout="59m",
                result_ttl=86400,
                args=(imanager.get_serialized_info(), parent_chunk[0], children_coords),
            )
            count += 1

    print(
        " ".join([f"{l}:{layer_counts_d[l]}" for l in range(2, imanager.n_layers + 1)])
    )
    print(f"queued {count} parents\n")


@ingest_cli.command("raw")
# @click.option("--agglomeration", required=True, type=str)
# @click.option("--watershed", required=True, type=str)
# @click.option("--edges", required=True, type=str)
# @click.option("--components", required=True, type=str)
# @click.option("--data-size", required=False, nargs=3, type=int)
@click.option("--graph-id", required=True, type=str)
# @click.option("--chunk-size", required=True, nargs=3, type=int)
# @click.option("--fanout", required=False, type=int)
# @click.option("--gcp-project-id", required=False, type=str)
# @click.option("--bigtable-instance-id", required=False, type=str)
# @click.option("--interval", required=False, type=float)
def run_ingest(
    # agglomeration,
    # watershed,
    # edges,
    # components,
    graph_id,
    # chunk_size,
    # data_size=None,
    # fanout=2,
    # gcp_project_id=None,
    # bigtable_instance_id=None,
    # interval=90.0
):
    global imanager
    agglomeration = "gs://ranl-scratch/190410_FAFB_v02_ws_size_threshold_200/agg"
    watershed = (
        "gs://microns-seunglab/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200"
    )
    edges = "gs://akhilesh-pcg/190410_FAFB_v02/edges"
    components = "gs://akhilesh-pcg/190410_FAFB_v02/components"
    use_raw_edges = False
    use_raw_components = False

    chunk_size = [256, 256, 512]
    fanout = 2
    gcp_project_id = None
    bigtable_instance_id = None

    data_source = DataSource(
        agglomeration, watershed, edges, components, use_raw_edges, use_raw_components
    )
    graph_config = GraphConfig(graph_id, chunk_size, fanout)
    bigtable_config = BigTableConfig(gcp_project_id, bigtable_instance_id)

    connection.flushdb()
    imanager = ingest_into_chunkedgraph(data_source, graph_config, bigtable_config)

    interval = 5.0
    print(f"\nChecking completed tasks every {interval} seconds.")
    loop_call = task.LoopingCall(enqueue_parent_tasks)
    loop_call.start(interval)
    
    enqueue_atomic_tasks(imanager, batch_size=10000, interval=60.0)
    reactor.run()


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
