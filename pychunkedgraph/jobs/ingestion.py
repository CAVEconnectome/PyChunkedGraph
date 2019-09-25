"""
cli for running ingest
"""
from collections import defaultdict, deque
from itertools import product
from typing import List

import numpy as np
import click
from flask import current_app
from flask.cli import AppGroup
from twisted.internet import task, reactor

from .chunk_task import ChunkTask
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


def _check_children_status(cg, layer, parent_coords) -> bool:
    """
    Checks if all the children chunks have been processed
        If yes, delete their entries from "results" hash, return True
        If no, return False
    """
    hash_name = "results"
    children_coords = []
    children_keys = []
    for dcoord in product(*[range(cg.fan_out)] * 3):
        child_coords = np.array(parent_coords[1:]) * cg.fan_out + np.array(dcoord)
        children_coords.append(child_coords)
        children_keys.append(f"{layer}_{'_'.join(map(str, child_coords))}")
    children_results = connection.hmget(hash_name, children_keys)
    completed = None not in children_results
    if completed:
        connection.hdel(hash_name, children_keys)
    return completed


def enqueue_parent_tasks():
    global connection
    global imanager
    results = connection.hvals("result")
    cg = imanager.cg

    # get parent chunks ids
    # get parent child chunks
    # check their existence in redis
    # if all exist, enqueue parent, delete children
    parent_chunks = set()
    for chunk_str in results:
        layer, x, y, z = map(int, chunk_str.split("_"))
        chunk_coord = np.array([x, y, z], np.uint64)
        parent_chunk_coord = chunk_coord // imanager.cg.fan_out
        x, y, z = parent_chunk_coord
        layer += 1
        parent_chunks.add((layer, x, y, z))

    for parent_chunk in parent_chunks:
        pass

    pass
    # with open(f"completed.txt", "a") as completed_f:
    #     completed_f.write(f"{args[0]['data'].decode('utf-8')}\n")
    # task_q.enqueue(
    #     create_parent_chunk,
    #     job_timeout="59m",
    #     result_ttl=86400,
    #     args=(
    #         imanager.get_serialized_info(),
    #         parent_task.layer,
    #         parent_task.children_coords,
    #     ),
    # )


@ingest_cli.command("raw")
# @click.option("--agglomeration", required=True, type=str)
# @click.option("--watershed", required=True, type=str)
# @click.option("--edges", required=True, type=str)
# @click.option("--components", required=True, type=str)
# @click.option("--data-size", required=False, nargs=3, type=int)
# @click.option("--graph-id", required=True, type=str)
# @click.option("--chunk-size", required=True, nargs=3, type=int)
# @click.option("--fanout", required=False, type=int)
# @click.option("--gcp-project-id", required=False, type=str)
# @click.option("--bigtable-instance-id", required=False, type=str)
def run_ingest(
    # agglomeration,
    # watershed,
    # edges,
    # components,
    # graph_id,
    # chunk_size,
    # data_size=None,
    # fanout=2,
    # gcp_project_id=None,
    # bigtable_instance_id=None,
):
    global imanager
    agglomeration = "gs://ranl-scratch/190410_FAFB_v02_ws_size_threshold_200/agg"
    watershed = (
        "gs://microns-seunglab/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200"
    )
    edges = "gs://akhilesh-pcg/190410_FAFB_v02/edges"
    components = "gs://akhilesh-pcg/190410_FAFB_v02/components"
    use_raw_edges = True
    use_raw_components = True

    graph_id = "akhilesh-190410_FAFB_v02-0"
    chunk_size = [256, 256, 512]
    fanout = 2
    gcp_project_id = None
    bigtable_instance_id = None

    data_source = DataSource(
        agglomeration, watershed, edges, components, use_raw_edges, use_raw_components
    )
    graph_config = GraphConfig(graph_id, chunk_size, fanout)
    bigtable_config = BigTableConfig(gcp_project_id, bigtable_instance_id)

    imanager = ingest_into_chunkedgraph(data_source, graph_config, bigtable_config)
    enqueue_atomic_tasks(imanager)

    timeout = 10.0
    loop_call = task.LoopingCall(enqueue_parent_tasks)
    loop_call.start(timeout)
    reactor.run()


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
