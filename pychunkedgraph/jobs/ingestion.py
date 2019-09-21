"""
cli for running ingest
"""
from collections import defaultdict, deque

import numpy as np
import click
from flask import current_app
from flask.cli import AppGroup

from .chunk_task import ChunkTask
from ..ingest.ran_ingestion_v2 import INGEST_CHANNEL
from ..ingest.ran_ingestion_v2 import INGEST_QUEUE
from ..ingest.ran_ingestion_v2 import ingest_into_chunkedgraph
from ..ingest.ran_ingestion_v2 import enqueue_atomic_tasks
from ..ingest.ran_ingestion_v2 import create_parent_chunk
from ..utils.redis import get_rq_queue
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig

ingest_cli = AppGroup("ingest")

imanager = None
task_count = 0
tasks_d = {}
task_q = get_rq_queue(INGEST_QUEUE)


# 1. ingest from raw data
#    raw_ag_path, raw_ws_path, edges_path, components_path, graph_config, bigtable_config
# 2. create intermediate data
# 3. ingest from intermediate data


def handle_job_result(*args, **kwargs):
    """
    handler function, listens to workers' return value
    queues parent chunk tasks when children chunks are complete
    """
    global tasks_d, task_count
    return_vals = args[0]["data"].decode("utf-8").split(",")
    task_id = return_vals[0]
    task_count += 1

    with open(f"completed.txt", "a") as completed_f:
        completed_f.write(f"{args[0]['data'].decode('utf-8')}\n")

    task = tasks_d[task_id]
    parent_id = task.parent_id
    if parent_id:
        parent_task = tasks_d[parent_id]
        parent_task.remove_child(task_id)
        if parent_task.dependencies == 0:
            task_q.enqueue(
                create_parent_chunk,
                job_timeout="59m",
                result_ttl=0,
                args=(
                    imanager.get_serialized_info(),
                    parent_task.layer,
                    parent_task.children_coords,
                ),
            )


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
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{INGEST_CHANNEL: handle_job_result})
    chunk_pubsub.run_in_thread(sleep_time=0.1)

    agglomeration = "gs://ranl-scratch/190410_FAFB_v02_ws_size_threshold_200"
    watershed = (
        "gs://microns-seunglab/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200"
    )
    edges = "gs://akhilesh-pcg/190410_FAFB_v02/edges"
    components = "gs://akhilesh-pcg/190410_FAFB_v02/components"
    graph_id = "akhilesh-190410_FAFB_v02-0"
    chunk_size = [256, 256, 512]
    fanout = 2
    gcp_project_id = None
    bigtable_instance_id = None

    data_source = DataSource(agglomeration, watershed, edges, components)
    graph_config = GraphConfig(graph_id, chunk_size, fanout)
    bigtable_config = BigTableConfig(gcp_project_id, bigtable_instance_id)

    imanager = ingest_into_chunkedgraph(data_source, graph_config, bigtable_config)
    root_task = _build_job_hierarchy()
    queue = deque([root_task.id])
    layer_counts = defaultdict(int)

    while queue:
        task_id = queue.pop()
        task = tasks_d[task_id]
        layer_counts[task.layer] += len(task.children)
        queue.extendleft(task.children)
    print(layer_counts)
    print(f"total jobs {sum(layer_counts.values())}")
    enqueue_atomic_tasks(imanager)
    return tasks_d


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)


def _build_job_hierarchy():
    global tasks_d
    root_task = ChunkTask(imanager.n_layers, np.array([0, 0, 0]))
    tasks_d[root_task.id] = root_task

    start_layer = imanager.n_layers
    stop_layer = 2
    for layer in range(start_layer, stop_layer, -1):
        _build_layer(layer)
    return root_task


def _build_layer(layer):
    global tasks_d
    layer_children_coords = imanager.chunk_coords // imanager.cg.fan_out ** (layer - 3)
    layer_children_coords = layer_children_coords.astype(np.int)
    layer_children_coords = np.unique(layer_children_coords, axis=0)

    layer_parents_coords = layer_children_coords // imanager.cg.fan_out
    layer_parents_coords = layer_parents_coords.astype(np.int)
    layer_parents_coords, indices = np.unique(
        layer_parents_coords, axis=0, return_inverse=True
    )
    parent_indices = np.arange(len(layer_parents_coords), dtype=np.int)
    for parent_idx in parent_indices:
        parent_coords = layer_parents_coords[parent_idx]
        parent_id = ChunkTask.get_id(layer, parent_coords)
        parent_task = tasks_d[parent_id]
        children_coords = layer_children_coords[indices == parent_idx]
        parent_task.children_coords = children_coords
        for child_coords in children_coords:
            child = ChunkTask(layer - 1, child_coords, parent_id=parent_id)
            tasks_d[child.id] = child
            parent_task.add_child(child.id)
