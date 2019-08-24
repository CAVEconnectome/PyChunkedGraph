"""
cli for running ingest
"""
from collections import defaultdict

import numpy as np
import click
from flask import current_app
from flask.cli import AppGroup

from .ran_ingestion_v2 import ingest_into_chunkedgraph
from .ran_ingestion_v2 import queue_parent

ingest_cli = AppGroup("ingest")
imanager = None
tasks = defaultdict(list)
layer_parent_children_counts = {}


def _get_children_count(chunk_coord, layer_id):
    global imanager
    child_chunk_coords = imanager.chunk_coords // imanager.cg.fan_out ** (layer_id - 3)
    child_chunk_coords = child_chunk_coords.astype(np.int)
    child_chunk_coords = np.unique(child_chunk_coords, axis=0)

    p_chunk_coords = child_chunk_coords // imanager.cg.fan_out
    p_chunk_coords = p_chunk_coords.astype(np.int)
    p_chunk_coords, counts = np.unique(p_chunk_coords, axis=0, return_counts=True)

    return dict(zip([str(coord) for coord in p_chunk_coords], counts))


def handle_job_result(*args, **kwargs):
    """handle worker return"""
    global imanager
    result = np.frombuffer(args[0]['data'], dtype=int)
    layer_id = result[0] + 1
    chunk_coord = result[1:]
    p_chunk_coord = chunk_coord // 2
    tasks[str(p_chunk_coord)].append(chunk_coord)
    children_count = len(tasks[str(p_chunk_coord)])

    if not layer_id in layer_parent_children_counts:
        layer_parent_children_counts[layer_id] = _get_children_count(p_chunk_coord, layer_id)
    n_children = layer_parent_children_counts[layer_id][str(p_chunk_coord)]
    
    if children_count == n_children:
        queue_parent(imanager, layer_id, p_chunk_coord, tasks.pop(str(p_chunk_coord)))
        with open("completed.txt", "a") as completed_f:
            completed_f.write(f"{p_chunk_coord}:{children_count}\n")        

    with open("results.txt", "a") as results_f:
        results_f.write(f"{chunk_coord}:{p_chunk_coord}:{children_count}\n")


@ingest_cli.command("atomic")
@click.argument("storage_path", type=str)
@click.argument("ws_cv_path", type=str)
@click.argument("edge_dir", type=str)
@click.argument("cg_table_id", type=str)
@click.argument("n_chunks", type=int, default=None)
def run_ingest(storage_path, ws_cv_path, cg_table_id, edge_dir=None, n_chunks=None):
    """
    run ingestion job
    eg: flask ingest atomic \
        gs://ranl/scratch/pinky100_ca_com/agg \
        gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com \
        gs://akhilesh-test/edges/pinky100-ingest \
        akhilesh-pinky100 \
        71400
    """
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{"ingest_channel": handle_job_result})
    chunk_pubsub.run_in_thread(sleep_time=0.1)

    global imanager
    imanager = ingest_into_chunkedgraph(
        storage_path=storage_path,
        ws_cv_path=ws_cv_path,
        cg_table_id=cg_table_id,
        edge_dir=edge_dir,
        n_chunks=n_chunks,
    )


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
