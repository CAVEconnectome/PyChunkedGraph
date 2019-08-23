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


def handle_job_result(*args, **kwargs):
    """handle worker return"""
    result = np.frombuffer(args[0]['data'], dtype=int)
    layer_id = result[0]
    chunk_coord = result[1:]
    p_chunk_coord = chunk_coord // 2

    tasks[str(p_chunk_coord)].append(chunk_coord)
    n_children = len(tasks[str(p_chunk_coord)])
    if n_children == 8:
        print(f"{p_chunk_coord} children done")
        queue_parent(imanager, layer_id+1, p_chunk_coord, tasks.pop(str(p_chunk_coord)))

    with open("results.txt", "a") as results_f:
        results_f.write(f"{chunk_coord}:{p_chunk_coord}:{n_children}\n")


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
        gs://akhilesh-test/edges/pinky-ingest-test \
        pinky100-compressed-all-layers
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
