"""
cli for running ingest
"""
from collections import defaultdict

import numpy as np
import click
from flask import current_app
from flask.cli import AppGroup

from .ran_ingestion_v2 import ingest_into_chunkedgraph

ingest_cli = AppGroup("ingest")
task_count = 0


def handle_job_result(*args, **kwargs):
    """handle worker return"""
    global task_count
    result = np.frombuffer(args[0]['data'], dtype=np.int32)
    layer = result[0]
    task_count += 1

    with open(f"completed_{layer}.txt", "w") as completed_f:
        completed_f.write(str(task_count))


@ingest_cli.command("table")
@click.argument("storage_path", type=str)
@click.argument("ws_cv_path", type=str)
@click.argument("edge_dir", type=str)
@click.argument("cg_table_id", type=str)
@click.argument("layer", type=int, default=None)
def run_ingest(storage_path, ws_cv_path, cg_table_id, edge_dir, layer):
    """
    run ingestion job
    eg: flask ingest table \
        gs://ranl/scratch/pinky100_ca_com/agg \
        gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com \
        gs://akhilesh-test/edges/pinky100-ingest \
        akhilesh-pinky100 \
        2
    """
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{"ingest_channel": handle_job_result})
    chunk_pubsub.run_in_thread(sleep_time=0.1)

    ingest_into_chunkedgraph(
        storage_path=storage_path,
        ws_cv_path=ws_cv_path,
        cg_table_id=cg_table_id,
        edge_dir=edge_dir,
        layer=layer
    )


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
