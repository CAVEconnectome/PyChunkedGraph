"""
cli for running ingest
"""
from collections import defaultdict

import click
from flask import current_app
from flask.cli import AppGroup

from .ran_ingestion_v2 import ingest_into_chunkedgraph

ingest_cli = AppGroup("ingest")

tasks = defaultdict(int)


def handle_job_result(*args, **kwargs):
    """handle worker return"""
    p_chunk_coord = str(args[0]['data'])
    tasks[p_chunk_coord] += 1
    if tasks[p_chunk_coord] == 8:
        print(f"{p_chunk_coord} done")
        with open("results.txt", "a") as results_f:
            results_f.write(f"{p_chunk_coord}\n")


@ingest_cli.command("atomic")
@click.argument("storage_path", type=str)
@click.argument("ws_cv_path", type=str)
@click.argument("edge_dir", type=str)
@click.argument("cg_table_id", type=str)
@click.argument("n_chunks", type=int)
def run_ingest(storage_path, ws_cv_path, cg_table_id, edge_dir=None, n_chunks=-1):
    """
    run ingestion job
    eg: flask ingest atomic \
        gs://ranl/scratch/pinky100_ca_com/agg \
        gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com \
        gs://akhilesh-test/edges/pinky-ingest-test \
        akhilesh-ingest-test \
        1
    """
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{"ingest_channel": handle_job_result})
    chunk_pubsub.run_in_thread(sleep_time=0.1)

    ingest_into_chunkedgraph(
        storage_path=storage_path,
        ws_cv_path=ws_cv_path,
        cg_table_id=cg_table_id,
        edge_dir=edge_dir,
        n_chunks=n_chunks,
    )


@ingest_cli.command("layer")
@click.argument("storage_path", type=str)
@click.argument("ws_cv_path", type=str)
@click.argument("cg_table_id", type=str)
@click.argument("layer_id", type=int)
def create_abstract(storage_path, ws_cv_path, cg_table_id, layer_id=3):
    """
    run ingestion job
    eg: flask ingest layer \
        gs://ranl/scratch/pinky100_ca_com/agg \
        gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com \
        akhilesh-pinky100-compressed \
        3
    """
    assert layer_id > 2
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{"ingest_channel": handle_job_result})
    chunk_pubsub.run_in_thread(sleep_time=0.1)

    ingest_into_chunkedgraph(
        storage_path=storage_path,
        ws_cv_path=ws_cv_path,
        cg_table_id=cg_table_id,
        start_layer=layer_id,
        is_new=False
    )


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
