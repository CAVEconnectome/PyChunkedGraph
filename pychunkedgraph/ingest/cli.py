"""
cli for running ingest
"""

import click
from flask import current_app
from flask.cli import AppGroup

from .ran_ingestion_v2 import ingest_into_chunkedgraph

ingest_cli = AppGroup("ingest")


def handle_job_result(*args, **kwargs):
    """handle worker return"""
    with open("results.txt", "a") as results_f:
        results_f.write(f"{str(args[0]['data'])}\n")


@ingest_cli.command("atomic")
@click.argument("storage_path", type=str)
@click.argument("ws_cv_path", type=str)
@click.argument("edge_dir", type=str)
@click.argument("cg_table_id", type=str)
@click.argument("n_chunks", type=int)
def run_ingest(storage_path, ws_cv_path, cg_table_id, edge_dir=None, n_chunks=-1):

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


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
