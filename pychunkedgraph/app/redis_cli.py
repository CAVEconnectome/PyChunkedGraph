"""
cli for redis jobs
"""
import os

import click
from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from flask import current_app
from flask.cli import AppGroup


REDIS_HOST = os.environ.get("REDIS_SERVICE_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_SERVICE_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

redis_cli = AppGroup("redis")
connection=Redis(
    host=os.environ['REDIS_SERVICE_HOST'],port=6379,db=0, password='dev')

@redis_cli.command("status")
@click.argument("queue", type=str, default="test")
def get_status(queue="test"):
    q=Queue(queue, connection=connection)
    workers = Worker.all(queue=q)
    print(f"Queue name \t: {queue}")
    print(f"Jobs queued \t: {len(q)}")
    print(f"Workers count \t: {len(workers)}")
    print(f"Jobs failed \t: {q.failed_job_registry.count}")


@redis_cli.command("failed_ids")
@click.argument("queue", type=str)
def failed_jobs(queue):
    q=Queue(queue, connection=connection)
    ids = q.failed_job_registry.get_job_ids()
    print("\n".join(ids))


@redis_cli.command("failed_info")
@click.argument("queue", type=str)
@click.argument("id", type=str)
def failed_job_info(queue, id):
    j=Job.fetch(id,connection=connection)
    print("kwargs")
    print(j.kwargs)
    print("args")
    print(j.args)
    print("exception")
    print(j.exc_info)


@redis_cli.command("empty")
@click.argument("queue", type=str)
def empty_queue(queue):
    q=Queue(queue, connection=connection)
    job_count = len(q)
    q.empty()
    print(f"{job_count} jobs removed from {queue}.")


def init_redis_cmds(app):
    app.cli.add_command(redis_cli)
