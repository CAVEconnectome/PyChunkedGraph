"""
cli for redis jobs
"""
import os
import sys

import click
from redis import Redis
from rq import Queue
from rq import Worker
from rq.worker import WorkerStatus
from rq.job import Job
from rq.exceptions import InvalidJobOperationError
from rq.registry import FailedJobRegistry
from flask import current_app
from flask.cli import AppGroup

from ..utils.redis import REDIS_HOST
from ..utils.redis import REDIS_PORT
from ..utils.redis import REDIS_PASSWORD


redis_cli = AppGroup("redis")
connection = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, password=REDIS_PASSWORD)


@redis_cli.command("status")
@click.argument("queue", type=str, default="test")
@click.option("--show-busy", is_flag=True)
def get_status(queue, show_busy):
    print("NOTE: Use --show-busy to display count of non idle workers\n")
    q = Queue(queue, connection=connection)
    print(f"Queue name \t: {queue}")
    print(f"Jobs queued \t: {len(q)}")
    print(f"Workers total \t: {Worker.count(queue=q)}")
    if show_busy:
        workers = Worker.all(queue=q)
        count = sum([worker.get_state() == WorkerStatus.BUSY for worker in workers])
        print(f"Workers busy \t: {count}")
    print(f"Jobs failed \t: {q.failed_job_registry.count}")


@redis_cli.command("failed_ids")
@click.argument("queue", type=str)
def failed_jobs(queue):
    q = Queue(queue, connection=connection)
    ids = q.failed_job_registry.get_job_ids()
    print("\n".join(ids))


@redis_cli.command("failed_info")
@click.argument("queue", type=str)
@click.argument("id", type=str)
def failed_job_info(queue, id):
    j = Job.fetch(id, connection=connection)
    print("KWARGS")
    print(j.kwargs)
    print("\nARGS")
    print(j.args)
    print("\nEXCEPTION")
    print(j.exc_info)


@redis_cli.command("empty")
@click.argument("queue", type=str)
def empty_queue(queue):
    q = Queue(queue, connection=connection)
    job_count = len(q)
    q.empty()
    print(f"{job_count} jobs removed from {queue}.")


@redis_cli.command("requeue")
@click.argument("queue", type=str)
@click.option("--all", "-a", is_flag=True, help="Requeue all failed jobs")
@click.argument("job_ids", nargs=-1)
def requeue(queue, all, job_ids):
    """Requeue failed jobs."""
    failed_job_registry = FailedJobRegistry(queue, connection=connection)
    if all:
        job_ids = failed_job_registry.get_job_ids()

    if not job_ids:
        click.echo("Nothing to do")
        sys.exit(0)

    click.echo("Requeueing {0} jobs from failed queue".format(len(job_ids)))
    fail_count = 0
    for job_id in job_ids:
        try:
            failed_job_registry.requeue(job_id)
        except InvalidJobOperationError:
            fail_count += 1

    if fail_count > 0:
        click.secho(
            "Unable to requeue {0} jobs from failed job registry".format(fail_count),
            fg="red",
        )


def init_redis_cmds(app):
    app.cli.add_command(redis_cli)
