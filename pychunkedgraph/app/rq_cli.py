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
from rq.registry import StartedJobRegistry
from rq.registry import FailedJobRegistry
from flask import current_app
from flask.cli import AppGroup

from ..utils.redis import REDIS_HOST
from ..utils.redis import REDIS_PORT
from ..utils.redis import REDIS_PASSWORD


rq_cli = AppGroup("rq")
connection = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, password=REDIS_PASSWORD)


@rq_cli.command("status")
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


@rq_cli.command("failed_ids")
@click.argument("queue", type=str)
def failed_jobs(queue):
    q = Queue(queue, connection=connection)
    ids = q.failed_job_registry.get_job_ids()
    print("\n".join(ids))


@rq_cli.command("failed_info")
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


@rq_cli.command("empty")
@click.argument("queue", type=str)
def empty_queue(queue):
    q = Queue(queue, connection=connection)
    job_count = len(q)
    q.empty()
    print(f"{job_count} jobs removed from {queue}.")


@rq_cli.command("reenqueue")
@click.argument("queue", type=str)
@click.argument("job_ids", nargs=-1, required=True)
def enqueue(queue, job_ids):
    """Enqueues existing jobs that are stuck for whatever reason."""
    print(queue, job_ids)
    pass


@rq_cli.command("requeue")
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

    click.echo(f"Requeueing {len(job_ids)} jobs from failed queue")
    fail_count = 0
    for job_id in job_ids:
        try:
            failed_job_registry.requeue(job_id)
        except InvalidJobOperationError:
            fail_count += 1

    if fail_count > 0:
        click.secho(
            f"Unable to requeue {fail_count} jobs from failed job registry", fg="red"
        )


@rq_cli.command("cleanup")
@click.argument("queue", type=str)
def clean_start_registry(queue):
    """
    Clean started job registry
    Sometimes started jobs are not moved to failed registry (network issues)
    This command takes the jobs off the started registry and reueues them
    """
    registry = StartedJobRegistry(name=queue, connection=connection)
    cleaned_jobs = registry.cleanup()
    print(f"Requeued {len(cleaned_jobs)} jobs from the started job registry.")


def init_redis_cmds(app):
    app.cli.add_command(rq_cli)


# import os
# import redis
# from rq import Queue
# from rq import Worker
# from rq.worker import WorkerStatus
# from rq.job import Job
# from rq.exceptions import InvalidJobOperationError
# from rq.registry import FailedJobRegistry, StartedJobRegistry

