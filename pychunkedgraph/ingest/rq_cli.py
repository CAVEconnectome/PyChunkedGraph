# pylint: disable=invalid-name, missing-function-docstring

"""
cli for redis jobs
"""
import sys

import click
from redis import Redis
from rq import Queue
from rq import Worker
from rq.worker import WorkerStatus
from rq.job import Job
from rq.exceptions import InvalidJobOperationError
from rq.exceptions import NoSuchJobError
from rq.registry import StartedJobRegistry
from rq.registry import FailedJobRegistry
from flask.cli import AppGroup

from ..utils.redis import REDIS_HOST
from ..utils.redis import REDIS_PORT
from ..utils.redis import REDIS_PASSWORD


# rq extended
rq_cli = AppGroup("rq")
connection = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, password=REDIS_PASSWORD)


@rq_cli.command("status")
@click.argument("queues", nargs=-1, type=str)
@click.option("--show-busy", is_flag=True)
def get_status(queues, show_busy):
    print("NOTE: Use --show-busy to display count of non idle workers\n")
    for queue in queues:
        q = Queue(queue, connection=connection)
        print(f"Queue name \t: {queue}")
        print(f"Jobs queued \t: {len(q)}")
        print(f"Workers total \t: {Worker.count(queue=q)}")
        if show_busy:
            workers = Worker.all(queue=q)
            count = sum([worker.get_state() == WorkerStatus.BUSY for worker in workers])
            print(f"Workers busy \t: {count}")
        print(f"Jobs failed \t: {q.failed_job_registry.count}\n")


@rq_cli.command("failed")
@click.argument("queue", type=str)
@click.argument("job_ids", nargs=-1)
def failed_jobs(queue, job_ids):
    if job_ids:
        for job_id in job_ids:
            j = Job.fetch(job_id, connection=connection)
            print(f"JOB ID {job_id}")
            print("KWARGS")
            print(j.kwargs)
            print("\nARGS")
            print(j.args)
            print("\nEXCEPTION")
            print(j.exc_info)
    else:
        q = Queue(queue, connection=connection)
        ids = q.failed_job_registry.get_job_ids()
        print("\n".join(ids))


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
    """Enqueues *existing* jobs that are stuck for whatever reason."""
    q = Queue(queue, connection=connection)
    for job_id in job_ids:
        q.push_job_id(job_id)


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
        except (InvalidJobOperationError, NoSuchJobError):
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


@rq_cli.command("clear_failed")
@click.argument("queue", type=str)
def clear_failed_registry(queue):
    failed_job_registry = FailedJobRegistry(queue, connection=connection)
    job_ids = failed_job_registry.get_job_ids()
    for job_id in job_ids:
        failed_job_registry.remove(job_id, delete_job=True)
    print(f"Deleted {len(job_ids)} jobs from the failed job registry.")


def init_rq_cmds(app):
    app.cli.add_command(rq_cli)
