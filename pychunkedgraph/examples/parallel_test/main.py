import click
import redis

from flask import current_app
from flask.cli import AppGroup
from pychunkedgraph.ingest.test_utils import independent_task

ingest_cli = AppGroup('parallel')

def handler(*args, **kwargs):
    '''
    Message handler function, called by redis
    when a message is received on pubsub channel
    '''
    print(args)
    print(kwargs)


@ingest_cli.command('test')
@click.argument('n_chunks', type=int)
@click.argument('chunk_size', type=int)
def create_atomic_chunks(n_chunks, chunk_size):
    print(f'Queueing {n_chunks} chunks of size {chunk_size} ...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'test-channel': handler})
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)

    for chunk_id in range(n_chunks):
        current_app.test_q.enqueue(
            independent_task,
            args=(chunk_id, chunk_size))
    return 'Queued'


def init_parallel_test_cmds(app):
    app.cli.add_command(ingest_cli)