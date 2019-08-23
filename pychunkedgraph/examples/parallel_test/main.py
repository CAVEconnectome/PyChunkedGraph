import click
import redis

from flask import current_app
from flask.cli import AppGroup
from pychunkedgraph.examples.parallel_test.tasks import independent_task

ingest_cli = AppGroup('parallel')

def handler(*args, **kwargs):
    '''
    Message handler function, called by redis
    when a message is received on pubsub channel
    '''
    print(args)
    print(kwargs)


@ingest_cli.command('test')
@click.argument('n', type=int)
@click.argument('size', type=int)
def create_atomic_chunks(n, size):
    print(f'Queueing {n} items of size {size} ...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'test-channel': handler})
    
    for item_id in range(n):
        current_app.test_q.enqueue(
            independent_task,
            args=(item_id, size))
    
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)
    return 'Queued'


def init_parallel_test_cmds(app):
    app.cli.add_command(ingest_cli)