import time
import click
import redis

from flask import current_app
from flask.cli import AppGroup
from pychunkedgraph.graph.chunkedgraph import ChunkedGraph
from pychunkedgraph.meshing import meshgen
import cloudvolume
import numpy as np
from datetime import datetime

ingest_cli = AppGroup('mesh')

num_messages = 0
messages = []
cg = ChunkedGraph(graph_id='minnie3_v0')

def exc_handler(job, exc_type, exc_value, tb):
    with open('exceptions.txt', 'a') as f:
        f.write(str(job.args)+'\n')
        f.write(str(exc_type) + '\n')
        f.write(str(exc_value) + '\n')


def handlerino_print(*args, **kwargs):
    with open('output.txt', 'a') as f:
        f.write(str(args[0]['data']) + '\n')


def handlerino_periodically_write_to_cloud(*args, **kwargs):
    global num_messages
    num_messages = num_messages + 1
    import pickle
    data = pickle.loads(args[0]['data'])
    print(num_messages, data)
    messages.append(data)
    with open('output.txt', 'a') as f:
        f.write(str(data) + '\n')
    if num_messages % 1000 == 0:
        print('Writing result data to cloud')
        cv_path = cg.meta._ws_cv.base_cloudpath
        filename = f'{datetime.now()}_meshes_{num_messages}'
        with cloudvolume.Storage(cv_path) as storage:
            storage.put_file(
                    file_path=f'meshing_run_data/{filename}',
                    content=','.join(map(str, messages)),
                    compress=False,
                    cache_control='no-cache'
                )


@ingest_cli.command('mesh_chunks_shuffled')
@click.argument('layer', type=int)
@click.argument('x_start', type=int)
@click.argument('y_start', type=int)
@click.argument('z_start', type=int)
@click.argument('x_end', type=int)
@click.argument('y_end', type=int)
@click.argument('z_end', type=int)
def mesh_chunks_shuffled(layer, x_start, y_start, z_start, x_end, y_end, z_end):
    print(f'Queueing...')

    if layer < 3:
        raise ValueError('Only layers >= 3 supported')

    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_periodically_write_to_cloud})
    chunks_arr = []
    for x in range(x_start,x_end):
        for y in range(y_start, y_end):
            for z in range(z_start, z_end):
                chunks_arr.append((x, y, z))

    print(f'Total jobs: {len(chunks_arr)}')
    np.random.shuffle(chunks_arr)

    for chunk in chunks_arr:
        chunk_id = cg.get_chunk_id(None, layer, chunk[0], chunk[1], chunk[2])
        current_app.test_q.enqueue(
            meshgen.chunk_initial_sharded_stitching_task,
            job_timeout='30m',
            args=(
                'minnie3_v0',
                chunk_id,
                2,
                'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v0',
                'graphene_meshes'
            ))
    
    print(f'Queued jobs: {len(current_app.test_q)}')
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)
                
    return 'Queued'


@ingest_cli.command('mesh_chunks_from_file')
@click.argument('filename', type=str)
def mesh_chunks_from_file(filename):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_periodically_write_to_cloud})
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)
    chunk_ids = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            chunk_ids.append(np.uint64(line))
            line = f.readline()
    for chunk_id in chunk_ids:
        current_app.test_q.enqueue(
            meshgen.chunk_initial_sharded_stitching_task,
            job_timeout='30m',
            args=(
                'minnie3_v0',
                chunk_id,
                2,
                'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v0',
                'graphene_meshes'
            ))
                
    return 'Queued'


@ingest_cli.command('mesh_chunks_exclude_file')
@click.argument('layer', type=int)
@click.argument('x_start', type=int)
@click.argument('y_start', type=int)
@click.argument('z_start', type=int)
@click.argument('x_end', type=int)
@click.argument('y_end', type=int)
@click.argument('z_end', type=int)
@click.argument('filename', type=str)
@click.argument('fragment_batch_size', type=int, default=None)
@click.argument('mesh_dir', type=str, default=None)
def mesh_chunks_exclude_file(layer, x_start, y_start, z_start, x_end, y_end, z_end, filename, fragment_batch_size, mesh_dir):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_periodically_write_to_cloud})
    chunk_ids = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            chunk_ids.append(np.uint64(line))
            line = f.readline()
    
    chunks_arr = []
    for x in range(x_start,x_end):
        for y in range(y_start, y_end):
            for z in range(z_start, z_end):
                chunk_id = cg.get_chunk_id(None, layer, x, y, z)
                if not chunk_id in chunk_ids:
                    chunks_arr.append(chunk_id)
    
    print(f'Total jobs: {len(chunks_arr)}')
    np.random.shuffle(chunks_arr)

    for chunk_id in chunks_arr:
        current_app.test_q.enqueue(
            meshgen.chunk_initial_sharded_stitching_task,
            job_timeout='30m',
            args=(
                'minnie3_v0',
                chunk_id,
                2,
                'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v0',
                'graphene_meshes'
            ))

    print(f'Queued jobs: {len(current_app.test_q)}')
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)
                
    return 'Queued' 


@ingest_cli.command('mesh_chunk_ids_shuffled')
@click.argument('chunk_ids_string')
# chunk_ids_string = comma separated string list of chunk ids, e.g. "376263874141234936,513410357520258150"
def mesh_chunk_ids_shuffled(chunk_ids_string):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_periodically_write_to_cloud})
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)

    chunk_ids = np.uint64(chunk_ids_string.split(','))
    np.random.shuffle(chunk_ids)

    for chunk_id in chunk_ids:
        current_app.test_q.enqueue(
            meshgen.chunk_initial_sharded_stitching_task,
            job_timeout='30m',
            args=(
                'minnie3_v0',
                chunk_id,
                2,
                'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v0',
                'graphene_meshes'
            ))
                
    return 'Queued'


@ingest_cli.command('listen')
@click.argument('channel', default='mesh_frag_test_channel')
def mesh_listen(channel):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{channel: handlerino_periodically_write_to_cloud})
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)
                
    return 'Queued'


def init_mesh_cmds(app):
    app.cli.add_command(ingest_cli)
