import time
import click
import redis

from flask import current_app
from flask.cli import AppGroup
from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
from pychunkedgraph.meshing import meshgen
import cloudvolume
import numpy as np
from datetime import datetime

ingest_cli = AppGroup('mesh')

num_messages = 0
messages = []
def handlerino_write_to_cloud(*args, **kwargs):
    global num_messages
    num_messages = num_messages + 1
    print(num_messages, args[0]['data'])
    messages.append(args[0]['data'])
    with open('output.txt', 'a') as f:
        f.write(str(args[0]['data']) + '\n')
    if num_messages == 1000:
        print('DONE')
        cv_path = 'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
        with cloudvolume.Storage(cv_path) as storage:
            storage.put_file(
                    file_path='frag_test/frag_test_summary_no_dust_threshold',
                    content=','.join(map(str, messages)),
                    compress=False,
                    cache_control='no-cache'
                )


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
    print(num_messages, args[0]['data'])
    messages.append(args[0]['data'])
    with open('output.txt', 'a') as f:
        f.write(str(args[0]['data']) + '\n')
    if num_messages % 1000 == 0:
        print('Writing result data to cloud')
        cv_path = 'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
        filename = f'{datetime.now()}_meshes_{num_messages}'
        with cloudvolume.Storage(cv_path) as storage:
            storage.put_file(
                    file_path=f'meshing_run_data/{filename}',
                    content=','.join(map(str, messages)),
                    compress=False,
                    cache_control='no-cache'
                )


@ingest_cli.command('mesh_chunks')
@click.argument('layer', type=int)
@click.argument('x_start', type=int)
@click.argument('y_start', type=int)
@click.argument('z_start', type=int)
@click.argument('x_end', type=int)
@click.argument('y_end', type=int)
@click.argument('z_end', type=int)
@click.argument('fragment_batch_size', type=int)
def mesh_chunks(layer, x_start, y_start, z_start, x_end, y_end, z_end, fragment_batch_size):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_print})
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)

    cg = ChunkedGraph('fly_v31')
    for x in range(x_start,x_end):
        for y in range(y_start, y_end):
            for z in range(z_start, z_end):
                chunk_id = cg.get_chunk_id(None, layer, x, y, z)
                current_app.test_q.enqueue(
                    meshgen.chunk_mesh_task_new_remapping,
                    job_timeout='20m',
                    args=(
                        cg.get_serialized_info(), 
                        chunk_id,
                        'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
                    ),
                    kwargs={
                        # 'cv_mesh_dir': 'mesh_testing/initial_testrun_meshes',
                        'mip': 1,
                        'max_err': 320,
                        'fragment_batch_size': fragment_batch_size
                        # 'dust_threshold': 100
                    })
                
    return 'Queued'

@ingest_cli.command('mesh_chunks_shuffled')
@click.argument('layer', type=int)
@click.argument('x_start', type=int)
@click.argument('y_start', type=int)
@click.argument('z_start', type=int)
@click.argument('x_end', type=int)
@click.argument('y_end', type=int)
@click.argument('z_end', type=int)
@click.argument('fragment_batch_size', type=int)
def mesh_chunks_shuffled(layer, x_start, y_start, z_start, x_end, y_end, z_end, fragment_batch_size):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_periodically_write_to_cloud})

    cg = ChunkedGraph('fly_v31')
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
            meshgen.chunk_mesh_task_new_remapping,
            job_timeout='300m',
            args=(
                cg.get_serialized_info(), 
                chunk_id,
                'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
            ),
            kwargs={
                # 'cv_mesh_dir': 'mesh_testing/initial_testrun_meshes',
                'mip': 1,
                'max_err': 320,
                'fragment_batch_size': fragment_batch_size
                # 'dust_threshold': 100
            })
    
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
    cg = ChunkedGraph('fly_v31')
    chunk_ids = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            chunk_ids.append(np.uint64(line))
            line = f.readline()
    for chunk_id in chunk_ids:
        current_app.test_q.enqueue(
            meshgen.chunk_mesh_task_new_remapping,
            job_timeout='20m',
            args=(
                cg.get_serialized_info(), 
                chunk_id,
                'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
            ),
            kwargs={
                # 'cv_mesh_dir': 'mesh_testing/initial_testrun_meshes',
                'mip': 1,
                'max_err': 320
                # 'dust_threshold': 100
            })
                
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
@click.argument('fragment_batch_size', type=int)
def mesh_chunks_exclude_file(layer, x_start, y_start, z_start, x_end, y_end, z_end, filename, fragment_batch_size):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_periodically_write_to_cloud})
    cg = ChunkedGraph('fly_v31')
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
            meshgen.chunk_mesh_task_new_remapping,
            job_timeout='180m',
            args=(
                cg.get_serialized_info(), 
                chunk_id,
                'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
            ),
            kwargs={
                # 'cv_mesh_dir': 'mesh_testing/initial_testrun_meshes',
                'mip': 1,
                'max_err': 320,
                'fragment_batch_size': fragment_batch_size
                # 'dust_threshold': 100
            })

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

    cg = ChunkedGraph('fly_v31')
    
    chunk_ids = np.uint64(chunk_ids_string.split(','))
    np.random.shuffle(chunk_ids)

    for chunk_id in chunk_ids:
        current_app.test_q.enqueue(
            meshgen.chunk_mesh_task_new_remapping,
            job_timeout='20m',
            args=(
                cg.get_serialized_info(),
                chunk_id,
                'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
            ),
            kwargs={
                # 'cv_mesh_dir': 'mesh_testing/initial_testrun_meshes',
                'mip': 1,
                'max_err': 320
                # 'dust_threshold': 100
            })
                
    return 'Queued'


@ingest_cli.command('frag_test')
@click.argument('n', type=int)
@click.argument('layer', type=int)
def mesh_frag_test(n, layer):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_write_to_cloud})
    thread = chunk_pubsub.run_in_thread(sleep_time=0.1)

    cg = ChunkedGraph('fly_v31')
    new_info = cg.cv.info

    dataset_size = np.array(new_info['scales'][0]['size'])
    dim_in_chunks = np.ceil(dataset_size / new_info['graph']['chunk_size'])
    num_chunks = np.prod(dim_in_chunks, dtype=np.int32)
    rand_chunks = np.random.choice(num_chunks, n, replace=False)
    for chunk in rand_chunks:
        x_span = dim_in_chunks[1] * dim_in_chunks[2]
        x = np.floor(chunk / x_span)
        rem = chunk - (x * x_span)
        y = np.floor(rem / dim_in_chunks[2])
        rem = rem - (y * dim_in_chunks[2])
        z = rem
        chunk_id = cg.get_chunk_id(None, layer, np.int32(x), np.int32(y), np.int32(z))
        current_app.test_q.enqueue(
            meshgen.chunk_mesh_task_new_remapping,
            job_timeout='60m',
            args=(
                cg.get_serialized_info(), 
                chunk_id,
                'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
            ),
            kwargs={
                # 'cv_mesh_dir': 'mesh_testing/initial_testrun_meshes',
                'mip': 1,
                'max_err': 320,
                'return_frag_count': True
            })
                
    return 'Queued'    


def init_mesh_cmds(app):
    app.cli.add_command(ingest_cli)
