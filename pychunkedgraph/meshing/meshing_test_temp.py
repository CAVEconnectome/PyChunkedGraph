import time
import click

import redis

from flask import current_app
from flask.cli import AppGroup
from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
from pychunkedgraph.meshing import meshgen
import cloudvolume
import numpy as np

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
    if num_messages == 50:
        print('DONE')
        cv_path = 'gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200'
        with cloudvolume.Storage(cv_path) as storage:
            storage.put_file(
                    file_path='frag_test/frag_test_summary',
                    content=','.join(map(str, messages)),
                    compress=False,
                    cache_control='no-cache'
                )

def handlerino_print(*args, **kwargs):
    with open('output.txt', 'a') as f:
        f.write(str(args[0]['data']))


@ingest_cli.command('mesh_chunks')
@click.argument('layer', type=int)
@click.argument('x_start', type=int)
@click.argument('y_start', type=int)
@click.argument('z_start', type=int)
@click.argument('x_end', type=int)
@click.argument('y_end', type=int)
@click.argument('z_end', type=int)
def mesh_chunks(layer, x_start, y_start, z_start, x_end, y_end, z_end):
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
                        'dust_threshold': 100
                    })
                
    return 'Queued'

@ingest_cli.command('frag_test')
@click.argument('n', type=int)
@click.argument('layer', type=int)
def mesh_frag_test(n, layer):
    print(f'Queueing...')
    chunk_pubsub = current_app.redis.pubsub()
    chunk_pubsub.subscribe(**{'mesh_frag_test_channel': handlerino_print})
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
                'dust_threshold': 100,
            })
                
    return 'Queued'    


def init_mesh_cmds(app):
    app.cli.add_command(ingest_cli)