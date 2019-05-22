import time
from flask import current_app
from pychunkedgraph.utils.general import redis_job

@redis_job(current_app.config['REDIS_URL'], 'test')
def independent_task(chunk_id, chunk_size):
    print(f' Working on chunk id: {chunk_id}, size {chunk_size}')
    i = 0
    while i < chunk_size:
        i += 1
    print('Done')
    return chunk_id