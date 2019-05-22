import os
import time
from flask import current_app
from pychunkedgraph.utils.general import redis_job

# not a good solution
# figure out how to use app context

REDIS_HOST = os.environ['REDIS_SERVICE_HOST']
REDIS_PORT = os.environ['REDIS_SERVICE_PORT']
REDIS_PASSWORD = os.environ['REDIS_PASSWORD']
REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'

@redis_job(REDIS_URL, 'test')
def independent_task(chunk_id, chunk_size):
    print(f' Working on chunk id: {chunk_id}, size {chunk_size}')
    i = 0
    while i < chunk_size:
        i += 1
    print('Done')
    return chunk_id