import os
import time
from flask import current_app
from pychunkedgraph.utils.general import redis_job

# not a good solution
# figure out how to use app context

REDIS_HOST = os.environ.get('REDIS_SERVICE_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_SERVICE_PORT', '6379')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', 'dev')
REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'

@redis_job(REDIS_URL, 'test-channel')
def independent_task(chunk_id, chunk_size):
    print(f' Working on chunk id: {chunk_id}, size {chunk_size}')
    i = 0
    while i < chunk_size:
        i += 1
    print('Done')
    return chunk_id