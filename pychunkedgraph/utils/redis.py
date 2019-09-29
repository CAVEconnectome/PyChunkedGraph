"""
generic helper funtions
"""

import os
import functools
from collections import namedtuple

import redis
from rq import Queue

# REDIS_SERVICE_HOST and REDIS_SERVICE_PORT are added by Kubernetes
REDIS_HOST = os.environ.get("REDIS_SERVICE_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_SERVICE_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

keys_fields = (
    "INGESTION_MANAGER",
    "PARENTS_HASH"
)
keys_defaults = (
    "pcg:imanager",
    "rq:enqueued:parents"
)
Keys = namedtuple(
    "keys",
    keys_fields,
    defaults=keys_defaults,
)

keys = Keys()


def get_redis_connection(redis_url=REDIS_URL):
    return redis.Redis.from_url(redis_url)


def redis_job(redis_url, redis_channel):
    """
    Decorator factory
    Returns a decorator that connects to a redis instance 
    and publish a message (return value of the function) when the job is done.
    """

    def redis_job_decorator(func):
        r = get_redis_connection()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            job_result = func(*args, **kwargs)
            if not job_result:
                job_result = str(job_result)
            r.publish(redis_channel, job_result)

        return wrapper

    return redis_job_decorator


def get_rq_queue(queue):
    connection = redis.Redis.from_url(REDIS_URL)
    return Queue(queue, connection=connection)
