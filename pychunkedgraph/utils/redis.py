"""
redis helper funtions
"""

import os
from collections import namedtuple

import redis
from rq import Queue

REDIS_HOST = os.environ.get(
    "REDIS_SERVICE_HOST",
    os.environ.get("REDIS_HOST", "localhost"),
)
REDIS_PORT = os.environ.get(
    "REDIS_SERVICE_PORT",
    os.environ.get("REDIS_PORT", "6379"),
)
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
CONNECTION = redis.Redis.from_url(REDIS_URL, socket_timeout=60)

keys_fields = ("INGESTION_MANAGER", "JOB_TYPE")
keys_defaults = ("pcg:imanager", "pcg:job_type")
Keys = namedtuple("keys", keys_fields, defaults=keys_defaults)

keys = Keys()


def get_redis_connection(redis_url=REDIS_URL):
    if redis_url == REDIS_URL:
        return CONNECTION
    return redis.Redis.from_url(redis_url, socket_timeout=60)


def get_rq_queue(queue):
    return Queue(queue, connection=CONNECTION)
