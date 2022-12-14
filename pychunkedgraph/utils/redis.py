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

keys_fields = ("INGESTION_MANAGER",)
keys_defaults = ("pcg:imanager",)
Keys = namedtuple("keys", keys_fields, defaults=keys_defaults)

keys = Keys()


def get_redis_connection(redis_url=REDIS_URL):
    return redis.Redis.from_url(redis_url)


def get_rq_queue(queue):
    connection = redis.Redis.from_url(REDIS_URL)
    return Queue(queue, connection=connection)


def get_sum(r, key):
    fc = r.smembers(key)
    fc = [x.decode() for x in fc]
    fc = [int(x if x != "0.0" else 0) for x in fc]
    return len(fc), sum(fc)


def get_sums():
    r = get_redis_connection()
    for k in r.keys():
        print(k, get_sum(r, k))
