# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel, broad-exception-caught

import os

import redis

REDIS_HOST = os.environ.get("MANIFEST_CACHE_REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("MANIFEST_CACHE_REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("MANIFEST_CACHE_REDIS_PASSWORD", "")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


REDIS = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1)
try:
    REDIS.ping()
    REDIS = redis.Redis.from_url(REDIS_URL)
except Exception:
    REDIS = None
