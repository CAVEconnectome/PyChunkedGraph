"""
generic helper funtions
"""

import numpy as np
import redis
import functools


REDIS_HOST = os.environ.get("REDIS_SERVICE_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_SERVICE_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


def redis_job(redis_url, redis_channel):
    """
    Decorator factory
    Returns a decorator that connects to a redis instance 
    and publish a message (return value of the function) when the job is done.
    """

    def redis_job_decorator(func):
        r = redis.Redis.from_url(redis_url)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            job_result = func(*args, **kwargs)
            if not job_result:
                job_result = str(job_result)
            r.publish(redis_channel, job_result)

        return wrapper

    return redis_job_decorator


def reverse_dictionary(dictionary):
    """
    given a dictionary - {key1 : [item1, item2 ...], key2 : [ite3, item4 ...]}
    return {item1: key1, item2: key1, item3: key2, item4: key2 }
    """
    keys = []
    vals = []
    for key, values in dictionary.items():
        keys.append([key] * len(values))
        vals.append(values)
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)

    return {k: v for k, v in zip(vals, keys)}
