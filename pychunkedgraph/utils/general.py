import redis
import functools

def redis_job(redis_url, redis_channel):
    '''
    Decorator factory
    Returns a decorator that connects to a redis instance 
    and publish a message (return value of the function) when the job is done.
    '''
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