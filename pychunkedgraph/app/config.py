import logging
import os


class BaseConfig(object):
    DEBUG = False
    TESTING = False
    HOME = os.path.expanduser("~")
    # TODO get this secret out of source control
    SECRET_KEY = '1d94e52c-1c89-4515-b87a-f48cf3cb7f0b'

    LOGGING_FORMAT = '{"source":"%(name)s","time":"%(asctime)s","severity":"%(levelname)s","message":"%(message)s"}'
    LOGGING_DATEFORMAT = '%Y-%m-%dT%H:%M:%S.0Z'
    LOGGING_LEVEL = logging.DEBUG

    CHUNKGRAPH_INSTANCE_ID = "pychunkedgraph"

    # TODO what is this suppose to be by default?
    CHUNKGRAPH_TABLE_ID = "pinky100_sv16"
    # CHUNKGRAPH_TABLE_ID = "pinky100_benchmark_v92"

    REDIS_URL = 'redis://localhost:6379/0'


class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    DEBUG = True


class DockerDeploymentConfig(BaseConfig):
    """Deployment configuration."""
    REDIS_HOST = os.environ['REDIS_SERVICE_HOST']
    REDIS_PORT = os.environ['REDIS_SERVICE_PORT']
    REDIS_PASSWORD = os.environ['REDIS_PASSWORD']
    REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'


class TestingConfig(BaseConfig):
    """Testing configuration."""
    TESTING = True
    PRESERVE_CONTEXT_ON_EXCEPTION = False
