import logging
import os


class BaseConfig(object):
    DEBUG = False
    TESTING = False
    HOME = os.path.expanduser("~")
    SECRET_KEY = 'CONFIGURE_THIS_VIA_INSTANCE_CONFIG'

    LOGGING_FORMAT = '{"source":"%(name)s","time":"%(asctime)s","severity":"%(levelname)s","message":"%(message)s"}'
    LOGGING_DATEFORMAT = '%Y-%m-%dT%H:%M:%S.0Z'
    LOGGING_LEVEL = logging.DEBUG

    CHUNKGRAPH_INSTANCE_ID = "pychunkedgraph"

    # TODO what is this suppose to be by default?
    CHUNKGRAPH_TABLE_ID = "pinky100_sv16"
    # CHUNKGRAPH_TABLE_ID = "pinky100_benchmark_v92"
