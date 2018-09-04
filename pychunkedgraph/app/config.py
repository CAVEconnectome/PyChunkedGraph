import logging
import os


class BaseConfig(object):
    DEBUG = False
    TESTING = False
    HOME = os.path.expanduser("~")
    # TODO get this secret out of source control
    SECRET_KEY = '1d94e52c-1c89-4515-b87a-f48cf3cb7f0b'
    LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOGGING_LOCATION = HOME + '/pychg_log/bookshelf.log'
    LOGGING_LEVEL = logging.DEBUG
    # TODO what is this suppose to be by default?
    CHUNKGRAPH_TABLE_ID = "pinky40_fanout2_v7"

    if not os.path.exists(os.path.dirname(LOGGING_LOCATION)):
        os.makedirs(os.path.dirname(LOGGING_LOCATION))
