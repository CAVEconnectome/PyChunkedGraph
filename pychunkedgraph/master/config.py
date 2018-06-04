import logging
import os

HOME = os.path.expanduser("~")


class BaseConfig(object):
    DEBUG = False
    TESTING = False
    # sqlite :memory: identifier is the default if no filepath is present
    SQLALCHEMY_DATABASE_URI = 'sqlite3://'
    SECRET_KEY = '1d94e52c-1c89-4515-b87a-f48cf3cb7f0b'
    LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOGGING_LOCATION = HOME + '/pychg_log/bookshelf.log'
    LOGGING_LEVEL = logging.DEBUG

    if not os.path.exists(os.path.dirname(LOGGING_LOCATION)):
        os.makedirs(os.path.dirname(LOGGING_LOCATION))