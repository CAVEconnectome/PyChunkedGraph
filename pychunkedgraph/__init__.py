from os import environ
from logging import getLogger

LOGGER_NAME = "pcg"
INFO_PRIORITY = 25

__version__ = "1.19.0"


# create logger instance
logger = getLogger(LOGGER_NAME)
logger.setLevel(environ.get("PCG_LOG_LEVEL", INFO_PRIORITY))
