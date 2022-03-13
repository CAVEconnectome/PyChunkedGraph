from os import environ
from logging import getLogger

LOGGER_NAME = "pcg"
INFO_PRIORITY = 25

__version__ = "2.1.1"


# create logger instance
logger = getLogger(LOGGER_NAME)
logger.setLevel(int(environ.get("PCG_LOG_LEVEL", INFO_PRIORITY)))
pcg_logger = logger
