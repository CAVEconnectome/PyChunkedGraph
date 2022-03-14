from os import environ
from logging import getLogger
from logging import basicConfig

LOGGER_NAME = "pcg"
INFO_PRIORITY = 25

__version__ = "1.19.0"


# create logger instance
logger = getLogger(LOGGER_NAME)
log_level = int(environ.get("PCG_LOG_LEVEL", INFO_PRIORITY))

# Set the root logger to write to stdout.
basicConfig(
    level=log_level,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)
logger.setLevel(log_level)
pcg_logger = logger
