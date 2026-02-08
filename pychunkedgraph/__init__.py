__version__ = "3.1.7"

import sys
import warnings
import logging as stdlib_logging  # Use alias to avoid conflict with pychunkedgraph.logging

# Suppress annoying warning from python_jsonschema_objects dependency
warnings.filterwarnings(
    "ignore", message="Schema id not specified", module="python_jsonschema_objects"
)

# Export logging levels for convenience
DEBUG = stdlib_logging.DEBUG
INFO = stdlib_logging.INFO
WARNING = stdlib_logging.WARNING
ERROR = stdlib_logging.ERROR

# Set up library-level logger with NullHandler (Python logging best practice)
stdlib_logging.getLogger(__name__).addHandler(stdlib_logging.NullHandler())


def configure_logging(level=stdlib_logging.INFO, format_str=None, stream=None):
    """
    Configure logging for pychunkedgraph. Call this to enable log output.

    Works in Jupyter notebooks and scripts.

    Args:
        level: Logging level (default: INFO). Use pychunkedgraph.DEBUG, .INFO, .WARNING, .ERROR
        format_str: Custom format string (optional)
        stream: Output stream (default: sys.stdout for Jupyter compatibility)

    Example:
        import pychunkedgraph
        pychunkedgraph.configure_logging()  # Enable INFO level logging
        pychunkedgraph.configure_logging(pychunkedgraph.DEBUG)  # Enable DEBUG level
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if stream is None:
        stream = sys.stdout

    # Get root logger for pychunkedgraph
    logger = stdlib_logging.getLogger(__name__)
    logger.setLevel(level)

    # Remove existing handlers and add fresh StreamHandler
    # This allows reconfiguring with different levels/formats
    for h in logger.handlers[:]:
        if isinstance(h, stdlib_logging.StreamHandler) and not isinstance(
            h, stdlib_logging.NullHandler
        ):
            logger.removeHandler(h)

    handler = stdlib_logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(stdlib_logging.Formatter(format_str))
    logger.addHandler(handler)

    return logger


configure_logging()
