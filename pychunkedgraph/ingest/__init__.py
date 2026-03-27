from collections import namedtuple

from pychunkedgraph import configure_logging, NOTICE

configure_logging(level=NOTICE)

_ingestconfig_fields = (
    "AGGLOMERATION",
    "WATERSHED",
    "USE_RAW_EDGES",
    "USE_RAW_COMPONENTS",
    "TEST_RUN",
)
_ingestconfig_defaults = (None, None, False, False, False)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
