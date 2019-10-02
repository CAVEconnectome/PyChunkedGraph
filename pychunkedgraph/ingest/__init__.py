from collections import namedtuple

from ..utils.redis import REDIS_URL

_ingestconfig_fields = ("build_graph", "task_q_name", "redis_url")
_ingestconfig_defaults = (True, None, REDIS_URL)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
