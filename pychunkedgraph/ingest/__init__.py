from collections import namedtuple

from ..utils.redis import REDIS_URL

_ingestconfig_fields = ("build_graph", "flush_redis_db", "task_q_name", "redis_url")
_ingestconfig_defaults = (True, False, "test", REDIS_URL)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
