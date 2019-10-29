from collections import namedtuple

from ..utils.redis import REDIS_URL

_ingestconfig_fields = (
    "build_graph",
    "cluster", # use workers in a cluster or run on single machine
    "redis_url",
    "atomic_q_name",
    "atomic_q_limit", # these limits ensure the queue won't use too much memory
    "atomic_q_interval", # sleep interval before queuing the next job when limit is reached
    "parents_q_name",
    "parents_q_limit",
    "parents_q_interval",
)
_ingestconfig_defaults = (True, False, REDIS_URL, "atomic", 100000, 60, "parents", 25000, 120)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
