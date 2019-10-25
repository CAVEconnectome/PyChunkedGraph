from collections import namedtuple

_ingestconfig_fields = (
    "build_graph",
    "redis_url",
    "atomic_q_name",
    "atomic_q_limit",
    "atomic_q_interval",
    "parents_q_name",
    "parents_q_limit",
    "parents_q_interval",
)
_ingestconfig_defaults = (True, "", "atomic", 100000, 60, "parents", 25000, 120)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
