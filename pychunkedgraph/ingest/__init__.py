from collections import namedtuple

from ..utils.redis import REDIS_URL


_cluster_ingest_config_fields = (
    "REDIS_URL",
    "FLUSH_REDIS",
    "ATOMIC_Q_NAME",
    "ATOMIC_Q_LIMIT",  # these limits ensure the queue won't use too much memory
    "ATOMIC_Q_INTERVAL",  # sleep interval before queuing the next job when limit is reached
    "PARENTS_Q_NAME",
    "PARENTS_Q_LIMIT",
    "PARENTS_Q_INTERVAL",
)
_cluster_ingest_defaults = (
    REDIS_URL,
    False,
    "atomic",
    100000,
    60,
    "parents",
    25000,
    120,
)
ClusterIngestConfig = namedtuple(
    "ClusterIngestConfig",
    _cluster_ingest_config_fields,
    defaults=_cluster_ingest_defaults,
)


_ingestconfig_fields = (
    "CLUSTER",  # run ingest on a single machine (simple) or on a cluster
    "AGGLOMERATION",
    "WATERSHED",
    "USE_RAW_EDGES",
    "USE_RAW_COMPONENTS",
)
_ingestconfig_defaults = (None, None, None, False, False)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
