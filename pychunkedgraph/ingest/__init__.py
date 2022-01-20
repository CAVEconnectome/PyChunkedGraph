from collections import namedtuple

from ..utils.redis import REDIS_URL


_cluster_ingest_config_fields = (
    "REDIS_URL",
    "ATOMIC_Q_NAME",
    "ATOMIC_Q_LIMIT",
    "ATOMIC_Q_INTERVAL",
)
_cluster_ingest_defaults = (
    REDIS_URL,
    "atomic",
    100000,
    60,
    "parents",
    100000,
    120,
)
ClusterIngestConfig = namedtuple(
    "ClusterIngestConfig",
    _cluster_ingest_config_fields,
    defaults=_cluster_ingest_defaults,
)


_ingestconfig_fields = (
    "CLUSTER",  # cluster config
    "AGGLOMERATION",
    "WATERSHED",
    "USE_RAW_EDGES",
    "USE_RAW_COMPONENTS",
    "TEST_RUN",
)
_ingestconfig_defaults = (None, None, None, False, False, False)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
