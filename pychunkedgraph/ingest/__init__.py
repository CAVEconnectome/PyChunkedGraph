import logging
from collections import namedtuple

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

_cluster_ingest_config_fields = (
    "ATOMIC_Q_NAME",
    "ATOMIC_Q_LIMIT",
    "ATOMIC_Q_INTERVAL",
)
_cluster_ingest_defaults = (
    "l2",
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
