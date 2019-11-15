from collections import namedtuple

from ..utils.redis import REDIS_URL


_cluster_ingest_config_fields = (
    "redis_url",
    "atomic_q_name",
    "atomic_q_limit",  # these limits ensure the queue won't use too much memory
    "atomic_q_interval",  # sleep interval before queuing the next job when limit is reached
    "parents_q_name",
    "parents_q_limit",
    "parents_q_interval",
)
_cluster_ingest_defaults = (REDIS_URL, "atomic", 100000, 60, "parents", 25000, 120)
ClusterIngestConfig = namedtuple(
    "ClusterIngestConfig",
    _cluster_ingest_config_fields,
    defaults=_cluster_ingest_defaults,
)


_ingestconfig_fields = (
    "build_graph",
    "cluster",  # run ingest on a single machine (simple) or on a cluster
    "agglomeration",
    "watershed",
    "data_version",
    "use_raw_edges",
    "use_raw_components",
)
_ingestconfig_defaults = (True, None, None, None, None, False, False)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
