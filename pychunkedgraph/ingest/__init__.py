from collections import namedtuple

from ..utils.redis import REDIS_URL


_cluster_config_fields = (
    "redis_url",
    "atomic_q_name",
    "atomic_q_limit",  # these limits ensure the queue won't use too much memory
    "atomic_q_interval",  # sleep interval before queuing the next job when limit is reached
    "parents_q_name",
    "parents_q_limit",
    "parents_q_interval",
)
_cluster_defaults = (REDIS_URL, "atomic", 100000, 60, "parents", 25000, 120)
ClusterConfig = namedtuple(
    "ClusterConfig", _cluster_config_fields, defaults=_cluster_defaults
)


_ingestconfig_fields = (
    "build_graph",
    "overwrite",  # overwrites existing graph
    "cluster",  # run ingest on a single machine (simple) or on a cluster
    "agglomeration",
    "watershed",
    "use_raw_edges",
    "use_raw_components",
    "data_version",
)
_ingestconfig_defaults = (True, False, None, None, None, False, False, None)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
