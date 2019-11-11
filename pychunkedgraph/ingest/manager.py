import pickle

from cloudvolume import CloudVolume

from . import IngestConfig
from ..graph import ChunkedGraphMeta
from ..graph.chunkedgraph import ChunkedGraph
from ..utils.redis import keys as r_keys
from ..utils.redis import get_rq_queue
from ..utils.redis import get_redis_connection


class IngestionManager:
    def __init__(self, config: IngestConfig, chunkedgraph_meta: ChunkedGraphMeta):
        self._config = config
        self._chunkedgraph_meta = chunkedgraph_meta
        self._cg = None
        self._redis = None
        self._task_queues = {}

    @property
    def config(self):
        return self._config

    @property
    def chunkedgraph_meta(self):
        return self._chunkedgraph_meta

    @property
    def cg(self):
        if self._cg is None:
            # TODO simplify ChunkedGraph class
            self._cg = ChunkedGraph(
                table_id=self._chunkedgraph_meta.graph_config.graph_id,
                project_id=self._chunkedgraph_meta.bigtable_config.project_id,
                instance_id=self._chunkedgraph_meta.bigtable_config.instance_id,
                s_bits_atomic_layer=self._chunkedgraph_meta.graph_config.s_bits_atomic_layer,
                n_bits_root_counter=8,
                meta=self._chunkedgraph_meta,
            )
        return self._cg

    @property
    def redis(self):
        if self._redis is not None:
            return self._redis
        self._redis = get_redis_connection(self._config.redis_url)
        self._redis.set(r_keys.INGESTION_MANAGER, self.serialized(pickled=True))
        return self._redis

    def serialized(self, pickled=False):
        params = {"config": self._config, "chunkedgraph_meta": self._chunkedgraph_meta}
        if pickled:
            return pickle.dumps(params)
        return params

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

    def get_task_queue(self, q_name):
        if q_name in self._task_queues:
            return self._task_queues[q_name]
        self._task_queues[q_name] = get_rq_queue(q_name)
        return self._task_queues[q_name]
