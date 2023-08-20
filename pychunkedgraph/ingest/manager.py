# pylint: disable=invalid-name, missing-docstring

import pickle

from . import IngestConfig
from ..graph.meta import ChunkedGraphMeta
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

        # initiate redis and cache info
        self.redis  # pylint: disable=pointless-statement

    @property
    def config(self):
        return self._config

    @property
    def cg_meta(self):
        return self._chunkedgraph_meta

    @property
    def cg(self):
        if self._cg is None:
            self._cg = ChunkedGraph(meta=self.cg_meta)
        return self._cg

    @property
    def redis(self):
        if self._redis is not None:
            return self._redis
        self._redis = get_redis_connection()
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
