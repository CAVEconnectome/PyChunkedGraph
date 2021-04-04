import itertools
import numpy as np
import pickle
from typing import Dict
from collections import defaultdict

from cloudvolume import CloudVolume

from . import IngestConfig
from ..backend import ChunkedGraphMeta
from ..backend.chunkedgraph import ChunkedGraph


class IngestionManager:
    def __init__(self, config: IngestConfig, cg_meta: ChunkedGraphMeta):
        self._config = config
        self._cg = None
        self._chunkedgraph_meta = cg_meta
        self._redis = None
        self._task_queues = {}

    @property
    def config(self):
        return self._config

    @property
    def cg_meta(self):
        return self._chunkedgraph_meta

    @property
    def cg(self):
        if self._cg is None:
            self._cg = ChunkedGraph(
                self._chunkedgraph_meta.graph_config.graph_id,
                project_id=self._chunkedgraph_meta.bigtable_config.project_id,
                instance_id=self._chunkedgraph_meta.bigtable_config.instance_id,
                meta=self._chunkedgraph_meta,
            )
        return self._cg

    @property
    def redis(self):
        from .redis import get_redis_connection
        from .redis import keys as r_keys

        if self._redis is not None:
            return self._redis
        self._redis = get_redis_connection(self._config.CLUSTER.REDIS_URL)
        self._redis.set(
            r_keys.INGESTION_MANAGER, self.get_serialized_info(pickled=True)
        )
        return self._redis

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

    def get_serialized_info(self, pickled=False):
        info = {"config": self._config, "cg_meta": self._chunkedgraph_meta}
        if pickled:
            return pickle.dumps(info)
        return info

    def get_task_queue(self, q_name):
        from .redis import get_rq_queue

        if q_name in self._task_queues:
            return self._task_queues[q_name]
        self._task_queues[q_name] = get_rq_queue(q_name)
        return self._task_queues[q_name]
