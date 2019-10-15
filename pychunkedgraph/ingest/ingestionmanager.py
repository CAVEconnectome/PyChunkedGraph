import itertools
import numpy as np
import pickle
from typing import Dict
from collections import defaultdict

from cloudvolume import CloudVolume

from . import IngestConfig
from .ingestion_utils import get_layer_count
from ..utils.redis import keys as r_keys
from ..utils.redis import get_redis_connection
from ..utils.redis import get_rq_queue
from ..backend import ChunkedGraphMeta
from ..backend.chunkedgraph import ChunkedGraph
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig


class IngestionManager(object):
    def __init__(self, config: IngestConfig, chunkedgraph_meta: ChunkedGraphMeta):

        self._config = config

        self._cg = None
        self._chunkedgraph_meta = chunkedgraph_meta
        self._ws_cv = CloudVolume(chunkedgraph_meta.data_source.watershed)
        self._chunk_coords = None
        self._layer_bounds_d = None

        self._task_queues = {}

        self._bitmasks = None
        self._bounds = None
        self._redis = None

    @property
    def config(self):
        return self._config

    @property
    def chunkedgraph_meta(self):
        return self._chunkedgraph_meta

    @property
    def cg(self):
        if self._cg is None:
            self._cg = ChunkedGraph(
                self._chunkedgraph_meta.graph_config.graph_id,
                self._chunkedgraph_meta.bigtable_config.project_id,
                self._chunkedgraph_meta.bigtable_config.instance_id,
                meta=self._chunkedgraph_meta,
            )
        return self._cg

    @property
    def redis(self):
        if self._redis:
            return self._redis
        self._redis = get_redis_connection(self._config.redis_url)
        self._redis.set(
            r_keys.INGESTION_MANAGER, self.get_serialized_info(pickled=True)
        )
        return self._redis

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

    def get_task_queue(self, q_name):
        if q_name in self._task_queues:
            return self._task_queues[q_name]
        self._task_queues[q_name] = get_rq_queue(q_name)
        return self._task_queues[q_name]

    def get_serialized_info(self, pickled=False):
        info = {"config": self._config, "chunkedgraph_meta": self._chunkedgraph_meta}
        if pickled:
            return pickle.dumps(info)
        return info

