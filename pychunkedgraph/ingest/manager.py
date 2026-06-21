# pylint: disable=invalid-name, missing-docstring

import pickle

from . import IngestConfig
from ..graph.meta import ChunkedGraphMeta
from ..graph.chunkedgraph import ChunkedGraph
from ..utils.redis import keys as r_keys
from ..utils.redis import get_rq_queue
from ..utils.redis import get_redis_connection


class IngestionManager:
    def __init__(
        self,
        config: IngestConfig,
        chunkedgraph_meta: ChunkedGraphMeta,
        ocdbt_config: dict = None,
        _from_pickle: bool = False,
    ):
        self._config = config
        self._chunkedgraph_meta = chunkedgraph_meta
        self._cg = None
        self._redis = None
        self._task_queues = {}
        self._from_pickle = _from_pickle
        self.ocdbt_config = ocdbt_config or {}

        if not _from_pickle:
            # initiate redis and store serialized state
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
        if not self._from_pickle:
            self._redis.set(r_keys.INGESTION_MANAGER, self.serialized(pickled=True))
        return self._redis

    @property
    def ocdbt_seg(self) -> bool:
        return bool(self.ocdbt_config.get("enabled"))

    @property
    def ocdbt_populate_base(self) -> bool:
        return bool(self.ocdbt_config.get("populate_base"))

    @property
    def ocdbt_populate_layer(self) -> int:
        return int(self.ocdbt_config.get("populate_layer", 3))

    def is_ocdbt_populate_layer(self, layer: int) -> bool:
        """True iff OCDBT is enabled, base-populate is on, AND the given
        layer matches the configured populate layer. Single guard for any
        code that branches on 'should this layer touch OCDBT?'.
        """
        return (
            self.ocdbt_seg
            and self.ocdbt_populate_base
            and layer == self.ocdbt_populate_layer
        )

    def serialized(self, pickled=False):
        params = {
            "config": self._config,
            "chunkedgraph_meta": self._chunkedgraph_meta,
            "ocdbt_config": self.ocdbt_config,
        }
        if pickled:
            return pickle.dumps(params)
        return params

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info), _from_pickle=True)

    def get_task_queue(self, q_name):
        if q_name in self._task_queues:
            return self._task_queues[q_name]
        self._task_queues[q_name] = get_rq_queue(q_name)
        return self._task_queues[q_name]
