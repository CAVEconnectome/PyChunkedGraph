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
from ..backend import GraphMeta
from ..backend.chunkedgraph_utils import compute_bitmasks
from ..backend.chunkedgraph import ChunkedGraph
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig


class IngestionManager(object):
    def __init__(
        self,
        config: IngestConfig,
        data_source: DataSource,
        graph_config: GraphConfig,
        bigtable_config: BigTableConfig,
    ):

        self._config = config
        self._data_source = data_source
        self._graph_config = graph_config
        self._bigtable_config = bigtable_config

        self._cg = None
        self._graph_meta = GraphMeta(data_source, graph_config, bigtable_config)
        self._ws_cv = CloudVolume(data_source.watershed)
        self._n_layers = None
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
    def graph_meta(self):
        return self._graph_meta

    @property
    def cg(self):
        if self._cg is None:
            self._cg = ChunkedGraph(
                self._graph_config.graph_id,
                self._bigtable_config.project_id,
                self._bigtable_config.instance_id,
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

    @property
    def edge_dtype(self):
        if self._data_source.data_version == 4:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff_x", np.float32),
                ("area_x", np.uint64),
                ("aff_y", np.float32),
                ("area_y", np.uint64),
                ("aff_z", np.float32),
                ("area_z", np.uint64),
            ]
        elif self._data_source.data_version == 3:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff_x", np.float64),
                ("area_x", np.uint64),
                ("aff_y", np.float64),
                ("area_y", np.uint64),
                ("aff_z", np.float64),
                ("area_z", np.uint64),
            ]
        elif self._data_source.data_version == 2:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff", np.float32),
                ("area", np.uint64),
            ]
        else:
            raise Exception()
        return dtype

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

    def get_task_queue(self, q_name):
        if q_name in self._task_queues:
            return self._task_queues[q_name]
        self._task_queues[q_name] = get_rq_queue(q_name)
        return self._task_queues[q_name]

    def get_serialized_info(self, pickled=False):
        info = {
            "config": self._config,
            "data_source": self._data_source,
            "graph_config": self._graph_config,
            "bigtable_config": self._bigtable_config,
        }
        if pickled:
            return pickle.dumps(info)
        return info

    def is_out_of_bounds(self, chunk_coordinate):
        if not self._bitmasks:
            self._bitmasks = compute_bitmasks(
                self._n_layers,
                self._graph_config.fanout,
                s_bits_atomic_layer=self._graph_config.s_bits_atomic_layer,
            )
        return np.any(chunk_coordinate < 0) or np.any(
            chunk_coordinate > 2 ** self._bitmasks[1]
        )

