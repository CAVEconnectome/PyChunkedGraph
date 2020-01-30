import itertools
import numpy as np
import pickle
from typing import Dict
from collections import defaultdict

from cloudvolume import CloudVolume

from . import IngestConfig
from ..backend import ChunkedGraphMeta
from ..backend.chunkedgraph import ChunkedGraph


class IngestionManager(object):
    def __init__(self, config: IngestConfig, cg_meta: ChunkedGraphMeta):

        self._config = config

        self._cg = None
        self._chunkedgraph_meta = cg_meta
        self._ws_cv = CloudVolume(cg_meta.data_source.watershed)
        self._chunk_coords = None
        self._layer_bounds_d = None

        self._bitmasks = None
        self._bounds = None
        self._redis = None

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

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

    def get_serialized_info(self, pickled=False):
        info = {"config": self._config, "cg_meta": self._chunkedgraph_meta}
        if pickled:
            return pickle.dumps(info)
        return info

