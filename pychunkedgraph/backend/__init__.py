from typing import Sequence, Dict

import numpy as np
from cloudvolume import CloudVolume


from .chunkedgraph_utils import get_voxels_boundary
from .chunkedgraph_utils import get_chunks_boundary
from .definitions.config import DataSource
from .definitions.config import GraphConfig
from .definitions.config import BigTableConfig
from .chunkedgraph_utils import log_n


class GraphMeta:
    def __init__(
        self,
        data_source: DataSource,
        graph_config: GraphConfig,
        bigtable_config: BigTableConfig,
    ):
        self._data_source = data_source
        self._graph_config = graph_config
        self._bigtable_config = bigtable_config

        self._ws_cv = CloudVolume(data_source.watershed)
        self._layer_bounds_d = None
        self._layer_count = None

    @property
    def layer_count(self) -> int:
        if self._layer_count:
            return self._layer_count
        bbox = np.array(self._ws_cv.bounds.to_list()).reshape(2, 3)
        n_chunks = ((bbox[1] - bbox[0]) / self._graph_config.chunk_size).astype(np.int)
        n_layers = int(np.ceil(log_n(np.max(n_chunks), self._graph_config.fanout))) + 2
        return n_layers

    @property
    def layer_chunk_bounds(self) -> Dict:
        if self._layer_bounds_d:
            return self._layer_bounds_d

        voxels_boundary = get_voxels_boundary(self._ws_cv)
        chunks_boundary = get_chunks_boundary(
            voxels_boundary, self._graph_config.chunk_size
        )

        layer_bounds_d = {}
        for layer in range(2, self.layer_count):
            layer_bounds = chunks_boundary / (2 ** (layer - 2))
            layer_bounds_d[layer] = np.ceil(layer_bounds).astype(np.int)
        self._layer_bounds_d = layer_bounds_d
        return self._layer_bounds_d
