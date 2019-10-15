from typing import Sequence
from typing import Dict
from typing import List

import numpy as np
from cloudvolume import CloudVolume


from .chunkedgraph_utils import get_voxels_boundary
from .chunkedgraph_utils import get_chunks_boundary
from .chunkedgraph_utils import compute_bitmasks
from .definitions.config import DataSource
from .definitions.config import GraphConfig
from .definitions.config import BigTableConfig
from .chunkedgraph_utils import log_n


class ChunkedGraphMeta:
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

        self._bitmasks = None

    @property
    def data_source(self):
        return self._data_source

    @property
    def graph_config(self):
        return self._graph_config

    @property
    def bigtable_config(self):
        return self._bigtable_config

    @property
    def layer_count(self) -> int:
        if self._layer_count:
            return self._layer_count
        bbox = np.array(self._ws_cv.bounds.to_list()).reshape(2, 3)
        n_chunks = ((bbox[1] - bbox[0]) / self._graph_config.chunk_size).astype(np.int)
        self._layer_count = (
            int(np.ceil(log_n(np.max(n_chunks), self._graph_config.fanout))) + 2
        )
        return self._layer_count

    @property
    def layer_chunk_bounds(self) -> Dict:
        """number of chunks in each dimension in each layer {layer: [x,y,z]}"""
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

    @property
    def layer_chunk_counts(self) -> List:
        """number of chunks in each layer"""
        counts = []
        for layer in range(2, self.layer_count):
            counts.append(np.prod(self.layer_chunk_bounds[layer]))
        return counts

    @property
    def edge_dtype(self):
        if self.data_source.data_version == 4:
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
        elif self.data_source.data_version == 3:
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
        elif self.data_source.data_version == 2:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff", np.float32),
                ("area", np.uint64),
            ]
        else:
            raise Exception()
        return dtype

    def is_out_of_bounds(self, chunk_coordinate):
        if not self._bitmasks:
            self._bitmasks = compute_bitmasks(
                self.layer_count,
                self.graph_config.fanout,
                s_bits_atomic_layer=self.graph_config.s_bits_atomic_layer,
            )
        return np.any(chunk_coordinate < 0) or np.any(
            chunk_coordinate > 2 ** self._bitmasks[1]
        )
