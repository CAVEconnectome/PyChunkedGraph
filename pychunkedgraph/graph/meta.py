from datetime import timedelta
from typing import Sequence
from typing import Dict
from typing import List
from collections import namedtuple

import numpy as np
from cloudvolume import CloudVolume

from .utils.generic import compute_bitmasks
from .utils.generic import log_n
from .chunks.utils import get_chunks_boundary


_datasource_fields = ("EDGES", "COMPONENTS", "WATERSHED", "CV_MIP")
_datasource_defaults = (None, None, None, 0)
DataSource = namedtuple(
    "DataSource", _datasource_fields, defaults=_datasource_defaults,
)


_bigtableconfig_fields = (
    "PROJECT",
    "INSTANCE",
    "ADMIN",
    "READ_ONLY",
)
_bigtableconfig_defaults = (
    "neuromancer-seung-import",
    "pychunkedgraph",
    "",
    True,
    False,
)
BigTableConfig = namedtuple(
    "BigTableConfig", _bigtableconfig_fields, defaults=_bigtableconfig_defaults
)


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = ("bigtable", BigTableConfig())
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)


_graphconfig_fields = (
    "ID",
    "ID_PREFIX",
    "CHUNK_SIZE",
    "FANOUT",
    "LAYER_ID_BITS",  # number of bits reserved for layer id
    "SPATIAL_BITS",  # number of bits used for each spatial in id creation on level 1
    "OVERWRITE",  # overwrites existing graph
    "ROOT_LOCK_EXPIRY",
)
_graphconfig_defaults = (
    None,
    None,
    None,
    2,
    8,
    10,
    False,
    timedelta(minutes=3, seconds=0),
)
GraphConfig = namedtuple(
    "GraphConfig", _graphconfig_fields, defaults=_graphconfig_defaults
)


class ChunkedGraphMeta:
    def __init__(
        self, graph_config: GraphConfig, data_source: DataSource,
    ):
        self._graph_config = graph_config
        self._data_source = data_source

        self._ws_cv = CloudVolume(data_source.WATERSHED)
        self._layer_bounds_d = None
        self._layer_count = None

        self._bitmasks = compute_bitmasks(
            self.layer_count, s_bits_atomic_layer=self._graph_config.SPATIAL_BITS,
        )

    @property
    def data_source(self):
        return self._data_source

    @property
    def graph_config(self):
        return self._graph_config

    @property
    def layer_count(self) -> int:
        if self._layer_count:
            return self._layer_count
        bbox = np.array(self._ws_cv.bounds.to_list()).reshape(2, 3)
        n_chunks = (
            (bbox[1] - bbox[0]) / np.array(self._graph_config.CHUNK_SIZE, dtype=int)
        ).astype(np.int)
        self._layer_count = (
            int(np.ceil(log_n(np.max(n_chunks), self._graph_config.FANOUT))) + 2
        )
        return self._layer_count

    @property
    def cv(self):
        return self._ws_cv

    @property
    def bitmasks(self):
        return self._bitmasks

    @property
    def voxel_bounds(self):
        return np.array(self._ws_cv.bounds.to_list()).reshape(2, -1).T

    @property
    def voxel_counts(self) -> Sequence[int]:
        """returns number of voxels in each dimension"""
        cv_bounds = np.array(self._ws_cv.bounds.to_list()).reshape(2, -1).T
        voxel_counts = cv_bounds.copy()
        voxel_counts -= cv_bounds[:, 0:1]
        voxel_counts = voxel_counts[:, 1]
        return voxel_counts

    @property
    def layer_chunk_bounds(self) -> Dict:
        """number of chunks in each dimension in each layer {layer: [x,y,z]}"""
        if self._layer_bounds_d:
            return self._layer_bounds_d

        chunks_boundary = get_chunks_boundary(
            self.voxel_counts, np.array(self._graph_config.CHUNK_SIZE, dtype=int)
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

    def __getnewargs__(self):
        return (self.graph_config, self.data_source)

    def __getstate__(self):
        return {"graph_config": self.graph_config, "data_source": self.data_source}

    def __setstate__(self, state):
        self._graph_config = state["graph_config"]
        self._data_source = state["data_source"]

    def is_out_of_bounds(self, chunk_coordinate):
        return np.any(chunk_coordinate < 0) or np.any(
            chunk_coordinate > 2 ** self.bitmasks[1]
        )
