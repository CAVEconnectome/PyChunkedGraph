import json
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Tuple
from typing import Sequence
from collections import namedtuple

import numpy as np
from cloudvolume import CloudVolume

from .utils.generic import compute_bitmasks
from .chunks.utils import get_chunks_boundary
from ..utils.redis import keys as r_keys
from ..utils.redis import get_rq_queue
from ..utils.redis import get_redis_connection


_datasource_fields = ("EDGES", "COMPONENTS", "WATERSHED", "DATA_VERSION", "CV_MIP")
_datasource_defaults = (None, None, None, None, 0)
DataSource = namedtuple(
    "DataSource",
    _datasource_fields,
    defaults=_datasource_defaults,
)


_graphconfig_fields = (
    "ID",  # ID_PREFIX and ID are together used when creating the graph
    "ID_PREFIX",
    "CHUNK_SIZE",
    "FANOUT",
    "LAYER_ID_BITS",  # number of bits reserved for layer id
    "SPATIAL_BITS",  # number of bits used for each spatial in id creation on level 1
    "OVERWRITE",  # overwrites existing graph
    "ROOT_LOCK_EXPIRY",
    "ROOT_COUNTERS",
)
_graphconfig_defaults = (
    None,
    "",
    None,
    2,
    8,
    10,
    False,
    timedelta(minutes=3, seconds=0),
    8,
)
GraphConfig = namedtuple(
    "GraphConfig", _graphconfig_fields, defaults=_graphconfig_defaults
)


class ChunkedGraphMeta:
    def __init__(
        self, graph_config: GraphConfig, data_source: DataSource, custom_data: Dict = {}
    ):
        """
        `custom_data`: stores arbitray key value information, for flexibility.
        """
        self._graph_config = graph_config
        self._data_source = data_source
        self._custom_data = custom_data

        self._ws_cv = None
        self._layer_bounds_d = None
        self._layer_count = None
        self._bitmasks = None

    @property
    def graph_config(self):
        return self._graph_config

    @property
    def data_source(self):
        return self._data_source

    @property
    def custom_data(self):
        return self._custom_data

    @property
    def ws_cv(self):
        if self._ws_cv:
            return self._ws_cv

        try:
            # try reading a cached info file for distributed workers
            # useful to avoid md5 errors on high gcs load
            cache_key = f"{self.graph_config.ID}:cv_info_cached"
            redis = get_redis_connection()
            cached_info = json.loads(redis.get(cache_key))
            self._ws_cv = CloudVolume(self._data_source.WATERSHED, info=cached_info)
        except Exception:
            self._ws_cv = CloudVolume(self._data_source.WATERSHED)
            try:
                redis.set(cache_key, json.dumps(self._ws_cv.info))
            except Exception:
                ...
        return self._ws_cv

    @property
    def resolution(self):
        return self.ws_cv.resolution  # pylint: disable=no-member

    @property
    def layer_count(self) -> int:
        from .utils.generic import log_n

        if self._layer_count:
            return self._layer_count
        bbox = np.array(self.ws_cv.bounds.to_list())  # pylint: disable=no-member
        bbox = bbox.reshape(2, 3)
        n_chunks = get_chunks_boundary(
            self.voxel_counts, np.array(self._graph_config.CHUNK_SIZE, dtype=int)
        )
        self._layer_count = (
            int(np.ceil(log_n(np.max(n_chunks), self._graph_config.FANOUT))) + 2
        )
        return self._layer_count

    @layer_count.setter
    def layer_count(self, count):
        self._layer_count = count
        self._bitmasks = compute_bitmasks(
            self._layer_count,
            s_bits_atomic_layer=self._graph_config.SPATIAL_BITS,
        )

    @property
    def cv(self):
        """Alias for watershed CV"""
        return self.ws_cv

    @property
    def bitmasks(self):
        if self._bitmasks:
            return self._bitmasks
        self._bitmasks = compute_bitmasks(
            self.layer_count,
            s_bits_atomic_layer=self._graph_config.SPATIAL_BITS,
        )
        return self._bitmasks

    @property
    def voxel_bounds(self):
        bounds = np.array(self.ws_cv.bounds.to_list())  # pylint: disable=no-member
        return bounds.reshape(2, -1).T

    @property
    def voxel_counts(self) -> Sequence[int]:
        """returns number of voxels in each dimension"""
        cv_bounds = np.array(self.ws_cv.bounds.to_list())  # pylint: disable=no-member
        cv_bounds = cv_bounds.reshape(2, -1).T
        voxel_counts = cv_bounds.copy()
        voxel_counts -= cv_bounds[:, 0:1]  # pylint: disable=unsubscriptable-object
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
            layer_bounds_d[layer] = np.ceil(layer_bounds).astype(int)
        self._layer_bounds_d = layer_bounds_d
        return self._layer_bounds_d

    @layer_chunk_bounds.setter
    def layer_chunk_bounds(self, layer_chunk_bounds_d):
        self._layer_bounds_d = layer_chunk_bounds_d

    @property
    def layer_chunk_counts(self) -> List:
        """number of chunks in each layer"""
        counts = []
        for layer in range(2, self.layer_count):
            counts.append(np.prod(self.layer_chunk_bounds[layer]))
        return counts + [1]

    @property
    def edge_dtype(self):
        if self.data_source.DATA_VERSION == 4:
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
        elif self.data_source.DATA_VERSION == 3:
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
        elif self.data_source.DATA_VERSION == 2:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff", np.float32),
                ("area", np.uint64),
            ]
        else:
            raise Exception()
        return dtype

    @property
    def READ_ONLY(self):
        return self.custom_data.get("READ_ONLY", False)

    @property
    def split_bounding_offset(self):
        return self.custom_data.get(
            "split_bounding_offset",
            (240, 240, 24),
        )

    @property
    def dataset_info(self) -> Dict:
        info = self.ws_cv.info  # pylint: disable=no-member

        info.update(
            {
                "chunks_start_at_voxel_offset": True,
                "data_dir": self.data_source.WATERSHED,
                "graph": {
                    "chunk_size": self.graph_config.CHUNK_SIZE,
                    "bounding_box": [2048, 2048, 512],
                    "n_bits_for_layer_id": self.graph_config.LAYER_ID_BITS,
                    "cv_mip": self.data_source.CV_MIP,
                    "n_layers": self.layer_count,
                    "spatial_bit_masks": self.bitmasks,
                },
            }
        )
        mesh_dir = self.custom_data.get("mesh", {}).get("dir", None)
        if mesh_dir is not None:
            info.update({"mesh": mesh_dir})
        return info

    def __getnewargs__(self):
        return (self.graph_config, self.data_source)

    def __getstate__(self):
        return {
            "graph_config": self.graph_config,
            "data_source": self.data_source,
            "custom_data": self.custom_data,
        }

    def __setstate__(self, state):
        self.__init__(
            state["graph_config"], state["data_source"], state.get("custom_data", {})
        )

    def __str__(self):
        from json import dumps

        meta_str = f"GRAPH_CONFIG\n{self.graph_config}\n"
        meta_str += f"\nDATA_SOURCE\n{self.data_source}\n"
        meta_str += f"\nCUSTOM_DATA\n{self.custom_data}\n"
        meta_str += f"\nBITMASKS\n{self.bitmasks}\n"
        meta_str += f"\nVOXEL_BOUNDS\n{self.voxel_bounds}\n"
        meta_str += f"\nVOXEL_COUNTS\n{self.voxel_counts}\n"
        meta_str += f"\nLAYER_CHUNK_BOUNDS\n{self.layer_chunk_bounds}\n"
        meta_str += f"\nLAYER_CHUNK_COUNTS\n{self.layer_chunk_counts}\n"
        meta_str += f"\nDATASET_INFO\n{dumps(self.dataset_info, indent=4)}\n"
        return meta_str

    def is_out_of_bounds(self, chunk_coordinate):
        return np.any(chunk_coordinate < 0) or np.any(
            chunk_coordinate > 2 ** self.bitmasks[1]
        )
