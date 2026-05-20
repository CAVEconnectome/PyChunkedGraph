import json
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Sequence
from collections import namedtuple

import numpy as np
from cloudvolume import CloudVolume

from pychunkedgraph.graph.ocdbt import (
    OcdbtConfig,
    build_cg_ocdbt_spec,
    fork_exists,
    get_seg_source_and_destination_ocdbt,
    read_populate_meta,
)

from .utils.generic import compute_bitmasks
from .chunks.utils import get_chunks_boundary
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


def _redis_cached_json(key: str, loader):
    """Return JSON-decoded value at ``key`` in Redis, or call ``loader()`` and
    write the result through. Spares distributed workers from re-fetching the
    same GCS object on every CG instantiation. Silently bypasses Redis if it
    is unreachable; returns ``loader()`` directly in that case.
    """
    redis = None
    try:
        redis = get_redis_connection()
        cached = redis.get(key)
        if cached is not None:
            return json.loads(cached)
    except Exception:
        redis = None
    value = loader()
    if value is not None and redis is not None:
        try:
            redis.set(key, json.dumps(value))
        except Exception:
            ...
    return value


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
        # Multi-scale OCDBT handles + per-scale resolutions, populated lazily
        # from source's info JSON. ws_ocdbt returns scale 0 for backward
        # compatibility; ws_ocdbt_scales exposes the full pyramid.
        self._ws_ocdbt_scales = None
        self._ws_ocdbt_resolutions = None
        self._layer_bounds_d = None
        self._layer_count = None
        self._bitmasks = None
        self._ocdbt_seg = None
        self._ocdbt_config_cached = None

    @property
    def graph_id(self):
        assert self._graph_config.ID is not None, "graph_id required"
        return self._graph_config.ID_PREFIX + self._graph_config.ID

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
        ws = self._data_source.WATERSHED
        info = _redis_cached_json(
            f"ws_cv_info_cached:{ws}",
            lambda: CloudVolume(ws, progress=False).info,
        )
        self._ws_cv = CloudVolume(ws, info=info, progress=False)
        return self._ws_cv

    @property
    def ocdbt_config(self) -> OcdbtConfig:
        """Per-CG OCDBT settings with precedence info-file > custom_data > defaults.

        The watershed's ``<ws>/ocdbt/.populated/meta.json`` is the authoritative
        on-disk source for fields that affect the OCDBT format (compression,
        max_inline_value_bytes, populate_layer). custom_data fills per-CG
        fields (enabled, sv_split_threshold) and anything the info file
        doesn't pin. Both layers fall through to dataclass defaults.

        The info-file fetch goes through a Redis cache (same pattern as
        ``ws_cv``) so distributed workers don't re-read the same GCS
        object on every CG instantiation. Result is also cached in
        instance state after first access. Legacy ``custom_data["seg"]``
        shape is read when ``"ocdbt_config"`` is absent so pre-refactor
        CGs still open.
        """
        if self._ocdbt_config_cached is not None:
            return self._ocdbt_config_cached

        meta_d = self._custom_data.get("ocdbt_config")
        if meta_d is None:
            seg = self._custom_data.get("seg", {})
            meta_d = {
                "enabled": bool(seg.get("ocdbt", False)),
                "sv_split_threshold": int(seg.get("sv_split_threshold", 10)),
            }

        info_d = None
        ws = self._data_source.WATERSHED
        if ws:
            info_d = _redis_cached_json(
                f"ocdbt_info_cached:{ws}",
                lambda: read_populate_meta(ws),
            )

        self._ocdbt_config_cached = OcdbtConfig.resolve(meta_d, info_d)
        return self._ocdbt_config_cached

    @property
    def ocdbt_seg(self) -> bool:
        if self._ocdbt_seg is None:
            self._ocdbt_seg = self.ocdbt_config.enabled
        return self._ocdbt_seg

    @property
    def ws_ocdbt(self):
        """Base scale (MIP 0) handle. Backward-compatible single-handle access."""
        return self.ws_ocdbt_scales[0]

    @property
    def ws_ocdbt_scales(self):
        """List of TensorStore handles, one per MIP level. Lazily initialized.

        Opens the CG's delta OCDBT via the kvstack-layered fork spec — reads
        merge the shared base + this CG's edits, writes go to the delta.
        """
        assert self.ocdbt_seg, "make sure this pcg has segmentation in ocdbt format"
        if self._ws_ocdbt_scales is None:
            ws = self.data_source.WATERSHED
            assert fork_exists(ws, self.graph_id), (
                f"ocdbt fork missing at {ws}/ocdbt/{self.graph_id}/ — "
                "create it via fork_base_manifest or the seg_ocdbt notebook"
            )
            _, self._ws_ocdbt_scales, self._ws_ocdbt_resolutions = (
                get_seg_source_and_destination_ocdbt(
                    ws, self.graph_id, self.ocdbt_config
                )
            )
        return self._ws_ocdbt_scales

    @property
    def ws_ocdbt_resolutions(self):
        """Per-scale [x,y,z] resolutions (used to derive downsample factors)."""
        # Trigger lazy init via ws_ocdbt_scales — both are populated together.
        _ = self.ws_ocdbt_scales
        return self._ws_ocdbt_resolutions

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
    def sv_split_threshold(self) -> int:
        return self.ocdbt_config.sv_split_threshold

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
                    "ocdbt_seg": self.ocdbt_seg,
                    # Full kvstore spec a reader hands to tensorstore's
                    # `neuroglancer_precomputed` driver. Server owns the
                    # contract — paths, data prefixes, and OCDBT config
                    # (e.g. `max_inline_value_bytes`) are all resolved
                    # here, so readers don't duplicate configuration and
                    # future schema changes are picked up on re-fetch.
                    # Readers pass this verbatim as `kvstore`; add a
                    # `version` field for time-travel reads.
                    "ocdbt_kvstore_spec": (
                        build_cg_ocdbt_spec(
                            self._data_source.WATERSHED,
                            self.graph_id,
                            self.ocdbt_config,
                        )
                        if self.ocdbt_seg and self._graph_config.ID
                        else None
                    ),
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
