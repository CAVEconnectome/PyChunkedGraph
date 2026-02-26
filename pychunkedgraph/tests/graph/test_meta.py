"""Tests for pychunkedgraph.graph.meta"""

import pickle

import numpy as np
import pytest

from pychunkedgraph.graph.meta import ChunkedGraphMeta, GraphConfig, DataSource


class TestChunkedGraphMeta:
    def test_init(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        assert meta.graph_config is not None
        assert meta.data_source is not None

    def test_graph_config_properties(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        assert meta.graph_config.FANOUT == 2
        assert meta.graph_config.SPATIAL_BITS == 10
        assert meta.graph_config.LAYER_ID_BITS == 8

    def test_layer_count_setter(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        meta.layer_count = 6
        assert meta.layer_count == 6
        assert meta.bitmasks is not None
        assert 1 in meta.bitmasks

    def test_bitmasks(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        bm = meta.bitmasks
        assert isinstance(bm, dict)
        assert 1 in bm
        assert 2 in bm

    def test_read_only_default(self, gen_graph):
        graph = gen_graph(n_layers=4)
        assert graph.meta.READ_ONLY is False

    def test_is_out_of_bounds(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        assert meta.is_out_of_bounds(np.array([-1, 0, 0]))
        assert not meta.is_out_of_bounds(np.array([0, 0, 0]))

    def test_pickle_roundtrip(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        state = meta.__getstate__()
        new_meta = ChunkedGraphMeta.__new__(ChunkedGraphMeta)
        new_meta.__setstate__(state)
        assert new_meta.graph_config == meta.graph_config
        assert new_meta.data_source == meta.data_source

    def test_split_bounding_offset_default(self, gen_graph):
        graph = gen_graph(n_layers=4)
        assert graph.meta.split_bounding_offset == (240, 240, 24)


class TestEdgeDtype:
    def test_v2(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=2)
        meta = ChunkedGraphMeta(gc, ds)
        # Manually set bitmasks/layer_count to avoid CloudVolume access
        meta._layer_count = 4
        meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
        dt = meta.edge_dtype
        names = [d[0] for d in dt]
        assert "sv1" in names
        assert "aff" in names
        assert "area" in names

    def test_v3(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=3)
        meta = ChunkedGraphMeta(gc, ds)
        meta._layer_count = 4
        meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
        dt = meta.edge_dtype
        names = [d[0] for d in dt]
        assert "aff_x" in names
        assert "area_x" in names

    def test_v4(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)
        meta._layer_count = 4
        meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
        dt = meta.edge_dtype
        # v4 uses float32 for affinities
        for name, dtype in dt:
            if name.startswith("aff"):
                assert dtype == np.float32


class TestDataSourceDefaults:
    def test_defaults(self):
        ds = DataSource()
        assert ds.EDGES is None
        assert ds.COMPONENTS is None
        assert ds.WATERSHED is None
        assert ds.DATA_VERSION is None
        assert ds.CV_MIP == 0


class TestGraphConfigDefaults:
    def test_defaults(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        assert gc.FANOUT == 2
        assert gc.LAYER_ID_BITS == 8
        assert gc.SPATIAL_BITS == 10
        assert gc.OVERWRITE is False
        assert gc.ROOT_COUNTERS == 8


class TestResolutionProperty:
    def test_resolution_returns_numpy_array(self, gen_graph):
        """meta.resolution should delegate to ws_cv.resolution and return a numpy array."""
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        res = meta.resolution
        assert isinstance(res, np.ndarray)
        # The mock CloudVolumeMock sets resolution to [1, 1, 1]
        np.testing.assert_array_equal(res, np.array([1, 1, 1]))

    def test_resolution_dtype(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        res = meta.resolution
        # Should be numeric
        assert np.issubdtype(res.dtype, np.integer) or np.issubdtype(
            res.dtype, np.floating
        )


class TestLayerChunkCounts:
    def test_layer_chunk_counts_length(self, gen_graph):
        """layer_chunk_counts should return a list with one entry per layer from 2..layer_count-1, plus [1] for root."""
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        counts = meta.layer_chunk_counts
        # layers 2, 3 contribute entries, plus the trailing [1] for root
        # layer_count=4, so range(2, 4) => layers 2, 3 => 2 entries + [1] = 3
        assert isinstance(counts, list)
        assert (
            len(counts) == meta.layer_count - 2 + 1
        )  # -2 for range start, +1 for root
        # The last entry should always be 1 (root layer)
        assert counts[-1] == 1

    def test_layer_chunk_counts_values(self, gen_graph):
        """Each count should be the product of chunk bounds for that layer."""
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        counts = meta.layer_chunk_counts
        for i, layer in enumerate(range(2, meta.layer_count)):
            expected = np.prod(meta.layer_chunk_bounds[layer])
            assert counts[i] == expected

    def test_layer_chunk_counts_n_layers_5(self, gen_graph):
        graph = gen_graph(n_layers=5)
        meta = graph.meta
        counts = meta.layer_chunk_counts
        # n_layers=5 => layers 2,3,4 + root => 4 entries
        assert len(counts) == 4
        assert counts[-1] == 1


class TestLayerChunkBoundsSetter:
    def test_setter_overrides_bounds(self, gen_graph):
        """Setting layer_chunk_bounds should override the computed value."""
        graph = gen_graph(n_layers=4)
        meta = graph.meta

        custom_bounds = {
            2: np.array([10, 10, 10]),
            3: np.array([5, 5, 5]),
        }
        meta.layer_chunk_bounds = custom_bounds
        assert meta.layer_chunk_bounds is custom_bounds
        np.testing.assert_array_equal(
            meta.layer_chunk_bounds[2], np.array([10, 10, 10])
        )
        np.testing.assert_array_equal(meta.layer_chunk_bounds[3], np.array([5, 5, 5]))

    def test_setter_with_none_clears(self, gen_graph):
        """Setting layer_chunk_bounds to None should clear cached value (next access recomputes)."""
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        # Access to populate the cache
        _ = meta.layer_chunk_bounds
        meta.layer_chunk_bounds = None
        # After clearing, the internal _layer_bounds_d is None
        assert meta._layer_bounds_d is None


class TestEdgeDtypeUnknownVersion:
    """Test that an unknown DATA_VERSION raises Exception in edge_dtype."""

    def test_unknown_version_raises(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=999)
        meta = ChunkedGraphMeta(gc, ds)
        meta._layer_count = 4
        meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
        with pytest.raises(Exception):
            _ = meta.edge_dtype

    def test_none_version_raises(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=None)
        meta = ChunkedGraphMeta(gc, ds)
        meta._layer_count = 4
        meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
        with pytest.raises(Exception):
            _ = meta.edge_dtype

    def test_version_1_raises(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=1)
        meta = ChunkedGraphMeta(gc, ds)
        meta._layer_count = 4
        meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
        with pytest.raises(Exception):
            _ = meta.edge_dtype


class TestGetNewArgs:
    """Test __getnewargs__ returns (graph_config, data_source)."""

    def test_getnewargs_returns_tuple(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)
        result = meta.__getnewargs__()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_getnewargs_contains_config_and_source(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2)
        ds = DataSource(DATA_VERSION=3, CV_MIP=1)
        meta = ChunkedGraphMeta(gc, ds)
        result = meta.__getnewargs__()
        assert result[0] is gc
        assert result[1] is ds
        assert result[0].CHUNK_SIZE == [64, 64, 64]
        assert result[1].DATA_VERSION == 3

    def test_getnewargs_with_gen_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        result = meta.__getnewargs__()
        assert result[0] == meta.graph_config
        assert result[1] == meta.data_source


class TestCustomData:
    """Test custom_data including READ_ONLY=True and mesh dir."""

    def test_read_only_true(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"READ_ONLY": True})
        assert meta.READ_ONLY is True

    def test_read_only_false_explicit(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"READ_ONLY": False})
        assert meta.READ_ONLY is False

    def test_read_only_default_no_custom_data(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)
        assert meta.READ_ONLY is False

    def test_mesh_dir_in_custom_data(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(
            gc, ds, custom_data={"mesh": {"dir": "gs://bucket/mesh"}}
        )
        assert meta.custom_data["mesh"]["dir"] == "gs://bucket/mesh"

    def test_split_bounding_offset_custom(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(
            gc, ds, custom_data={"split_bounding_offset": (100, 100, 10)}
        )
        assert meta.split_bounding_offset == (100, 100, 10)

    def test_custom_data_preserved_through_getstate(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        custom = {"READ_ONLY": True, "mesh": {"dir": "gs://bucket/mesh"}}
        meta = ChunkedGraphMeta(gc, ds, custom_data=custom)
        state = meta.__getstate__()
        assert state["custom_data"] == custom


class TestCvAlias:
    """Test that cv property returns the same object as ws_cv."""

    def test_cv_returns_same_as_ws_cv(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        assert meta.cv is meta.ws_cv

    def test_cv_is_not_none(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        assert meta.cv is not None


class TestStr:
    """Test __str__ returns a non-empty string with expected sections."""

    def _add_info_to_mock(self, meta):
        """Add an info dict to the CloudVolumeMock so dataset_info works."""
        meta._ws_cv.info = {"scales": [{"resolution": [1, 1, 1]}]}

    def test_str_not_empty(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        result = str(meta)
        assert len(result) > 0

    def test_str_contains_sections(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        result = str(meta)
        assert "GRAPH_CONFIG" in result
        assert "DATA_SOURCE" in result
        assert "CUSTOM_DATA" in result
        assert "BITMASKS" in result
        assert "VOXEL_BOUNDS" in result
        assert "VOXEL_COUNTS" in result
        assert "LAYER_CHUNK_BOUNDS" in result
        assert "LAYER_CHUNK_COUNTS" in result
        assert "DATASET_INFO" in result

    def test_str_is_string_type(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        result = str(meta)
        assert isinstance(result, str)


class TestDatasetInfo:
    """Test dataset_info returns dict with expected keys."""

    def _add_info_to_mock(self, meta):
        """Add an info dict to the CloudVolumeMock so dataset_info works."""
        meta._ws_cv.info = {"scales": [{"resolution": [1, 1, 1]}]}

    def test_dataset_info_is_dict(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        info = meta.dataset_info
        assert isinstance(info, dict)

    def test_dataset_info_has_expected_keys(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        info = meta.dataset_info
        assert "chunks_start_at_voxel_offset" in info
        assert info["chunks_start_at_voxel_offset"] is True
        assert "data_dir" in info
        assert "graph" in info

    def test_dataset_info_graph_section(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        info = meta.dataset_info
        graph_info = info["graph"]
        assert "chunk_size" in graph_info
        assert "n_bits_for_layer_id" in graph_info
        assert "cv_mip" in graph_info
        assert "n_layers" in graph_info
        assert "spatial_bit_masks" in graph_info
        assert graph_info["n_layers"] == meta.layer_count

    def test_dataset_info_with_mesh_dir(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        meta._custom_data = {"mesh": {"dir": "gs://bucket/mesh"}}
        info = meta.dataset_info
        assert "mesh" in info
        assert info["mesh"] == "gs://bucket/mesh"

    def test_dataset_info_without_mesh_dir(self, gen_graph):
        graph = gen_graph(n_layers=4)
        meta = graph.meta
        self._add_info_to_mock(meta)
        meta._custom_data = {}
        info = meta.dataset_info
        assert "mesh" not in info


# =====================================================================
# Pure unit tests (no BigTable emulator needed) - mock CloudVolume & Redis
# =====================================================================
import json
from unittest.mock import MagicMock, patch, PropertyMock


class TestWsCvRedisCached:
    """Test ws_cv property with Redis caching."""

    @patch("pychunkedgraph.graph.meta.CloudVolume")
    @patch("pychunkedgraph.graph.meta.get_redis_connection")
    def test_ws_cv_redis_cached(self, mock_get_redis, mock_cv_cls):
        """When redis has cached info, ws_cv uses cached CloudVolume."""
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        cached_info = {"scales": [{"resolution": [8, 8, 40]}]}
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached_info)
        mock_get_redis.return_value = mock_redis

        mock_cv_instance = MagicMock()
        mock_cv_cls.return_value = mock_cv_instance

        result = meta.ws_cv

        assert result is mock_cv_instance
        mock_cv_cls.assert_called_once_with(
            "gs://bucket/ws", info=cached_info, progress=False
        )

    @patch("pychunkedgraph.graph.meta.CloudVolume")
    @patch("pychunkedgraph.graph.meta.get_redis_connection")
    def test_ws_cv_redis_failure_fallback(self, mock_get_redis, mock_cv_cls):
        """When redis raises, ws_cv falls back to direct CloudVolume."""
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        mock_get_redis.side_effect = Exception("Redis connection failed")

        mock_cv_instance = MagicMock()
        mock_cv_instance.info = {"scales": []}
        mock_cv_cls.return_value = mock_cv_instance

        result = meta.ws_cv

        assert result is mock_cv_instance
        # Should have been called without info kwarg (fallback)
        mock_cv_cls.assert_called_with("gs://bucket/ws", progress=False)

    @patch("pychunkedgraph.graph.meta.CloudVolume")
    @patch("pychunkedgraph.graph.meta.get_redis_connection")
    def test_ws_cv_caches_to_redis(self, mock_get_redis, mock_cv_cls):
        """When redis is available but cache miss, ws_cv caches info to redis."""
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        mock_redis = MagicMock()
        # Make redis.get raise to simulate cache miss on json.loads
        mock_redis.get.return_value = None  # This will make json.loads fail
        mock_get_redis.return_value = mock_redis

        mock_cv_instance = MagicMock()
        mock_cv_instance.info = {"scales": [{"resolution": [8, 8, 40]}]}
        mock_cv_cls.return_value = mock_cv_instance

        result = meta.ws_cv

        assert result is mock_cv_instance
        # The fallback CloudVolume call (no info= kwarg)
        mock_cv_cls.assert_called_with("gs://bucket/ws", progress=False)
        # Should try to cache in redis
        mock_redis.set.assert_called_once()

    @patch("pychunkedgraph.graph.meta.CloudVolume")
    @patch("pychunkedgraph.graph.meta.get_redis_connection")
    def test_ws_cv_returns_cached_instance(self, mock_get_redis, mock_cv_cls):
        """Once ws_cv has been set, subsequent calls return the cached instance."""
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        # Pre-set the cached ws_cv
        mock_cv = MagicMock()
        meta._ws_cv = mock_cv

        result = meta.ws_cv
        assert result is mock_cv
        # Should not try to create a new CloudVolume
        mock_cv_cls.assert_not_called()


class TestLayerCountComputed:
    """Test layer_count property computation from CloudVolume bounds."""

    def test_layer_count_computed_from_cv(self):
        """layer_count should be computed from ws_cv.bounds."""
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2, SPATIAL_BITS=10)
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        # Create a mock ws_cv with bounds
        mock_cv = MagicMock()
        # bounds.to_list() returns [x_min, y_min, z_min, x_max, y_max, z_max]
        # With a 256x256x256 volume and 64x64x64 chunks: 4 chunks per dim
        # log_2(4) = 2, +2 = 4 layers
        mock_cv.bounds.to_list.return_value = [0, 0, 0, 256, 256, 256]
        meta._ws_cv = mock_cv

        count = meta.layer_count
        assert isinstance(count, int)
        assert count >= 3  # at least 3 layers for any reasonable volume

    def test_layer_count_cached_after_first_access(self):
        """After layer_count is computed, it should be cached."""
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2, SPATIAL_BITS=10)
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        meta._layer_count = 5

        assert meta.layer_count == 5


class TestBitmasksLazy:
    """Test bitmasks property lazy computation."""

    def test_bitmasks_lazy_computed(self):
        """bitmasks should be computed lazily from layer_count."""
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2, SPATIAL_BITS=10)
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        # Set layer_count directly to avoid needing ws_cv for layer_count
        meta._layer_count = 5

        bm = meta.bitmasks
        assert isinstance(bm, dict)
        assert 1 in bm
        assert 2 in bm

    def test_bitmasks_cached_after_first_access(self):
        """Once computed, bitmasks should be cached."""
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2, SPATIAL_BITS=10)
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)
        meta._layer_count = 5

        bm1 = meta.bitmasks
        bm2 = meta.bitmasks
        assert bm1 is bm2


class TestOcdbtSeg:
    """Test ocdbt_seg property."""

    def test_ocdbt_seg_false_by_default(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)
        assert meta.ocdbt_seg is False

    def test_ocdbt_seg_true_from_custom_data(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})
        assert meta.ocdbt_seg is True

    def test_ocdbt_seg_false_explicit(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": False}})
        assert meta.ocdbt_seg is False

    def test_ocdbt_seg_cached(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})
        val1 = meta.ocdbt_seg
        val2 = meta.ocdbt_seg
        assert val1 is val2

    def test_ws_ocdbt_asserts_when_not_ocdbt(self):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)
        with pytest.raises(AssertionError, match="ocdbt"):
            _ = meta.ws_ocdbt

    @patch("pychunkedgraph.graph.meta.get_seg_source_and_destination_ocdbt")
    def test_ws_ocdbt_returns_destination(self, mock_get_ocdbt):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})

        mock_src = MagicMock()
        mock_dst = MagicMock()
        mock_get_ocdbt.return_value = (mock_src, mock_dst)

        result = meta.ws_ocdbt
        assert result is mock_dst
        mock_get_ocdbt.assert_called_once_with("gs://bucket/ws")

    @patch("pychunkedgraph.graph.meta.get_seg_source_and_destination_ocdbt")
    def test_ws_ocdbt_cached(self, mock_get_ocdbt):
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})

        mock_dst = MagicMock()
        mock_get_ocdbt.return_value = (MagicMock(), mock_dst)

        result1 = meta.ws_ocdbt
        result2 = meta.ws_ocdbt
        assert result1 is result2
        mock_get_ocdbt.assert_called_once()


class TestLayerChunkBoundsComputed:
    """Test layer_chunk_bounds property computation."""

    def test_layer_chunk_bounds_computed(self):
        """layer_chunk_bounds should be computed from voxel_counts and chunk_size."""
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2, SPATIAL_BITS=10)
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        mock_cv = MagicMock()
        mock_cv.bounds.to_list.return_value = [0, 0, 0, 256, 256, 256]
        meta._ws_cv = mock_cv
        # layer_count needs to be set to avoid recursive calls
        meta._layer_count = 4

        bounds = meta.layer_chunk_bounds
        assert isinstance(bounds, dict)
        # For layer_count=4, should have entries for layers 2 and 3
        assert 2 in bounds
        assert 3 in bounds
        # With 256/64=4 chunks, layer 2 should have 4 chunks per dim
        np.testing.assert_array_equal(bounds[2], np.array([4, 4, 4]))
        # layer 3: 4/2 = 2 chunks per dim
        np.testing.assert_array_equal(bounds[3], np.array([2, 2, 2]))

    def test_layer_chunk_bounds_cached(self):
        """After first access, layer_chunk_bounds should be cached."""
        gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], FANOUT=2, SPATIAL_BITS=10)
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        mock_cv = MagicMock()
        mock_cv.bounds.to_list.return_value = [0, 0, 0, 256, 256, 256]
        meta._ws_cv = mock_cv
        meta._layer_count = 4

        bounds1 = meta.layer_chunk_bounds
        bounds2 = meta.layer_chunk_bounds
        assert bounds1 is bounds2
