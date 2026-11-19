"""Tests for pychunkedgraph.graph.meta"""

import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pychunkedgraph.graph import meta as meta_module
from pychunkedgraph.graph.meta import ChunkedGraphMeta, GraphConfig, DataSource
from pychunkedgraph.meshing.mesh_meta import MeshMeta
from pychunkedgraph.tests.helpers import mock_ws_info


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
# Pure unit tests (no BigTable emulator needed) - mock CloudVolume
# =====================================================================
import json
from unittest.mock import MagicMock, patch, PropertyMock


class TestWsCv:
    """Test the ws_cv property: one CloudVolume, opened at the watershed and kept."""

    @patch("cloudvolume.CloudVolume")
    def test_ws_cv_opens_the_watershed(self, mock_cv_cls):
        """ws_cv opens the watershed path itself, with progress off."""
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds)

        assert meta.ws_cv is mock_cv_cls.return_value
        mock_cv_cls.assert_called_once_with("gs://bucket/ws", progress=False)

    @patch("cloudvolume.CloudVolume")
    def test_ws_cv_returns_cached_instance(self, mock_cv_cls):
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

        # 256^3 volume, 64^3 chunks -> 4 chunks/dim -> log2(4)=2, +2 = 4 layers
        meta._ws_info_d = mock_ws_info(size=(256, 256, 256))

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

    @patch("pychunkedgraph.graph.meta.read_populate_meta", return_value=None)
    @patch("pychunkedgraph.graph.meta.ensure_fork_synced")
    @patch("pychunkedgraph.graph.meta.fork_exists", return_value=True)
    @patch("pychunkedgraph.graph.meta.get_seg_source_and_destination_ocdbt")
    def test_ws_ocdbt_returns_base_scale(
        self,
        mock_get_ocdbt,
        _mock_fork_exists,
        _mock_ensure_synced,
        _mock_read_populate_meta,
    ):
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})

        mock_src = MagicMock()
        mock_dst_base = MagicMock()
        mock_dst_mip1 = MagicMock()
        mock_get_ocdbt.return_value = (
            [mock_src, MagicMock()],
            [mock_dst_base, mock_dst_mip1],
            [[4, 4, 40], [8, 8, 40]],
        )

        assert meta.ws_ocdbt is mock_dst_base
        assert meta.ws_ocdbt_scales == [mock_dst_base, mock_dst_mip1]
        assert meta.ws_ocdbt_resolutions == [[4, 4, 40], [8, 8, 40]]
        # `get_seg_source_and_destination_ocdbt` is called with a third positional
        # arg (the resolved OcdbtConfig) — verify only the (ws, graph_id) part;
        # the OcdbtConfig.resolve contract has its own unit tests.
        mock_get_ocdbt.assert_called_once()
        assert mock_get_ocdbt.call_args.args[:2] == ("gs://bucket/ws", "test_graph")

    @patch("pychunkedgraph.graph.meta.read_populate_meta", return_value=None)
    @patch("pychunkedgraph.graph.meta.ensure_fork_synced")
    @patch("pychunkedgraph.graph.meta.fork_exists", return_value=True)
    @patch("pychunkedgraph.graph.meta.get_seg_source_and_destination_ocdbt")
    def test_ws_ocdbt_cached(
        self,
        mock_get_ocdbt,
        _mock_fork_exists,
        _mock_ensure_synced,
        _mock_read_populate_meta,
    ):
        gc = GraphConfig(ID="test_graph", CHUNK_SIZE=[64, 64, 64])
        ds = DataSource(WATERSHED="gs://bucket/ws", DATA_VERSION=4)
        meta = ChunkedGraphMeta(gc, ds, custom_data={"seg": {"ocdbt": True}})

        mock_dst = MagicMock()
        mock_get_ocdbt.return_value = ([MagicMock()], [mock_dst], [[4, 4, 40]])

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

        meta._ws_info_d = mock_ws_info(size=(256, 256, 256))
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

        meta._ws_info_d = mock_ws_info(size=(256, 256, 256))
        meta._layer_count = 4

        bounds1 = meta.layer_chunk_bounds
        bounds2 = meta.layer_chunk_bounds
        assert bounds1 is bounds2


# =====================================================================
# repoint_watershed
# =====================================================================
OLD, NEW = "gs://old/seg", "gs://new/seg"


@pytest.fixture
def infos(monkeypatch):
    """Watershed infos by path, served in place of the bucket: ``ts.KvStore.open`` reads
    an info, and ``ts.open`` answers the path it was opened at."""
    by_path = {OLD: {"at": OLD}, NEW: {"at": NEW}}

    def done(value):
        return SimpleNamespace(result=lambda: value)

    def kvstore(path):
        info = SimpleNamespace(value=json.dumps(by_path[path]))
        return done(SimpleNamespace(read=lambda _key: done(info)))

    stub = SimpleNamespace(
        KvStore=SimpleNamespace(open=kvstore), open=lambda spec: done(spec["kvstore"])
    )
    monkeypatch.setattr(meta_module, "ts", stub)
    return by_path


def _meta(custom_data):
    return ChunkedGraphMeta(GraphConfig(ID="g"), DataSource(WATERSHED=OLD), custom_data)


def test_repoint_reads_the_new_watershed(infos):
    meta = _meta({})
    assert meta.ws_info == {"at": OLD}
    assert meta.repoint_watershed(NEW) is meta
    assert meta.data_source.WATERSHED == NEW
    assert meta.ws_info == {"at": NEW}


def test_repoint_reopens_what_the_old_watershed_fed(infos, monkeypatch):
    """A volume or store opened at the old watershed does not outlive the repoint."""
    monkeypatch.setattr("cloudvolume.CloudVolume", lambda path, progress: path)
    meta = _meta({})
    assert (meta.ws_cv, meta.ws_ts) == (OLD, OLD)
    meta.repoint_watershed(NEW)
    assert (meta.ws_cv, meta.ws_ts) == (NEW, NEW)


def test_repoint_keeps_meshes_where_they_resolved(infos):
    mm = MeshMeta(_meta({"mesh": {"dir": "graphene_meshes"}}).repoint_watershed(NEW))
    assert mm.initial_path == f"{OLD}/graphene_meshes/initial"
    assert mm.dynamic_path == f"{OLD}/graphene_meshes/dynamic"


def test_repoint_keeps_the_old_watershed_grid_keys(infos):
    infos[OLD] = {
        "mesh_metadata": {"uniform_draco_grid_size": 7.0, "max_meshed_layer": 9}
    }
    infos[NEW] = {"mesh_metadata": {"max_meshed_layer": 3}}
    meta = _meta({"mesh": {"dir": "graphene_meshes", "max_meshed_layer": 5}})
    client = MeshMeta(meta.repoint_watershed(NEW)).client_metadata
    assert client["uniform_draco_grid_size"] == 7.0
    assert client["max_meshed_layer"] == 5


def test_a_table_repointed_away_and_back_publishes_its_dynamic_dir_by_name(infos):
    meta = _meta({"mesh": {"dir": "graphene_meshes", "dynamic_mesh_dir": "dynamic_g"}})
    meta.repoint_watershed(NEW).repoint_watershed(OLD)
    assert MeshMeta(meta).client_metadata["unsharded_mesh_dir"] == "dynamic_g"


def test_a_moved_table_publishes_its_dynamic_dir_url(infos):
    meta = _meta({"mesh": {"dir": "graphene_meshes"}}).repoint_watershed(NEW)
    published = MeshMeta(meta).client_metadata["unsharded_mesh_dir"]
    assert published == f"{OLD}/graphene_meshes/dynamic"


@pytest.mark.parametrize("custom_data", [{"READ_ONLY": True}, {"mesh": {"mip": 2}}])
def test_repoint_of_a_table_without_meshes_keeps_its_custom_data(infos, custom_data):
    meta = _meta(dict(custom_data)).repoint_watershed(NEW)
    assert meta.custom_data == custom_data


def test_repoint_reopens_the_segmentation_store_at_the_new_watershed(monkeypatch):
    """OCDBT settings and handles read after a repoint come from the new watershed."""
    populate = {OLD: {"populate_layer": 3}, NEW: {"populate_layer": 5}}
    monkeypatch.setattr(meta_module, "read_populate_meta", populate.get)
    monkeypatch.setattr(meta_module, "fork_exists", lambda ws, graph_id: True)
    monkeypatch.setattr(meta_module, "ensure_fork_synced", lambda ws, graph_id: False)
    monkeypatch.setattr(
        meta_module,
        "get_seg_source_and_destination_ocdbt",
        lambda ws, graph_id, config: ([], [ws], [config.populate_layer]),
    )
    meta = _meta({"ocdbt_config": {"enabled": True}})
    assert meta.ws_ocdbt_scales == [OLD]
    meta.repoint_watershed(NEW)
    assert meta.ocdbt_config.populate_layer == 5
    assert (meta.ws_ocdbt_scales, meta.ws_ocdbt_resolutions) == ([NEW], [5])


# =====================================================================
# for_copied_graph
# =====================================================================
def test_a_copy_keeps_its_sources_initial_meshes(infos):
    meta = _meta({"mesh": {"dir": "graphene_meshes"}}).for_copied_graph("c")
    assert meta.graph_config.ID == "c"
    assert MeshMeta(meta).initial_path == f"{OLD}/graphene_meshes/initial"


def test_a_copy_reads_its_sources_dynamic_dir_and_writes_its_own(infos, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1234.5)
    meta = _meta({"mesh": {"dir": "graphene_meshes"}}).for_copied_graph("c")
    mm = MeshMeta(meta)
    assert mm.dynamic_path == f"{OLD}/graphene_meshes/dynamic_c"
    assert [(one.path, one.until) for one in mm.inherited] == [
        (f"{OLD}/graphene_meshes/dynamic", 1234)
    ]


def test_a_copy_stores_its_own_dir_absolute(infos):
    meta = _meta({"mesh": {"dir": "graphene_meshes"}}).for_copied_graph("c")
    own = f"{OLD}/graphene_meshes/dynamic_c"
    assert meta.custom_data["mesh"]["dynamic_mesh_dir"] == own


def test_a_copy_of_a_copy_reads_both_sources_oldest_first(infos):
    root = f"{OLD}/graphene_meshes"
    meta = _meta({"mesh": {"dir": "graphene_meshes"}})
    mm = MeshMeta(meta.for_copied_graph("c").for_copied_graph("d"))
    assert [one.path for one in mm.inherited] == [
        f"{root}/dynamic",
        f"{root}/dynamic_c",
    ]
    assert mm.dynamic_path == f"{root}/dynamic_d"


@pytest.mark.parametrize("custom_data", [{"other": 1}, {"mesh": {"mip": 2}}])
def test_a_copy_of_a_table_without_meshes_keeps_its_custom_data(infos, custom_data):
    meta = _meta(dict(custom_data)).for_copied_graph("c")
    assert meta.custom_data == custom_data
