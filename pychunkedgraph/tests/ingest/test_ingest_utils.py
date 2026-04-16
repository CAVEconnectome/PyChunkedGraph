"""Tests for pychunkedgraph.ingest.utils"""

import io
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pychunkedgraph.ingest.utils import (
    bootstrap,
    chunk_id_str,
    get_chunks_not_done,
    job_type_guard,
    move_up,
    postprocess_edge_data,
    randomize_grid_points,
)


class TestBootstrap:
    def test_from_config(self):
        from google.auth import credentials

        config = {
            "data_source": {
                "EDGES": "gs://test/edges",
                "COMPONENTS": "gs://test/components",
                "WATERSHED": "gs://test/ws",
            },
            "graph_config": {
                "CHUNK_SIZE": [64, 64, 64],
                "FANOUT": 2,
                "SPATIAL_BITS": 10,
            },
            "backend_client": {
                "TYPE": "bigtable",
                "CONFIG": {
                    "ADMIN": True,
                    "READ_ONLY": False,
                    "PROJECT": "test-project",
                    "INSTANCE": "test-instance",
                    "CREDENTIALS": credentials.AnonymousCredentials(),
                },
            },
            "ingest_config": {},
        }
        meta, ingest_config, client_info = bootstrap("test_graph", config=config)
        assert meta.graph_config.ID == "test_graph"
        assert meta.graph_config.FANOUT == 2
        assert ingest_config.USE_RAW_EDGES is False


class TestPostprocessEdgeData:
    def test_v2_passthrough(self):
        class FakeMeta:
            class data_source:
                DATA_VERSION = 2

            resolution = np.array([1, 1, 1])

        class FakeIM:
            cg_meta = FakeMeta()

        edge_dict = {"test": {"sv1": [1], "sv2": [2], "aff": [0.5], "area": [10]}}
        result = postprocess_edge_data(FakeIM(), edge_dict)
        assert result == edge_dict

    def test_v3(self):
        class FakeMeta:
            class data_source:
                DATA_VERSION = 3

            resolution = np.array([4, 4, 40])

        class FakeIM:
            cg_meta = FakeMeta()

        edge_dict = {
            "test": {
                "sv1": np.array([1]),
                "sv2": np.array([2]),
                "aff_x": np.array([0.1]),
                "aff_y": np.array([0.2]),
                "aff_z": np.array([0.3]),
                "area_x": np.array([10]),
                "area_y": np.array([20]),
                "area_z": np.array([30]),
            }
        }
        result = postprocess_edge_data(FakeIM(), edge_dict)
        assert "aff" in result["test"]
        assert "area" in result["test"]
        # aff = 0.1*4 + 0.2*4 + 0.3*40 = 0.4 + 0.8 + 12 = 13.2
        np.testing.assert_almost_equal(result["test"]["aff"][0], 13.2)

    def test_empty_data(self):
        class FakeMeta:
            class data_source:
                DATA_VERSION = 3

            resolution = np.array([1, 1, 1])

        class FakeIM:
            cg_meta = FakeMeta()

        edge_dict = {"test": {}}
        result = postprocess_edge_data(FakeIM(), edge_dict)
        assert result["test"] == {}


class TestRandomizeGridPoints:
    def test_basic(self):
        points = list(randomize_grid_points(2, 2, 2))
        assert len(points) == 8
        # All coordinates should be valid
        for x, y, z in points:
            assert 0 <= x < 2
            assert 0 <= y < 2
            assert 0 <= z < 2

    def test_covers_all(self):
        points = list(randomize_grid_points(3, 2, 1))
        assert len(points) == 6
        coords = {(x, y, z) for x, y, z in points}
        assert len(coords) == 6


class TestPostprocessEdgeDataUnknownVersion:
    def test_version5_raises(self):
        """Version 5 is not supported and should raise ValueError."""

        class FakeMeta:
            class data_source:
                DATA_VERSION = 5

            resolution = np.array([1, 1, 1])

        class FakeIM:
            cg_meta = FakeMeta()

        edge_dict = {"test": {"sv1": [1], "sv2": [2]}}
        with pytest.raises(ValueError, match="Unknown data_version"):
            postprocess_edge_data(FakeIM(), edge_dict)


class TestPostprocessEdgeDataV4SameAsV3:
    def test_v4_same_code_path(self):
        """Version 4 should use the same processing logic as v3 (combine xyz components)."""

        class FakeMetaV3:
            class data_source:
                DATA_VERSION = 3

            resolution = np.array([2, 2, 20])

        class FakeMetaV4:
            class data_source:
                DATA_VERSION = 4

            resolution = np.array([2, 2, 20])

        class FakeIMv3:
            cg_meta = FakeMetaV3()

        class FakeIMv4:
            cg_meta = FakeMetaV4()

        edge_dict_v3 = {
            "test": {
                "sv1": np.array([10]),
                "sv2": np.array([20]),
                "aff_x": np.array([0.1]),
                "aff_y": np.array([0.2]),
                "aff_z": np.array([0.3]),
                "area_x": np.array([5]),
                "area_y": np.array([6]),
                "area_z": np.array([7]),
            }
        }
        edge_dict_v4 = {
            "test": {
                "sv1": np.array([10]),
                "sv2": np.array([20]),
                "aff_x": np.array([0.1]),
                "aff_y": np.array([0.2]),
                "aff_z": np.array([0.3]),
                "area_x": np.array([5]),
                "area_y": np.array([6]),
                "area_z": np.array([7]),
            }
        }

        result_v3 = postprocess_edge_data(FakeIMv3(), edge_dict_v3)
        result_v4 = postprocess_edge_data(FakeIMv4(), edge_dict_v4)

        # Both versions should produce the same combined aff and area values
        np.testing.assert_array_almost_equal(
            result_v3["test"]["aff"], result_v4["test"]["aff"]
        )
        np.testing.assert_array_almost_equal(
            result_v3["test"]["area"], result_v4["test"]["area"]
        )
        np.testing.assert_array_equal(
            result_v3["test"]["sv1"], result_v4["test"]["sv1"]
        )
        np.testing.assert_array_equal(
            result_v3["test"]["sv2"], result_v4["test"]["sv2"]
        )


class TestChunkIdStr:
    def test_basic(self):
        result = chunk_id_str(3, [1, 2, 3])
        assert result == "3_1_2_3"

    def test_layer_zero(self):
        result = chunk_id_str(0, [0, 0, 0])
        assert result == "0_0_0_0"

    def test_tuple_coords(self):
        result = chunk_id_str(5, (10, 20, 30))
        assert result == "5_10_20_30"

    def test_single_coord(self):
        result = chunk_id_str(2, [7])
        assert result == "2_7"


class TestMoveUp:
    def test_writes_escape_code_to_stdout(self):
        """move_up() writes the ANSI escape code for cursor-up to stdout."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            move_up(3)
        finally:
            sys.stdout = old_stdout
        assert captured.getvalue() == "\033[3A"

    def test_default_one_line(self):
        """move_up() with no argument moves up 1 line."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            move_up()
        finally:
            sys.stdout = old_stdout
        assert captured.getvalue() == "\033[1A"


class TestGetChunksNotDone:
    def _make_mock_imanager(self):
        imanager = MagicMock()
        imanager.redis = MagicMock()
        return imanager

    def test_all_completed_returns_empty(self):
        """When all coords are completed in redis, returns empty list."""
        imanager = self._make_mock_imanager()
        coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        # All marked as completed (1 = member of the set)
        imanager.redis.smismember.return_value = [1, 1, 1]
        result = get_chunks_not_done(imanager, layer=2, coords=coords)
        assert result == []

    def test_some_not_completed_returns_those(self):
        """When some coords are not completed, returns those coords."""
        imanager = self._make_mock_imanager()
        coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        # First is completed, second and third are not
        imanager.redis.smismember.return_value = [1, 0, 0]
        result = get_chunks_not_done(imanager, layer=2, coords=coords)
        assert result == [[1, 0, 0], [0, 1, 0]]

    def test_redis_exception_returns_all_coords(self):
        """When redis raises an exception, returns all coords as fallback."""
        imanager = self._make_mock_imanager()
        coords = [[0, 0, 0], [1, 0, 0]]
        imanager.redis.smismember.side_effect = Exception("Redis down")
        result = get_chunks_not_done(imanager, layer=2, coords=coords)
        assert result == coords


class TestJobTypeGuard:
    @patch("pychunkedgraph.ingest.utils.get_redis_connection")
    def test_same_job_type_runs_normally(self, mock_get_redis):
        """When current job_type matches, decorated function runs normally."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"ingest"
        mock_get_redis.return_value = mock_redis

        @job_type_guard("ingest")
        def my_func():
            return "success"

        assert my_func() == "success"

    @patch("pychunkedgraph.ingest.utils.get_redis_connection")
    def test_different_job_type_calls_exit(self, mock_get_redis):
        """When current job_type differs, exit(1) is called."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"upgrade"
        mock_get_redis.return_value = mock_redis

        @job_type_guard("ingest")
        def my_func():
            return "success"

        with pytest.raises(SystemExit) as exc_info:
            my_func()
        assert exc_info.value.code == 1

    @patch("pychunkedgraph.ingest.utils.get_redis_connection")
    def test_no_current_type_runs_normally(self, mock_get_redis):
        """When no current job_type is set in redis, decorated function runs normally."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        @job_type_guard("ingest")
        def my_func():
            return "success"

        assert my_func() == "success"


# =====================================================================
# Additional pure unit tests
# =====================================================================
from pychunkedgraph.ingest.utils import start_ocdbt_server


class TestGetChunksNotDoneWithSplits:
    """Test get_chunks_not_done with splits > 0."""

    def _make_mock_imanager(self):
        imanager = MagicMock()
        imanager.redis = MagicMock()
        return imanager

    def test_get_chunks_not_done_with_splits(self):
        """When splits > 0, should expand coords with split indices."""
        imanager = self._make_mock_imanager()
        coords = [[0, 0, 0], [1, 0, 0]]
        splits = 2
        # With 2 coords and 2 splits, we get 4 entries:
        # (0,0,0) split 0, (0,0,0) split 1, (1,0,0) split 0, (1,0,0) split 1
        # All completed
        imanager.redis.smismember.return_value = [1, 1, 1, 1]
        result = get_chunks_not_done(imanager, layer=2, coords=coords, splits=splits)
        assert result == []

    def test_get_chunks_not_done_with_splits_some_incomplete(self):
        """When splits > 0 and some are not done, return the incomplete (coord, split) tuples."""
        imanager = self._make_mock_imanager()
        coords = [[0, 0, 0], [1, 0, 0]]
        splits = 2
        # 4 entries, only first is completed
        imanager.redis.smismember.return_value = [1, 0, 1, 0]
        result = get_chunks_not_done(imanager, layer=2, coords=coords, splits=splits)
        # Should return the (coord, split) tuples that are not done
        assert len(result) == 2
        assert result[0] == ([0, 0, 0], 1)
        assert result[1] == ([1, 0, 0], 1)

    def test_get_chunks_not_done_splits_redis_error(self):
        """When redis raises with splits > 0, should return split_coords as fallback."""
        imanager = self._make_mock_imanager()
        coords = [[0, 0, 0]]
        splits = 2
        imanager.redis.smismember.side_effect = Exception("Redis down")
        result = get_chunks_not_done(imanager, layer=2, coords=coords, splits=splits)
        # Should return all (coord, split) tuples
        assert len(result) == 2
        assert result[0] == ([0, 0, 0], 0)
        assert result[1] == ([0, 0, 0], 1)

    def test_get_chunks_not_done_splits_coord_str_format(self):
        """With splits, redis keys should include the split index."""
        imanager = self._make_mock_imanager()
        coords = [[2, 3, 4]]
        splits = 1
        imanager.redis.smismember.return_value = [0]
        get_chunks_not_done(imanager, layer=3, coords=coords, splits=splits)
        # Check the coords_strs passed to smismember
        call_args = imanager.redis.smismember.call_args
        assert call_args[0][0] == "3c"
        assert call_args[0][1] == ["2_3_4_0"]


class TestStartOcdbtServer:
    """Test start_ocdbt_server function."""

    @patch("pychunkedgraph.ingest.utils.ts")
    @patch.dict("os.environ", {"MY_POD_IP": "10.0.0.1"})
    def test_start_ocdbt_server(self, mock_ts):
        """start_ocdbt_server should open a KvStore and set redis keys."""
        imanager = MagicMock()
        imanager.cg.meta.data_source.EDGES = "gs://bucket/edges"
        mock_redis = MagicMock()
        imanager.redis = mock_redis

        server = MagicMock()
        server.port = 12345

        mock_kv_future = MagicMock()
        mock_ts.KvStore.open.return_value = mock_kv_future

        start_ocdbt_server(imanager, server)

        # Verify tensorstore was called with the right spec
        call_args = mock_ts.KvStore.open.call_args[0][0]
        assert call_args["driver"] == "ocdbt"
        assert "gs://bucket/edges/ocdbt" in call_args["base"]
        assert call_args["coordinator"]["address"] == "localhost:12345"
        mock_kv_future.result.assert_called_once()

        # Verify redis keys were set
        mock_redis.set.assert_any_call("OCDBT_COORDINATOR_PORT", "12345")
        mock_redis.set.assert_any_call("OCDBT_COORDINATOR_HOST", "10.0.0.1")

    @patch("pychunkedgraph.ingest.utils.ts")
    @patch.dict("os.environ", {}, clear=True)
    def test_start_ocdbt_server_default_host(self, mock_ts):
        """When MY_POD_IP is not set, should default to localhost."""
        imanager = MagicMock()
        imanager.cg.meta.data_source.EDGES = "gs://bucket/edges"
        mock_redis = MagicMock()
        imanager.redis = mock_redis

        server = MagicMock()
        server.port = 9999

        mock_kv_future = MagicMock()
        mock_ts.KvStore.open.return_value = mock_kv_future

        start_ocdbt_server(imanager, server)

        mock_redis.set.assert_any_call("OCDBT_COORDINATOR_HOST", "localhost")


class TestPostprocessEdgeDataNoneValues:
    """Test postprocess_edge_data when edge_dict values are None."""

    def test_postprocess_edge_data_none_values(self):
        """When edge_dict[k] is None, the key should be in result with empty dict."""

        class FakeMeta:
            class data_source:
                DATA_VERSION = 3

            resolution = np.array([4, 4, 40])

        class FakeIM:
            cg_meta = FakeMeta()

        edge_dict = {"test_key": None}
        result = postprocess_edge_data(FakeIM(), edge_dict)
        assert "test_key" in result
        assert result["test_key"] == {}

    def test_postprocess_edge_data_v4_none_values(self):
        """Version 4 with None values should also produce empty dict."""

        class FakeMeta:
            class data_source:
                DATA_VERSION = 4

            resolution = np.array([4, 4, 40])

        class FakeIM:
            cg_meta = FakeMeta()

        edge_dict = {
            "a": None,
            "b": {
                "sv1": np.array([1]),
                "sv2": np.array([2]),
                "aff_x": np.array([0.1]),
                "aff_y": np.array([0.2]),
                "aff_z": np.array([0.3]),
                "area_x": np.array([10]),
                "area_y": np.array([20]),
                "area_z": np.array([30]),
            },
        }
        result = postprocess_edge_data(FakeIM(), edge_dict)
        assert result["a"] == {}
        assert "aff" in result["b"]
        assert "area" in result["b"]
