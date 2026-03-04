"""Tests for pychunkedgraph.ingest.manager"""

import pickle
import pytest
from unittest.mock import MagicMock, patch

from pychunkedgraph.ingest import IngestConfig
from pychunkedgraph.graph.meta import ChunkedGraphMeta, GraphConfig, DataSource


def _make_config_and_meta():
    config = IngestConfig()
    gc = GraphConfig(CHUNK_SIZE=[64, 64, 64], ID="test")
    ds = DataSource(
        EDGES="gs://test/edges",
        COMPONENTS="gs://test/comp",
        WATERSHED="gs://test/ws",
        DATA_VERSION=2,
    )
    meta = ChunkedGraphMeta(gc, ds)
    meta._layer_count = 4
    meta._bitmasks = {1: 10, 2: 10, 3: 1, 4: 1}
    return config, meta


def _make_manager():
    """Create an IngestionManager with mocked redis connection."""
    config, meta = _make_config_and_meta()
    with patch("pychunkedgraph.ingest.manager.get_redis_connection") as mock_redis_conn:
        mock_redis = MagicMock()
        mock_redis_conn.return_value = mock_redis
        from pychunkedgraph.ingest.manager import IngestionManager

        manager = IngestionManager(config=config, chunkedgraph_meta=meta)
    return manager, config, meta, mock_redis


class TestIngestionManagerSerialization:
    def test_serialized_dict(self):
        config, meta = _make_config_and_meta()
        # Test the serialized dict path without needing Redis
        params = {"config": config, "chunkedgraph_meta": meta}
        assert "config" in params
        assert "chunkedgraph_meta" in params
        assert params["config"] == config

    def test_serialized_pickle_roundtrip(self):
        config, meta = _make_config_and_meta()
        params = {"config": config, "chunkedgraph_meta": meta}
        serialized = pickle.dumps(params)
        restored = pickle.loads(serialized)
        assert restored["config"] == config
        assert restored["chunkedgraph_meta"].graph_config.ID == "test"


class TestSerializedDict:
    def test_serialized_returns_dict_with_correct_keys(self):
        """serialized() returns a dict with config and chunkedgraph_meta keys."""
        manager, config, meta, _ = _make_manager()
        result = manager.serialized()
        assert isinstance(result, dict)
        assert "config" in result
        assert "chunkedgraph_meta" in result
        assert result["config"] is config
        assert result["chunkedgraph_meta"] is meta


class TestSerializedPickleRoundtrip:
    def test_serialized_pickled_roundtrips(self):
        """serialized(pickled=True) produces bytes that pickle-load back correctly."""
        manager, config, meta, _ = _make_manager()
        pickled = manager.serialized(pickled=True)
        assert isinstance(pickled, bytes)
        loaded = pickle.loads(pickled)
        assert isinstance(loaded, dict)
        assert loaded["config"] == config
        assert isinstance(loaded["chunkedgraph_meta"], ChunkedGraphMeta)
        assert loaded["chunkedgraph_meta"].graph_config == meta.graph_config
        assert loaded["chunkedgraph_meta"].data_source == meta.data_source


class TestConfigProperty:
    def test_config_property_returns_injected_config(self):
        """config property returns the IngestConfig passed to __init__."""
        manager, config, _, _ = _make_manager()
        assert manager.config is config


class TestCgMetaProperty:
    def test_cg_meta_property_returns_injected_meta(self):
        """cg_meta property returns the ChunkedGraphMeta passed to __init__."""
        manager, _, meta, _ = _make_manager()
        assert manager.cg_meta is meta


class TestGetTaskQueueCaching:
    def test_get_task_queue_returns_cached_on_second_call(self):
        """Calling get_task_queue twice with the same name returns the same cached object."""
        manager, _, _, _ = _make_manager()
        with patch("pychunkedgraph.ingest.manager.get_rq_queue") as mock_get_rq:
            mock_queue = MagicMock()
            mock_get_rq.return_value = mock_queue

            q1 = manager.get_task_queue("test_queue")
            q2 = manager.get_task_queue("test_queue")

            assert q1 is q2
            mock_get_rq.assert_called_once_with("test_queue")


class TestRedisPropertyCaching:
    def test_redis_returns_cached_connection(self):
        """redis property returns cached value on second access; get_redis_connection not called again."""
        config, meta = _make_config_and_meta()
        with patch(
            "pychunkedgraph.ingest.manager.get_redis_connection"
        ) as mock_redis_conn:
            mock_redis = MagicMock()
            mock_redis_conn.return_value = mock_redis
            from pychunkedgraph.ingest.manager import IngestionManager

            manager = IngestionManager(config=config, chunkedgraph_meta=meta)
            call_count_after_init = mock_redis_conn.call_count

            r1 = manager.redis
            r2 = manager.redis

            # No additional calls to get_redis_connection after init
            assert mock_redis_conn.call_count == call_count_after_init
            assert r1 is r2
            assert r1 is mock_redis
