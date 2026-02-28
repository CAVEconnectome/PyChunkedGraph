"""Tests for pychunkedgraph.ingest IngestConfig"""

from pychunkedgraph.ingest import IngestConfig


class TestIngestConfig:
    def test_defaults(self):
        config = IngestConfig()
        assert config.AGGLOMERATION is None
        assert config.WATERSHED is None
        assert config.USE_RAW_EDGES is False
        assert config.USE_RAW_COMPONENTS is False
        assert config.TEST_RUN is False

    def test_custom_values(self):
        config = IngestConfig(
            AGGLOMERATION="gs://bucket/agg",
            WATERSHED="gs://bucket/ws",
            USE_RAW_EDGES=True,
            USE_RAW_COMPONENTS=True,
            TEST_RUN=True,
        )
        assert config.AGGLOMERATION == "gs://bucket/agg"
        assert config.WATERSHED == "gs://bucket/ws"
        assert config.USE_RAW_EDGES is True
        assert config.USE_RAW_COMPONENTS is True
        assert config.TEST_RUN is True
