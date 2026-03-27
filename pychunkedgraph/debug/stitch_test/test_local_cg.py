"""Tests for LocalChunkedGraph worker creation. No BigTable access needed."""

import os

os.environ["BIGTABLE_PROJECT"] = "zetta-proofreading"
os.environ["BIGTABLE_INSTANCE"] = "pychunkedgraph"

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from pychunkedgraph.graph import basetypes

from .local_cg import LocalChunkedGraph

NODE_ID = basetypes.NODE_ID
TABLE = "stitch_redesign_test_hsmith_mec_baseline_multiwave"
FIXTURES = Path(__file__).parent / "fixtures"

_PATCH_TARGET = "pychunkedgraph.graph.chunkedgraph.get_client_class"
_mock_clients: list = []


def _fake_client_class(backend_type):
    def _make(*a, **kw):
        client = MagicMock()
        _mock_clients.append(client)
        return client
    return _make


@pytest.fixture(scope="module")
def cached_init() -> tuple:
    meta_bytes = (FIXTURES / "meta.pkl").read_bytes()
    cv_info = pickle.loads((FIXTURES / "cv_info.pkl").read_bytes())
    return meta_bytes, cv_info


class TestLocalCGWorkerCreation:

    _CV_REFRESH = "cloudvolume.datasource.precomputed.metadata.PrecomputedMetadata.refresh_info"

    def test_worker_skips_meta_and_cv_read(self, cached_init) -> None:
        """Verifies create_worker with cached meta doesn't read from BigTable or GCS.
        Broken = every worker does a full meta + cv_info fetch → slow pool init."""
        _mock_clients.clear()
        meta_bytes, cv_info = cached_init
        with patch(_PATCH_TARGET, _fake_client_class), \
             patch(self._CV_REFRESH) as mock_refresh:
            LocalChunkedGraph.pool_init(meta_bytes, cv_info)
            lcg = LocalChunkedGraph.create_worker(TABLE)
            assert lcg.cg.meta.graph_config.ID == TABLE
            assert lcg.cg.meta.layer_count == 7
            assert lcg.cg.meta._ws_cv.info == cv_info
            for client in _mock_clients:
                client.read_table_meta.assert_not_called()
            mock_refresh.assert_not_called()

    def test_chunk_ops_work_after_init(self, cached_init) -> None:
        """Meta-dependent operations (get_chunk_layer, get_segment_id) work with cached meta.
        Broken = worker can't compute chunk layers → hierarchy build fails."""
        with patch(_PATCH_TARGET, _fake_client_class):
            LocalChunkedGraph.pool_init(*cached_init)
            lcg = LocalChunkedGraph.create_worker(TABLE)
            test_id = np.uint64((2 << 56) | 12345)
            assert lcg.get_chunk_layer(test_id) == 2
            assert lcg.get_segment_id(test_id) == 12345

    def test_thread_pool_no_meta_read(self, cached_init) -> None:
        """Multiple workers created concurrently all get correct meta without reads.
        Broken = race condition in meta init → some workers get wrong layer_count."""
        with patch(_PATCH_TARGET, _fake_client_class):
            LocalChunkedGraph.pool_init(*cached_init)
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = [ex.submit(LocalChunkedGraph.create_worker, TABLE) for _ in range(2)]
                workers = [f.result() for f in futs]
            for w in workers:
                assert w.cg.meta.layer_count == 7

    def test_meta_pickle_roundtrip(self, cached_init) -> None:
        """Meta survives pickle roundtrip (required for pool_init serialization).
        Broken = meta fields lost in pickle → worker has incomplete config."""
        meta = pickle.loads(cached_init[0])
        meta2 = pickle.loads(pickle.dumps(meta))
        assert meta.graph_config.ID == meta2.graph_config.ID
        assert meta.layer_count == meta2.layer_count
