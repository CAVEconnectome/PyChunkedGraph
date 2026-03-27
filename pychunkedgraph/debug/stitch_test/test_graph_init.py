"""
Tests that CG worker creation uses only cached meta — no BigTable or GCS reads.

Run: BIGTABLE_PROJECT=zetta-proofreading BIGTABLE_INSTANCE=pychunkedgraph pytest test_graph_init.py -v
"""

import os

os.environ["BIGTABLE_PROJECT"] = "zetta-proofreading"
os.environ["BIGTABLE_INSTANCE"] = "pychunkedgraph"

import pickle
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from unittest.mock import patch

import pytest

from pychunkedgraph.debug.stitch_test.graph_init import (
    create_cg,
    pool_init,
    prepare_shared_init,
)

TABLE = "stitch_redesign_test_hsmith_mec_baseline_multiwave"


@pytest.fixture(scope="module")
def shared_init() -> tuple:
    return prepare_shared_init(TABLE)


def _create_in_worker(args: tuple) -> str:
    """Worker function for process pool test."""
    graph_id = args[0]
    cg = create_cg(graph_id)
    return cg.graph_id


class TestNoRemoteReads:

    def test_create_cg_no_bigtable_meta_read(self, shared_init: tuple) -> None:
        meta_bytes, cv_info = shared_init
        pool_init(meta_bytes, cv_info)

        with patch("kvdbclient.bigtable.client.Client.read_table_meta") as mock_meta:
            cg = create_cg(TABLE)
            mock_meta.assert_not_called()
            assert cg.graph_id == TABLE

    def test_create_cg_no_gcs_read(self, shared_init: tuple) -> None:
        meta_bytes, cv_info = shared_init
        pool_init(meta_bytes, cv_info)

        with patch("cloudvolume.CloudVolume.__init__") as mock_cv:
            cg = create_cg(TABLE)
            mock_cv.assert_not_called()

    def test_thread_pool_no_remote_reads(self, shared_init: tuple) -> None:
        meta_bytes, cv_info = shared_init
        pool_init(meta_bytes, cv_info)

        with patch("kvdbclient.bigtable.client.Client.read_table_meta") as mock_meta:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(create_cg, TABLE) for _ in range(4)]
                results = [f.result() for f in futs]
            mock_meta.assert_not_called()
            assert all(cg.graph_id == TABLE for cg in results)

    def test_process_pool_no_remote_reads(self, shared_init: tuple) -> None:
        meta_bytes, cv_info = shared_init
        with Pool(2, initializer=pool_init, initargs=(meta_bytes, cv_info)) as p:
            results = p.map(_create_in_worker, [(TABLE,)] * 4)
        assert all(gid == TABLE for gid in results)

    def test_meta_survives_pickle_roundtrip(self, shared_init: tuple) -> None:
        meta_bytes, cv_info = shared_init
        meta = pickle.loads(meta_bytes)
        meta2 = pickle.loads(pickle.dumps(meta))
        assert meta.graph_config.ID == meta2.graph_config.ID
        assert meta.data_source.WATERSHED == meta2.data_source.WATERSHED
