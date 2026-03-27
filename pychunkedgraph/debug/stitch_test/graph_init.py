"""
ChunkedGraph instance creation with cached metadata.

Meta and CloudVolume info are read once in the parent process, then passed
to workers via pickle (fork COW). Workers create CG instances without
any BigTable or GCS reads.
"""

import os
import pickle

from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph

_worker_meta = None
_worker_cv_info = None


def prepare_shared_init(table_name: str) -> tuple[bytes, dict]:
    """Read meta + cv_info once in parent. Returns (meta_bytes, cv_info)."""
    cg = ChunkedGraph(graph_id=table_name)
    meta_bytes = pickle.dumps(cg.meta)
    cv = CloudVolume(cg.meta.data_source.WATERSHED, mip=0)
    return meta_bytes, cv.info


def pool_init(meta_bytes: bytes, cv_info: dict) -> None:
    """Pool initializer: deserialize meta, cache globally."""
    global _worker_meta, _worker_cv_info
    _worker_meta = pickle.loads(meta_bytes)
    _worker_cv_info = cv_info
    setup_env()


def create_cg(graph_id: str) -> ChunkedGraph:
    """Create CG in worker using cached meta. No BigTable/GCS reads."""
    assert _worker_meta is not None, "pool_init not called"
    return ChunkedGraph(graph_id=graph_id, meta=_worker_meta)


def setup_env() -> None:
    """Set BigTable env vars."""
    os.environ.setdefault("BIGTABLE_PROJECT", "zetta-proofreading")
    os.environ.setdefault("BIGTABLE_INSTANCE", "pychunkedgraph")
