"""Shared utilities for multiprocessing workers: CG creation, edge loading, wave listing."""

import os
import pickle
import time

import numpy as np
from cloudfiles import CloudFile, CloudFiles
from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph, basetypes

from .tables import EDGES_SRC

_CG_INIT_RETRIES = 5
_CG_INIT_DELAY = 2

_worker_meta = None
_worker_cv_info = None


def pool_init(meta_bytes: bytes, cv_info: dict) -> None:
    """Pool initializer: deserialize meta + cv_info once per worker process."""
    global _worker_meta, _worker_cv_info
    _worker_meta = pickle.loads(meta_bytes)
    _worker_cv_info = cv_info


def create_cg_worker(graph_id: str) -> ChunkedGraph:
    """Create ChunkedGraph in worker using pre-loaded meta. No BigTable or GCS reads."""
    if _worker_meta is not None:
        cg = ChunkedGraph(meta=_worker_meta)
        if _worker_cv_info is not None:
            cg.meta._ws_cv = CloudVolume(
                cg.meta._data_source.WATERSHED, info=_worker_cv_info, progress=False
            )
        return cg
    return create_cg(graph_id)


def create_cg(graph_id: str) -> ChunkedGraph:
    """Create ChunkedGraph with retry — meta read can transiently fail."""
    for attempt in range(_CG_INIT_RETRIES):
        cg = ChunkedGraph(graph_id=graph_id)
        if cg.meta is not None:
            return cg
        time.sleep(_CG_INIT_DELAY * (attempt + 1))
    raise RuntimeError(f"ChunkedGraph meta is None after {_CG_INIT_RETRIES} retries for {graph_id}")


def prepare_shared_init(table_name: str) -> tuple:
    """Read meta + cv info once in parent process, return bytes for Pool initializer."""
    cg = create_cg(table_name)
    cv_info = cg.meta.ws_cv.info
    meta_bytes = pickle.dumps(cg.meta)
    return meta_bytes, cv_info


def load_edges(path: str) -> np.ndarray:
    return np.asarray(pickle.loads(CloudFile(path).get()), dtype=basetypes.NODE_ID)


def default_n_workers(n_tasks: int) -> int:
    return min(n_tasks, 3 * os.cpu_count())


def list_wave_files(wave: int = 0) -> list[str]:
    """List all edge files for a wave, sorted by subtask index."""
    cf = CloudFiles(EDGES_SRC)
    prefix = f"task_{wave}_"
    files = [f for f in cf.list() if f.startswith(prefix)]
    files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]))
    return [f"{EDGES_SRC}/{f}" for f in files]


def list_all_waves() -> list[int]:
    """List all wave indices available."""
    cf = CloudFiles(EDGES_SRC)
    waves = set()
    for f in cf.list():
        parts = f.replace(".edges", "").split("_")
        if len(parts) >= 3 and parts[0] == "task":
            waves.add(int(parts[1]))
    return sorted(waves)
