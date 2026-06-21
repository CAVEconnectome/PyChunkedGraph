"""Public API for the OCDBT-backed segmentation store.

See ``main.py`` for the architectural notes. This module just re-exports
the names that external callers (ingest, edits, runtime, tests) reach for.
"""

from .meta import OcdbtConfig
from .utils import (
    _layer_bbox,
    _read_source_scales,
    base_exists,
    fork_exists,
    is_chunk_populated,
    mark_chunk_populated,
    read_populate_meta,
    write_populate_meta,
)
from .main import (
    _mode_downsample,
    build_cg_ocdbt_spec,
    copy_ws_bbox_multiscale,
    copy_ws_chunk,
    copy_ws_chunk_multiscale,
    create_base_ocdbt,
    ensure_fork_synced,
    fork_base_manifest,
    get_seg_source_and_destination_ocdbt,
    open_base_ocdbt,
    propagate_to_coarser_scales,
    wipe_base_ocdbt,
    write_seg_chunks,
)

__all__ = [
    "OcdbtConfig",
    "_layer_bbox",
    "_mode_downsample",
    "_read_source_scales",
    "base_exists",
    "build_cg_ocdbt_spec",
    "copy_ws_bbox_multiscale",
    "copy_ws_chunk",
    "copy_ws_chunk_multiscale",
    "create_base_ocdbt",
    "ensure_fork_synced",
    "fork_base_manifest",
    "fork_exists",
    "get_seg_source_and_destination_ocdbt",
    "is_chunk_populated",
    "mark_chunk_populated",
    "open_base_ocdbt",
    "propagate_to_coarser_scales",
    "read_populate_meta",
    "wipe_base_ocdbt",
    "write_populate_meta",
    "write_seg_chunks",
]
