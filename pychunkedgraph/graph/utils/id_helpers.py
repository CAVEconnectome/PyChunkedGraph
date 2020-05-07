"""
Utils functions for node and segment IDs.
"""

from typing import Optional

import numpy as np

from . import basetypes
from ..meta import ChunkedGraphMeta
from ..chunks import utils as chunk_utils


def get_segment_id_limit(
    meta: ChunkedGraphMeta, node_or_chunk_id: basetypes.CHUNK_ID
) -> basetypes.SEGMENT_ID:
    """Get maximum possible Segment ID for given Node ID or Chunk ID."""
    layer = chunk_utils.get_chunk_layer(meta, node_or_chunk_id)
    chunk_offset = 64 - meta.graph_config.LAYER_ID_BITS - 3 * meta.bitmasks[layer]
    return np.uint64(2 ** chunk_offset - 1)


def get_segment_id(
    meta: ChunkedGraphMeta, node_id: basetypes.NODE_ID
) -> basetypes.SEGMENT_ID:
    """Extract Segment ID from Node ID."""
    return node_id & get_segment_id_limit(meta, node_id)


def get_node_id(
    meta: ChunkedGraphMeta,
    segment_id: basetypes.SEGMENT_ID,
    chunk_id: basetypes.CHUNK_ID = None,
    layer: int = None,
    x: int = None,
    y: int = None,
    z: int = None,
) -> basetypes.NODE_ID:
    """
    (1) Build Node ID from Segment ID and Chunk ID
    (2) Build Node ID from Segment ID, Layer, X, Y and Z components
    """
    if chunk_id is not None:
        return chunk_id | segment_id
    else:
        return chunk_utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z) | segment_id


def get_atomic_id_from_coord(
    meta: ChunkedGraphMeta,
    get_root: callable,
    x: int,
    y: int,
    z: int,
    parent_id: np.uint64,
    n_tries: int = 5,
) -> np.uint64:
    """Determines atomic id given a coordinate."""
    x = int(x / 2 ** meta.data_source.CV_MIP)
    y = int(y / 2 ** meta.data_source.CV_MIP)

    checked = []
    atomic_id = None
    root_id = get_root(parent_id)

    for i_try in range(n_tries):
        # Define block size -- increase by one each try
        x_l = x - (i_try - 1) ** 2
        y_l = y - (i_try - 1) ** 2
        z_l = z - (i_try - 1) ** 2

        x_h = x + 1 + (i_try - 1) ** 2
        y_h = y + 1 + (i_try - 1) ** 2
        z_h = z + 1 + (i_try - 1) ** 2

        x_l = 0 if x_l < 0 else x_l
        y_l = 0 if y_l < 0 else y_l
        z_l = 0 if z_l < 0 else z_l

        # Get atomic ids from cloudvolume
        atomic_id_block = meta.cv[x_l:x_h, y_l:y_h, z_l:z_h]
        atomic_ids, atomic_id_count = np.unique(atomic_id_block, return_counts=True)

        # sort by frequency and discard those ids that have been checked
        # previously
        sorted_atomic_ids = atomic_ids[np.argsort(atomic_id_count)]
        sorted_atomic_ids = sorted_atomic_ids[~np.in1d(sorted_atomic_ids, checked)]

        # For each candidate id check whether its root id corresponds to the
        # given root id
        for candidate_atomic_id in sorted_atomic_ids:
            ass_root_id = get_root(candidate_atomic_id)
            if ass_root_id == root_id:
                # atomic_id is not None will be our indicator that the
                # search was successful
                atomic_id = candidate_atomic_id
                break
            else:
                checked.append(candidate_atomic_id)
        if atomic_id is not None:
            break
    # Returns None if unsuccessful
    return atomic_id
