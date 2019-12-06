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
    chunk_id: Optional[basetypes.CHUNK_ID] = None,
    layer: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
) -> basetypes.NODE_ID:
    """
    (1) Build Node ID from Segment ID and Chunk ID
    (2) Build Node ID from Segment ID, Layer, X, Y and Z components
    """
    if chunk_id is not None:
        return chunk_id | segment_id
    else:
        return chunk_utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z) | segment_id
