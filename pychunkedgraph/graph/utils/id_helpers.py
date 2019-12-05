from typing import Optional

import numpy as np

from ..meta import ChunkedGraphMeta
from ..chunks import utils as chunk_utils


def get_segment_id_limit(
    meta: ChunkedGraphMeta, node_or_chunk_id: np.uint64
) -> np.uint64:
    """ Get maximum possible Segment ID for given Node ID or Chunk ID
    :param node_or_chunk_id: np.uint64
    :return: np.uint64
    """
    layer = chunk_utils.get_chunk_layer(meta, node_or_chunk_id)
    chunk_offset = 64 - meta.graph_config.LAYER_ID_BITS - 3 * meta.bitmasks[layer]
    return np.uint64(2 ** chunk_offset - 1)


def get_segment_id(meta: ChunkedGraphMeta, node_id: np.uint64) -> np.uint64:
    """ Extract Segment ID from Node ID
    :param node_id: np.uint64
    :return: np.uint64
    """
    return node_id & get_segment_id_limit(meta, node_id)


def get_node_id(
    meta: ChunkedGraphMeta,
    segment_id: np.uint64,
    chunk_id: Optional[np.uint64] = None,
    layer: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
) -> np.uint64:
    """
    (1) Build Node ID from Segment ID and Chunk ID
    (2) Build Node ID from Segment ID, Layer, X, Y and Z components
    """
    if chunk_id is not None:
        return chunk_id | segment_id
    else:
        return chunk_utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z) | segment_id
