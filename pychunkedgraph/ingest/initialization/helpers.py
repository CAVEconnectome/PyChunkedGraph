from typing import Sequence

import numpy as np

from ...backend import ChunkedGraphMeta
from ...backend.chunkedgraph_utils import get_children_chunk_coords


def get_touching_atomic_chunks(
    chunkedgraph_meta: ChunkedGraphMeta, layer: int, chunk_coords: Sequence[int]
):
    """get atomic chunks along touching faces of children chunks of a parent chunk"""

    parent_atomic_chunk_count = chunkedgraph_meta.graph_config.fanout ** (layer - 2)

    layer2_chunk_boundaries = chunkedgraph_meta.layer_chunk_bounds[2]
    children_layer = layer - 1



