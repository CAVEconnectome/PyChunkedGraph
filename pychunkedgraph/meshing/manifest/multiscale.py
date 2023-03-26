# pylint: disable=invalid-name, missing-docstring, line-too-long

from collections import deque
from typing import Dict

import numpy as np

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.types import empty_1d
from pychunkedgraph.graph.utils.basetypes import NODE_ID


def _get_hierarchy(cg: ChunkedGraph, node_id: NODE_ID) -> Dict:
    node_children = {}
    node_ids = np.array([node_id], dtype=NODE_ID)
    while node_ids.size > 0:
        children = cg.get_children(node_ids)
        node_children.update(children)

        _ids = np.concatenate(list(children.values())) if children else empty_1d.copy()
        node_layers = cg.get_chunk_layers(_ids)
        node_ids = _ids[node_layers > 2]

        for l2id in _ids[node_layers == 2]:
            node_children[l2id] = empty_1d.copy()
    return node_children


def build_octree(cg: ChunkedGraph, node_id: NODE_ID):
    """
    From neuroglancer multiscale specification:
      Row-major `[n, 5]` array where each row is of the form `[x, y, z, start, end_and_empty]`, where
      `x`, `y`, and `z` are the chunk grid coordinates of the entry at a particular level of detail.
      Row `n-1` corresponds to level of detail `lodScales.length - 1`, the root of the octree.  Given
      a row corresponding to an octree node at level of detail `lod`, bits `start` specifies the row
      number of the first child octree node at level of detail `lod-1`, and bits `[0,30]` of
      `end_and_empty` specify one past the row number of the last child octree node.  Bit `31` of
      `end_and_empty` is set to `1` if the mesh for the octree node is empty and should not be
      requested/rendered.
    """
    node_children = _get_hierarchy(cg, node_id)
    node_ids = np.array(list(node_children.keys()), dtype=NODE_ID)

    node_coords = {}
    node_layers = cg.get_chunk_layers(node_ids)
    for layer in set(node_layers):
        layer_mask = node_layers == layer
        coords = cg.get_chunk_coordinates_multiple(node_ids[layer_mask])
        _node_coords = dict(zip(node_ids[layer_mask], coords))
        node_coords.update(_node_coords)

    ROW_TOTAL = len(node_ids)
    row_count = len(node_ids)
    octree_size = 5 * row_count
    octree = np.zeros(octree_size, dtype=np.uint32)

    que = deque()
    que.append(node_id)
    rows_used = 1
    while len(que) > 0:
        row_count -= 1
        offset = 5 * row_count
        current_node = que.popleft()

        x, y, z = node_coords[current_node]
        octree[offset + 0] = x
        octree[offset + 1] = y
        octree[offset + 2] = z

        children = node_children[current_node]
        start = 0
        end_empty = 0
        if children.size > 0:
            rows_used += children.size
            start = ROW_TOTAL - rows_used
            end_empty = start + children.size

        octree[offset + 3] = start
        octree[offset + 4] = end_empty

        for child in children:
            que.append(child)
    return octree
