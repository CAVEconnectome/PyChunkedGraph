# pylint: disable=invalid-name, missing-docstring, line-too-long, no-member

from collections import deque
from typing import Dict, Set, Tuple

import numpy as np
from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.types import empty_1d
from pychunkedgraph.graph.utils.basetypes import NODE_ID
from .cache import ManifestCache
from .sharded import normalize_fragments
from .utils import del_none_keys
from ..meshgen_utils import get_json_info


def _get_hierarchy(cg: ChunkedGraph, node_id: NODE_ID) -> Dict:
    node_children = {}
    layer = cg.get_chunk_layer(node_id)
    if layer < 2:
        return node_children
    if layer == 2:
        node_children[node_id] = empty_1d.copy()
        return node_children

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


def _get_skipped_and_missing_leaf_nodes(
    node_children: Dict, mesh_fragments: Dict
) -> Tuple[Set, Set]:
    """
    Returns nodes with only one child and leaves (l2ids).
    Nodes with one child do not have a mesh fragment, because it would be identical to child fragment.
    Leaves are used to determine correct size for the octree.
    """
    skipped = set()
    leaves = set()
    for node_id, children in node_children.items():
        if children.size == 1:
            skipped.add(node_id)
        elif children.size == 0 and not node_id in mesh_fragments:
            leaves.add(node_id)
    return skipped, leaves


def _get_node_coords_and_layers_map(
    cg: ChunkedGraph, node_children: Dict
) -> Tuple[Dict, Dict]:
    node_ids = np.fromiter(node_children.keys(), dtype=NODE_ID)
    node_coords = {}
    node_layers = cg.get_chunk_layers(node_ids)
    for layer in set(node_layers):
        layer_mask = node_layers == layer
        coords = cg.get_chunk_coordinates_multiple(node_ids[layer_mask])
        _node_coords = dict(zip(node_ids[layer_mask], coords))
        node_coords.update(_node_coords)
    return node_coords, dict(zip(node_ids, node_layers))


def _normalize_coordinates(coords, layer, bfs_depth, max_layer):
    node_depth = max_layer - layer
    depth_diff = node_depth - bfs_depth
    return coords // 2**depth_diff


def build_octree(
    cg: ChunkedGraph, node_id: NODE_ID, node_children: Dict, mesh_fragments: Dict
):
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
    node_ids = np.fromiter(mesh_fragments.keys(), dtype=NODE_ID)
    node_coords_d, node_layers_d = _get_node_coords_and_layers_map(cg, node_children)
    skipped, leaves = _get_skipped_and_missing_leaf_nodes(node_children, mesh_fragments)

    OCTREE_NODE_SIZE = 5
    ROW_TOTAL = len(node_ids) + len(skipped) + len(leaves) + 1
    row_counter = len(node_ids) + len(skipped) + len(leaves) + 1
    octree_size = OCTREE_NODE_SIZE * ROW_TOTAL
    octree = np.zeros(octree_size, dtype=np.uint32)

    octree_node_ids = ROW_TOTAL * [0]
    octree_fragments = ROW_TOTAL * [""]

    que = deque()
    rows_used = 1
    que.append((node_id, 0))

    while len(que) > 0:
        row_counter -= 1
        current_node, depth = que.popleft()
        children = node_children[current_node]
        node_layer = node_layers_d[current_node]
        node_coords = node_coords_d[current_node]

        # node_coords = _normalize_coordinates(
        #     coords=node_coords,
        #     layer=node_layer,
        #     bfs_depth=depth,
        #     max_layer=cg.meta.layer_count,
        # )

        x, y, z = node_coords
        # x, y, z = node_coords * np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=int)
        offset = OCTREE_NODE_SIZE * row_counter
        octree[offset + 0] = x
        octree[offset + 1] = y
        octree[offset + 2] = z

        rows_used += children.size
        start = ROW_TOTAL - rows_used
        end_empty = start + children.size

        octree[offset + 3] = start
        octree[offset + 4] = end_empty

        octree_node_ids[row_counter] = current_node
        try:
            if children.size == 1:
                # map to child fragment
                octree_fragments[row_counter] = mesh_fragments[children[0]]
            else:
                octree_fragments[row_counter] = mesh_fragments[current_node]
        except KeyError:
            # no mesh, mark node empty
            octree[offset + 4] |= 1 << 31

        for child in children:
            que.append((child, depth + 1))
    return octree, octree_node_ids, octree_fragments


def get_manifest(cg: ChunkedGraph, node_id: NODE_ID) -> Dict:
    node_children = _get_hierarchy(cg, node_id)
    node_ids = np.fromiter(node_children.keys(), dtype=NODE_ID)
    manifest_cache = ManifestCache(cg.graph_id, initial=True)

    cv = CloudVolume(
        "graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=cg.meta.custom_data.get("mesh", {}).get("dir", "graphene_meshes"),
        info=get_json_info(cg),
        progress=False,
    )

    fragments_d, _not_cached, _ = manifest_cache.get_fragments(node_ids)
    initial_meshes = cv.mesh.initial_exists(_not_cached, return_byte_range=True)
    _fragments_d, _ = del_none_keys(initial_meshes)
    manifest_cache.set_fragments(_fragments_d)
    fragments_d.update(_fragments_d)

    octree, node_ids, fragments = build_octree(cg, node_id, node_children, fragments_d)
    max_layer = min(cg.get_chunk_layer(node_id) + 1, cg.meta.layer_count)

    response = {
        "chunkShape": np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=np.dtype("<f4")),
        "chunkGridSpatialOrigin": np.array([0, 0, 0], dtype=np.dtype("<f4")),
        "lodScales": 2 ** np.arange(max_layer, dtype=np.dtype("<f4")) * 1,
        "fragments": normalize_fragments(fragments),
        "octree": octree,
    }
    return node_ids, response
