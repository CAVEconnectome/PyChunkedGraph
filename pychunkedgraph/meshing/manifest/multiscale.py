# pylint: disable=invalid-name, missing-docstring, line-too-long, no-member

import functools
from collections import deque

import numpy as np
from cloudvolume import CloudVolume

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.types import empty_1d
from pychunkedgraph.graph.utils.basetypes import NODE_ID
from .cache import ManifestCache
from .sharded import normalize_fragments
from .utils import del_none_keys
from ..meshgen_utils import get_json_info

OCTREE_NODE_SIZE = 5


def _get_hierarchy(cg: ChunkedGraph, node_id: np.uint64) -> dict:
    children_map = {}
    layer = cg.get_chunk_layer(node_id)
    if layer < 2:
        return children_map
    if layer == 2:
        children_map[node_id] = empty_1d.copy()
        return children_map

    node_ids = np.array([node_id], dtype=NODE_ID)
    while node_ids.size > 0:
        children = cg.get_children(node_ids)
        children_map.update(children)

        _ids = np.concatenate(list(children.values())) if children else empty_1d.copy()
        node_layers = cg.get_chunk_layers(_ids)
        node_ids = _ids[node_layers > 2]

        for l2id in _ids[node_layers == 2]:
            children_map[l2id] = empty_1d.copy()
    return children_map


def _get_node_coords_and_layers_map(
    cg: ChunkedGraph, children_map: dict
) -> tuple[dict, dict]:
    node_ids = np.fromiter(children_map.keys(), dtype=NODE_ID)
    coords_map = {}
    node_layers = cg.get_chunk_layers(node_ids)
    for layer in set(node_layers):
        layer_mask = node_layers == layer
        coords = cg.get_chunk_coordinates_multiple(node_ids[layer_mask])
        _node_coords = dict(zip(node_ids[layer_mask], coords))
        coords_map.update(_node_coords)
    return coords_map, dict(zip(node_ids, node_layers))


def _insert_skipped_nodes(
    cg: ChunkedGraph, children_map: dict, coords_map: dict, layers_map: dict
):
    new_children_map = {}
    for node, children in children_map.items():
        nl = layers_map[node]
        if len(children) > 1 or nl == 2:
            new_children_map[node] = children
        else:
            assert (
                len(children) == 1
            ), f"Skipped hierarchy must have exactly 1 child: {node} - {children}."
            cl = layers_map[children[0]]
            layer_diff = nl - cl
            if layer_diff == 1:
                new_children_map[node] = children
                continue

            cx, cy, cz = coords_map[children[0]]
            skipped_hierarchy = [node]
            count = 1
            while layer_diff:
                height = layer_diff - 1
                x, y, z = cx >> height, cy >> height, cz >> height
                skipped_layer = nl - count
                skipped_child = cg.get_chunk_id(layer=skipped_layer, x=x, y=y, z=z)
                skipped_hierarchy.append(skipped_child)
                coords_map[skipped_child] = np.array((x, y, z), dtype=int)
                layers_map[skipped_child] = skipped_layer
                count += 1
                layer_diff -= 1
            skipped_hierarchy.append(children[0])

            for i in range(len(skipped_hierarchy) - 1):
                node = skipped_hierarchy[i]
                if node in new_children_map:
                    node += 1
                child = skipped_hierarchy[i + 1]
                if child in new_children_map:
                    child += 1
                new_children_map[node] = np.array([child], dtype=NODE_ID)
    return new_children_map, coords_map, layers_map


def _sort_octree_row(cg: ChunkedGraph, children: np.ndarray):
    """
    Sort children by their morton code.
    """
    if children.size == 0:
        return children
    children_coords = []

    for child in children:
        children_coords.append(cg.get_chunk_coordinates(child))

    def cmp_zorder(lhs, rhs) -> bool:
        # https://en.wikipedia.org/wiki/Z-order_curve
        # https://github.com/google/neuroglancer/issues/272
        def less_msb(x: int, y: int) -> bool:
            return x < y and x < (x ^ y)

        msd = 2
        for dim in [1, 0]:
            if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
                msd = dim
        return lhs[msd] - rhs[msd]

    children, _ = zip(
        *sorted(
            zip(children, children_coords),
            key=functools.cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1])),
        )
    )
    return children


def _validate_octree(octree: np.ndarray, octree_node_ids: np.ndarray):
    assert octree.size % 5 == 0, "Invalid octree size."
    num_nodes = octree.size // 5
    seen_nodes = set()

    def _explore_node(node: int):
        if node in seen_nodes:
            raise ValueError(f"Previsouly seen node {node}.")
        seen_nodes.add(node)

        if node < 0 or node >= num_nodes:
            raise ValueError(f"Invalid node reference {node}.")

        x, y, z = octree[node * 5 : node * 5 + 3]
        child_begin = octree[node * 5 + 3] & ~(1 << 31)
        child_end = octree[node * 5 + 4] & ~(1 << 31)
        p = octree_node_ids[node]

        if (
            child_begin < 0
            or child_end < 0
            or child_end < child_begin
            or child_end > num_nodes
        ):
            print(octree[node * 5 : node * 5 + 5])
            raise ValueError(
                f"Invalid child references: {(node, p)} specifies child_begin={child_begin} and child_end={child_end}."
            )

        for child in range(child_begin, child_end):
            cx, cy, cz = octree[child * 5 : child * 5 + 3]
            c = octree_node_ids[child]
            msg = f"Invalid child position: parent {(node, p)} at {(x, y, z)}, child {(child, c)} at {(cx, cy ,cz)}."
            assert cx >> 1 == x and cy >> 1 == y and cz >> 1 == z, msg
            _explore_node(child)

    if num_nodes == 0:
        return
    _explore_node(num_nodes - 1)
    if len(seen_nodes) != num_nodes:
        raise ValueError(f"Orphan nodes in tree {num_nodes - len(seen_nodes)}")


def build_octree(
    cg: ChunkedGraph, node_id: np.uint64, children_map: dict, mesh_fragments: dict
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
    node_q = deque()
    node_q.append(node_id)
    coords_map, layers_map = _get_node_coords_and_layers_map(cg, children_map)
    children_map, coords_map, layers_map = _insert_skipped_nodes(
        cg, children_map, coords_map, layers_map
    )

    ROW_TOTAL = len(children_map)
    row_counter = len(children_map)
    octree_size = OCTREE_NODE_SIZE * ROW_TOTAL
    octree = np.zeros(octree_size, dtype=np.uint32)

    octree_node_ids = ROW_TOTAL * [0]
    octree_fragments = ROW_TOTAL * [""]
    rows_used = 1

    while len(node_q) > 0:
        row_counter -= 1
        current_node = node_q.popleft()
        children = children_map[current_node]
        x, y, z = coords_map[current_node]

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
                # mark node virtual
                octree[offset + 3] |= 1 << 31
            else:
                octree_fragments[row_counter] = mesh_fragments[current_node]
        except KeyError:
            # no mesh, mark node empty
            octree[offset + 4] |= 1 << 31

        children = _sort_octree_row(cg, children)
        for child in children:
            if layers_map[child] > 2:
                node_q.append(child)

    _validate_octree(octree, octree_node_ids)
    return octree, octree_node_ids, octree_fragments


def get_manifest(cg: ChunkedGraph, node_id: np.uint64) -> dict:
    children_map = _get_hierarchy(cg, node_id)
    node_ids = np.fromiter(children_map.keys(), dtype=NODE_ID)
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

    octree, node_ids, fragments = build_octree(cg, node_id, children_map, fragments_d)
    max_layer = min(cg.get_chunk_layer(node_id) + 1, cg.meta.layer_count)

    chunk_shape = np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=np.dtype("<f4"))
    chunk_shape *= cg.meta.resolution
    clip_bounds = cg.meta.voxel_bounds.T * cg.meta.resolution
    response = {
        "chunkShape": chunk_shape,
        "chunkGridSpatialOrigin": np.array([0, 0, 0], dtype=np.dtype("<f4")),
        "lodScales": np.arange(max_layer - 2, dtype=np.dtype("<f4")) * 1,
        "fragments": normalize_fragments(fragments),
        "octree": octree,
        "clipLowerBound": np.array(clip_bounds[0], dtype=np.dtype("<f4")),
        "clipUpperBound": np.array(clip_bounds[1], dtype=np.dtype("<f4")),
    }
    return node_ids, response
