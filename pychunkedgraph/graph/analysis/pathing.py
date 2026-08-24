import os
import typing

import graph_tool
import numpy as np

from pychunkedgraph.graph.utils import flatgraph

from .. import exceptions as cg_exceptions
from ..edges import in_sorted
from ..subgraph import get_subgraph_nodes


def get_first_shared_parent(
    cg, first_node_id: np.uint64, second_node_id: np.uint64, time_stamp=None
):
    """
    Get the common parent of first_node_id and second_node_id with the lowest layer.
    Returns None if the two nodes belong to different root ids.
    :param first_node_id: np.uint64
    :param second_node_id: np.uint64
    :return: np.uint64 or None
    """
    first_node_parent_ids = set()
    second_node_parent_ids = set()
    cur_first_node_parent = first_node_id
    cur_second_node_parent = second_node_id
    while cur_first_node_parent is not None or cur_second_node_parent is not None:
        if cur_first_node_parent is not None:
            first_node_parent_ids.add(cur_first_node_parent)
        if cur_second_node_parent is not None:
            second_node_parent_ids.add(cur_second_node_parent)
        if cur_first_node_parent in second_node_parent_ids:
            return cur_first_node_parent
        if cur_second_node_parent in first_node_parent_ids:
            return cur_second_node_parent
        if cur_first_node_parent is not None:
            cur_first_node_parent = cg.get_parent(
                cur_first_node_parent, time_stamp=time_stamp
            )
        if cur_second_node_parent is not None:
            cur_second_node_parent = cg.get_parent(
                cur_second_node_parent, time_stamp=time_stamp
            )
    return None


def get_children_at_layer(
    cg,
    agglomeration_id: np.uint64,
    layer: int,
    allow_lower_layers: bool = False,
):
    """
    Get the children of agglomeration_id that have layer = layer.
    :param agglomeration_id: np.uint64
    :param layer: int
    :return: [np.uint64]
    """
    nodes_to_query = [agglomeration_id]
    children_at_layer = []
    while True:
        children = cg.get_children(nodes_to_query, flatten=True)
        children_layers = cg.get_chunk_layers(children)
        if allow_lower_layers:
            stop_layer_mask = children_layers <= layer
        else:
            stop_layer_mask = children_layers == layer
        continue_layer_mask = children_layers > layer
        found_children_at_layer = children[stop_layer_mask]
        children_at_layer.append(found_children_at_layer)
        nodes_to_query = children[continue_layer_mask]
        if not np.any(nodes_to_query):
            break
    return np.concatenate(children_at_layer)


def get_lvl2_edge_list(
    cg,
    node_id: np.uint64,
    bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
    max_num_lvl2_ids: typing.Optional[int] = None,
):
    """get an edge list of lvl2 ids for a particular node

    :param cg: ChunkedGraph object
    :param node_id: np.uint64 that you want the edge list for
    :param bbox: Optional[Sequence[Sequence[int]]] a bounding box to limit the search
    :param max_num_lvl2_ids: Optional[int] reject the request (raising BadRequest) when the
        node resolves to more than this many level 2 ids. Guards against pathologically large
        objects (e.g. erroneous mega-merges) whose induced level 2 edge list would be many GB
        and can OOM the worker. ``None`` disables the guard.
    """

    if bbox is None:
        # maybe temporary, this was the old implementation
        lvl2_ids = get_children_at_layer(cg, node_id, 2)
    else:
        lvl2_ids = get_subgraph_nodes(
            cg,
            node_id,
            bbox=bbox,
            bbox_is_coordinate=True,
            return_layers=[2],
            return_flattened=True,
        )

    # Enforce the size guard *before* the (potentially multi-GB) induced-edge computation
    # below. The level 2 id count is the cheap proxy we already have in hand; the edge read
    # in _get_edges_for_lvl2_ids scales with it and is what actually exhausts memory.
    if max_num_lvl2_ids is not None and len(lvl2_ids) > max_num_lvl2_ids:
        hint = (
            "Provide a smaller bounding box ('bounds')."
            if bbox is not None
            else "Provide a bounding box ('bounds') to restrict the query to a sub-region."
        )
        raise cg_exceptions.BadRequest(
            f"The level 2 graph for {node_id} has {len(lvl2_ids)} level 2 nodes, which exceeds "
            f"the maximum of {max_num_lvl2_ids}. {hint}"
        )

    edges = _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True)
    return edges


# Level 2 ids per call to cg.get_atomic_cross_edges. The read is the one thing here that
# scales with object size rather than with the answer, so it is sliced. Swept on the
# profiled object below (peak rss): 3k/5k/10k all land within noise of each other, 25k
# drifts up. Flat enough that this is not a knife edge, so the default rarely needs
# changing -- PCG_CROSS_EDGE_READ_BATCH is there for tuning a deployment whose objects or
# pod memory differ, without a redeploy of the code.
try:
    CROSS_EDGE_READ_BATCH = int(os.environ.get("PCG_CROSS_EDGE_READ_BATCH", 10_000))
except ValueError:
    CROSS_EDGE_READ_BATCH = 10_000
if CROSS_EDGE_READ_BATCH < 1:
    # Zero or negative would make the range() below yield no slices at all, so the
    # function would return an empty edge list instead of failing: a wrong answer rather
    # than an error. Ignore it and keep the default.
    CROSS_EDGE_READ_BATCH = 10_000


def _sorted_by_key(keys, values):
    """Sort a key/value pair of arrays together, by key.

    Each slice contributes a sorted fragment but their ranges interleave, so the
    concatenation is not sorted. Both the miss scan and the remap binary-search it.
    """
    order = np.argsort(keys, kind="stable")
    return keys[order], values[order]


def _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=False, batch_size=None):
    """Cross-chunk edges between `lvl2_ids`, remapped from supervoxels to level 2 ids.

    Memory, not speed, is the binding constraint: the intermediates dwarf the answer.
    Profiled on api6 for aibs_v1dd root 864691132764660103, the object that trips uwsgi's
    reload-on-rss: 240,928 level 2 ids whose cross edges are 18,349,472 supervoxel pairs
    (280MB), collapsing to 276,472 level 2 edges (4.4MB).

    The costly part is not the edges but the dict they arrive in -- one small array per
    (level 2 id, layer), ~480k of them, 930MB for 280MB of data, nearly all numpy object
    overhead. So the read is sliced and each slice is concatenated and its dict dropped
    before the next is fetched, which keeps only one slice's worth of that overhead alive.
    The edges themselves are all still held, so this is still a single pass over Bigtable.

    Measured in the read pod against the previous single-shot version: peak rss
    1861-1887MB -> 933-952MB, wall clock 33.3-35.3s -> 30.3-31.0s, output byte-identical.
    It gets faster rather than slower because the working set shrinks and because of the
    miss scan below.

    The read slice size defaults to CROSS_EDGE_READ_BATCH (the PCG_CROSS_EDGE_READ_BATCH
    environment variable, 10,000) and can be overridden per call with `batch_size`. It
    changes the peak and nothing else -- the result is independent of it.

    Two things that look like obvious wins here and are not, both measured:
      - Slicing the *reduce* as well, so the edges need not be held either, drops the peak
        to ~700MB but needs a second pass over Bigtable and costs 7s. Not worth it.
      - Swapping fastremap for searchsorted on its own, without slicing the read, is a
        wash (1834-1869MB). It pays only in combination, because the reduce loop below
        would otherwise rebuild fastremap's 6.95M-key table once per slice.
    """
    # protect in case there are no lvl2 ids
    if len(lvl2_ids) == 0:
        return np.empty((0, 2), dtype=np.uint64)

    # Sorted and deduplicated once, up front. It is the reference set for the induced
    # filter below, and slicing it keeps each supervoxel's mapping entry in exactly one
    # slice. Reading a dict keyed by id already collapsed duplicates, so this does not
    # change which edges are fetched.
    lvl2_ids = np.unique(lvl2_ids)
    if batch_size is None:
        batch_size = CROSS_EDGE_READ_BATCH
    elif batch_size < 1:
        # Same trap as above, reached explicitly instead of through the environment.
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batches = []
    map_keys = []
    map_values = []
    partner_ids = []
    for start in range(0, len(lvl2_ids), batch_size):
        cce_dict = cg.get_atomic_cross_edges(lvl2_ids[start : start + batch_size])
        blocks = []
        owners = []
        lengths = []
        for lvl2_id in cce_dict:
            for level in cce_dict[lvl2_id]:
                block = cce_dict[lvl2_id][level]
                if len(block) == 0:
                    continue
                blocks.append(block)
                owners.append(lvl2_id)
                lengths.append(len(block))
        # The whole point: the slice's edges are copied out by the concatenate, so its
        # dict can go before the next read allocates another one.
        cce_dict.clear()
        if not blocks:
            continue
        edges = np.concatenate(blocks)
        del blocks

        # Column 0 is a supervoxel of the block's owner, column 1 its partner across the
        # chunk boundary; both are views into `edges`. One np.repeat gives the parent of
        # every row, rather than an np.full per block.
        owner_column = np.repeat(
            np.array(owners, dtype=edges.dtype), np.array(lengths, dtype=np.int64)
        )
        keys, first = np.unique(edges[:, 0], return_index=True)
        map_keys.append(keys)
        map_values.append(owner_column[first])
        partner_ids.append(np.unique(edges[:, 1]))
        batches.append(edges)
        del edges, owner_column, keys, first

    # protect in case there are no edges
    if not batches:
        return np.empty((0, 2), dtype=np.uint64)

    # A supervoxel belongs to exactly one level 2 node, so the fragments are disjoint.
    known_supervoxel_array = np.concatenate(map_keys)
    known_l2_array = np.concatenate(map_values)
    del map_keys, map_values
    known_supervoxel_array, known_l2_array = _sorted_by_key(
        known_supervoxel_array, known_l2_array
    )

    # Partners with no mapping entry belong to level 2 nodes outside `lvl2_ids` and have
    # to be looked up. For a whole object there are none -- cross edges are stored from
    # both sides, so every partner also appears in some slice's column 0 -- but for a
    # subset (a 'bounds' query, or a shared parent below the root) there are.
    #
    # Scanned one slice at a time on purpose. Concatenating every slice's partners and
    # handing that to np.setdiff1d, which sorts both sides again, cost 477MB on the
    # profiled object to produce an empty answer.
    misses = []
    for partners in partner_ids:
        missed = partners[~in_sorted(partners, known_supervoxel_array)]
        if len(missed):
            misses.append(missed)
    del partner_ids
    if misses:
        supervoxels_to_query_parent = np.unique(np.concatenate(misses))
        del misses
        missing_l2_ids = cg.get_parents(supervoxels_to_query_parent)
        known_supervoxel_array = np.concatenate(
            (known_supervoxel_array, supervoxels_to_query_parent)
        )
        known_l2_array = np.concatenate((known_l2_array, missing_l2_ids))
        del supervoxels_to_query_parent, missing_l2_ids
        known_supervoxel_array, known_l2_array = _sorted_by_key(
            known_supervoxel_array, known_l2_array
        )

    # Reduce each slice to level 2 pairs and free it before moving on, so the 66:1
    # collapse happens per slice and the accumulator stays at answer size.
    reduced = []
    while batches:
        edges = batches.pop()
        # Every supervoxel here is in the mapping -- column 0 by construction, column 1
        # via the get_parents step above -- so a binary search is exact. searchsorted
        # rather than fastremap.remap_from_array_kv, which would rebuild a lookup over
        # the whole mapping on every iteration of this loop.
        flat = edges.reshape(-1)
        flat[:] = known_l2_array[np.searchsorted(known_supervoxel_array, flat)]
        del flat
        # In place: `edges` is ours, np.concatenate having copied it out of the dict.
        edges.sort(axis=1)
        if induced:
            # make this an induced subgraph
            # keep only the edges that are between the lvl2 ids asked for. Filtering
            # before the dedup is worth it when it bites -- a subset query -- and costs
            # little when it does not. in_sorted rather than np.isin, which would re-sort
            # lvl2_ids on each of the two calls.
            edges = edges[
                in_sorted(edges[:, 0], lvl2_ids) & in_sorted(edges[:, 1], lvl2_ids)
            ]
        reduced.append(np.unique(edges, axis=0))
        del edges

    del known_supervoxel_array, known_l2_array
    if len(reduced) == 1:
        return reduced[0]
    # Slices reduce independently, so the same level 2 pair can come out of more than one
    # of them; this is what makes the result independent of batch_size.
    return np.unique(np.concatenate(reduced), axis=0)


def find_l2_shortest_path(
    cg, source_l2_id: np.uint64, target_l2_id: np.uint64, time_stamp=None
):
    """
    Find a path of level 2 ids that connect two level 2 node ids through cross chunk edges.
    Return a list of level 2 ids representing this path.
    Return None if the two level 2 ids do not belong to the same object.
    :param cg: ChunkedGraph object
    :param source_l2_id: np.uint64
    :param target_l2_id: np.uint64
    :return: [np.uint64] or None
    """
    # Get the cross-chunk edges that we need to build the graph
    shared_parent_id = get_first_shared_parent(
        cg, source_l2_id, target_l2_id, time_stamp
    )
    if shared_parent_id is None:
        return None

    edge_array = get_lvl2_edge_list(cg, shared_parent_id)
    # Create a graph-tool graph of the mapped cross-chunk-edges
    weighted_graph, _, _, graph_indexed_l2_ids = flatgraph.build_gt_graph(
        edge_array, is_directed=False
    )

    # Find the shortest path from the source_l2_id to the target_l2_id
    source_graph_id = np.where(graph_indexed_l2_ids == source_l2_id)[0][0]
    target_graph_id = np.where(graph_indexed_l2_ids == target_l2_id)[0][0]
    source_vertex = weighted_graph.vertex(source_graph_id)
    target_vertex = weighted_graph.vertex(target_graph_id)
    vertex_list, _ = graph_tool.topology.shortest_path(
        weighted_graph, source=source_vertex, target=target_vertex
    )

    # Remap the graph-tool ids to lvl2 ids and return the path
    vertex_indices = [weighted_graph.vertex_index[vertex] for vertex in vertex_list]
    l2_traversal_path = graph_indexed_l2_ids[vertex_indices]
    return l2_traversal_path


def compute_rough_coordinate_path(cg, l2_ids):
    """
    Given a list of l2_ids, return a list of rough coordinates representing
    the path the l2_ids form.
    :param cg: ChunkedGraph object
    :param l2_ids: Sequence[np.uint64]
    :return: [np.ndarray]
    """
    coordinate_path = []
    for l2_id in l2_ids:
        chunk_center = cg.get_chunk_coordinates(l2_id) + np.array([0.5, 0.5, 0.5])
        coordinate = chunk_center * np.array(
            cg.meta.graph_config.CHUNK_SIZE
        ) + np.array(cg.meta.cv.mip_voxel_offset(0))
        coordinate = coordinate * np.array(cg.meta.cv.mip_resolution(0))
        coordinate = coordinate.astype(np.float32)
        coordinate_path.append(coordinate)
    return coordinate_path
