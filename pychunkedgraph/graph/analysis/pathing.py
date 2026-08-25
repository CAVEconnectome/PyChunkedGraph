import os
import typing

import graph_tool
import numpy as np

from pychunkedgraph.graph.utils import flatgraph

from .. import attributes
from .. import exceptions as cg_exceptions
from .. import lineage
from ..edges import in_sorted
from ..subgraph import get_subgraph_nodes
from ..utils import basetypes


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


def _positive_int_from_env(name, default, minimum=1):
    """Read an int of at least `minimum` from the environment, else `default`.

    Anything unparseable or below `minimum` is ignored rather than honoured. For most of
    these a zero would silently produce an empty read or an empty corridor -- a wrong
    answer instead of an error -- which is why the floor is 1 by default. The dilation
    passes minimum=0, where zero is a meaningful setting rather than a broken one.
    """
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value >= minimum else default


# How many subtrees find_l2_shortest_path aims to cut the object into for its coarse
# graph. The cut is chosen by descending until the frontier is at least this wide, rather
# than by fixing a layer: the hierarchy is not uniform (a layer 7 node can have level 2
# children directly) and layer sizes differ per dataset, so a fixed layer is a constant
# that would need retuning per deployment and would still be wrong for an unusual object.
# Descending to a target width costs nothing extra -- the descent has to happen anyway to
# collect the level 2 ids. On the profiled object this lands on the same cut that layer 5
# gave (16,220 groups against 240,928 level 2 nodes).
FIND_PATH_TARGET_GROUPS = _positive_int_from_env("PCG_FIND_PATH_TARGET_GROUPS", 15_000)

# Objects at or below this many level 2 ids skip the coarse pass entirely. The exact
# search is already cheap there, and the coarse pass would add a second read for nothing.
FIND_PATH_MIN_LVL2_IDS = _positive_int_from_env("PCG_FIND_PATH_MIN_LVL2_IDS", 50_000)

# How many hops of coarse neighbours to include around the coarse path. This is the whole
# safety margin between a near-optimal and a hop-minimal path: a wider corridor is more
# likely to contain the true shortest route, at the cost of reading more of the object.
#
# Swept on api6 aibs_v1dd root 864691132764660103. Timing and memory are one real request
# end to end (deployed baseline 34.6s / 1006MB); the path column is over 9 pairs spanning
# 251 to 1616 hops:
#
#   dilation   corridor    time    peak   exact   worst    note
#          0    1-9% sub  15.0s   410MB    4/9    +4.0%   dominated by 1: no faster
#          1    2-9%      14.7s   441MB    4/9    +4.0%   the default
#          2    2-14%     17.5s   484MB    7/9    +4.0%
#          3    4-20%     18.3s   517MB      -        -   dominated by 2 on most pairs
#          5    6-32%     20.9s   561MB    9/9     0.0%   exactness, at ~2x the corridor
#
# 1 is the default because this endpoint exists to get a user from A to B, not to certify
# a minimum. Paths came in under 1% long on most pairs and 4% at worst, for 2.4x the speed
# and 56% less memory than reading the whole object. Raise it if optimality starts to
# matter -- 5 was exact on every pair measured -- but note that no finite dilation
# guarantees it.
#
# Going below 1 buys nothing. At dilation 1 the corridor is already only ~7% of the
# object, so roughly 9 of those 15 seconds are the fixed cost of the coarse pass -- the
# tree descent plus the CrossChunkEdge[coarse_layer:] read, both O(object) however narrow
# the corridor gets. Narrowing it further trades path quality for noise. That fixed cost
# is also the floor on this approach; removing it needs the coarse graph cached per
# subtree id rather than rebuilt per request.
#
# What is NOT a risk at low dilation: a corridor too narrow to contain any route, which
# would fall back to the full graph and cost more than never pruning. Zero fallbacks over
# those 9 pairs at dilation 0, 1 and 2.
#
# 0 is still accepted rather than silently replaced by the default, since it is a
# meaningful setting -- the coarse path and nothing around it.
FIND_PATH_DILATION = _positive_int_from_env("PCG_FIND_PATH_DILATION", 1, minimum=0)

# Whether a cold object should pay the full edge read once, to seed the root level cache
# that every later call on that neuron patches. Off by default: it roughly doubles the
# first call (the corridor search exists precisely to avoid that read), and a drive-by
# query on a neuron nobody is editing would pay it for nothing. Turn it on where the
# workload is proofreading sessions rather than one-off lookups.
FIND_PATH_SEED_CACHE = os.environ.get("PCG_FIND_PATH_SEED_CACHE", "0") != "0"

# Smallest subtree whose edge list is worth storing when the unpruned search computes one
# anyway. Below this a query is already well under a second and caching it would only
# churn keys; above it, a subtree that bypasses the coarse pass would otherwise stay at
# its first-call cost forever -- 2.1s at 13k level 2 ids, and up to ~8s at the bypass
# threshold itself.
FIND_PATH_CACHE_MIN_LVL2_IDS = _positive_int_from_env(
    "PCG_FIND_PATH_CACHE_MIN_LVL2_IDS", 2_000
)

# Largest object worth seeding an edge list for. Building one is the full read the corridor
# search exists to avoid -- 47s and ~1GB peak at 240,928 level 2 ids -- so it grows out of
# reach quickly and would be refused outright by LVL2_GRAPH_MAX_NODES beyond that. Objects
# past this cap still get cached at whatever shared parent a query lands on, which is what
# makes mid-tree queries on a huge neuron fast without ever materialising the whole thing.
FIND_PATH_SEED_MAX_LVL2_IDS = _positive_int_from_env(
    "PCG_FIND_PATH_SEED_MAX_LVL2_IDS", 300_000
)


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


def _shortest_path_between(edge_array, source_id, target_id):
    """Unweighted shortest path through `edge_array`, or None if not connected.

    `edge_array` is an n x 2 array of node ids. Returns the ids along the path.
    """
    if len(edge_array) == 0:
        return None
    graph, _, _, indexed_ids = flatgraph.build_gt_graph(edge_array, is_directed=False)
    source_idx = np.flatnonzero(indexed_ids == source_id)
    target_idx = np.flatnonzero(indexed_ids == target_id)
    if not len(source_idx) or not len(target_idx):
        # an endpoint is not in this edge set at all -- it has no cross edges inside it
        return None
    vertex_list, _ = graph_tool.topology.shortest_path(
        graph,
        source=graph.vertex(source_idx[0]),
        target=graph.vertex(target_idx[0]),
    )
    if not len(vertex_list):
        return None
    return indexed_ids[[graph.vertex_index[v] for v in vertex_list]]


def _make_cache(cg):
    """A SubtreeCache for this graph, or None if it is not usable.

    Everything about the cache is optional: a redis that is absent, unreachable or turned
    off must leave find_path working exactly as it would without one.
    """
    try:
        from . import path_cache

        cache = path_cache.SubtreeCache(cg.graph_id)
        return cache if cache.available else None
    except Exception:
        return None


def _coarse_cut(cg, node_id, target_groups):
    """Descend to an antichain of subtrees wide enough to serve as a coarse graph.

    Stops as soon as the cut is at least `target_groups` wide, so which *layer* it lands
    on is chosen per object rather than fixed: layer sizes differ between datasets and
    objects, and a shallow shared parent should produce a small cut instead of a
    degenerate one.

    The cut stays layer aligned, which is load bearing rather than incidental. Every group
    root has a parent strictly above `coarse_layer`, so two groups can only be separated
    at `coarse_layer` or above -- exactly what lets the caller read only the
    CrossChunkEdge[coarse_layer:] columns. A cut chosen purely by width is not layer
    aligned and loses that: measured on the profiled object it gave a similar 19,982
    groups but forced coarse_layer to 3, which is 51% of the cross edge payload instead
    of 11.7%.

    Returns (group_roots, coarse_layer).
    """
    frontier = np.array([node_id], dtype=basetypes.NODE_ID)
    while True:
        layers = cg.get_chunk_layers(frontier)
        top = int(layers.max())
        if top <= 2 or len(frontier) >= target_groups:
            break
        # Expand only the deepest layer still present. Stepping one layer number at a time
        # would stall on the layers this hierarchy skips; taking whatever is deepest keeps
        # the cut layer aligned and always makes progress.
        children = cg.get_children(frontier[layers == top], flatten=True)
        keep = frontier[layers < top]
        grown = np.concatenate([keep, children]) if len(keep) else children
        if not len(grown) or len(grown) <= len(frontier):
            break
        frontier = grown
    if not len(frontier):
        return np.empty(0, dtype=basetypes.NODE_ID), 2
    # The tightest layer the cut is aligned to, which is the fewest cross edge columns
    # the caller can get away with reading.
    return frontier, max(2, int(cg.get_chunk_layers(frontier).max()))


def _expand_groups(cg, group_roots):
    """Level 2 descendants of each group root, as {group root: level 2 ids}."""
    if not len(group_roots):
        return {}
    lvl2_ids = []
    labels = []
    frontier = group_roots
    frontier_labels = group_roots
    while len(frontier):
        layers = cg.get_chunk_layers(frontier)
        at_lvl2 = layers == 2
        lvl2_ids.append(frontier[at_lvl2])
        labels.append(frontier_labels[at_lvl2])
        above = frontier[~at_lvl2]
        labels_above = frontier_labels[~at_lvl2]
        if not len(above):
            break
        children_d = cg.get_children(above)
        counts = np.array([len(children_d[n]) for n in above], dtype=np.int64)
        keep = counts > 0
        if not keep.any():
            break
        frontier = np.concatenate([children_d[n] for n in above[keep]])
        frontier_labels = np.repeat(labels_above[keep], counts[keep])

    all_l2 = np.concatenate(lvl2_ids)
    all_labels = np.concatenate(labels)
    if not len(all_l2):
        return {}
    order = np.argsort(all_labels, kind="stable")
    all_l2 = all_l2[order]
    all_labels = all_labels[order]
    splits = np.flatnonzero(np.diff(all_labels)) + 1
    starts = np.concatenate([[0], splits])
    return {
        int(all_labels[s]): part
        for s, part in zip(starts, np.split(all_l2, splits))
    }


def _boundary_edges(cg, lvl2_by_group, coarse_layer, batch_size=None):
    """Cross edges leaving each group, as {group root: (n, 3) array}.

    Columns are (own level 2 id, own supervoxel, partner supervoxel). The partner is left
    as a supervoxel deliberately -- see path_cache: the level 2 node owning it lives
    outside this subtree and can be replaced by an edit there, while the supervoxel id
    cannot. Resolution to a current level 2 id happens in _coarse_context.

    Two level 2 nodes in different groups must be joined by an edge crossing a chunk
    boundary at `coarse_layer` or above -- a lower boundary would put them in the same
    group -- so only the CrossChunkEdge[coarse_layer:] columns are read. Conversely
    everything in those columns is a boundary edge, by the same argument.
    """
    if not lvl2_by_group:
        return {}
    if batch_size is None:
        batch_size = CROSS_EDGE_READ_BATCH

    owners = np.concatenate(list(lvl2_by_group.values()))
    owner_groups = np.concatenate(
        [
            np.full(len(part), group, dtype=basetypes.NODE_ID)
            for group, part in lvl2_by_group.items()
        ]
    )
    order = np.argsort(owners, kind="stable")
    owners = owners[order]
    owner_groups = owner_groups[order]
    del order

    properties = [
        attributes.Connectivity.CrossChunkEdge[layer]
        for layer in range(coarse_layer, max(coarse_layer + 1, cg.meta.layer_count))
    ]

    all_triples = []
    all_groups = []
    for start in range(0, len(owners), batch_size):
        # Read the restricted column set directly. cg.get_atomic_cross_edges always reads
        # every layer, which is the cost this function exists to avoid.
        rows = cg.client.read_nodes(
            node_ids=owners[start : start + batch_size], properties=properties
        )
        blocks = []
        block_owners = []
        lengths = []
        for lvl2_id, columns in rows.items():
            for cells in columns.values():
                block = cells[0].value
                if len(block) == 0:
                    continue
                blocks.append(block)
                block_owners.append(lvl2_id)
                lengths.append(len(block))
        rows.clear()
        if not blocks:
            continue
        edges = np.concatenate(blocks)
        del blocks
        owner_column = np.repeat(
            np.array(block_owners, dtype=edges.dtype),
            np.array(lengths, dtype=np.int64),
        )
        all_triples.append(np.column_stack([owner_column, edges[:, 0], edges[:, 1]]))
        all_groups.append(owner_groups[np.searchsorted(owners, owner_column)])
        del edges, owner_column

    empty = np.empty((0, 3), dtype=basetypes.NODE_ID)
    if not all_triples:
        return {int(group): empty for group in lvl2_by_group}

    triples = np.concatenate(all_triples)
    groups = np.concatenate(all_groups)
    del all_triples, all_groups

    # One sort and one split, rather than a boolean mask per group. Masking per group is
    # O(groups x rows) -- with ~27k groups over millions of rows it cost more than the
    # Bigtable read it accompanies, which made a warm cache slower than no cache at all.
    order = np.argsort(groups, kind="stable")
    groups = groups[order]
    triples = triples[order]
    del order
    splits = np.flatnonzero(np.diff(groups)) + 1
    starts = np.concatenate([[0], splits])
    per_group = {int(group): empty for group in lvl2_by_group}
    # No per-group np.unique here. Bigtable returns each (level 2 id, layer) row once, so
    # there is nothing to dedupe, and _coarse_graph uniques the group pairs at the end
    # regardless. Doing it per group meant ~27k np.unique calls on tiny arrays, which cost
    # more than the read itself.
    for begin, part in zip(starts, np.split(triples, splits)):
        per_group[int(groups[begin])] = part
    return per_group


def _coarse_membership(cg, node_id, target_groups, cache=None):
    """The cut and which level 2 ids fall under each of its subtrees.

    Deliberately does not touch cross edges: a small object skips the coarse pass
    entirely, and reading its boundary columns first would be pure waste.

    Returns (group_roots, lvl2_by_group, coarse_layer, cache, fresh), where `fresh` is
    the subset the caller still has to persist if it decides the cache is worth writing.
    """
    group_roots, coarse_layer = _coarse_cut(cg, node_id, target_groups)
    if not len(group_roots):
        return group_roots, {}, coarse_layer, cache
    if cache is None:
        cache = _make_cache(cg)

    cached = cache.get_descendants(group_roots) if cache is not None else {}
    missed = np.array(
        [g for g in group_roots if int(g) not in cached], dtype=basetypes.NODE_ID
    )
    fresh = _expand_groups(cg, missed)
    cached.update(fresh)
    # `fresh` is handed back rather than written here: a small object bypasses the coarse
    # pass entirely a moment from now, and its cut degenerates to one group per level 2
    # node, so persisting it would store thousands of single-element lists that nothing
    # will ever read. The caller writes them only once it knows it is on the path that
    # benefits.
    return group_roots, cached, coarse_layer, cache, fresh


def _flatten_groups(group_roots, lvl2_by_group):
    """(lvl2_ids, lvl2_groups), positionally aligned."""
    parts = []
    labels = []
    for group in group_roots:
        part = lvl2_by_group.get(int(group))
        if part is None or not len(part):
            continue
        parts.append(part)
        labels.append(np.full(len(part), group, dtype=basetypes.NODE_ID))
    if not parts:
        empty = np.empty(0, dtype=basetypes.NODE_ID)
        return empty, empty
    return np.concatenate(parts), np.concatenate(labels)


def _coarse_graph(
    cg, group_roots, lvl2_by_group, lvl2_ids, lvl2_groups, coarse_layer, cache=None
):
    """Group-level edges of the coarse graph, from cache where possible."""
    no_edges = np.empty((0, 2), dtype=basetypes.NODE_ID)
    if not len(group_roots):
        return no_edges

    # Not cached. These rows are 65MB per object against a 4.4MB answer, and the answer
    # itself is cached at root level instead -- so this only runs when there is no usable
    # predecessor to patch from, i.e. on a genuinely cold object.
    wanted = {
        int(g): lvl2_by_group[int(g)]
        for g in group_roots
        if len(lvl2_by_group.get(int(g), ()))
    }
    parts = [rows for rows in _boundary_edges(cg, wanted, coarse_layer).values() if len(rows)]
    if not parts:
        return no_edges
    boundary = np.concatenate(parts)
    del parts

    # Resolve each partner supervoxel to the level 2 node that owns it *now*. Column 1 of
    # every row is an own supervoxel paired with its own level 2 id in column 0, and cross
    # edges are stored from both sides, so the union of those pairs covers every partner
    # inside this object. This is the step that makes a cached row survive an edit on the
    # far side of the boundary: the supervoxel is stable, the level 2 node owning it is not.
    supervoxels, first = np.unique(boundary[:, 1], return_index=True)
    owners_of = boundary[first, 0]
    partners = boundary[:, 2]
    idx = np.searchsorted(supervoxels, partners)
    idx[idx == len(supervoxels)] = 0
    inside = supervoxels[idx] == partners
    ends = np.column_stack([boundary[:, 0], owners_of[idx]])[inside]
    del supervoxels, owners_of, partners, idx, inside, boundary, first
    if not len(ends):
        return no_edges

    # ...then map both ends onto the grouping this request actually has. A partner that
    # stays unresolved here is outside the object, which is what makes the graph induced.
    order = np.argsort(lvl2_ids, kind="stable")
    sorted_lvl2 = lvl2_ids[order]
    lookup = lvl2_groups[order]
    del order
    flat = ends.reshape(-1)
    idx = np.searchsorted(sorted_lvl2, flat)
    idx[idx == len(sorted_lvl2)] = 0
    known = (sorted_lvl2[idx] == flat).reshape(-1, 2).all(axis=1)
    pairs = lookup[idx].reshape(-1, 2)[known]
    del flat, idx, known, ends
    if not len(pairs):
        return no_edges
    pairs.sort(axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0) if len(pairs) else no_edges


def _splice_edge_list(cg, old_lvl2, old_edges, new_lvl2):
    """Patch a cached edge list from one root onto the level 2 ids of another.

    An edit replaces a handful of level 2 nodes and leaves the rest alone, so the whole
    edge list can be brought forward by dropping what left and reading only what arrived:

      * edges between two surviving nodes are untouched, so they carry over verbatim;
      * every edge involving a new node appears in that node's own cross edge rows, since
        cross edges are stored from both sides -- so reading only the new ids finds them
        all, including the ones joining a new node to a surviving one.

    Measured at ~0.6s for 50 changed ids against 31.5s to rebuild from scratch.
    """
    old_sorted = np.unique(old_lvl2)
    new_sorted = np.unique(new_lvl2)
    added = new_sorted[~in_sorted(new_sorted, old_sorted)]
    survived = old_sorted[in_sorted(old_sorted, new_sorted)]

    if len(old_edges):
        keep = in_sorted(old_edges[:, 0], survived) & in_sorted(
            old_edges[:, 1], survived
        )
        carried = old_edges[keep]
    else:
        carried = np.empty((0, 2), dtype=basetypes.NODE_ID)
    if not len(added):
        return carried

    # induced=False, then filtered here: an edge from a new node to a surviving one has
    # only one endpoint in `added`, so inducing on `added` would drop exactly the edges
    # that stitch the new material onto the old.
    fresh = _get_edges_for_lvl2_ids(cg, added, induced=False)
    if len(fresh):
        inside = in_sorted(fresh[:, 0], new_sorted) & in_sorted(fresh[:, 1], new_sorted)
        fresh = fresh[inside]
    if not len(fresh):
        return carried
    if not len(carried):
        return np.unique(fresh, axis=0)
    return np.unique(np.concatenate([carried, fresh]), axis=0)


def _edges_from_root_cache(cg, shared_parent_id, lvl2_ids, cache, time_stamp=None):
    """Serve a mid-tree query by narrowing the whole object's cached edge list.

    A subtree's level 2 ids are a subset of its root's, so its induced edges are a subset
    of the root's induced edges and the narrowing is exact rather than approximate.
    Measured on api6: fetching the root's 6.4MB entry and filtering it to an 11,578 node
    layer 8 subtree took 0.022s against 1.83s to rebuild that subtree from Bigtable, and
    produced element-identical output.

    Deliberately a direct lookup with no splice. Patching a root entry forward would need
    the root's *current* level 2 ids, and obtaining those is the 5.5s descent this path
    exists to avoid -- far more than the ~1.8s rebuild it is trying to beat. So after an
    edit this simply misses until some root level query refreshes the entry, and the
    caller falls back to computing the subtree.
    """
    if cache is None:
        return None
    try:
        root_id = cg.get_root(shared_parent_id, time_stamp=time_stamp)
    except Exception:
        return None
    if root_id is None or int(root_id) == int(shared_parent_id):
        return None  # already a root query; the caller's own lookup covers it
    found = cache.get_edges([root_id])
    entry = found.get(int(root_id))
    if entry is None:
        return None
    root_lvl2, root_edges = entry
    if not len(root_edges):
        return None
    subset = np.unique(lvl2_ids)
    # The subset relationship is what makes narrowing exact, so it is checked rather than
    # assumed. If the entry does not cover every level 2 id in this subtree, narrowing it
    # would silently drop the edges touching the ones it is missing -- a shorter edge list
    # that still yields a path, just the wrong one. Falling back costs a rebuild; getting
    # this wrong costs a wrong answer.
    if not in_sorted(subset, np.unique(root_lvl2)).all():
        return None
    return root_edges[
        in_sorted(root_edges[:, 0], subset) & in_sorted(root_edges[:, 1], subset)
    ]


def _edge_list_from_cache(cg, node_id, lvl2_ids, cache):
    """The full level 2 edge list for `node_id`, from cache or patched from a predecessor.

    Returns None when neither is available, leaving the caller to prune instead.
    """
    if cache is None:
        return None
    direct = cache.get_edges([node_id])
    if int(node_id) in direct:
        return direct[int(node_id)][1]
    try:
        previous = lineage.get_previous_root_ids(cg, [node_id])
    except Exception:
        return None
    candidates = [int(r) for ids in previous.values() for r in np.asarray(ids).reshape(-1)]
    if not candidates:
        return None
    for root_id, (old_lvl2, old_edges) in cache.get_edges(candidates).items():
        return _splice_edge_list(cg, old_lvl2, old_edges, lvl2_ids)
    return None


def _dilate(edge_array, seed_ids, hops):
    """All node ids within `hops` of `seed_ids` in `edge_array`."""
    selected = np.unique(seed_ids)
    for _ in range(hops):
        touching = in_sorted(edge_array[:, 0], selected) | in_sorted(
            edge_array[:, 1], selected
        )
        if not touching.any():
            break
        grown = np.unique(edge_array[touching].reshape(-1))
        if len(grown) == len(selected) and np.array_equal(grown, selected):
            break
        selected = np.union1d(selected, grown)
    return selected


def find_l2_shortest_path(
    cg,
    source_l2_id: np.uint64,
    target_l2_id: np.uint64,
    time_stamp=None,
    max_num_lvl2_ids: typing.Optional[int] = None,
    cache=None,
):
    """
    Find a path of level 2 ids that connect two level 2 node ids through cross chunk edges.
    Return a list of level 2 ids representing this path.
    Return None if the two level 2 ids do not belong to the same object.

    For a large object this does not read the whole level 2 graph. It first builds a
    coarse graph over subtrees (see `_coarse_context`), finds the
    route through that, dilates it, and only then reads the level 2 cross edges under the
    resulting corridor. Measured on api6 aibs_v1dd root 864691132764660103, the widest
    possible pair -- the graph diameter, a 1,616 node path -- passes through 195 of the
    object's ~16,220 layer 5 nodes, so the corridor is on the order of 1% of the object.

    The path this returns is not hop-minimal, unlike the previous full-graph search: a
    shorter route can leave the corridor. At the default dilation it came in under 1% long
    on most pairs measured and 4% at worst, and was exact on several (see
    FIND_PATH_DILATION for the sweep and for how to trade speed back for optimality). That
    is the intended trade -- the endpoint exists to get a user from A to B -- but it does
    mean the result can differ from what the old implementation returned for the same two
    nodes. If no route is found in the corridor at all, this falls back to the full graph
    rather than reporting the two nodes as unconnected.

    :param cg: ChunkedGraph object
    :param source_l2_id: np.uint64
    :param target_l2_id: np.uint64
    :param max_num_lvl2_ids: Optional[int] reject the request when the object has more
        than this many level 2 ids, matching the guard on the lvl2_graph endpoint.
    :return: [np.uint64] or None
    """
    shared_parent_id = get_first_shared_parent(
        cg, source_l2_id, target_l2_id, time_stamp
    )
    if shared_parent_id is None:
        return None

    group_roots, lvl2_by_group, coarse_layer, cache, fresh_descendants = _coarse_membership(
        cg, shared_parent_id, FIND_PATH_TARGET_GROUPS, cache=cache
    )
    lvl2_ids, lvl2_groups = _flatten_groups(group_roots, lvl2_by_group)
    if max_num_lvl2_ids is not None and len(lvl2_ids) > max_num_lvl2_ids:
        raise cg_exceptions.BadRequest(
            f"The level 2 graph for {shared_parent_id} has {len(lvl2_ids)} level 2 nodes, "
            f"which exceeds the maximum of {max_num_lvl2_ids}."
        )

    def _full_search():
        # The level 2 ids are already in hand, so this repeats the read but not the descent.
        edges = _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True)
        # Free to cache: this *is* the full edge list, already paid for. Without it a
        # subtree just under the bypass threshold stays slow forever -- measured at 2.1s
        # first call and 2.1s on requery, with nothing between it and the ~8s the
        # threshold allows. Skipped for trivially small subtrees, which are already
        # sub-second and would only churn keys.
        if (
            cache is not None
            and len(lvl2_ids) >= FIND_PATH_CACHE_MIN_LVL2_IDS
        ):
            if fresh_descendants:
                cache.set_descendants(fresh_descendants)
            cache.set_edges(shared_parent_id, lvl2_ids, edges)
        return _shortest_path_between(edges, source_l2_id, target_l2_id)

    # Below this size the corridor machinery costs more than it saves, and the exact
    # answer is cheap. This is the common case: two nearby level 2 ids share a parent low
    # in the hierarchy, so the subtree under it is small and this is where they land.
    # Consulted before the size bypass, not after: a cached edge list answers in ~12ms
    # whatever the subtree's size, so there is no reason to make small objects redo work
    # that is already sitting in redis.
    cached_edges = _edge_list_from_cache(cg, shared_parent_id, lvl2_ids, cache)
    if cached_edges is None:
        # Nothing under this shared parent, but the whole object may be cached from an
        # earlier end-to-end query. Narrowing that is ~80x cheaper than rebuilding.
        cached_edges = _edges_from_root_cache(
            cg, shared_parent_id, lvl2_ids, cache, time_stamp
        )
    if cached_edges is not None:
        if cache is not None:
            cache.set_edges(shared_parent_id, lvl2_ids, cached_edges)
        path = _shortest_path_between(cached_edges, source_l2_id, target_l2_id)
        if path is not None:
            return path
        # An edge list that cannot connect the two is either stale or genuinely
        # disconnected; fall through and let the unpruned search settle it.

    if len(lvl2_ids) <= FIND_PATH_MIN_LVL2_IDS:
        return _full_search()

    if cache is not None and fresh_descendants:
        cache.set_descendants(fresh_descendants)


    # Both degenerate ends of the cut make the coarse pass pure cost. One group cannot
    # separate anything, so the coarse graph comes out empty and the corridor is the whole
    # object. A group per level 2 id is no coarser than the graph we are trying to avoid
    # building, so it prunes nothing. The second is what a shallow shared parent produces:
    # descending from it reaches level 2 in one step.
    n_groups = len(np.unique(lvl2_groups))
    if n_groups < 2 or n_groups >= len(lvl2_ids):
        return _full_search()

    # Only now, past every bypass, is the boundary read worth issuing.
    coarse_edges = _coarse_graph(
        cg, group_roots, lvl2_by_group, lvl2_ids, lvl2_groups, coarse_layer, cache=cache
    )
    sorted_lvl2 = np.sort(lvl2_ids)
    group_lookup = lvl2_groups[np.argsort(lvl2_ids, kind="stable")]
    source_group = group_lookup[np.searchsorted(sorted_lvl2, source_l2_id)]
    target_group = group_lookup[np.searchsorted(sorted_lvl2, target_l2_id)]

    if source_group != target_group:
        coarse_path = _shortest_path_between(
            coarse_edges, source_group, target_group
        )
        if coarse_path is None:
            return _full_search()
        corridor_groups = _dilate(coarse_edges, coarse_path, FIND_PATH_DILATION)
    else:
        corridor_groups = _dilate(
            coarse_edges, np.array([source_group], dtype=basetypes.NODE_ID),
            FIND_PATH_DILATION,
        )
    del coarse_edges

    # corridor_groups is sorted (np.unique / np.union1d), which is what in_sorted needs;
    # lvl2_groups is tested elementwise and does not need to be.
    corridor_lvl2 = lvl2_ids[in_sorted(lvl2_groups, corridor_groups)]

    path = _shortest_path_between(
        _get_edges_for_lvl2_ids(cg, corridor_lvl2, induced=True),
        source_l2_id,
        target_l2_id,
    )
    if path is None:
        # The corridor was too tight, or the only route leaves it. Never report the two
        # nodes as unconnected on the strength of a pruned search.
        path = _full_search()
    _seed_edge_cache(cg, shared_parent_id, lvl2_ids, cache)
    return path


def _seed_edge_cache(cg, node_id, lvl2_ids, cache):
    """Compute and store the full edge list so later calls can patch it.

    Deliberately separate from answering the request: it costs the full ~31.5s read that
    the corridor search exists to avoid, and buys nothing for the call that pays it. It is
    only worth doing when more calls on the same neuron are expected, which is why it is
    opt-in.
    """
    if cache is None or not FIND_PATH_SEED_CACHE:
        return
    if len(lvl2_ids) > FIND_PATH_SEED_MAX_LVL2_IDS:
        # Seeding builds the whole edge list, and that does not scale: 240,928 level 2 ids
        # cost 47s and ~1GB of peak rss, so a million-node object would be minutes and
        # several GB -- on a pod shared by 8-16 workers. Past this size the corridor search
        # is the only sane way to answer, and intermediate entries carry the caching.
        return
    try:
        cache.set_edges(
            node_id, lvl2_ids, _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True)
        )
    except Exception:
        # Seeding is best effort; never let it fail a request that already succeeded.
        pass


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
