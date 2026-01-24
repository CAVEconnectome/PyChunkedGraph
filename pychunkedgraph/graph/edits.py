# pylint: disable=invalid-name, missing-docstring, too-many-locals, c-extension-no-member

import datetime, logging, random
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Set
from collections import defaultdict
from contextlib import contextmanager

import fastremap
import numpy as np

from pychunkedgraph.debug.profiler import HierarchicalProfiler, get_profiler

from . import types
from . import attributes
from . import cache as cache_utils
from .edges import get_latest_edges_wrapper, get_new_nodes
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts
from .utils import basetypes
from .utils import flatgraph
from .utils.serializers import serialize_uint64
from ..utils.general import in2d
from ..debug.utils import sanity_check, sanity_check_single

logger = logging.getLogger(__name__)


def _init_old_hierarchy(cg, l2ids: np.ndarray, parent_ts: datetime.datetime = None):
    """
    Populates old hierarcy from child to root and also gets children of intermediate nodes.
    These will be needed later and cached in cg.cache used during an edit.
    """
    all_parents = []
    old_hierarchy_d = {id_: {2: id_} for id_ in l2ids}
    node_layer_parent_map = cg.get_all_parents_dict_multiple(
        l2ids, time_stamp=parent_ts
    )
    for id_ in l2ids:
        layer_parent_d = node_layer_parent_map[id_]
        old_hierarchy_d[id_].update(layer_parent_d)
        for parent in layer_parent_d.values():
            all_parents.append(parent)
            old_hierarchy_d[parent] = old_hierarchy_d[id_]
    children = cg.get_children(all_parents, flatten=True)
    _ = cg.get_parents(children, time_stamp=parent_ts)
    return old_hierarchy_d


def flip_ids(id_map, node_ids):
    """
    returns old or new ids according to the map
    """
    ids = [np.asarray(list(id_map[id_]), dtype=basetypes.NODE_ID) for id_ in node_ids]
    ids.append(types.empty_1d)  # concatenate needs at least one array
    return np.concatenate(ids).astype(basetypes.NODE_ID)


def _analyze_affected_edges(
    cg, atomic_edges: Iterable[np.ndarray], parent_ts: datetime.datetime = None
) -> Tuple[Iterable, Dict]:
    """
    Returns l2 edges within chunk and self edges for nodes in cross chunk edges.

    Also returns new cross edges dicts for nodes crossing chunk boundary.
    """
    profiler = get_profiler()

    supervoxels = np.unique(atomic_edges)
    with profiler.profile("analyze_get_parents"):
        parents = cg.get_parents(supervoxels, time_stamp=parent_ts)
    sv_parent_d = dict(zip(supervoxels.tolist(), parents))
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    parent_edges = [
        [sv_parent_d[edge_[0]], sv_parent_d[edge_[1]]]
        for edge_ in atomic_edges[edge_layers == 1]
    ]

    cross_edges_d = defaultdict(lambda: defaultdict(list))
    for layer in range(2, cg.meta.layer_count):
        layer_edges = atomic_edges[edge_layers == layer]
        if not layer_edges.size:
            continue
        for edge in layer_edges:
            parent0 = sv_parent_d[edge[0]]
            parent1 = sv_parent_d[edge[1]]
            cross_edges_d[parent0][layer].append([parent0, parent1])
            cross_edges_d[parent1][layer].append([parent1, parent0])
            parent_edges.extend([[parent0, parent0], [parent1, parent1]])
    # Convert inner Python lists to typed numpy arrays to avoid
    # dtype promotion issues when concatenated with uint64 arrays.
    for node_id in cross_edges_d:
        for layer in cross_edges_d[node_id]:
            cross_edges_d[node_id][layer] = np.array(
                cross_edges_d[node_id][layer], dtype=basetypes.NODE_ID
            ).reshape(-1, 2)
    return parent_edges, cross_edges_d


def _get_relevant_components(edges: np.ndarray, svs: np.ndarray) -> Tuple:
    edges = np.concatenate([edges, np.vstack([svs, svs]).T]).astype(basetypes.NODE_ID)
    graph, _, _, graph_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)
    relevant_ccs = []
    # remove if connected component contains no sources or no sinks
    # when merging, there must be only two components
    for cc_idx in ccs:
        cc = graph_ids[cc_idx]
        if np.any(np.isin(svs, cc)):
            relevant_ccs.append(cc)
    assert len(relevant_ccs) == 2, "must be 2 components"
    return relevant_ccs


def merge_preprocess(
    cg,
    *,
    subgraph_edges: np.ndarray,
    supervoxels: np.ndarray,
    parent_ts: datetime.datetime = None,
) -> np.ndarray:
    """
    Check and return inactive edges in the subgraph.
    """
    edge_layers = cg.get_cross_chunk_edges_layer(subgraph_edges)
    active_edges = [types.empty_2d]
    inactive_edges = [types.empty_2d]
    for layer in np.unique(edge_layers):
        _edges = subgraph_edges[edge_layers == layer]
        edge_nodes = fastremap.unique(_edges.ravel())
        roots = cg.get_roots(edge_nodes, time_stamp=parent_ts, stop_layer=layer + 1)
        parent_map = dict(zip(edge_nodes, roots))
        parent_edges = fastremap.remap(_edges, parent_map, preserve_missing_labels=True)

        active_mask = parent_edges[:, 0] == parent_edges[:, 1]
        active, inactive = _edges[active_mask], _edges[~active_mask]
        active_edges.append(active)
        inactive_edges.append(inactive)

    active_edges = np.concatenate(active_edges).astype(basetypes.NODE_ID)
    inactive_edges = np.concatenate(inactive_edges).astype(basetypes.NODE_ID)
    relevant_ccs = _get_relevant_components(active_edges, supervoxels)
    _inactive = [types.empty_2d]
    # source to sink edges
    source_mask = np.isin(inactive_edges[:, 0], relevant_ccs[0])
    sink_mask = np.isin(inactive_edges[:, 1], relevant_ccs[1])
    _inactive.append(inactive_edges[source_mask & sink_mask])

    # sink to source edges
    sink_mask = np.isin(inactive_edges[:, 1], relevant_ccs[0])
    source_mask = np.isin(inactive_edges[:, 0], relevant_ccs[1])
    _inactive.append(inactive_edges[source_mask & sink_mask])
    _inactive = np.concatenate(_inactive).astype(basetypes.NODE_ID)
    return np.unique(_inactive, axis=0) if _inactive.size else types.empty_2d


def check_fake_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    inactive_edges: Iterable[np.ndarray],
    time_stamp: datetime.datetime,
    parent_ts: datetime.datetime = None,
) -> Tuple[Iterable[np.ndarray], Iterable]:
    """if no inactive edges found, add user input as fake edge."""
    if inactive_edges.size:
        roots = np.unique(
            cg.get_roots(
                np.unique(inactive_edges),
                assert_roots=True,
                time_stamp=parent_ts,
            )
        )
        assert len(roots) == 2, "edges must be from 2 roots"
        return inactive_edges, []

    rows = []
    supervoxels = atomic_edges.ravel()
    # fake edges are stored with l2 chunks
    chunk_ids = cg.get_chunk_ids_from_node_ids(
        cg.get_parents(supervoxels, time_stamp=parent_ts)
    )
    sv_l2chunk_id_d = dict(zip(supervoxels.tolist(), chunk_ids))
    for edge in atomic_edges:
        id1, id2 = sv_l2chunk_id_d[edge[0]], sv_l2chunk_id_d[edge[1]]
        val_dict = {}
        val_dict[attributes.Connectivity.FakeEdges] = np.array(
            [[edge]], dtype=basetypes.NODE_ID
        )
        id1 = serialize_uint64(id1, fake_edges=True)
        rows.append(
            cg.client.mutate_row(
                id1,
                val_dict,
                time_stamp=time_stamp,
            )
        )
        val_dict = {}
        val_dict[attributes.Connectivity.FakeEdges] = np.array(
            [edge[::-1]], dtype=basetypes.NODE_ID
        )
        id2 = serialize_uint64(id2, fake_edges=True)
        rows.append(
            cg.client.mutate_row(
                id2,
                val_dict,
                time_stamp=time_stamp,
            )
        )
    return atomic_edges, rows


def add_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    operation_id: np.uint64 = None,
    time_stamp: datetime.datetime = None,
    parent_ts: datetime.datetime = None,
    allow_same_segment_merge=False,
    stitch_mode: bool = False,
    do_sanity_check: bool = True,
):
    profiler = get_profiler()
    profiler.reset()  # Reset for fresh profiling

    with profiler.profile("add_edges"):
        with profiler.profile("analyze_affected_edges"):
            edges, l2_cross_edges_d = _analyze_affected_edges(
                cg, atomic_edges, parent_ts=parent_ts
            )

        l2ids = np.unique(edges)
        if not allow_same_segment_merge and not stitch_mode:
            with profiler.profile("validate_roots"):
                roots = cg.get_roots(l2ids, assert_roots=True, time_stamp=parent_ts)
                assert np.unique(roots).size >= 2, "L2 IDs must belong to different roots."

        new_old_id_d = defaultdict(set)
        old_new_id_d = defaultdict(set)

    old_hierarchy_d = _init_old_hierarchy(cg, l2ids, parent_ts=parent_ts)
    atomic_children_d = cg.get_children(l2ids)
    cross_edges_d = merge_cross_edge_dicts(
        cg.get_cross_chunk_edges(l2ids, time_stamp=parent_ts), l2_cross_edges_d
    )
    graph, _, _, graph_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    components = flatgraph.connected_components(graph)

    chunk_count_map = defaultdict(int)
    for cc_indices in components:
        l2ids_ = graph_ids[cc_indices]
        chunk = cg.get_chunk_id(l2ids_[0])
        chunk_count_map[chunk] += 1

    chunk_ids = list(chunk_count_map.keys())
    random.shuffle(chunk_ids)
    chunk_new_ids_map = {}
    for chunk_id in chunk_ids:
        new_ids = cg.id_client.create_node_ids(chunk_id, size=chunk_count_map[chunk_id])
        chunk_new_ids_map[chunk_id] = list(new_ids)

    new_l2_ids = []
    for cc_indices in components:
        l2ids_ = graph_ids[cc_indices]
        new_id = chunk_new_ids_map[cg.get_chunk_id(l2ids_[0])].pop()
        new_l2_ids.append(new_id)
        new_old_id_d[new_id].update(l2ids_)
        for id_ in l2ids_:
            old_new_id_d[id_].add(new_id)

        # update cache
        # map parent to new merged children and vice versa
        merged_children = [atomic_children_d[l2id] for l2id in l2ids_]
        merged_children = np.concatenate(merged_children).astype(basetypes.NODE_ID)
        cg.cache.children_cache[new_id] = merged_children
        cache_utils.update(cg.cache.parents_cache, merged_children, new_id)

        # update cross chunk edges by replacing old_ids with new
        # this can be done only after all new IDs have been created
        with profiler.profile("update_cross_edges"):
            for new_id, cc_indices in zip(new_l2_ids, components):
                l2ids_ = graph_ids[cc_indices]
                new_cx_edges_d = {}
                cx_edges = [cross_edges_d[l2id] for l2id in l2ids_]
                cx_edges_d = concatenate_cross_edge_dicts(cx_edges, unique=True)
                temp_map = {k: next(iter(v)) for k, v in old_new_id_d.items()}
                for layer, edges in cx_edges_d.items():
                    edges = fastremap.remap(edges, temp_map, preserve_missing_labels=True)
                    new_cx_edges_d[layer] = edges
                    assert np.all(edges[:, 0] == new_id)
                cg.cache.cross_chunk_edges_cache[new_id] = new_cx_edges_d

    profiler = get_profiler()
    profiler.reset()
    with profiler.profile("run"):
        create_parents = CreateParentNodes(
            cg,
            new_l2_ids=new_l2_ids,
            old_hierarchy_d=old_hierarchy_d,
            new_old_id_d=new_old_id_d,
            old_new_id_d=old_new_id_d,
            operation_id=operation_id,
            time_stamp=time_stamp,
            parent_ts=parent_ts,
            stitch_mode=stitch_mode,
            do_sanity_check=do_sanity_check,
            profiler=profiler,
        )
        new_roots = create_parents.run()

    if do_sanity_check:
        sanity_check(cg, new_roots, operation_id)
    create_parents.create_new_entries()
    profiler.print_report(operation_id)
    return new_roots, new_l2_ids, create_parents.new_entries


def _split_l2_agglomeration(
    cg,
    operation_id: int,
    agg: types.Agglomeration,
    removed_edges: np.ndarray,
    parent_ts: datetime.datetime = None,
):
    """
    For a given L2 id, remove given edges; calculate new connected components.
    """
    chunk_edges = agg.in_edges.get_pairs()
    chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]

    cross_edges = agg.cross_edges.get_pairs()
    # we must avoid the cache to read roots to get segment state before edit began
    parents = cg.get_parents(cross_edges[:, 0], time_stamp=parent_ts, raw_only=True)

    # if there are cross edges, there must be a single parent.
    # if there aren't any, there must be no parents. XOR these 2 conditions.
    err = f"got cross edges from more than one l2 node; op {operation_id}"
    assert (np.unique(parents).size == 1) != (cross_edges.size == 0), err

    if cross_edges.size:
        # inactive edges must be filtered out
        root = cg.get_root(parents[0], time_stamp=parent_ts, raw_only=True)
        neighbor_roots = cg.get_roots(
            cross_edges[:, 1], raw_only=True, time_stamp=parent_ts
        )
        active_mask = neighbor_roots == root
        cross_edges = cross_edges[active_mask]
        cross_edges = cross_edges[~in2d(cross_edges, removed_edges)]
    isolated_ids = agg.supervoxels[~np.isin(agg.supervoxels, chunk_edges)]
    isolated_edges = np.column_stack((isolated_ids, isolated_ids))
    _edges = np.concatenate([chunk_edges, isolated_edges]).astype(basetypes.NODE_ID)
    graph, _, _, graph_ids = flatgraph.build_gt_graph(_edges, make_directed=True)
    return flatgraph.connected_components(graph), graph_ids, cross_edges


def _filter_component_cross_edges(
    component_ids: np.ndarray, cross_edges: np.ndarray, cross_edge_layers: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Filters cross edges for a connected component `cc_ids`
    from `cross_edges` of the complete chunk.
    """
    mask = np.isin(cross_edges[:, 0], component_ids)
    cross_edges_ = cross_edges[mask]
    cross_edge_layers_ = cross_edge_layers[mask]
    edges_d = {}
    for layer in np.unique(cross_edge_layers_):
        edge_m = cross_edge_layers_ == layer
        _cross_edges = cross_edges_[edge_m]
        if _cross_edges.size:
            edges_d[layer] = _cross_edges
    return edges_d


def remove_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    operation_id: basetypes.OPERATION_ID = None,  # type: ignore
    time_stamp: datetime.datetime = None,
    parent_ts: datetime.datetime = None,
    do_sanity_check: bool = True,
):
    edges, _ = _analyze_affected_edges(cg, atomic_edges, parent_ts=parent_ts)
    l2ids = np.unique(edges)
    roots = cg.get_roots(l2ids, assert_roots=True, time_stamp=parent_ts)
    assert np.unique(roots).size == 1, "L2 IDs must belong to same root."

    l2id_agglomeration_d, _ = cg.get_l2_agglomerations(
        l2ids, active=True, time_stamp=parent_ts
    )
    new_old_id_d = defaultdict(set)
    old_new_id_d = defaultdict(set)
    old_hierarchy_d = _init_old_hierarchy(cg, l2ids, parent_ts=parent_ts)
    chunk_id_map = dict(zip(l2ids.tolist(), cg.get_chunk_ids_from_node_ids(l2ids)))

    removed_edges = [atomic_edges, atomic_edges[:, ::-1]]
    removed_edges = np.concatenate(removed_edges, axis=0).astype(basetypes.NODE_ID)
    new_l2_ids = []
    for id_ in l2ids:
        agg = l2id_agglomeration_d[id_]
        ccs, graph_ids, cross_edges = _split_l2_agglomeration(
            cg, operation_id, agg, removed_edges, parent_ts
        )
        new_parents = cg.id_client.create_node_ids(chunk_id_map[agg.node_id], len(ccs))

        cross_edge_layers = cg.get_cross_chunk_edges_layer(cross_edges)
        for i_cc, cc in enumerate(ccs):
            new_id = new_parents[i_cc]
            new_l2_ids.append(new_id)
            new_old_id_d[new_id].add(id_)
            old_new_id_d[id_].add(new_id)
            cg.cache.children_cache[new_id] = graph_ids[cc]
            cache_utils.update(cg.cache.parents_cache, graph_ids[cc], new_id)
            cg.cache.cross_chunk_edges_cache[new_id] = _filter_component_cross_edges(
                graph_ids[cc], cross_edges, cross_edge_layers
            )

    cx_edges_d = cg.get_cross_chunk_edges(new_l2_ids, time_stamp=parent_ts)
    for new_id in new_l2_ids:
        new_cx_edges_d = cx_edges_d.get(new_id, {})
        for layer, edges in new_cx_edges_d.items():
            svs = np.unique(edges)
            parents = cg.get_parents(svs, time_stamp=parent_ts)
            temp_map = dict(zip(svs, parents))

            edges = fastremap.remap(edges, temp_map, preserve_missing_labels=True)
            edges = np.unique(edges, axis=0)
            new_cx_edges_d[layer] = edges
            assert np.all(edges[:, 0] == new_id)
        cg.cache.cross_chunk_edges_cache[new_id] = new_cx_edges_d

    create_parents = CreateParentNodes(
        cg,
        new_l2_ids=new_l2_ids,
        old_hierarchy_d=old_hierarchy_d,
        new_old_id_d=new_old_id_d,
        old_new_id_d=old_new_id_d,
        operation_id=operation_id,
        time_stamp=time_stamp,
        parent_ts=parent_ts,
        do_sanity_check=do_sanity_check,
    )
    new_roots = create_parents.run()

    if do_sanity_check:
        sanity_check(cg, new_roots, operation_id)
    create_parents.create_new_entries()
    return new_roots, new_l2_ids, create_parents.new_entries


def _get_descendants_batch(cg, node_ids):
    """Get all descendants at layers >= 2 for multiple node_ids.
    Batches get_children calls by level to reduce IO.
    Returns dict {node_id: np.ndarray of descendants}.
    """
    if not node_ids:
        return {}
    results = {nid: [] for nid in node_ids}
    # expand_map: {node_to_expand: root_node_id}
    expand_map = {nid: nid for nid in node_ids}

    while expand_map:
        next_expand = {}
        children_d = cg.get_children(list(expand_map.keys()))
        for parent, root in expand_map.items():
            children = children_d[parent]
            layers = cg.get_chunk_layers(children)
            mask = layers >= 2
            results[root].extend(children[mask])
            for c in children[layers > 2]:
                next_expand[c] = root
        expand_map = next_expand
    return {
        nid: np.array(desc, dtype=basetypes.NODE_ID) for nid, desc in results.items()
    }


def _get_counterparts(
    cg, node_id: int, cx_edges_d: dict
) -> Tuple[List[int], Dict[int, int]]:
    """
    Extract counterparts and their corresponding layers from cross chunk edges.
    Returns (counterparts list, counterpart_layers dict).
    """
    node_layer = cg.get_chunk_layer(node_id)
    counterparts = []
    counterpart_layers = {}
    for layer in range(node_layer, cg.meta.layer_count):
        layer_edges = cx_edges_d.get(layer, types.empty_2d)
        if layer_edges.size == 0:
            continue
        counterparts.extend(layer_edges[:, 1])
        layers_d = dict(zip(layer_edges[:, 1], [layer] * len(layer_edges[:, 1])))
        counterpart_layers.update(layers_d)
    return counterparts, counterpart_layers


def _update_neighbor_cx_edges_single(
    cg,
    new_id: int,
    node_map: dict,
    counterpart_layers: dict,
    all_counterparts_cx_edges_d: dict,
    descendants_d: dict,
) -> dict:
    """
    For each new_id, update cross chunk edges of its counterparts.
    Some of them maybe updated multiple times so we need to collect them first
    and then write to storage to consolidate the mutations.
    Returns updated counterparts.
    """
    node_layer = cg.get_chunk_layer(new_id)
    counterparts = list(counterpart_layers.keys())
    cp_cx_edges_d = {cp: all_counterparts_cx_edges_d.get(cp, {}) for cp in counterparts}
    updated_counterparts = {}
    for counterpart, edges_d in cp_cx_edges_d.items():
        val_dict = {}
        counterpart_layer = counterpart_layers[counterpart]
        for layer in range(node_layer, cg.meta.layer_count):
            edges = edges_d.get(layer, types.empty_2d)
            if edges.size == 0:
                continue
            assert np.all(edges[:, 0] == counterpart)
            edges = fastremap.remap(edges, node_map, preserve_missing_labels=True)
            if layer == counterpart_layer:
                flip_edge = np.array([counterpart, new_id], dtype=basetypes.NODE_ID)
                edges = np.concatenate([edges, [flip_edge]]).astype(basetypes.NODE_ID)
                descendants = descendants_d[new_id]
                mask = np.isin(edges[:, 1], descendants)
                if np.any(mask):
                    masked_edges = edges[mask]
                    masked_edges[:, 1] = new_id
                    edges[mask] = masked_edges
            edges = np.unique(edges, axis=0)
            edges_d[layer] = edges
            val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = edges
        if not val_dict:
            continue
        cg.cache.cross_chunk_edges_cache[counterpart] = edges_d
        updated_counterparts[counterpart] = val_dict
    return updated_counterparts


def _update_neighbor_cx_edges(
    cg,
    new_ids: List[int],
    new_old_id: dict,
    old_new_id,
    *,
    time_stamp,
    parent_ts,
) -> List:
    """
    For each new_id, get counterparts and update its cross chunk edges.
    Some of them maybe updated multiple times so we need to collect them first
    and then write to storage to consolidate the mutations.
    Returns mutations to updated counterparts/partner nodes.
    """
    profiler = get_profiler()
    updated_counterparts = {}

    with profiler.profile("neighbor_get_cross_chunk_edges"):
        newid_cx_edges_d = cg.get_cross_chunk_edges(new_ids, time_stamp=parent_ts)

    node_map = {}
    for k, v in old_new_id.items():
        if len(v) == 1:
            node_map[k] = next(iter(v))

    all_cps = set()
    newid_counterpart_info = {}
    for _id in new_ids:
        counterparts, cp_layers = _get_counterparts(cg, _id, newid_cx_edges_d[_id])
        all_cps.update(counterparts)
        newid_counterpart_info[_id] = cp_layers

    all_cx_edges_d = cg.get_cross_chunk_edges(list(all_cps), time_stamp=parent_ts)
    descendants_d = _get_descendants_batch(cg, new_ids)
    for new_id in new_ids:
        m = {old_id: new_id for old_id in flip_ids(new_old_id, [new_id])}
        node_map.update(m)
        cp_layers = newid_counterpart_info[new_id]
        result = _update_neighbor_cx_edges_single(
            cg, new_id, node_map, cp_layers, all_cx_edges_d, descendants_d
        )
        updated_counterparts.update(result)

    with profiler.profile("neighbor_create_mutations"):
        updated_entries = []
        for node, val_dict in updated_counterparts.items():
            rowkey = serialize_uint64(node)
            row = cg.client.mutate_row(rowkey, val_dict, time_stamp=time_stamp)
            updated_entries.append(row)

    return updated_entries


class CreateParentNodes:
    def __init__(
        self,
        cg,
        *,
        new_l2_ids: Iterable,
        operation_id: basetypes.OPERATION_ID,  # type: ignore
        time_stamp: datetime.datetime,
        new_old_id_d: Dict[np.uint64, Set[np.uint64]] = None,
        old_new_id_d: Dict[np.uint64, Set[np.uint64]] = None,
        old_hierarchy_d: Dict[np.uint64, Dict[int, np.uint64]] = None,
        parent_ts: datetime.datetime = None,
        stitch_mode: bool = False,
        do_sanity_check: bool = True,
        profiler: HierarchicalProfiler = None,
    ):
        self.cg = cg
        self.new_entries = []
        self._new_l2_ids = new_l2_ids
        self._old_hierarchy_d = old_hierarchy_d
        self._new_old_id_d = new_old_id_d
        self._old_new_id_d = old_new_id_d
        self._new_ids_d = defaultdict(list)
        self._opid = operation_id
        self._time_stamp = time_stamp
        self._last_ts = parent_ts
        self.stitch_mode = stitch_mode
        self.do_sanity_check = do_sanity_check
        self._profiler = profiler if profiler else get_profiler()

    def _update_id_lineage(
        self,
        parent: basetypes.NODE_ID,  # type: ignore
        children: np.ndarray,
        layer: int,
        parent_layer: int,
    ):
        # update newly created children; mask others
        mask = np.isin(children, self._new_ids_d[layer])
        for child_id in children[mask]:
            child_old_ids = self._new_old_id_d[child_id]
            for id_ in child_old_ids:
                old_id = self._old_hierarchy_d[id_].get(parent_layer, id_)
                self._new_old_id_d[parent].add(old_id)
                self._old_new_id_d[old_id].add(parent)

    def _get_connected_components(self, node_ids: np.ndarray, layer: int):
        cross_edges_d = self.cg.get_cross_chunk_edges(
            node_ids, time_stamp=self._last_ts
        )
        cx_edges = [types.empty_2d]
        for id_ in node_ids:
            edges_ = cross_edges_d[id_].get(layer, types.empty_2d)
            cx_edges.append(edges_)

        cx_edges = [*cx_edges, np.vstack([node_ids, node_ids]).T]
        cx_edges = np.concatenate(cx_edges).astype(basetypes.NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(cx_edges, make_directed=True)
        components = flatgraph.connected_components(graph)
        return components, graph_ids

    def _get_layer_node_ids(
        self, new_ids: np.ndarray, layer: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # get old identities of new IDs
        old_ids = flip_ids(self._new_old_id_d, new_ids)
        # get their parents, then children of those parents
        old_parents = self.cg.get_parents(old_ids, time_stamp=self._last_ts)
        siblings = self.cg.get_children(np.unique(old_parents), flatten=True)

        # replace old identities with new IDs
        mask = np.isin(siblings, old_ids)
        node_ids = [flip_ids(self._old_new_id_d, old_ids), siblings[~mask], new_ids]
        node_ids = np.concatenate(node_ids).astype(basetypes.NODE_ID)
        node_ids = np.unique(node_ids)
        layer_mask = self.cg.get_chunk_layers(node_ids) == layer
        return node_ids[layer_mask]

    def _update_cross_edge_cache_batched(self, new_ids: list):
        """
        Batch update cross chunk edges in cache for all new IDs at a layer.
        """
        updated_entries = []
        if not new_ids:
            return updated_entries

        parent_layer = self.cg.get_chunk_layer(new_ids[0])
        if parent_layer == 2:
            # L2 cross edges have already been updated
            return updated_entries

        all_children_d = self.cg.get_children(new_ids)
        all_children = np.concatenate(list(all_children_d.values()))
        all_cx_edges_raw = self.cg.get_cross_chunk_edges(
            all_children, time_stamp=self._last_ts
        )
        combined_cx_edges = concatenate_cross_edge_dicts(all_cx_edges_raw.values())
        with self._profiler.profile("latest"):
            updated_cx_edges, edge_nodes = get_latest_edges_wrapper(
                self.cg, combined_cx_edges, parent_ts=self._last_ts
            )

        # update cache with resolved stale edges
        val_ds = defaultdict(dict)
        children_cx_edges = defaultdict(dict)
        for lyr in range(2, self.cg.meta.layer_count):
            edges = updated_cx_edges.get(lyr, types.empty_2d)
            if len(edges) == 0:
                continue
            children, inverse = np.unique(edges[:, 0], return_inverse=True)
            masks = inverse == np.arange(len(children))[:, None]
            for child, mask in zip(children, masks):
                children_cx_edges[child][lyr] = edges[mask]
                val_ds[child][attributes.Connectivity.CrossChunkEdge[lyr]] = edges[mask]

        for c, cx_edges_map in children_cx_edges.items():
            self.cg.cache.cross_chunk_edges_cache[c] = cx_edges_map
            rowkey = serialize_uint64(c)
            row = self.cg.client.mutate_row(rowkey, val_ds[c], time_stamp=self._last_ts)
            updated_entries.append(row)

        # Distribute results back to each parent's cache
        # Key insight: edges[:, 0] are children, map them to their parent
        edge_parents = get_new_nodes(self.cg, edge_nodes, parent_layer, self._last_ts)
        edge_parents_d = dict(zip(edge_nodes, edge_parents))
        for new_id in new_ids:
            children_set = set(all_children_d[new_id])
            parent_cx_edges_d = {}
            for layer in range(parent_layer, self.cg.meta.layer_count):
                edges = updated_cx_edges.get(layer, types.empty_2d)
                if len(edges) == 0:
                    continue
                # Filter to edges whose source is one of this parent's children
                mask = np.isin(edges[:, 0], list(children_set))
                if not np.any(mask):
                    continue

                pedges = edges[mask].copy()
                pedges = fastremap.remap(
                    pedges, edge_parents_d, preserve_missing_labels=True
                )
                parent_cx_edges_d[layer] = np.unique(pedges, axis=0)
                assert np.all(
                    pedges[:, 0] == new_id
                ), f"OP {self._opid}: mismatch {new_id} != {np.unique(pedges[:, 0])}"
            self.cg.cache.cross_chunk_edges_cache[new_id] = parent_cx_edges_d
        return updated_entries

    def _get_new_ids(self, chunk_id, count, is_root):
        batch_size = count
        new_ids = []
        while len(new_ids) < count:
            candidate_ids = self.cg.id_client.create_node_ids(
                chunk_id, batch_size, root_chunk=is_root
            )
            existing = self.cg.client.read_nodes(node_ids=candidate_ids)
            non_existing = set(candidate_ids) - existing.keys()
            new_ids.extend(non_existing)
            batch_size = min(batch_size * 2, 2**16)
        return new_ids[:count]

    def _get_new_parents(self, layer, ccs, graph_ids) -> tuple[dict, dict]:
        cc_layer_chunk_map = {}
        size_map = defaultdict(int)
        for i, cc_idx in enumerate(ccs):
            parent_layer = layer + 1  # must be reset for each connected component
            cc_ids = graph_ids[cc_idx]
            if len(cc_ids) == 1:
                # skip connection
                parent_layer = self.cg.meta.layer_count
                cx_edges_d = self.cg.get_cross_chunk_edges(
                    [cc_ids[0]], time_stamp=self._last_ts
                )
                for l in range(layer + 1, self.cg.meta.layer_count):
                    if len(cx_edges_d[cc_ids[0]].get(l, types.empty_2d)) > 0:
                        parent_layer = l
                        break
            chunk_id = self.cg.get_parent_chunk_id(cc_ids[0], parent_layer)
            cc_layer_chunk_map[i] = (parent_layer, chunk_id)
            size_map[chunk_id] += 1

        chunk_ids = list(size_map.keys())
        random.shuffle(chunk_ids)
        chunk_new_ids_map = {}
        layers = self.cg.get_chunk_layers(chunk_ids)
        for c, l in zip(chunk_ids, layers):
            is_root = l == self.cg.meta.layer_count
            chunk_new_ids_map[c] = self._get_new_ids(c, size_map[c], is_root)
        return chunk_new_ids_map, cc_layer_chunk_map

    def _create_new_parents(self, layer: int):
        """
        keep track of old IDs
        merge - one new ID has 2 old IDs
        split - two/more new IDs have the same old ID
        get parents of old IDs, their children are the siblings
        those siblings include old IDs, replace with new
        get cross edges of all, find connected components
        update parent old IDs
        """
        new_ids = self._new_ids_d[layer]
        layer_node_ids = self._get_layer_node_ids(new_ids, layer)
        ccs, _ids = self._get_connected_components(layer_node_ids, layer)
        new_parents_map, cc_layer_chunk_map = self._get_new_parents(layer, ccs, _ids)

        for i, cc_indices in enumerate(ccs):
            cc_ids = _ids[cc_indices]
            parent_layer, chunk_id = cc_layer_chunk_map[i]
            parent = new_parents_map[chunk_id].pop()

            self._new_ids_d[parent_layer].append(parent)
            self._update_id_lineage(parent, cc_ids, layer, parent_layer)
            self.cg.cache.children_cache[parent] = cc_ids
            cache_utils.update(self.cg.cache.parents_cache, cc_ids, parent)
            if not self.do_sanity_check:
                continue

            try:
                sanity_check_single(self.cg, parent, self._opid)
            except AssertionError:
                pairs = [
                    (a, b) for idx, a in enumerate(cc_ids) for b in cc_ids[idx + 1 :]
                ]
                for c1, c2 in pairs:
                    l2c1 = self.cg.get_l2children([c1])
                    l2c2 = self.cg.get_l2children([c2])
                    if np.intersect1d(l2c1, l2c2).size:
                        c = np.intersect1d(l2c1, l2c2)
                        msg = f"{self._opid}: {layer} {c1} {c2} common children {c}"
                        raise ValueError(msg)

    def run(self) -> Iterable:
        """
        After new level 2 IDs are created, create parents in higher layers.
        Cross edges are used to determine existing siblings.
        """
        self._new_ids_d[2] = self._new_l2_ids
        for layer in range(2, self.cg.meta.layer_count):
            new_nodes = self._new_ids_d[layer]
            if len(new_nodes) == 0:
                continue
            self.cg.cache.new_ids.update(new_nodes)
            # all new IDs in this layer have been created
            # update their cross chunk edges and their neighbors'
            with self._profiler.profile(f"l{layer}_update_cx_cache"):
                entries = self._update_cross_edge_cache_batched(new_nodes)
                self.new_entries.extend(entries)

            with self._profiler.profile(f"l{layer}_update_neighbor_cx"):
                entries = _update_neighbor_cx_edges(
                    self.cg,
                    new_nodes,
                    self._new_old_id_d,
                    self._old_new_id_d,
                    time_stamp=self._time_stamp,
                    parent_ts=self._last_ts,
                )
                self.new_entries.extend(entries)
            with self._profiler.profile(f"l{layer}_create_new_parents"):
                self._create_new_parents(layer)
        return self._new_ids_d[self.cg.meta.layer_count]

    def _update_root_id_lineage(self):
        if self.stitch_mode:
            return
        new_roots = self._new_ids_d[self.cg.meta.layer_count]
        former_roots = flip_ids(self._new_old_id_d, new_roots)
        former_roots = np.unique(former_roots)

        err = f"new roots are inconsistent; op {self._opid}"
        assert len(former_roots) < 2 or len(new_roots) < 2, err
        for new_root_id in new_roots:
            val_dict = {
                attributes.Hierarchy.FormerParent: former_roots,
                attributes.OperationLogs.OperationID: self._opid,
            }
            self.new_entries.append(
                self.cg.client.mutate_row(
                    serialize_uint64(new_root_id),
                    val_dict,
                    time_stamp=self._time_stamp,
                )
            )

        for former_root_id in former_roots:
            val_dict = {
                attributes.Hierarchy.NewParent: np.array(
                    new_roots, dtype=basetypes.NODE_ID
                ),
                attributes.OperationLogs.OperationID: self._opid,
            }
            self.new_entries.append(
                self.cg.client.mutate_row(
                    serialize_uint64(former_root_id),
                    val_dict,
                    time_stamp=self._time_stamp,
                )
            )

    def _get_cross_edges_val_dicts(self):
        val_dicts = {}
        for layer in range(2, self.cg.meta.layer_count):
            new_ids = np.array(self._new_ids_d[layer], dtype=basetypes.NODE_ID)
            cross_edges_d = self.cg.get_cross_chunk_edges(
                new_ids, time_stamp=self._last_ts
            )
            for id_ in new_ids:
                val_dict = {}
                for layer, edges in cross_edges_d[id_].items():
                    val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = edges
                val_dicts[id_] = val_dict
        return val_dicts

    def create_new_entries(self) -> List:
        max_layer = self.cg.meta.layer_count
        val_dicts = self._get_cross_edges_val_dicts()
        for layer in range(2, max_layer + 1):
            new_ids = self._new_ids_d[layer]
            for id_ in new_ids:
                if self.do_sanity_check:
                    root_layer = self.cg.get_chunk_layer(self.cg.get_root(id_))
                    assert root_layer == max_layer, (id_, self.cg.get_root(id_))

                    if layer < max_layer:
                        try:
                            _parent = self.cg.get_parent(id_)
                            _children = self.cg.get_children(_parent)
                            assert id_ in _children, (layer, id_, _parent, _children)
                        except TypeError as e:
                            logger.error(id_, _parent, self.cg.get_root(id_))
                            raise TypeError from e

                val_dict = val_dicts.get(id_, {})
                children = self.cg.get_children(id_)
                err = f"parent layer less than children; op {self._opid}"
                assert np.max(
                    self.cg.get_chunk_layers(children)
                ) < self.cg.get_chunk_layer(id_), err
                val_dict[attributes.Hierarchy.Child] = children
                self.new_entries.append(
                    self.cg.client.mutate_row(
                        serialize_uint64(id_),
                        val_dict,
                        time_stamp=self._time_stamp,
                    )
                )
                for child_id in children:
                    self.new_entries.append(
                        self.cg.client.mutate_row(
                            serialize_uint64(child_id),
                            {attributes.Hierarchy.Parent: id_},
                            time_stamp=self._time_stamp,
                        )
                    )

        with self._profiler.profile("update_root_id_lineage"):
            self._update_root_id_lineage()
