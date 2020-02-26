import datetime
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Sequence
from collections import defaultdict

from . import cache
from . import types
from .utils import basetypes
from .utils import flatgraph
from .utils.context_managers import TimeIt
from .connectivity.nodes import edge_exists
from .connectivity.search import check_reachability
from .edges.utils import filter_min_layer_cross_edges
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple
from ..utils.general import in2d
from ..utils.general import reverse_dictionary


def _init_old_hierarchy(cg, l2ids: np.ndarray):
    new_old_id_d = defaultdict(list)
    old_new_id_d = defaultdict(list)
    old_hierarchy_d = {id_: {2: id_} for id_ in l2ids}
    for id_ in l2ids:
        parents_d = cg.get_all_parents_dict(id_)
        old_hierarchy_d[id_].update(parents_d)
        for parent in parents_d.values():
            old_hierarchy_d[parent] = old_hierarchy_d[id_]
    return new_old_id_d, old_new_id_d, old_hierarchy_d


def _analyze_atomic_edges(
    cg, atomic_edges: Iterable[np.ndarray]
) -> Tuple[Iterable, Dict]:
    """
    Determine if atomic edges are within the chunk.
    If not, they are cross edges between two L2 IDs in adjacent chunks.
    Returns edges between L2 IDs and atomic cross edges.
    """
    nodes = np.unique(atomic_edges)
    parents = cg.get_parents(nodes)
    parents_d = dict(zip(nodes, parents))
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    mask = edge_layers == 1

    # initialize with in-chunk edges
    with TimeIt("get_parents edges"):
        parent_edges = [
            [parents_d[edge_[0]], parents_d[edge_[1]]] for edge_ in atomic_edges[mask]
        ]

    # cross chunk edges
    atomic_cross_edges_d = {}
    for edge, layer in zip(atomic_edges[~mask], edge_layers[~mask]):
        parent_1 = parents_d[edge[0]]
        parent_2 = parents_d[edge[1]]
        atomic_cross_edges_d[parent_1] = {layer: [edge]}
        atomic_cross_edges_d[parent_2] = {layer: [edge[::-1]]}
        parent_edges.append([parent_1, parent_1])
        parent_edges.append([parent_2, parent_2])
    return (parent_edges, atomic_cross_edges_d)


def merge_preprocess(
    cg,
    edges: Iterable[np.ndarray],
    subgraph_edges: Iterable[np.ndarray],
    root_l2ids_d: Dict,
    l2id_agglomeration_d: Dict,
):
    """
    Determine if a fake edge needs to be added.
    Get subgraph within the bounding box
        If there is a path between the supervoxels, fake edge is not necessary.
            Determine all the inactive edges between the L2 chidlren of the 2 node IDs.
        If no path exists, add user input as a fake edge.
    """
    node_ids = np.unique(edges)
    subgraph_edges = np.concatenate([subgraph_edges, np.vstack([node_ids, node_ids]).T])
    graph, _, _, node_ids = flatgraph.build_gt_graph(subgraph_edges, is_directed=False)
    reachable = check_reachability(graph, edges[:, 0], edges[:, 1], node_ids)
    if reachable[0]:
        sv_parent_d = {}
        out_cross_edges = [types.empty_2d]
        for l2_agg in l2id_agglomeration_d.values():
            edges_ = (l2_agg.out_edges + l2_agg.cross_edges).get_pairs()
            sv_parent_d.update(dict(zip(edges_[:, 0], [l2_agg.node_id] * edges_.size)))
            out_cross_edges.append(edges_)

        out_cross_edges_sv = np.concatenate(out_cross_edges)
        get_sv_parents = np.vectorize(
            lambda x: sv_parent_d.get(x, 0), otypes=[np.uint64]
        )

        out_cross_edges = get_sv_parents(out_cross_edges_sv)
        del sv_parent_d

        out_cross_edges_unique = np.unique(out_cross_edges, axis=0)
        mask = np.in1d(out_cross_edges, out_cross_edges_unique)

        add_edges = [types.empty_2d]

        for edge in out_cross_edges_unique:
            mask = np.in1d(out_cross_edges, edge)
            print("mask", mask)
            add_edges.append(out_cross_edges_sv[mask])
        return np.concatenate(add_edges)
    return edges


def add_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    operation_id: np.uint64 = None,
    time_stamp: datetime.datetime = None,
):
    # TODO add docs
    cache.clear()
    cg.cache = cache.CacheService(cg)
    edges, l2_atomic_cross_edges_d = _analyze_atomic_edges(cg, atomic_edges)
    l2ids = np.unique(edges)
    new_old_id_d, old_new_id_d, old_hierarchy_d = _init_old_hierarchy(cg, l2ids)
    atomic_children_d = cg.get_children(l2ids)
    atomic_cross_edges_d = merge_cross_edge_dicts_multiple(
        cg.get_atomic_cross_edges(l2ids), l2_atomic_cross_edges_d
    )

    graph, _, _, graph_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    components = flatgraph.connected_components(graph)
    new_l2_ids = []
    for cc_indices in components:
        l2ids_ = graph_ids[cc_indices]
        new_id = cg.id_client.create_node_id(cg.get_chunk_id(l2ids_[0]))
        cache.CHILDREN[new_id] = np.concatenate(
            [atomic_children_d[l2id] for l2id in l2ids_]
        )
        cache.ATOMIC_CX_EDGES[new_id] = concatenate_cross_edge_dicts(
            [atomic_cross_edges_d[l2id] for l2id in l2ids_]
        )
        cache.update(cache.PARENTS, cache.CHILDREN[new_id], new_id)
        new_l2_ids.append(new_id)
        new_old_id_d[new_id].extend(l2ids_)
        for id_ in l2ids_:
            old_new_id_d[id_].append(new_id)

    create_parents = CreateParentNodes(
        cg,
        new_l2_ids=new_l2_ids,
        old_hierarchy_d=old_hierarchy_d,
        new_old_id_d=new_old_id_d,
        old_new_id_d=old_new_id_d,
        operation_id=operation_id,
        time_stamp=time_stamp,
    )
    return create_parents.run()


def _process_l2_agglomeration(
    agg: types.Agglomeration,
    removed_edges: np.ndarray,
    atomic_cross_edges_d: Dict[int, np.ndarray],
):
    """
    For a given L2 id, remove given edges
    and calculate new connected components.
    """
    chunk_edges = agg.in_edges.get_pairs()
    cross_edges = np.concatenate([*atomic_cross_edges_d[agg.node_id].values()])
    chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]
    cross_edges = cross_edges[~in2d(cross_edges, removed_edges)]

    isolated_ids = agg.supervoxels[~np.in1d(agg.supervoxels, chunk_edges)]
    isolated_edges = np.column_stack((isolated_ids, isolated_ids))
    graph, _, _, graph_ids = flatgraph.build_gt_graph(
        np.concatenate([chunk_edges, isolated_edges]), make_directed=True
    )
    return flatgraph.connected_components(graph), graph_ids, cross_edges


def _filter_component_cross_edges(
    cc_ids: np.ndarray, cross_edges: np.ndarray, cross_edge_layers: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Filters cross edges for a connected component `cc_ids`
    from `cross_edges` of the complete chunk.
    """
    mask = np.in1d(cross_edges[:, 0], cc_ids)
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
    l2id_agglomeration_d: Dict,
    operation_id: basetypes.OPERATION_ID = None,
    time_stamp: datetime.datetime = None,
):
    # TODO add docs
    cache.clear()
    cg.cache = cache.CacheService(cg)
    edges, _ = _analyze_atomic_edges(cg, atomic_edges)
    l2ids = np.unique(edges)
    new_old_id_d, old_new_id_d, old_hierarchy_d = _init_old_hierarchy(cg, l2ids)
    l2id_chunk_id_d = dict(zip(l2ids, cg.get_chunk_ids_from_node_ids(l2ids)))
    atomic_cross_edges_d = cg.get_atomic_cross_edges(l2ids)

    removed_edges = np.concatenate([atomic_edges, atomic_edges[:, ::-1]], axis=0)
    new_l2_ids = []
    for id_ in l2ids:
        l2_agg = l2id_agglomeration_d[id_]
        ccs, graph_ids, cross_edges = _process_l2_agglomeration(
            l2_agg, removed_edges, atomic_cross_edges_d
        )
        cross_edge_layers = cg.get_cross_chunk_edges_layer(cross_edges)
        new_parent_ids = cg.id_client.create_node_ids(
            l2id_chunk_id_d[l2_agg.node_id], len(ccs)
        )
        for i_cc, cc in enumerate(ccs):
            new_id = new_parent_ids[i_cc]
            cache.CHILDREN[new_id] = graph_ids[cc]
            cache.ATOMIC_CX_EDGES[new_id] = _filter_component_cross_edges(
                cache.CHILDREN[new_id], cross_edges, cross_edge_layers
            )
            cache.update(cache.PARENTS, cache.CHILDREN[new_id], new_id)
            new_l2_ids.append(new_id)
            new_old_id_d[new_id].append(id_)
            old_new_id_d[id_].append(new_id)

    create_parents = CreateParentNodes(
        cg,
        new_l2_ids=new_l2_ids,
        old_hierarchy_d=old_hierarchy_d,
        new_old_id_d=new_old_id_d,
        old_new_id_d=old_new_id_d,
        operation_id=operation_id,
        time_stamp=time_stamp,
    )
    return atomic_edges, create_parents.run()


class CreateParentNodes:
    def __init__(
        self,
        cg,
        *,
        new_l2_ids: Iterable,
        old_hierarchy_d: Dict[np.uint64, Dict[int, np.uint64]] = None,
        new_old_id_d: Dict[np.uint64, Iterable[np.uint64]] = None,
        old_new_id_d: Dict[np.uint64, Iterable[np.uint64]] = None,
        operation_id: basetypes.OPERATION_ID,
        time_stamp: datetime.datetime,
    ):
        # TODO add docs
        self.cg = cg
        self._new_l2_ids = new_l2_ids
        self._old_hierarchy_d = old_hierarchy_d
        self._new_old_id_d = new_old_id_d
        self._old_new_id_d = old_new_id_d
        self._new_ids_d = defaultdict(list)  # new IDs in each layer
        self._cross_edges_d = {}
        self.operation_id = operation_id
        self.time_stamp = time_stamp

    def _update_id_lineage(
        self,
        parent: basetypes.NODE_ID,
        children: np.ndarray,
        layer: int,
        parent_layer: int,
    ):
        mask = np.in1d(children, self._new_ids_d[layer], assume_unique=True)
        for child_id in children[mask]:
            child_old_ids = self._new_old_id_d[child_id]
            for id_ in child_old_ids:
                try:
                    old_id = self._old_hierarchy_d[id_][parent_layer]
                    self._new_old_id_d[parent].append(old_id)
                    self._old_new_id_d[old_id].append(parent)
                except KeyError:
                    pass

    def _get_connected_components(self, node_ids: np.ndarray, layer: int):
        cached = np.fromiter(self._cross_edges_d.keys(), dtype=basetypes.NODE_ID)
        not_cached = node_ids[~np.in1d(node_ids, cached)]
        self._cross_edges_d.update(
            self.cg.get_cross_chunk_edges(not_cached, uplift=False)
        )
        sv_parent_d = {}
        sv_cross_edges = [types.empty_2d]
        for id_ in node_ids:
            edges_ = self._cross_edges_d[id_].get(layer, types.empty_2d)
            sv_parent_d.update(dict(zip(edges_[:, 0], [id_] * edges_.size)))
            sv_cross_edges.append(edges_)

        get_sv_parents = np.vectorize(sv_parent_d.get, otypes=[np.uint64])
        cross_edges = get_sv_parents(np.concatenate(sv_cross_edges))
        del sv_parent_d
        cross_edges = np.concatenate([cross_edges, np.vstack([node_ids, node_ids]).T])
        graph, _, _, graph_ids = flatgraph.build_gt_graph(
            np.unique(cross_edges, axis=0), make_directed=True
        )
        return flatgraph.connected_components(graph), graph_ids

    def _get_layer_node_ids(self, new_ids: np.ndarray):
        old_ids = [self._new_old_id_d[id_] for id_ in new_ids]
        old_ids = np.unique(np.array(old_ids, dtype=basetypes.NODE_ID))
        children_d = self.cg.get_children(np.unique(self.cg.get_parents(old_ids)))
        node_ids = np.concatenate([*children_d.values()])
        mask = np.in1d(node_ids, old_ids, assume_unique=True)
        return np.concatenate(
            [
                np.array(self._old_new_id_d[id_], dtype=basetypes.NODE_ID)
                for id_ in node_ids[mask]
            ]
            + [node_ids[~mask]]
        )

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
        components, graph_ids = self._get_connected_components(
            self._get_layer_node_ids(new_ids), layer
        )
        for cc_indices in components:
            parent_layer = layer + 1
            cc_ids = graph_ids[cc_indices]
            if len(cc_ids) == 1:
                parent_layer = list(self._cross_edges_d[cc_ids[0]].keys())[0]
            parent_id = self.cg.id_client.create_node_id(
                self.cg.get_parent_chunk_id(cc_ids[0], parent_layer)
            )
            self._new_ids_d[parent_layer].append(parent_id)
            cache.CHILDREN[parent_id] = cc_ids
            cache.update(cache.PARENTS, cache.CHILDREN[parent_id], parent_id)
            self._update_id_lineage(parent_id, cc_ids, layer, parent_layer)

    def run(self) -> Iterable:
        """
        After new level 2 IDs are created, create parents in higher layers.
        Cross edges are used to determine existing siblings.
        """
        self._new_ids_d[2] = self._new_l2_ids
        for layer in range(2, self.cg.meta.layer_count):
            # print()
            # print("*" * 100)
            # print("layer", layer, self._new_ids_d[layer])
            if len(self._new_ids_d[layer]) == 0:
                continue
            self._create_new_parents(layer)
        return self._new_ids_d
        # return self._new_ids_d[self.cg.meta.layer_count]
