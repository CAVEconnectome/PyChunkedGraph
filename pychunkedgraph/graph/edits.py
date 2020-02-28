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
from . import attributes
from .utils import basetypes
from .utils import flatgraph
from .utils.context_managers import TimeIt
from .utils.serializers import serialize_uint64
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


def _analyze_edges_to_add(
    cg, atomic_edges: Iterable[np.ndarray]
) -> Tuple[Iterable, Dict]:
    """
    Determine if atomic edges are within the chunk.
    If not, they are cross edges between two L2 IDs in adjacent chunks.
    Returns edges between L2 IDs and atomic cross edges.
    """
    nodes = np.unique(atomic_edges)
    parents = cg.get_parents(nodes)
    sv_parent_d = dict(zip(nodes, parents))
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    parent_edges = [
        [sv_parent_d[edge_[0]], sv_parent_d[edge_[1]]]
        for edge_ in atomic_edges[edge_layers == 1]
    ]

    # cross chunk edges
    atomic_cross_edges_d = defaultdict(lambda: defaultdict(list))
    for layer in np.unique(edge_layers[edge_layers > 1]):
        layer_edges = atomic_edges[edge_layers == layer]
        for edge in layer_edges:
            parent_1 = sv_parent_d[edge[0]]
            parent_2 = sv_parent_d[edge[1]]
            atomic_cross_edges_d[parent_1][layer].append(edge)
            atomic_cross_edges_d[parent_2][layer].append(edge[::-1])
            parent_edges.extend([[parent_1, parent_1], [parent_2, parent_2]])
    return (parent_edges, atomic_cross_edges_d)


def _get_relevant_components(edges: np.ndarray, supervoxels: np.ndarray) -> Tuple:
    graph, _, _, graph_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)
    relevant_ccs = []
    # remove if connected component contains no sources or no sinks
    # when merging, there must be only two components
    for cc_idx in ccs:
        cc = graph_ids[cc_idx]
        if np.any(np.in1d(supervoxels, cc)):
            relevant_ccs.append(cc)
    assert len(relevant_ccs) == 2
    return relevant_ccs


def merge_preprocess(
    cg, *, subgraph_edges: np.ndarray, supervoxels: np.ndarray
) -> np.ndarray:
    """
    Determine if a fake edge needs to be added.
    Get subgraph within the bounding box
    Add fake edge if there are no inactive edges between two components.
    """
    edges_roots = cg.get_roots(subgraph_edges.ravel()).reshape(-1, 2)
    active_mask = edges_roots[:, 0] == edges_roots[:, 1]
    active, inactive = subgraph_edges[active_mask], subgraph_edges[~active_mask]
    relevant_ccs = _get_relevant_components(active, supervoxels)
    source_mask = np.in1d(inactive[:, 0], relevant_ccs[0])
    sink_mask = np.in1d(inactive[:, 1], relevant_ccs[1])
    return inactive[source_mask & sink_mask]


def _check_fake_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    inactive_edges: Iterable[np.ndarray],
    time_stamp: datetime.datetime,
) -> Tuple[Iterable[np.ndarray], Iterable]:
    if inactive_edges.size:
        return inactive_edges, []

    rows = []
    supervoxels = atomic_edges.ravel()
    chunk_ids = cg.get_chunk_ids_from_node_ids(cg.get_parents(supervoxels))
    sv_chunk_id_d = dict(zip(supervoxels, chunk_ids))
    for edge in atomic_edges:
        id1, id2 = sv_chunk_id_d[edge[0]], sv_chunk_id_d[edge[1]]
        val_dict = {}
        val_dict[attributes.Connectivity.FakeEdges] = np.array(
            [[edge]], dtype=basetypes.NODE_ID
        )
        id1 = serialize_uint64(id1, fake_edges=True)
        rows.append(cg.client.mutate_row(id1, val_dict, time_stamp=time_stamp,))
        val_dict = {}
        val_dict[attributes.Connectivity.FakeEdges] = np.array(
            [edge[::-1]], dtype=basetypes.NODE_ID
        )
        id2 = serialize_uint64(id2, fake_edges=True)
        rows.append(cg.client.mutate_row(id2, val_dict, time_stamp=time_stamp,))
    return atomic_edges, rows


def add_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    inactive_edges: Iterable[np.ndarray],
    operation_id: np.uint64 = None,
    time_stamp: datetime.datetime = None,
):
    # TODO add docs
    atomic_edges, rows = _check_fake_edges(
        cg,
        atomic_edges=atomic_edges,
        inactive_edges=inactive_edges,
        time_stamp=time_stamp,
    )
    edges, l2_atomic_cross_edges_d = _analyze_edges_to_add(cg, atomic_edges)
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
        cg.cache.children_cache[new_id] = np.concatenate(
            [atomic_children_d[l2id] for l2id in l2ids_]
        )
        cg.cache.atomic_cx_edges_cache[new_id] = concatenate_cross_edge_dicts(
            [atomic_cross_edges_d[l2id] for l2id in l2ids_]
        )
        cache.update(cg.cache.parents_cache, cg.cache.children_cache[new_id], new_id)
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
    return create_parents.run(), new_l2_ids, rows + create_parents.create_new_entries()


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
    cg.cache = cache.CacheService(cg)
    edges, _ = _analyze_edges_to_add(cg, atomic_edges)
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
            cg.cache.children_cache[new_id] = graph_ids[cc]
            cg.cache.atomic_cx_edges_cache[new_id] = _filter_component_cross_edges(
                cg.cache.children_cache[new_id], cross_edges, cross_edge_layers
            )
            cache.update(
                cg.cache.parents_cache, cg.cache.children_cache[new_id], new_id
            )
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
    return create_parents.run(), new_l2_ids, create_parents.create_new_entries()


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
        self._operation_id = operation_id
        self._time_stamp = time_stamp

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
                self.cg.get_parent_chunk_id(cc_ids[0], parent_layer),
                root_chunk=parent_layer == self.cg.meta.layer_count,
            )
            self._new_ids_d[parent_layer].append(parent_id)
            self.cg.cache.children_cache[parent_id] = cc_ids
            cache.update(
                self.cg.cache.parents_cache,
                self.cg.cache.children_cache[parent_id],
                parent_id,
            )
            self._update_id_lineage(parent_id, cc_ids, layer, parent_layer)

    def run(self) -> Iterable:
        """
        After new level 2 IDs are created, create parents in higher layers.
        Cross edges are used to determine existing siblings.
        """
        self._new_ids_d[2] = self._new_l2_ids
        for layer in range(2, self.cg.meta.layer_count):
            if len(self._new_ids_d[layer]) == 0:
                continue
            self._create_new_parents(layer)
        return self._new_ids_d[self.cg.meta.layer_count]

    def _update_root_id_lineage(self):
        new_root_ids = self._new_ids_d[self.cg.meta.layer_count]
        former_root_ids = [self._new_old_id_d[id_] for id_ in new_root_ids]
        former_root_ids = np.unique(np.array(former_root_ids, dtype=basetypes.NODE_ID))
        assert len(former_root_ids) < 2 or len(new_root_ids) < 2
        rows = []
        for new_root_id in new_root_ids:
            val_dict = {
                attributes.Hierarchy.FormerParent: np.array(former_root_ids),
                attributes.OperationLogs.OperationID: self._operation_id,
            }
            rows.append(
                self.cg.client.mutate_row(
                    serialize_uint64(new_root_id),
                    val_dict,
                    time_stamp=self._time_stamp,
                )
            )

        for former_root_id in former_root_ids:
            val_dict = {
                attributes.Hierarchy.NewParent: np.array(new_root_ids),
                attributes.OperationLogs.OperationID: self._operation_id,
            }
            rows.append(
                self.cg.client.mutate_row(
                    serialize_uint64(former_root_id),
                    val_dict,
                    time_stamp=self._time_stamp,
                )
            )
        return rows

    def _get_atomic_cross_edges_val_dict(self):
        new_ids = np.array(self._new_ids_d[2], dtype=basetypes.NODE_ID)
        val_dicts = {}
        atomic_cross_edges_d = self.cg.get_atomic_cross_edges(new_ids)
        for id_ in new_ids:
            val_dict = {}
            for layer, edges in atomic_cross_edges_d[id_].items():
                val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = edges
            val_dicts[id_] = val_dict
        return val_dicts

    def create_new_entries(self):
        rows = []
        val_dicts = self._get_atomic_cross_edges_val_dict()
        for layer in range(2, self.cg.meta.layer_count + 1):
            new_ids = self._new_ids_d[layer]
            for id_ in new_ids:
                val_dict = val_dicts.get(id_, {})
                children = self.cg.get_children(id_)
                assert np.max(
                    self.cg.get_chunk_layers(children)
                ) < self.cg.get_chunk_layer(id_), "Parent layer less than children."
                val_dict[attributes.Hierarchy.Child] = children
                rows.append(
                    self.cg.client.mutate_row(
                        serialize_uint64(id_), val_dict, time_stamp=self._time_stamp,
                    )
                )
                for child_id in children:
                    val_dict = {attributes.Hierarchy.Parent: id_}
                    rows.append(
                        self.cg.client.mutate_row(
                            serialize_uint64(child_id),
                            val_dict,
                            time_stamp=self._time_stamp,
                        )
                    )
        return rows + self._update_root_id_lineage()
