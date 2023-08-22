# pylint: disable=invalid-name, missing-docstring, too-many-locals, c-extension-no-member

import datetime
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from collections import defaultdict

import fastremap
import numpy as np
import fastremap

from . import types
from . import attributes
from . import cache as cache_utils
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts
from .utils import basetypes
from .utils import flatgraph
from .utils.serializers import serialize_uint64
from ..logging.log_db import TimeIt
from ..utils.general import in2d


def _init_old_hierarchy(cg, l2ids: np.ndarray, parent_ts: datetime.datetime = None):
    new_old_id_d = defaultdict(set)
    old_new_id_d = defaultdict(set)
    old_hierarchy_d = {id_: {2: id_} for id_ in l2ids}
    for id_ in l2ids:
        layer_parent_d = cg.get_all_parents_dict(id_, time_stamp=parent_ts)
        old_hierarchy_d[id_].update(layer_parent_d)
        for parent in layer_parent_d.values():
            old_hierarchy_d[parent] = old_hierarchy_d[id_]
    return new_old_id_d, old_new_id_d, old_hierarchy_d


def _analyze_affected_edges(
    cg, atomic_edges: Iterable[np.ndarray], parent_ts: datetime.datetime = None
) -> Tuple[Iterable, Dict]:
    """
    Returns l2 edges within chunk and self edges for nodes in cross chunk edges.

    Also returns new cross edges dicts for nodes crossing chunk boundary.
    """
    supervoxels = np.unique(atomic_edges)
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
    return parent_edges, cross_edges_d


def _get_relevant_components(edges: np.ndarray, supervoxels: np.ndarray) -> Tuple:
    edges = np.concatenate([edges, np.vstack([supervoxels, supervoxels]).T])
    graph, _, _, graph_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)
    relevant_ccs = []
    # remove if connected component contains no sources or no sinks
    # when merging, there must be only two components
    for cc_idx in ccs:
        cc = graph_ids[cc_idx]
        if np.any(np.in1d(supervoxels, cc)):
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

    relevant_ccs = _get_relevant_components(np.concatenate(active_edges), supervoxels)
    inactive = np.concatenate(inactive_edges)
    _inactive = [types.empty_2d]
    # source to sink edges
    source_mask = np.in1d(inactive[:, 0], relevant_ccs[0])
    sink_mask = np.in1d(inactive[:, 1], relevant_ccs[1])
    _inactive.append(inactive[source_mask & sink_mask])

    # sink to source edges
    sink_mask = np.in1d(inactive[:, 1], relevant_ccs[0])
    source_mask = np.in1d(inactive[:, 0], relevant_ccs[1])
    _inactive.append(inactive[source_mask & sink_mask])
    _inactive = np.concatenate(_inactive)
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
):
    edges, l2_cross_edges_d = _analyze_affected_edges(
        cg, atomic_edges, parent_ts=parent_ts
    )
    l2ids = np.unique(edges)
    if not allow_same_segment_merge:
        roots = cg.get_roots(l2ids, assert_roots=True, time_stamp=parent_ts)
        assert np.unique(roots).size == 2, "L2 IDs must belong to different roots."
    new_old_id_d, old_new_id_d, old_hierarchy_d = _init_old_hierarchy(
        cg, l2ids, parent_ts=parent_ts
    )
    atomic_children_d = cg.get_children(l2ids)
    cross_edges_d = merge_cross_edge_dicts(
        cg.get_cross_chunk_edges(l2ids), l2_cross_edges_d
    )

    graph, _, _, graph_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    components = flatgraph.connected_components(graph)
    new_l2_ids = []
    for cc_indices in components:
        l2ids_ = graph_ids[cc_indices]
        new_id = cg.id_client.create_node_id(cg.get_chunk_id(l2ids_[0]))
        new_l2_ids.append(new_id)
        new_old_id_d[new_id].update(l2ids_)
        for id_ in l2ids_:
            old_new_id_d[id_].add(new_id)

        # update cache
        # map parent to new merged children and vice versa
        merged_children = np.concatenate([atomic_children_d[l2id] for l2id in l2ids_])
        cg.cache.children_cache[new_id] = merged_children
        cache_utils.update(cg.cache.parents_cache, merged_children, new_id)

    # update cross chunk edges by replacing old_ids with new
    # this can be done only after all new IDs have been created
    for new_id, cc_indices in zip(new_l2_ids, components):
        l2ids_ = graph_ids[cc_indices]
        new_cx_edges_d = {}
        cx_edges = [cross_edges_d[l2id] for l2id in l2ids_]
        cx_edges_d = concatenate_cross_edge_dicts(cx_edges, unique=True)
        temp_map = {k: next(iter(v)) for k, v in old_new_id_d.items()}
        for layer, edges in cx_edges_d.items():
            edges = fastremap.remap(edges, temp_map, preserve_missing_labels=True)
            new_cx_edges_d[layer] = edges
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
    )

    new_roots = create_parents.run()
    print()
    print("layers", cg.meta.layer_count, "new_roots", new_roots)
    new_entries = create_parents.create_new_entries()
    return new_roots, new_l2_ids, new_entries


def _process_l2_agglomeration(agg: types.Agglomeration, removed_edges: np.ndarray):
    """
    For a given L2 id, remove given edges; calculate new connected components.
    """
    chunk_edges = agg.in_edges.get_pairs()
    chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]

    isolated_ids = agg.supervoxels[~np.in1d(agg.supervoxels, chunk_edges)]
    isolated_edges = np.column_stack((isolated_ids, isolated_ids))
    graph, _, _, graph_ids = flatgraph.build_gt_graph(
        np.concatenate([chunk_edges, isolated_edges]), make_directed=True
    )
    return flatgraph.connected_components(graph), graph_ids


def remove_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    l2id_agglomeration_d: Dict,
    operation_id: basetypes.OPERATION_ID = None,
    time_stamp: datetime.datetime = None,
    parent_ts: datetime.datetime = None,
):
    edges, _ = _analyze_affected_edges(cg, atomic_edges, parent_ts=parent_ts)
    l2ids = np.unique(edges)
    roots = cg.get_roots(l2ids, assert_roots=True, time_stamp=parent_ts)
    assert np.unique(roots).size == 1, "L2 IDs must belong to same root."

    new_old_id_d, old_new_id_d, old_hierarchy_d = _init_old_hierarchy(
        cg, l2ids, parent_ts=parent_ts
    )
    l2id_chunk_id_d = dict(zip(l2ids.tolist(), cg.get_chunk_ids_from_node_ids(l2ids)))
    cross_edges_d = cg.get_cross_chunk_edges(l2ids)

    removed_edges = np.concatenate([atomic_edges, atomic_edges[:, ::-1]], axis=0)
    new_l2_ids = []
    for id_ in l2ids:
        l2_agg = l2id_agglomeration_d[id_]
        ccs, graph_ids = _process_l2_agglomeration(l2_agg, removed_edges)
        new_parent_ids = cg.id_client.create_node_ids(
            l2id_chunk_id_d[l2_agg.node_id], len(ccs)
        )
        for i_cc, cc in enumerate(ccs):
            new_id = new_parent_ids[i_cc]
            cg.cache.children_cache[new_id] = graph_ids[cc]
            cg.cache.atomic_cx_edges_cache[new_id] = None
            cache_utils.update(cg.cache.parents_cache, graph_ids[cc], new_id)
            new_l2_ids.append(new_id)
            new_old_id_d[new_id].add(id_)
            old_new_id_d[id_].add(new_id)

    create_parents = CreateParentNodes(
        cg,
        new_l2_ids=new_l2_ids,
        old_hierarchy_d=old_hierarchy_d,
        new_old_id_d=new_old_id_d,
        old_new_id_d=old_new_id_d,
        operation_id=operation_id,
        time_stamp=time_stamp,
        parent_ts=parent_ts,
    )
    new_roots = create_parents.run()
    new_entries = create_parents.create_new_entries()
    return new_roots, new_l2_ids, new_entries


class CreateParentNodes:
    def __init__(
        self,
        cg,
        *,
        new_l2_ids: Iterable,
        operation_id: basetypes.OPERATION_ID,
        time_stamp: datetime.datetime,
        new_old_id_d: Dict[np.uint64, Iterable[np.uint64]] = None,
        old_new_id_d: Dict[np.uint64, Iterable[np.uint64]] = None,
        old_hierarchy_d: Dict[np.uint64, Dict[int, np.uint64]] = None,
        parent_ts: datetime.datetime = None,
    ):
        self.cg = cg
        self._new_l2_ids = new_l2_ids
        self._old_hierarchy_d = old_hierarchy_d
        self._new_old_id_d = new_old_id_d
        self._old_new_id_d = old_new_id_d
        self._new_ids_d = defaultdict(list)  # new IDs in each layer
        self._operation_id = operation_id
        self._time_stamp = time_stamp
        self._last_successful_ts = parent_ts

    def _update_id_lineage(
        self,
        parent: basetypes.NODE_ID,
        children: np.ndarray,
        layer: int,
        parent_layer: int,
    ):
        mask = np.in1d(children, self._new_ids_d[layer])
        for child_id in children[mask]:
            child_old_ids = self._new_old_id_d[child_id]
            for id_ in child_old_ids:
                old_id = self._old_hierarchy_d[id_].get(parent_layer, id_)
                self._new_old_id_d[parent].add(old_id)
                self._old_new_id_d[old_id].add(parent)

    def _get_old_ids(self, new_ids):
        old_ids = [
            np.array(list(self._new_old_id_d[id_]), dtype=basetypes.NODE_ID)
            for id_ in new_ids
        ]
        return np.concatenate(old_ids)

    def _get_new_ids(self, old_ids):
        old_ids = [
            np.array(list(self._old_new_id_d[id_]), dtype=basetypes.NODE_ID)
            for id_ in old_ids
        ]
        return np.concatenate(old_ids)

    def _get_connected_components(self, node_ids: np.ndarray, layer: int):
        with TimeIt(
            f"get_cross_chunk_edges.{layer}",
            self.cg.graph_id,
            self._operation_id,
        ):
            cross_edges_d = self.cg.get_cross_chunk_edges(node_ids)

        cx_edges = [types.empty_2d]
        for id_ in node_ids:
            edges_ = cross_edges_d[id_].get(layer, types.empty_2d)
            cx_edges.append(edges_)

        cx_edges = np.concatenate([*cx_edges, np.vstack([node_ids, node_ids]).T])
        temp_map = {k: next(iter(v)) for k, v in self._old_new_id_d.items()}
        cx_edges = fastremap.remap(cx_edges, temp_map, preserve_missing_labels=True)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(cx_edges, make_directed=True)
        return flatgraph.connected_components(graph), graph_ids

    def _get_layer_node_ids(
        self, new_ids: np.ndarray, layer: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # get old identities of new IDs
        old_ids = self._get_old_ids(new_ids)
        # get their parents, then children of those parents
        parents = self.cg.get_parents(old_ids, time_stamp=self._last_successful_ts)
        node_ids = self.cg.get_children(np.unique(parents), flatten=True)
        # replace old identities with new IDs
        mask = np.in1d(node_ids, old_ids)
        node_ids = np.concatenate(
            [self._get_new_ids(node_ids[mask]), node_ids[~mask], new_ids]
        )
        node_ids = np.unique(node_ids)
        layer_mask = self.cg.get_chunk_layers(node_ids) == layer
        return node_ids[layer_mask]
        # return node_ids

    def _update_cross_edge_cache(self, parent, children):
        """
        updates cross chunk edges in cache;
        this can only be done after all new components at a layer have IDs
        """
        cx_edges_d = self.cg.get_cross_chunk_edges(children)
        cx_edges_d = concatenate_cross_edge_dicts(cx_edges_d.values(), unique=True)

        parent_layer = self.cg.get_chunk_layer(parent)
        edge_nodes = np.unique(np.concatenate([*cx_edges_d.values(), types.empty_2d]))
        edge_parents = self.cg.get_roots(
            edge_nodes, stop_layer=parent_layer, ceil=False
        )
        edge_parents_d = dict(zip(edge_nodes, edge_parents))

        new_cx_edges_d = {}
        for layer in range(parent_layer, self.cg.meta.layer_count):
            layer_edges = cx_edges_d.get(layer, types.empty_2d)
            if len(layer_edges) == 0:
                continue
            new_cx_edges_d[layer] = fastremap.remap(
                layer_edges, edge_parents_d, preserve_missing_labels=True
            )
        self.cg.cache.cross_chunk_edges_cache[parent] = new_cx_edges_d

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
        parent_layer = layer + 1
        new_ids = self._new_ids_d[layer]
        layer_node_ids = self._get_layer_node_ids(new_ids, layer)
        print(layer, layer_node_ids)
        components, graph_ids = self._get_connected_components(layer_node_ids, layer)
        new_parent_ids = []
        for cc_indices in components:
            cc_ids = graph_ids[cc_indices]
            if len(cc_ids) == 1:
                # skip connection
                parent_layer = self.cg.meta.layer_count
                for l in range(layer + 1, self.cg.meta.layer_count):
                    cx_edges_d = self.cg.get_cross_chunk_edges([cc_ids[0]])
                    if len(cx_edges_d[cc_ids[0]].get(l, types.empty_2d)) > 0:
                        parent_layer = l
                        break
            parent_id = self.cg.id_client.create_node_id(
                self.cg.get_parent_chunk_id(cc_ids[0], parent_layer),
                root_chunk=parent_layer == self.cg.meta.layer_count,
            )
            self._new_ids_d[parent_layer].append(parent_id)
            self._update_id_lineage(parent_id, cc_ids, layer, parent_layer)
            new_parent_ids.append(parent_id)

            self.cg.cache.children_cache[parent_id] = cc_ids
            cache_utils.update(
                self.cg.cache.parents_cache,
                cc_ids,
                parent_id,
            )

        for new_id in new_parent_ids:
            children = self.cg.get_children(new_id)
            self._update_cross_edge_cache(new_id, children)

    def run(self) -> Iterable:
        """
        After new level 2 IDs are created, create parents in higher layers.
        Cross edges are used to determine existing siblings.
        """
        self._new_ids_d[2] = self._new_l2_ids
        for layer in range(2, self.cg.meta.layer_count):
            if len(self._new_ids_d[layer]) == 0:
                continue
            with TimeIt(
                f"create_new_parents_layer.{layer}",
                self.cg.graph_id,
                self._operation_id,
            ):
                self._create_new_parents(layer)
        return self._new_ids_d[self.cg.meta.layer_count]

    def _update_root_id_lineage(self):
        new_root_ids = self._new_ids_d[self.cg.meta.layer_count]
        former_root_ids = self._get_old_ids(new_root_ids)
        former_root_ids = np.unique(former_root_ids)

        print()
        print(former_root_ids, "->", new_root_ids)
        print(self.cg.get_children(former_root_ids))
        print(self.cg.get_children(np.array(new_root_ids, dtype=np.uint64)))
        assert (
            len(former_root_ids) < 2 or len(new_root_ids) < 2
        ), "Result inconsistent with either split or merge effects."
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

    def _get_cross_edges_val_dict(self):
        print("haha", self.cg.get_cross_chunk_edges([216172782113783809]))
        val_dicts = {}
        for layer in range(2, self.cg.meta.layer_count):
            new_ids = np.array(self._new_ids_d[layer], dtype=basetypes.NODE_ID)
            cross_edges_d = self.cg.get_cross_chunk_edges(new_ids)
            print()
            print(layer, new_ids)
            print("cx", cross_edges_d)
            print("ch", self.cg.get_children(new_ids))
            for id_ in new_ids:
                val_dict = {}
                for layer, edges in cross_edges_d[id_].items():
                    val_dict[attributes.Connectivity.CrossChunkEdge[layer]] = edges
                val_dicts[id_] = val_dict
        return val_dicts

    def create_new_entries(self) -> List:
        rows = []
        val_dicts = self._get_cross_edges_val_dict()
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
                        serialize_uint64(id_),
                        val_dict,
                        time_stamp=self._time_stamp,
                    )
                )
                for child_id in children:
                    rows.append(
                        self.cg.client.mutate_row(
                            serialize_uint64(child_id),
                            {attributes.Hierarchy.Parent: id_},
                            time_stamp=self._time_stamp,
                        )
                    )
        return rows + self._update_root_id_lineage()
