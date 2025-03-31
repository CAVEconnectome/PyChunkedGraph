# pylint: disable=invalid-name, missing-docstring, too-many-locals, c-extension-no-member

import datetime
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from collections import defaultdict

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
    Determine if atomic edges are within the chunk.
    If not, they are cross edges between two L2 IDs in adjacent chunks.
    Returns edges between L2 IDs and atomic cross edges.
    """
    supervoxels = np.unique(atomic_edges)
    parents = cg.get_parents(supervoxels, time_stamp=parent_ts)
    sv_parent_d = dict(zip(supervoxels.tolist(), parents))
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    parent_edges = [
        [sv_parent_d[edge_[0]], sv_parent_d[edge_[1]]]
        for edge_ in atomic_edges[edge_layers == 1]
    ]

    # cross chunk edges
    atomic_cross_edges_d = defaultdict(lambda: defaultdict(list))
    for layer in range(2, cg.meta.layer_count):
        layer_edges = atomic_edges[edge_layers == layer]
        if not layer_edges.size:
            continue
        for edge in layer_edges:
            parent_1 = sv_parent_d[edge[0]]
            parent_2 = sv_parent_d[edge[1]]
            atomic_cross_edges_d[parent_1][layer].append(edge)
            atomic_cross_edges_d[parent_2][layer].append(edge[::-1])
            parent_edges.extend([[parent_1, parent_1], [parent_2, parent_2]])
    return (parent_edges, atomic_cross_edges_d)


def _get_relevant_components(edges: np.ndarray, supervoxels: np.ndarray) -> Tuple:
    edges = np.concatenate([edges, np.vstack([supervoxels, supervoxels]).T]).astype(
        basetypes.NODE_ID
    )
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
    Determine if a fake edge needs to be added.
    Get subgraph within the bounding box
    Add fake edge if there are no inactive edges between two components.
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

    relevant_ccs = _get_relevant_components(
        np.concatenate(active_edges).astype(basetypes.NODE_ID), supervoxels
    )
    inactive = np.concatenate(inactive_edges).astype(basetypes.NODE_ID)
    _inactive = [types.empty_2d]
    # source to sink edges
    source_mask = np.in1d(inactive[:, 0], relevant_ccs[0])
    sink_mask = np.in1d(inactive[:, 1], relevant_ccs[1])
    _inactive.append(inactive[source_mask & sink_mask])

    # sink to source edges
    sink_mask = np.in1d(inactive[:, 1], relevant_ccs[0])
    source_mask = np.in1d(inactive[:, 0], relevant_ccs[1])
    _inactive.append(inactive[source_mask & sink_mask])
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
        print("found inactive", len(inactive_edges))
        return inactive_edges, []

    rows = []
    supervoxels = atomic_edges.ravel()
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
    print("no inactive", len(atomic_edges))
    return atomic_edges, rows


def add_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    operation_id: np.uint64 = None,
    time_stamp: datetime.datetime = None,
    parent_ts: datetime.datetime = None,
    allow_same_segment_merge: bool = False,
    stitch_mode: bool = False,
):
    edges, l2_atomic_cross_edges_d = _analyze_affected_edges(
        cg, atomic_edges, parent_ts=parent_ts
    )
    l2ids = np.unique(edges)
    if not allow_same_segment_merge and not stitch_mode:
        assert (
            np.unique(cg.get_roots(l2ids, assert_roots=True, time_stamp=parent_ts)).size
            >= 2
        ), "L2 IDs must belong to different roots."
    new_old_id_d, old_new_id_d, old_hierarchy_d = _init_old_hierarchy(
        cg, l2ids, parent_ts=parent_ts
    )
    atomic_children_d = cg.get_children(l2ids)
    atomic_cross_edges_d = merge_cross_edge_dicts(
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
        ).astype(basetypes.NODE_ID)
        cg.cache.atomic_cx_edges_cache[new_id] = concatenate_cross_edge_dicts(
            [atomic_cross_edges_d[l2id] for l2id in l2ids_]
        )
        cache_utils.update(
            cg.cache.parents_cache, cg.cache.children_cache[new_id], new_id
        )
        new_l2_ids.append(new_id)
        new_old_id_d[new_id].update(l2ids_)
        for id_ in l2ids_:
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
        stitch_mode=stitch_mode,
    )

    new_roots = create_parents.run()
    new_entries = create_parents.create_new_entries()
    return new_roots, new_l2_ids, new_entries


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
    cross_edges = np.concatenate(
        [types.empty_2d, *atomic_cross_edges_d.values()]
    ).astype(basetypes.NODE_ID)
    chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]
    cross_edges = cross_edges[~in2d(cross_edges, removed_edges)]

    isolated_ids = agg.supervoxels[~np.in1d(agg.supervoxels, chunk_edges)]
    isolated_edges = np.column_stack((isolated_ids, isolated_ids))
    _edges = np.concatenate([chunk_edges, isolated_edges]).astype(basetypes.NODE_ID)
    graph, _, _, graph_ids = flatgraph.build_gt_graph(_edges, make_directed=True)
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
    operation_id: basetypes.OPERATION_ID = None,  # type: ignore
    time_stamp: datetime.datetime = None,
    parent_ts: datetime.datetime = None,
):
    edges, _ = _analyze_affected_edges(cg, atomic_edges, parent_ts=parent_ts)
    l2ids = np.unique(edges)
    assert (
        np.unique(cg.get_roots(l2ids, assert_roots=True, time_stamp=parent_ts)).size
        == 1
    ), "L2 IDs must belong to same root."
    new_old_id_d, old_new_id_d, old_hierarchy_d = _init_old_hierarchy(
        cg, l2ids, parent_ts=parent_ts
    )
    l2id_chunk_id_d = dict(zip(l2ids.tolist(), cg.get_chunk_ids_from_node_ids(l2ids)))
    atomic_cross_edges_d = cg.get_atomic_cross_edges(l2ids)

    removed_edges = np.concatenate(
        [atomic_edges, atomic_edges[:, ::-1]], axis=0
    ).astype(basetypes.NODE_ID)
    new_l2_ids = []
    for id_ in l2ids:
        l2_agg = l2id_agglomeration_d[id_]
        ccs, graph_ids, cross_edges = _process_l2_agglomeration(
            l2_agg, removed_edges, atomic_cross_edges_d[id_]
        )
        # calculated here to avoid repeat computation in loop
        cross_edge_layers = cg.get_cross_chunk_edges_layer(cross_edges)
        new_parent_ids = cg.id_client.create_node_ids(
            l2id_chunk_id_d[l2_agg.node_id], len(ccs)
        )
        for i_cc, cc in enumerate(ccs):
            new_id = new_parent_ids[i_cc]
            cg.cache.children_cache[new_id] = graph_ids[cc]
            cg.cache.atomic_cx_edges_cache[new_id] = _filter_component_cross_edges(
                graph_ids[cc], cross_edges, cross_edge_layers
            )
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
        operation_id: basetypes.OPERATION_ID,  # type: ignore
        time_stamp: datetime.datetime,
        new_old_id_d: Dict[np.uint64, Iterable[np.uint64]] = None,
        old_new_id_d: Dict[np.uint64, Iterable[np.uint64]] = None,
        old_hierarchy_d: Dict[np.uint64, Dict[int, np.uint64]] = None,
        parent_ts: datetime.datetime = None,
        stitch_mode: bool = False,
    ):
        self.cg = cg
        self._new_l2_ids = new_l2_ids
        self._old_hierarchy_d = old_hierarchy_d
        self._new_old_id_d = new_old_id_d
        self._old_new_id_d = old_new_id_d
        self._new_ids_d = defaultdict(list)  # new IDs in each layer
        self._cross_edges_d = {}
        self._operation_id = operation_id
        self._time_stamp = time_stamp
        self._last_successful_ts = parent_ts
        self.stitch_mode = stitch_mode

    def _update_id_lineage(
        self,
        parent: basetypes.NODE_ID,  # type: ignore
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
        return np.concatenate(old_ids).astype(basetypes.NODE_ID)

    def _map_sv_to_parent(self, node_ids, layer, node_map=None):
        sv_parent_d = {}
        sv_cross_edges = [types.empty_2d]
        if node_map is None:
            node_map = {}
        for id_ in node_ids:
            id_eff = node_map.get(id_, id_)
            edges_ = self._cross_edges_d[id_].get(layer, types.empty_2d)
            sv_parent_d.update(dict(zip(edges_[:, 0], [id_eff] * len(edges_))))
            sv_cross_edges.append(edges_)
        return sv_parent_d, np.concatenate(sv_cross_edges).astype(basetypes.NODE_ID)

    def _get_connected_components(
        self, node_ids: np.ndarray, layer: int, lower_layer_ids: np.ndarray
    ):
        _node_ids = np.concatenate([node_ids, lower_layer_ids]).astype(
            basetypes.NODE_ID
        )
        cached = np.fromiter(self._cross_edges_d.keys(), dtype=basetypes.NODE_ID)
        not_cached = _node_ids[~np.in1d(_node_ids, cached)]

        with TimeIt(
            f"get_cross_chunk_edges.{layer}",
            self.cg.graph_id,
            self._operation_id,
        ):
            self._cross_edges_d.update(
                self.cg.get_cross_chunk_edges(not_cached, all_layers=True)
            )

        sv_parent_d, sv_cross_edges = self._map_sv_to_parent(node_ids, layer)
        get_sv_parents = np.vectorize(sv_parent_d.get, otypes=[np.uint64])
        try:
            cross_edges = get_sv_parents(sv_cross_edges)
        except TypeError:  # NoneType error
            # if there is a missing parent, try including lower layer ids
            # this can happen due to skip connections

            # we want to map all these lower IDs to the current layer
            lower_layer_to_layer = self.cg.get_roots(
                lower_layer_ids, stop_layer=layer, ceil=False
            )
            node_map = {k: v for k, v in zip(lower_layer_ids, lower_layer_to_layer)}
            sv_parent_d, sv_cross_edges = self._map_sv_to_parent(
                _node_ids, layer, node_map=node_map
            )
            get_sv_parents = np.vectorize(sv_parent_d.get, otypes=[np.uint64])
            cross_edges = get_sv_parents(sv_cross_edges)

        cross_edges = np.concatenate(
            [cross_edges, np.vstack([node_ids, node_ids]).T]
        ).astype(basetypes.NODE_ID)
        graph, _, _, graph_ids = flatgraph.build_gt_graph(
            cross_edges, make_directed=True
        )
        return flatgraph.connected_components(graph), graph_ids

    def _get_layer_node_ids(
        self, new_ids: np.ndarray, layer: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # get old identities of new IDs
        old_ids = self._get_old_ids(new_ids)
        # get their parents, then children of those parents
        node_ids = self.cg.get_children(
            np.unique(
                self.cg.get_parents(old_ids, time_stamp=self._last_successful_ts)
            ),
            flatten=True,
        )
        # replace old identities with new IDs
        mask = np.in1d(node_ids, old_ids)
        node_ids = np.concatenate(
            [
                np.array(list(self._old_new_id_d[id_]), dtype=basetypes.NODE_ID)
                for id_ in node_ids[mask]
            ]
            + [node_ids[~mask], new_ids]
        ).astype(basetypes.NODE_ID)
        node_ids = np.unique(node_ids)
        layer_mask = self.cg.get_chunk_layers(node_ids) == layer
        return node_ids[layer_mask], node_ids[~layer_mask]

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
        layer_node_ids, lower_layer_ids = self._get_layer_node_ids(new_ids, layer)
        components, graph_ids = self._get_connected_components(
            layer_node_ids, layer, lower_layer_ids
        )
        for cc_indices in components:
            parent_layer = layer + 1
            cc_ids = graph_ids[cc_indices]
            if len(cc_ids) == 1:
                # skip connection
                parent_layer = self.cg.meta.layer_count
                for l in range(layer + 1, self.cg.meta.layer_count):
                    if len(self._cross_edges_d[cc_ids[0]].get(l, types.empty_2d)) > 0:
                        parent_layer = l
                        break

            while True:
                parent_id = self.cg.id_client.create_node_id(
                    self.cg.get_parent_chunk_id(cc_ids[0], parent_layer),
                    root_chunk=parent_layer == self.cg.meta.layer_count,
                )
                _entry = self.cg.client.read_node(parent_id)
                if _entry == {}:
                    break
            self._new_ids_d[parent_layer].append(parent_id)
            self.cg.cache.children_cache[parent_id] = cc_ids
            cache_utils.update(
                self.cg.cache.parents_cache,
                cc_ids,
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
            with TimeIt(
                f"create_new_parents_layer.{layer}",
                self.cg.graph_id,
                self._operation_id,
            ):
                self._create_new_parents(layer)
        return self._new_ids_d[self.cg.meta.layer_count]

    def _update_root_id_lineage(self):
        rows = []
        if self.stitch_mode:
            return rows
        new_root_ids = self._new_ids_d[self.cg.meta.layer_count]
        former_root_ids = self._get_old_ids(new_root_ids)
        former_root_ids = np.unique(former_root_ids)
        assert (
            len(former_root_ids) < 2 or len(new_root_ids) < 2
        ), "Something went wrong."
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

    def create_new_entries(self) -> List:
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
