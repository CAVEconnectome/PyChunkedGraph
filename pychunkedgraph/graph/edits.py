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
from .edges.utils import filter_min_layer_cross_edges
from .edges.utils import concatenate_cross_edge_dicts
from .edges.utils import merge_cross_edge_dicts_multiple
from ..utils.general import in2d


"""
TODO
1. get split working
2. handle fake edges 
3. unit tests, edit old and create new
4. split merge manual tests
5. performance
6. meshing
7. ingest instructions, pinky test run


Their children might be "too much" due to the split; even within one chunk. How do you deal with that?

a good way to test this is to check all intermediate nodes from the component before the split and then after the split. Basically, get all childrens in all layers of the one component before and the (hopefully) two components afterwards. Check (1) are all intermediate nodes from before in a list after and (2) do all intermediate nodes appear exactly one time after the split (aka is there overlap between the resulting components). (edited) 

for (2) Overlap can be real but then they have to be exactly the same. In that case the removed edges did not split the component in two

"""


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


def add_edges(
    cg,
    *,
    atomic_edges: Iterable[np.ndarray],
    operation_id: np.uint64 = None,
    source_coords: Sequence[np.uint64] = None,
    sink_coords: Sequence[np.uint64] = None,
    time_stamp: datetime.datetime = None,
):
    """
    Problem: Update parent and children of the new level 2 id
    For each layer >= 2
        get cross edges
        get parents
            get children
        above children + new ID will form a new component
        update parent, former parents and new parents for all affected IDs
    """
    edges, l2_atomic_cross_edges_d = _analyze_atomic_edges(cg, atomic_edges)
    l2ids = np.unique(edges)

    # setup relevant children and atomic cross edges
    atomic_children_d = cg.get_children(l2ids)
    atomic_cross_edges_d = merge_cross_edge_dicts_multiple(
        cg.get_atomic_cross_edges(l2ids), l2_atomic_cross_edges_d
    )

    graph, _, _, graph_node_ids = flatgraph.build_gt_graph(edges, make_directed=True)
    ccs = flatgraph.connected_components(graph)
    new_l2_ids = []
    for cc in ccs:
        l2ids_ = graph_node_ids[cc]
        new_id = cg.id_client.create_node_id(cg.get_chunk_id(l2ids_[0]))

        cache.CHILDREN[new_id] = np.concatenate(
            [atomic_children_d[l2id] for l2id in l2ids_]
        )
        cache.ATOMIC_CX_EDGES[new_id] = concatenate_cross_edge_dicts(
            [atomic_cross_edges_d[l2id] for l2id in l2ids_]
        )
        cache.update(cache.PARENTS, cache.CHILDREN[new_id], new_id)
        new_l2_ids.append(new_id)

    # for k in cache.ATOMIC_CX_EDGES.keys():
    #     print(k)
    create_parents = CreateParentNodes(
        cg, new_l2_ids=new_l2_ids, operation_id=operation_id, time_stamp=time_stamp,
    )
    return create_parents.run()


class CreateParentNodes:
    def __init__(
        self,
        cg,
        *,
        new_l2_ids: Iterable,
        operation_id: basetypes.OPERATION_ID,
        time_stamp: datetime.datetime,
    ):
        self.cg = cg
        self.new_l2_ids = new_l2_ids
        self.operation_id = operation_id
        self.time_stamp = time_stamp
        self._layer_new_ids_d = defaultdict(list)
        self._done = set()

        self.cg.cache = cache.CacheService(self.cg)

    def _create_new_sibling(self, child_id, sibling_layer) -> basetypes.NODE_ID:
        """
        `child_id` child ID of the missing sibling
        `layer` layer at which the missing sibling needs to be created
        """
        # current parent skipped this layer, so it would be grand parent
        grandpa_id = self.cg.get_parent(child_id)
        new_sibling_id = self.cg.id_client.create_node_id(
            self.cg.get_parent_chunk_id(child_id, sibling_layer)
        )

        old_children = self.cg.get_children(grandpa_id)
        cache.CHILDREN[grandpa_id] = np.setdiff1d(
            old_children, [child_id], assume_unique=True,
        )
        cache.update(
            cache.PARENTS, cache.CHILDREN[grandpa_id], grandpa_id,
        )
        cache.CHILDREN[new_sibling_id] = np.array([child_id], dtype=basetypes.NODE_ID)
        cache.PARENTS[child_id] = new_sibling_id
        # print("new_sibling_id", child_id, new_sibling_id)
        return new_sibling_id

    def _handle_missing_siblings(self, layer, new_id_ce_siblings) -> np.ndarray:
        """Create new sibling when a new ID has none because of skip connections."""
        # print("before", layer, new_id_ce_siblings)
        mask = self.cg.get_chunk_layers(new_id_ce_siblings) < layer
        missing = new_id_ce_siblings[mask]
        for id_ in missing:
            self._layer_new_ids_d[layer].append(self._create_new_sibling(id_, layer))
        new_id_ce_siblings[mask] = self.cg.get_parents(missing)
        # print("after", layer, new_id_ce_siblings)
        return new_id_ce_siblings

    def _get_all_siblings(
        self,
        new_id: basetypes.NODE_ID,
        new_parent_id: basetypes.NODE_ID,
        new_id_ce_siblings: Iterable,
    ) -> List:
        """
        Get parents of `new_id_ce_siblings`
        Children of these parents will include all siblings (filter by chunk IDs)
        """
        parents = self.cg.get_parents(new_id_ce_siblings)
        cache.update(cache.PARENTS, new_id_ce_siblings, parents)
        chunk_ids = self.cg.get_children_chunk_ids(new_parent_id)
        chunk_ids = np.setdiff1d(chunk_ids, [self.cg.get_chunk_id(new_id)])
        children = self.cg.get_children(np.unique(parents), flatten=True)
        children_chunk_ids = self.cg.get_chunk_ids_from_node_ids(children)
        return children[np.in1d(children_chunk_ids, chunk_ids)]

    def _create_parent(
        self,
        new_id: basetypes.NODE_ID,
        layer: int,
        ce_layer: int,
        ce_siblings: np.ndarray,
    ) -> None:
        """Helper function."""
        if new_id in self._done:
            # parent already updated
            return
        # print("ce_layer", new_id, ce_layer, ce_siblings)
        if ce_layer > layer:
            # skip connection
            new_parent_id = self.cg.id_client.create_node_id(
                self.cg.get_parent_chunk_id(new_id, ce_layer)
            )
            cache.CHILDREN[new_parent_id] = np.array([new_id], dtype=basetypes.NODE_ID)
            self._layer_new_ids_d[ce_layer].append(new_parent_id)
        else:
            new_parent_id = self.cg.id_client.create_node_id(
                self.cg.get_parent_chunk_id(new_id, layer + 1)
            )
            ce_siblings = self._handle_missing_siblings(ce_layer, ce_siblings)

            # siblings that are also new IDs
            common = np.intersect1d(
                ce_siblings, self._layer_new_ids_d[layer], assume_unique=True
            )
            # they do not have parents yet so exclude them
            siblings = self._get_all_siblings(
                new_id,
                new_parent_id,
                np.setdiff1d(ce_siblings, common, assume_unique=True),
            )
            cache.CHILDREN[new_parent_id] = np.unique(
                np.concatenate([[new_id], common, siblings])
            )
            self._layer_new_ids_d[layer + 1].append(new_parent_id)

        cache.update(cache.PARENTS, cache.CHILDREN[new_parent_id], new_parent_id)
        # print(new_parent_id, cache.CHILDREN[new_parent_id])
        # print()
        self._done.add(new_id)

    def run(self) -> Iterable:
        """
        After new level 2 IDs are created, create parents in higher layers.
        Cross edges are used to determine existing siblings.
        """
        # cache for convenience, if `node_id` exists in this
        # no need to call `get_cross_chunk_edges`
        cross_edges_d = {}
        self._layer_new_ids_d[2] = self.new_l2_ids
        for current_layer in range(2, self.cg.meta.layer_count):
            if len(self._layer_new_ids_d[current_layer]) == 0:
                continue

            new_ids = np.array(self._layer_new_ids_d[current_layer], basetypes.NODE_ID)
            cached = np.fromiter(cross_edges_d.keys(), dtype=basetypes.NODE_ID)
            not_cached = new_ids[~np.in1d(new_ids, cached)]
            cross_edges_d.update(self.cg.get_cross_chunk_edges(not_cached))

            # print("\n", "*" * 50)
            # print(current_layer, new_ids)
            for new_id in new_ids:
                ce_layer = list(cross_edges_d[new_id].keys())[0]
                ce_siblings = cross_edges_d[new_id][ce_layer][:, 1]
                self._create_parent(new_id, current_layer, ce_layer, ce_siblings)
        return self._done, self._layer_new_ids_d[self.cg.meta.layer_count]


# def _process_l2_agglomeration(agg: types.Agglomeration, removed_edges: np.ndarray):
#     """
#     For a given L2 id, remove given edges
#     and calculate new connected components.
#     """
#     chunk_edges = agg.in_edges.get_pairs()
#     cross_edges = agg.cross_edges.get_pairs()
#     chunk_edges = chunk_edges[~in2d(chunk_edges, removed_edges)]
#     cross_edges = cross_edges[~in2d(cross_edges, removed_edges)]

#     isolated_ids = agg.supervoxels[~np.in1d(agg.supervoxels, chunk_edges)]
#     isolated_edges = np.column_stack((isolated_ids, isolated_ids))
#     graph, _, _, unique_graph_ids = flatgraph.build_gt_graph(
#         np.concatenate([chunk_edges, isolated_edges]), make_directed=True
#     )
#     return flatgraph.connected_components(graph), unique_graph_ids, cross_edges


# def _filter_component_cross_edges(
#     cc_ids: np.ndarray, cross_edges: np.ndarray, cross_edge_layers: np.ndarray
# ) -> Dict[int, np.ndarray]:
#     """
#     Filters cross edges for a connected component `cc_ids`
#     from `cross_edges` of the complete chunk.
#     """
#     mask = np.in1d(cross_edges[:, 0], cc_ids)
#     cross_edges_ = cross_edges[mask]
#     cross_edge_layers_ = cross_edge_layers[mask]

#     edges_d = {}
#     for layer in np.unique(cross_edge_layers_):
#         edge_m = cross_edge_layers_ == layer
#         _cross_edges = cross_edges_[edge_m]
#         if _cross_edges.size:
#             edges_d[layer] = _cross_edges
#     return edges_d


# def remove_edges(
#     cg,
#     *,
#     operation_id: basetypes.OPERATION_ID,
#     atomic_edges: Iterable[np.ndarray],
#     l2id_agglomeration_d: Dict,
#     time_stamp: datetime.datetime,
# ):
#     edges, _ = _analyze_atomic_edges(cg, atomic_edges)
#     l2ids = np.unique(edges)
#     l2id_chunk_id_d = dict(zip(l2ids, cg.get_chunk_ids_from_node_ids(l2ids)))

#     # This view of the to be removed edges helps us to
#     # compute the mask of retained edges in chunk
#     removed_edges = np.concatenate([atomic_edges, atomic_edges[:, ::-1]], axis=0)

#     new_l2_ids = []
#     new_hierarchy_d = {}
#     for id_ in l2ids:
#         l2_agg = l2id_agglomeration_d[id_]
#         ccs, unique_graph_ids, cross_edges = _process_l2_agglomeration(
#             l2_agg, removed_edges
#         )
#         cross_edge_layers = cg.get_cross_chunk_edges_layer(cross_edges)
#         new_parent_ids = cg.id_client.create_node_ids(
#             l2id_chunk_id_d[l2_agg.node_id], len(ccs)
#         )
#         for i_cc, cc in enumerate(ccs):
#             new_id = new_parent_ids[i_cc]
#             new_node = types.Node(new_id)
#             new_node.children = unique_graph_ids[cc]
#             new_node.atomic_cross_edges = _filter_component_cross_edges(
#                 new_node.children, cross_edges, cross_edge_layers
#             )
#             new_hierarchy_d[new_id] = new_node
#             for child_id in new_node.children:
#                 new_hierarchy_d[child_id] = types.Node(child_id, parent_id=new_id)
#             new_l2_ids.append(new_id)
#     cg.node_cache = new_hierarchy_d
#     create_parents = CreateParentNodes(
#         cg, new_l2_ids=new_l2_ids, operation_id=operation_id, time_stamp=time_stamp,
#     )
#     return create_parents.run()

