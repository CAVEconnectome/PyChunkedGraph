import datetime
import numpy as np
import collections

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union,\
    NamedTuple

from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.backend.chunkedgraph_utils \
    import get_google_compatible_time_stamp, combine_cross_chunk_edge_dicts
from pychunkedgraph.backend.utils import column_keys, serializers
from pychunkedgraph.backend import chunkedgraph, flatgraph_utils

def add_edges(cg, operation_id: np.uint64,
              atomic_edges: Sequence[Sequence[np.uint64]],
              time_stamp: datetime.datetime,
              affinities: Optional[Sequence[np.float32]] = None
              ):

    atomic_edges = np.array(atomic_edges, dtype=np.uint64)

    # Comply to resolution of BigTables TimeRange
    time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                  round_up=False)

    if affinities is None:
        affinities = np.ones(len(atomic_edges),
                             dtype=column_keys.Connectivity.Affinity.basetype)

    assert len(affinities) == len(atomic_edges)

    rows = []

def old_parent_childrens(eh, node_ids, layer):
    # 1 - gather all next layer parents
    old_next_layer_node_ids = []
    old_this_layer_node_ids = []
    for node_id in node_ids:
        old_next_layer_node_ids.extend(
            eh.get_old_node_ids(node_id, layer + 1))
        old_this_layer_node_ids.extend(
            eh.get_old_node_ids(node_id, layer))

    old_next_layer_node_ids = np.unique(old_next_layer_node_ids)
    old_this_layer_node_ids = np.unique(old_this_layer_node_ids)

    # 2 - acquire their children
    old_this_layer_partner_ids = []
    for old_next_layer_node_id in old_next_layer_node_ids:
        partner_ids = eh.get_layer_children(old_next_layer_node_id, layer)
        partner_ids = partner_ids[~np.in1d(partner_ids,
                                           old_this_layer_node_ids)]
        old_this_layer_partner_ids.extend(partner_ids)

    old_this_layer_partner_ids = np.unique(old_this_layer_partner_ids)

    return old_this_layer_node_ids, old_next_layer_node_ids, \
           old_this_layer_partner_ids


def compute_cross_chunk_connected_components(eh, node_ids, layer):

    # On each layer we build the a graph with all cross chunk edges
    # that involve the nodes on the current layer
    # To do this efficiently, we acquire all candidate same layer nodes
    # that were previously connected to any of the currently assessed
    # nodes. In practice, we (1) gather all relevant parents in the next
    # layer and then (2) acquire their children

    old_this_layer_node_ids, old_next_layer_node_ids, \
        old_this_layer_partner_ids = \
            old_parent_childrens(eh, node_ids, layer)

    # Build network from cross chunk edges
    edge_id_map = {}
    cross_edges_lvl1 = []
    for new_node_id in node_ids:
        node_cross_edges = eh.read_cross_chunk_edges(new_node_id)[layer]
        edge_id_map.update(dict(zip(node_cross_edges[:, 0],
                                    [new_node_id] * len(node_cross_edges))))
        cross_edges_lvl1.extend(node_cross_edges)

    for old_partner_id in old_this_layer_partner_ids:
        node_cross_edges = eh.read_cross_chunk_edges(old_partner_id)[layer]

        edge_id_map.update(dict(zip(node_cross_edges[:, 0],
                                    [old_partner_id] * len(node_cross_edges))))
        cross_edges_lvl1.extend(node_cross_edges)

    cross_edges_lvl1 = np.array(cross_edges_lvl1)
    edge_id_map_vec = np.vectorize(edge_id_map.get)
    cross_edges = edge_id_map_vec(cross_edges_lvl1)

    graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
        cross_edges, make_directed=True)

    ccs = flatgraph_utils.connected_components(graph)

    return ccs, unique_graph_ids


def create_parent_children_rows(eh, parent_id, children_ids,
                                parent_cross_chunk_edge_dict, former_root_ids,
                                operation_id, time_stamp):
    rows = []

    val_dict = {}
    for l, layer_edges in parent_cross_chunk_edge_dict.items():
        val_dict[column_keys.Connectivity.CrossChunkEdge[l]] = layer_edges

    assert np.max(eh.cg.get_chunk_layers(children_ids)) < eh.cg.get_chunk_layer(
        parent_id)

    if former_root_ids is not None:
        val_dict[column_keys.Hierarchy.FormerParent] = np.array(former_root_ids)
        val_dict[column_keys.OperationLogs.OperationID] = operation_id

    val_dict = {column_keys.Hierarchy.Child: children_ids}

    rows.append(eh.cg.mutate_row(serializers.serialize_uint64(parent_id),
                                 val_dict, time_stamp=time_stamp))

    for child_id in children_ids:
        val_dict = {column_keys.Hierarchy.Parent: parent_id}
        rows.append(eh.cg.mutate_row(serializers.serialize_uint64(child_id),
                                     val_dict, time_stamp=time_stamp))

    return rows

def propagate_edits_to_root(cg: chunkedgraph.ChunkedGraph,
                            lvl2_dict: Dict,
                            operation_id: np.uint64,
                            merge_edges: Sequence[Tuple[np.uint64]],
                            time_stamp: datetime.datetime):
    """

    :param cg: ChunkedGraph
    :param lvl2_dict: dict
        maps
    :param operation_id:
    :param time_stamp:
    :return:
    """
    rows = []

    # Initialization
    eh = EditHelper(cg, lvl2_dict)
    eh.bulk_family_read()
    # eh.bulk_cross_chunk_edge_read()

    # Insert new nodes if missing (due to skip connections)
    rows.extend(eh.add_missing_nodes(merge_edges, time_stamp))

    # Setup loop variables
    layer_dict = collections.defaultdict(list)
    layer_dict[2] = list(lvl2_dict.keys())
    new_root_ids = []
    # Loop over all layers up to the top - there might be layers where there is
    # nothing to do
    for current_layer in range(2, eh.cg.n_layers):
        new_node_ids = layer_dict[current_layer]

        # Calculate connected components based on cross chunk edges ------------
        ccs, unique_graph_ids = \
            compute_cross_chunk_connected_components(eh, new_node_ids,
                                                     current_layer)

        # Build a dictionary of new connected components -----------------------
        cc_collection = collections.defaultdict(list)
        for cc in ccs:
            cc_node_ids = unique_graph_ids[cc]

            cc_cross_edge_dict = collections.defaultdict(list)
            for cc_node_id in cc_node_ids:
                node_cross_edges = eh.read_cross_chunk_edges(cc_node_id)
                cc_cross_edge_dict = \
                    combine_cross_chunk_edge_dicts(cc_cross_edge_dict,
                                                   node_cross_edges,
                                                   start_layer=current_layer + 1)

            if len(cc_cross_edge_dict[current_layer + 1]) == 0 and \
                    len(cc_node_ids) == 1:
                # Skip connection
                next_layer = None
                for l in range(current_layer, eh.cg.n_layers):
                    if len(cc_cross_edge_dict[l]) > 0:
                        next_layer = l
                        break

                if next_layer is None:
                    next_layer = eh.cg.n_layers
            else:
                next_layer = current_layer + 1

            next_layer_chunk_id = eh.cg.get_parent_chunk_id_dict(cc_node_ids[0])[next_layer]

            cc_collection[next_layer_chunk_id].append(
                [cc_node_ids, cc_cross_edge_dict])

        # At this point we extracted all relevant data - now we just need to
        # create the new rows --------------------------------------------------
        for next_layer_chunk_id in cc_collection:
            n_ids = len(cc_collection[next_layer_chunk_id])
            new_parent_ids = eh.cg.get_unique_node_id_range(next_layer_chunk_id,
                                                            n_ids)

            for new_parent_id, cc_collection in \
                    zip(new_parent_ids, cc_collection[next_layer_chunk_id]):

                if eh.cg.get_chunk_layer(next_layer_chunk_id) == eh.cg.n_layers:
                    new_root_ids.append(new_parent_id)
                    former_root_ids = eh.get_old_node_ids(new_parent_ids,
                                                          eh.cg.n_layers)
                else:
                    former_root_ids = None

                cc_rows = create_parent_children_rows(eh, new_parent_id,
                                                      cc_collection[0],
                                                      cc_collection[1],
                                                      former_root_ids,
                                                      operation_id,
                                                      time_stamp)
                rows.extend(cc_rows)

    return new_root_ids


class EditHelper(object):
    def __init__(self, cg, lvl2_dict):
        """

        :param cg: ChunkedGraph isntance
        :param lvl2_dict: maps new lvl2 ids to old lvl2 ids
        """
        self._cg = cg
        self._lvl2_dict = lvl2_dict

        self._parent_dict = {}
        self._children_dict = {}
        self._cross_chunk_edge_dict = {}

    @property
    def cg(self):
        return self._cg

    @property
    def lvl2_dict(self):
        return self._lvl2_dict

    def get_children(self, node_id):
        """ Cache around the get_children call to the chunkedgraph

        :param node_id: np.uint64
        :return: np.uint64
        """
        if not node_id in self._children_dict:
            self._children_dict[node_id] = self.get_children(node_id)
            for child_id in self._children_dict[node_id]:
                if not child_id in self._parent_dict:
                    self._parent_dict[child_id] = node_id
                else:
                    assert self._parent_dict[child_id] == node_id

        return self._children_dict[node_id]


    def get_parent(self, node_id):
        """ Cache around the get_parent call to the chunkedgraph

        :param node_id: np.uint64
        :return: np.uint64
        """
        if not node_id in self._parent_dict:
            self._parent_dict[node_id] = self.get_parent(node_id)

        return self._children_dict[node_id]


    def get_root(self, node_id, get_all_parents=False):
        parents = [node_id]

        while self.get_parent(parents[-1]) is not None:
            parents.append(self.get_parent(parents[-1]))

        if get_all_parents:
            return np.array(parents)
        else:
            return parents[-1]

    def get_layer_children(self, node_id, layer):
        assert layer > 0
        assert layer <= self.cg.get_chunk_layer(node_id)

        if self.cg.get_chunk_layer(node_id) == layer:
            return [node_id]

        layer_children_ids = []
        next_children_ids = [node_id]

        while len(next_children_ids) > 0:
            next_children_id = next_children_ids[0]
            del next_children_ids[0]

            children_ids = self.get_children(next_children_id)

            if self.cg.get_chunk_layer(children_ids[0]) == layer:
                layer_children_ids.extend(children_ids)
            else:
                next_children_ids.extend(children_ids)

        return layer_children_ids

    def get_layer_parent(self, node_id, layer, layer_only=False):
        assert layer >= self.cg.get_chunk_layer(node_id)
        assert layer < self.cg.n_layers

        if self.cg.get_chunk_layer(node_id) == layer:
            return [node_id]

        layer_parent_ids = []
        next_parent_ids = [node_id]

        while len(next_parent_ids) > 0:
            next_parent_id = next_parent_ids[0]
            del next_parent_ids[0]

            parent_id = self.get_parent(next_parent_id)

            if self.cg.get_chunk_layer(parent_id) < layer:
                next_parent_ids.extend(parent_id)
            elif self.cg.get_chunk_layer(parent_id) == layer:
                layer_parent_ids.append(parent_id)
            elif self.cg.get_chunk_layer(parent_id) > layer and not layer_only:
                layer_parent_ids.append(parent_id)

        return layer_parent_ids

    def get_old_node_ids(self, node_id, layer):
        lvl2_children = self.get_layer_children(node_id, layer=2)

        old_lvl2_ids = []
        for lvl2_child in lvl2_children:
            old_lvl2_ids.extend(self.lvl2_dict[lvl2_child])

        old_lvl2_ids = np.unique(old_lvl2_ids)
        old_parents = []
        for old_lvl2_id in old_lvl2_ids:
            old_parents.append(self.get_layer_parent(old_lvl2_id, layer))

        old_parents = np.unique(old_parents)
        return old_parents

    def read_cross_chunk_edges(self, node_id):
        """ Cache around the read_cross_chunk_edges call to the chunkedgraph

        :param node_id: np.uint64
        :return: dict
        """
        if not node_id in self._cross_chunk_edge_dict:
            self._cross_chunk_edge_dict[node_id] = \
                self.cg.read_cross_chunk_edges(node_id)

        return self._cross_chunk_edge_dict[node_id]

    def bulk_family_read(self):
        """ Caches parent and children information that will be needed later

        :return:
        """
        def _get_root_thread(lvl2_node_id):
            p_ids = self.cg.get_root(lvl2_node_id, get_all_parents=True)
            p_ids = np.concatenate([[lvl2_node_id], p_ids])

            for i_parent in range(len(p_ids) - 1):
                self._parent_dict[p_ids[i_parent]] = p_ids[i_parent+1]

        lvl2_node_ids = []
        for v in self.lvl2_dict.values():
            lvl2_node_ids.extend(v)

        mu.multithread_func(_get_root_thread, lvl2_node_ids,
                            n_threads=len(lvl2_node_ids), debug=False)

        parent_ids = list(self._parent_dict.keys())
        child_dict = self.cg.get_children(parent_ids, flatten=False)

        for parent_id in child_dict:
            self._children_dict[parent_id] = child_dict[parent_id]

            for child_id in self._children_dict[parent_id]:
                if not child_id in self._parent_dict:
                    self._parent_dict[child_id] = parent_id
                else:
                    assert self._parent_dict[child_id] == parent_id

    def bulk_cross_chunk_edge_read(self):
        raise NotImplementedError

    def _add_skip_node(self, atomic_id, layer, time_stamp):
        rows = []

        parents = self.get_layer_parent(atomic_id, layer, False)
        parent_layers = self.cg.get_chunk_layers(parents)
        if not layer in parents:
            upper_parents = parents[parent_layers > layer]
            upper_parent_id = upper_parents[np.argmin(upper_parents)]

            children_ids = self.get_children(upper_parent_id)
            chunk_id = self.cg.get_parent_chunk_id_dict(children_ids[0])[layer]

            new_node_id = self.cg.get_unique_node_id(chunk_id)

            val_dict = {column_keys.Hierarchy.Child:
                            np.array(children_ids, dtype=np.uint64)}
            rows.append(
                self.cg.mutate_row(serializers.serialize_uint64(new_node_id),
                                   val_dict, time_stamp=time_stamp))

            val_dict = {column_keys.Hierarchy.Parent:
                            np.array([new_node_id], dtype=np.uint64)}

            self._children_dict[new_node_id] = []
            for child_id in children_ids:
                rows.append(
                    self.cg.mutate_row(serializers.serialize_uint64(child_id),
                                       val_dict, time_stamp=time_stamp))
                self._parent_dict[child_id] = new_node_id
                self._children_dict[new_node_id].append(child_id)

            self._children_dict[upper_parent_id] = [new_node_id]
            self._parent_dict[new_node_id] = upper_parent_id

        return rows

    def add_missing_nodes(self, merge_edges, time_stamp):
        rows = []

        edge_layers = self.cg.get_cross_chunk_edges_layer(merge_edges)
        edge_layers_m = edge_layers > 1

        for layer, edge in zip(edge_layers[edge_layers_m],
                               merge_edges[edge_layers_m]):
            rows.extend(self._add_skip_node(edge[0], layer, time_stamp))
            rows.extend(self._add_skip_node(edge[1], layer, time_stamp))

        return rows
