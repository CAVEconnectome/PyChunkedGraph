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

def add_edges(cg: chunkedgraph.ChunkedGraph,
              operation_id: np.uint64,
              atomic_edges: Sequence[Sequence[np.uint64]],
              time_stamp: datetime.datetime,
              affinities: Optional[Sequence[np.float32]] = None
              ):
    """ Add edges to chunkedgraph

    Computes all new rows to be written to the chunkedgraph

    :param cg: ChunkedGraph instance
    :param operation_id: np.uint64
    :param atomic_edges: list of list of np.uint64
        edges between supervoxels
    :param time_stamp: datetime.datetime
    :param affinities: list of np.float32
    :return: list
    """
    #TODO: add return tuple type

    atomic_edges = np.array(atomic_edges, dtype=np.uint64)

    # Comply to resolution of BigTables TimeRange
    time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                  round_up=False)

    if affinities is None:
        affinities = np.ones(len(atomic_edges),
                             dtype=column_keys.Connectivity.Affinity.basetype)

    assert len(affinities) == len(atomic_edges)

    rows = [] # list of rows to be written to BigTable
    lvl2_dict = {}
    lvl2_cross_chunk_edge_dict = {}

    lvl2_edges = []
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    edge_layer_m = edge_layers > 1

    # New edges are either within or across chunks. If an edge is across a
    # chunk boundary we need to store it as new cross edge. Otherwise, this
    # edge will combine two formerly disconnected lvl2 segments.
    new_cross_edge_dict = {}
    for atomic_edge in atomic_edges[~edge_layer_m]:
        lvl2_edges.append([cg.get_parent(atomic_edge[0]),
                           cg.get_parent(atomic_edge[1])])

    for atomic_edge, layer in zip(atomic_edges[edge_layer_m],
                                  edge_layers[edge_layer_m]):
        parent_id_0 = cg.get_parent(atomic_edge[0])
        parent_id_1 = cg.get_parent(atomic_edge[1])

        new_cross_edge_dict[parent_id_0] = {layer: atomic_edge}
        new_cross_edge_dict[parent_id_1] = {layer: atomic_edge[::-1]}

        lvl2_edges.append([parent_id_0, parent_id_0])
        lvl2_edges.append([parent_id_1, parent_id_1])

    # Compute connected components on lvl2
    graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
        lvl2_edges, make_directed=True)

    ccs = flatgraph_utils.connected_components(graph)
    for cc in ccs:
        lvl2_ids = unique_graph_ids[cc]
        chunk_id = cg.get_chunk_id(lvl2_ids[0])

        new_node_id = cg.get_unique_node_id(chunk_id)
        lvl2_dict[new_node_id] = lvl2_ids

        cross_chunk_edge_dict = {}
        for lvl2_id in lvl2_ids:
            lvl2_id_cross_chunk_edges = cg.read_cross_chunk_edges(lvl2_id)
            cross_chunk_edge_dict = \
                combine_cross_chunk_edge_dicts(
                    cross_chunk_edge_dict,
                    lvl2_id_cross_chunk_edges)

            if lvl2_id in new_cross_edge_dict:
                cross_chunk_edge_dict = \
                    combine_cross_chunk_edge_dicts(
                        new_cross_edge_dict[lvl2_id],
                        lvl2_id_cross_chunk_edges)

        lvl2_cross_chunk_edge_dict[new_node_id] = cross_chunk_edge_dict

    # Propagate changes up the tree
    new_root_ids, new_rows = propagate_edits_to_root(
        cg, lvl2_dict, lvl2_cross_chunk_edge_dict, operation_id=operation_id,
        time_stamp=time_stamp)
    rows.extend(new_rows)

    return new_root_ids, rows


def old_parent_childrens(eh, node_ids, layer):
    """ Retrieves the former partners of new nodes

    Two steps
        1. acquire old parents
        2. read children of those old parents

    :param eh: EditHelper instance
    :param node_ids: list of np.uint64s
    :param layer: np.int
    :return:
    """
    assert len(node_ids) > 0
    assert np.sum(np.in1d(node_ids, eh.new_node_ids)) == len(node_ids)

    # 1 - gather all next layer parents
    old_next_layer_node_ids = []
    old_this_layer_node_ids = []
    for node_id in node_ids:
        old_next_layer_node_ids.extend(
            eh.get_old_node_ids(node_id, layer + 1))

        old_this_layer_node_ids.extend(
            eh.get_old_node_ids(node_id, layer))

    old_next_layer_node_ids = np.unique(old_next_layer_node_ids)
    next_layer_m = eh.cg.get_chunk_layers(old_next_layer_node_ids) == layer + 1
    old_next_layer_node_ids = old_next_layer_node_ids[next_layer_m]

    old_this_layer_node_ids = np.unique(old_this_layer_node_ids)
    this_layer_m = eh.cg.get_chunk_layers(old_this_layer_node_ids) == layer
    old_this_layer_node_ids = old_this_layer_node_ids[this_layer_m]

    # 2 - acquire their children
    old_this_layer_partner_ids = []
    for old_next_layer_node_id in old_next_layer_node_ids:
        partner_ids = eh.get_layer_children(old_next_layer_node_id, layer,
                                            layer_only=True)

        partner_ids = partner_ids[~np.in1d(partner_ids,
                                           old_this_layer_node_ids)]
        old_this_layer_partner_ids.extend(partner_ids)

    old_this_layer_partner_ids = np.unique(old_this_layer_partner_ids)

    return old_this_layer_node_ids, old_next_layer_node_ids, \
           old_this_layer_partner_ids


def compute_cross_chunk_connected_components(eh, node_ids, layer):
    """ Computes connected component for next layer

    :param eh: EditHelper
    :param node_ids: list of np.uint64s
    :param layer: np.int
    :return:
    """
    assert len(node_ids) > 0

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
    for node_id in node_ids:
        node_cross_edges = eh.read_cross_chunk_edges(node_id)[layer]
        edge_id_map.update(dict(zip(node_cross_edges[:, 0],
                                    [node_id] * len(node_cross_edges))))
        cross_edges_lvl1.extend(node_cross_edges)

    for old_partner_id in old_this_layer_partner_ids:
        node_cross_edges = eh.read_cross_chunk_edges(old_partner_id)[layer]

        edge_id_map.update(dict(zip(node_cross_edges[:, 0],
                                    [old_partner_id] * len(node_cross_edges))))
        cross_edges_lvl1.extend(node_cross_edges)

    cross_edges_lvl1 = np.array(cross_edges_lvl1)
    edge_id_map_vec = np.vectorize(edge_id_map.get)

    if len(cross_edges_lvl1) > 0:
        cross_edges = edge_id_map_vec(cross_edges_lvl1)
    else:
        cross_edges = np.empty([0, 2], dtype=np.uint64)

    assert np.sum(np.in1d(eh.old_node_ids, cross_edges)) == 0

    cross_edges = np.concatenate([cross_edges,
                                  np.vstack([node_ids, node_ids]).T])

    graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
        cross_edges, make_directed=True)

    ccs = flatgraph_utils.connected_components(graph)

    return ccs, unique_graph_ids


def create_parent_children_rows(eh, parent_id, children_ids,
                                parent_cross_chunk_edge_dict, former_root_ids,
                                operation_id, time_stamp):
    """ Generates BigTable rows

    :param eh: EditHelper
    :param parent_id: np.uint64
    :param children_ids: list of np.uint64s
    :param parent_cross_chunk_edge_dict: dict
    :param former_root_ids: list of np.uint64s
    :param operation_id: np.uint64
    :param time_stamp: datetime.datetime
    :return:
    """

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

    if former_root_ids is not None:
        for former_root_id in former_root_ids:
            val_dict = {column_keys.Hierarchy.NewParent: parent_id}

            rows.append(eh.cg.mutate_row(
                serializers.serialize_uint64(former_root_id),
                val_dict, time_stamp=time_stamp))

    for child_id in children_ids:
        val_dict = {column_keys.Hierarchy.Parent: parent_id}
        rows.append(eh.cg.mutate_row(serializers.serialize_uint64(child_id),
                                     val_dict, time_stamp=time_stamp))

    return rows

def propagate_edits_to_root(cg: chunkedgraph.ChunkedGraph,
                            lvl2_dict: Dict,
                            lvl2_cross_chunk_edge_dict: Dict,
                            operation_id: np.uint64,
                            time_stamp: datetime.datetime):
    """ Propagates changes through layers

    :param cg: ChunkedGraph instance
    :param lvl2_dict: dict
        maps new ids to old ids
    :param lvl2_cross_chunk_edge_dict: dict
    :param operation_id: np.uint64
    :param time_stamp: datetime.datetime
    :return:
    """
    rows = []

    # Initialization
    eh = EditHelper(cg, lvl2_dict, lvl2_cross_chunk_edge_dict)
    eh.bulk_family_read()
    # eh.bulk_cross_chunk_edge_read()

    # Insert new nodes if missing (due to skip connections)
    # rows.extend(eh.add_missing_nodes(merge_edges, time_stamp))

    # Setup loop variables
    layer_dict = collections.defaultdict(list)
    layer_dict[2] = list(lvl2_dict.keys())
    new_root_ids = []
    # Loop over all layers up to the top - there might be layers where there is
    # nothing to do
    for current_layer in range(2, eh.cg.n_layers):
        print(f"CURRENT LAYER: {current_layer}")

        if len(layer_dict[current_layer]) == 0:
            continue

        new_node_ids = layer_dict[current_layer]

        # Calculate connected components based on cross chunk edges ------------
        ccs, unique_graph_ids = \
            compute_cross_chunk_connected_components(eh, new_node_ids,
                                                     current_layer)

        # Build a dictionary of new connected components -----------------------
        cc_collections = collections.defaultdict(list)
        for cc in ccs:
            cc_node_ids = unique_graph_ids[cc]
            cc_cross_edge_dict = collections.defaultdict(list)
            for cc_node_id in cc_node_ids:
                node_cross_edges = eh.read_cross_chunk_edges(cc_node_id)
                cc_cross_edge_dict = \
                    combine_cross_chunk_edge_dicts(cc_cross_edge_dict,
                                                   node_cross_edges,
                                                   start_layer=current_layer + 1)

            if (not current_layer + 1 in cc_cross_edge_dict or
                len(cc_cross_edge_dict[current_layer + 1]) == 0) and \
                    len(cc_node_ids) == 1:
                # Skip connection
                next_layer = None
                for l in range(current_layer + 1, eh.cg.n_layers):
                    if len(cc_cross_edge_dict[l]) > 0:
                        next_layer = l
                        break

                if next_layer is None:
                    next_layer = eh.cg.n_layers
            else:
                next_layer = current_layer + 1

            next_layer_chunk_id = eh.cg.get_parent_chunk_id_dict(cc_node_ids[0])[next_layer]

            cc_collections[next_layer_chunk_id].append(
                [cc_node_ids, cc_cross_edge_dict])

        # At this point we extracted all relevant data - now we just need to
        # create the new rows --------------------------------------------------
        for next_layer_chunk_id in cc_collections:
            n_ids = len(cc_collections[next_layer_chunk_id])
            new_parent_ids = eh.cg.get_unique_node_id_range(next_layer_chunk_id,
                                                            n_ids)
            next_layer = eh.cg.get_chunk_layer(next_layer_chunk_id)

            for new_parent_id, cc_collection in \
                    zip(new_parent_ids, cc_collections[next_layer_chunk_id]):
                layer_dict[next_layer].append(new_parent_id)
                eh.add_new_layer_node(new_parent_id, cc_collection[0],
                                      cc_collection[1])

                if eh.cg.get_chunk_layer(next_layer_chunk_id) == eh.cg.n_layers:
                    new_root_ids.append(new_parent_id)
                    former_root_ids = []
                    for new_parent_id in new_parent_ids:
                        former_root_ids.extend(
                            eh.get_old_node_ids(new_parent_id, eh.cg.n_layers))

                    former_root_ids = np.array(former_root_ids)
                else:
                    former_root_ids = None

                cc_rows = create_parent_children_rows(eh, new_parent_id,
                                                      cc_collection[0],
                                                      cc_collection[1],
                                                      former_root_ids,
                                                      operation_id,
                                                      time_stamp)
                rows.extend(cc_rows)

    return new_root_ids, rows


class EditHelper(object):
    def __init__(self, cg, lvl2_dict, cross_chunk_edge_dict):
        """

        :param cg: ChunkedGraph isntance
        :param lvl2_dict: maps new lvl2 ids to old lvl2 ids
        """
        self._cg = cg
        self._lvl2_dict = lvl2_dict

        self._parent_dict = {}
        self._children_dict = {}
        self._cross_chunk_edge_dict = cross_chunk_edge_dict
        self._new_node_ids = list(lvl2_dict.keys())
        self._old_node_dict = lvl2_dict

    @property
    def cg(self):
        return self._cg

    @property
    def lvl2_dict(self):
        return self._lvl2_dict

    @property
    def old_node_dict(self):
        return self._old_node_dict

    @property
    def old_node_ids(self):
        return np.concatenate(list(self.old_node_dict.values()))

    @property
    def new_node_ids(self):
        return self._new_node_ids

    def get_children(self, node_id):
        """ Cache around the get_children call to the chunkedgraph

        :param node_id: np.uint64
        :return: np.uint64
        """
        if not node_id in self._children_dict:
            self._children_dict[node_id] = self.cg.get_children(node_id)
            for child_id in self._children_dict[node_id]:
                if not child_id in self._parent_dict:
                    self._parent_dict[child_id] = node_id
                else:
                    assert self._parent_dict[child_id] == node_id

            print(f"MISS CHILDREN -- node_id {node_id}")

        return self._children_dict[node_id]

    def get_parent(self, node_id):
        """ Cache around the get_parent call to the chunkedgraph

        :param node_id: np.uint64
        :return: np.uint64
        """
        if not node_id in self._parent_dict:
            self._parent_dict[node_id] = self.cg.get_parent(node_id)

            print(f"MISS PARENT -- node_id {node_id}")

        return self._parent_dict[node_id]

    def get_root(self, node_id, get_all_parents=False):
        parents = [node_id]

        while self.get_parent(parents[-1]) is not None:
            parents.append(self.get_parent(parents[-1]))

        if get_all_parents:
            return np.array(parents)
        else:
            return parents[-1]

    def get_layer_children(self, node_id, layer, layer_only=False):
        """ Get 

        :param node_id:
        :param layer:
        :param layer_only:
        :return:
        """
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
            child_id = children_ids[0]

            if self.cg.get_chunk_layer(child_id) > layer:
                next_children_ids.extend(children_ids)
            elif self.cg.get_chunk_layer(child_id) == layer:
                layer_children_ids.extend(children_ids)
            elif self.cg.get_chunk_layer(child_id) < layer and not layer_only:
                layer_children_ids.extend(children_ids)

        return np.array(layer_children_ids, dtype=np.uint64)

    def get_layer_parent(self, node_id, layer, layer_only=False,
                         choose_lower_layer=False):
        """ Gets parent in particular layer

        :param node_id: np.uint64
        :param layer: np.int
        :param layer_only: bool
        :param choose_lower_layer: bool
        :return:
        """
        assert layer >= self.cg.get_chunk_layer(node_id)
        assert layer <= self.cg.n_layers

        if self.cg.get_chunk_layer(node_id) == layer:
            return [node_id]

        layer_parent_ids = []
        next_parent_ids = [node_id]

        while len(next_parent_ids) > 0:
            next_parent_id = next_parent_ids[0]
            del next_parent_ids[0]

            parent_id = self.get_parent(next_parent_id)

            if parent_id is None:
                raise()

            if self.cg.get_chunk_layer(parent_id) < layer:
                next_parent_ids.append(parent_id)
            elif self.cg.get_chunk_layer(parent_id) == layer:
                layer_parent_ids.append(parent_id)
            elif self.cg.get_chunk_layer(parent_id) > layer and not layer_only:
                if choose_lower_layer:
                    layer_parent_ids.append(next_parent_id)
                else:
                    layer_parent_ids.append(parent_id)

        return layer_parent_ids

    def _get_lower_old_node_ids(self, node_id):
        if not node_id in self._new_node_ids:
            return []
        elif node_id in self._old_node_dict:
            return self._old_node_dict[node_id]
        else:
            assert self.cg.get_chunk_layer(node_id) > 1

            old_node_ids = []
            for child_id in self.get_children(node_id):
                old_node_ids.extend(self._get_lower_old_node_ids(child_id))

            return np.unique(old_node_ids)

    def get_old_node_ids(self, node_id, layer):
        """ Acquires old node ids for new node id

        :param node_id: np.uint64
        :param layer: np.int
        :return:
        """
        lower_old_node_ids = self._get_lower_old_node_ids(node_id)

        old_node_ids = []
        for lower_old_node_id in lower_old_node_ids:
            old_node_ids.extend(self.get_layer_parent(lower_old_node_id, layer,
                                                      choose_lower_layer=True))

        old_node_ids = np.unique(old_node_ids)
        return old_node_ids

    def read_cross_chunk_edges(self, node_id):
        """ Cache around the read_cross_chunk_edges call to the chunkedgraph

        :param node_id: np.uint64
        :return: dict
        """
        if not node_id in self._cross_chunk_edge_dict:
            self._cross_chunk_edge_dict[node_id] = \
                self.cg.read_cross_chunk_edges(node_id)
            print(f"NO HIT -- {len(self._cross_chunk_edge_dict)} -- {node_id} -- "
                  f"{node_id in self._children_dict} -- {len(self._cross_chunk_edge_dict[node_id])}")

        return self._cross_chunk_edge_dict[node_id]

    def bulk_family_read(self):
        """ Caches parent and children information that will be needed later """
        def _get_root_thread(lvl2_node_id):
            p_ids = self.cg.get_root(lvl2_node_id, get_all_parents=True)
            p_ids = np.concatenate([[lvl2_node_id], p_ids])

            for i_parent in range(len(p_ids) - 1):
                self._parent_dict[p_ids[i_parent]] = p_ids[i_parent+1]

        def _read_cc_edges_thread(node_ids):
            for node_id in node_ids:
                if self.cg.get_chunk_layer(node_id) == self.cg.n_layers:
                    continue

                self.read_cross_chunk_edges(node_id)

        lvl2_node_ids = []
        for v in self.lvl2_dict.values():
            lvl2_node_ids.extend(v)

        mu.multithread_func(_get_root_thread, lvl2_node_ids,
                            n_threads=len(lvl2_node_ids), debug=False)

        parent_ids = list(self._parent_dict.values())
        child_dict = self.cg.get_children(parent_ids, flatten=False)
        node_ids = []

        for parent_id in child_dict:
            self._children_dict[parent_id] = child_dict[parent_id]

            if self.cg.get_chunk_layer(parent_id) > 2:
                node_ids.extend(child_dict[parent_id])

            node_ids.append(parent_id)

            for child_id in self._children_dict[parent_id]:
                if not child_id in self._parent_dict:
                    self._parent_dict[child_id] = parent_id
                else:
                    assert self._parent_dict[child_id] == parent_id

        node_ids = np.unique(node_ids)
        n_threads = int(len(node_ids) / 5)

        print(f"n_threads: {n_threads}")

        node_id_blocks = np.array_split(node_ids, n_threads)

        mu.multithread_func(_read_cc_edges_thread, node_id_blocks,
                            n_threads=len(child_dict), debug=False)

    def bulk_cross_chunk_edge_read(self):
        raise NotImplementedError

    def add_new_layer_node(self, node_id, children_ids, cross_chunk_edge_dict):
        """ Adds a new node to the helper infrastructure

        :param node_id: np.uint64
        :param children_ids: list of np.uint64s
        :param cross_chunk_edge_dict: dict
        :return:
        """
        self._cross_chunk_edge_dict[node_id] = cross_chunk_edge_dict

        self._children_dict[node_id] = children_ids
        for child_id in children_ids:
            self._parent_dict[child_id] = node_id

        self._new_node_ids.append(node_id)
        layer = self.cg.get_chunk_layer(node_id)
        self._old_node_dict[node_id] = self.get_old_node_ids(node_id, layer)
