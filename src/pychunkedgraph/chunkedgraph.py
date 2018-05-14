import collections
import numpy as np
import time
import datetime
import os
import networkx as nx
import pytz

from google.cloud import bigtable

# global variables
HOME = os.path.expanduser("~")
N_DIGITS_UINT64 = len(str(np.iinfo(np.uint64).max))
UTC = pytz.UTC

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = HOME + "/.cloudvolume/secrets/google-secret.json"


def serialize_node_id(node_id):
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    s_node_id = "%.20d" % node_id
    s_node_id = serialize_key(s_node_id)
    return s_node_id


def serialize_key(key):
    """ Serializes a key to be ingested by a bigtable table row

    :param key: str
    :return: str
    """
    return key.encode("utf-8")


def mutate_row(table, row_key, column_family_id, val_dict, time_stamp=None):
    """

    :param table: bigtable table instance
    :param row_key: serialized bigtable row key
    :param column_family_id: str
        serialized column family id
    :param val_dict: dict
    :param time_stamp: None or datetime
    :return: list
    """
    row = table.row(row_key)

    for column, value in val_dict.items():
        row.set_cell(column_family_id=column_family_id, column=column,
                     value=value, timestamp=time_stamp)
    return row


def get_chunk_id_from_node_id(node_id, dtype=np.uint8):
    """ Extracts z, y, x, l

    :param node_id: int
    :return: list of ints
    """

    if dtype == np.uint8:
        return np.frombuffer(np.uint64(node_id), dtype=np.uint8)[4:]
    elif dtype == np.uint32:
        return np.frombuffer(np.uint64(node_id), dtype=np.uint32)[1:]
    else:
        raise NotImplementedError()


def get_chunk_ids_from_node_ids(node_id, dtype=np.uint8):
    """ Extracts z, y, x, l

    :param node_id: array of ints
    :return: list of ints
    """

    if dtype == np.uint8:
        return np.frombuffer(np.uint64(node_id).copy(order='C'),
                             dtype=np.uint8).reshape(-1, 8)[:, 4:]
    elif dtype == np.uint32:
        return np.frombuffer(np.uint64(node_id).copy(order='C'),
                             dtype=np.uint32).reshape(-1, 2)[:, 1:]
    else:
        raise NotImplementedError()


def test_if_nodes_are_in_same_chunk(node_ids):
    """ Test whether two nodes are in the same chunk

    :param node_ids: list of two ints
    :return: bool
    """
    assert len(node_ids) == 2

    return np.frombuffer(node_ids[0], dtype=np.uint32)[1] == \
           np.frombuffer(node_ids[1], dtype=np.uint32)[1]


class ChunkedGraph(object):
    def __init__(self, instance_id="pychunkedgraph",
                 project_id="neuromancer-seung-import",
                 chunk_size=(512, 512, 64), dev_mode=False,
                 table_id=None):

        self._client = bigtable.Client(project=project_id, admin=True)
        self._instance = self.client.instance(instance_id)

        if table_id is None:
            if dev_mode:
                self._table = self.instance.table("pychgtable_dev")
            else:
                self._table = self.instance.table("pychgtable")
        else:
            self._table = self.instance.table(table_id)

        self._fan_out = 2
        self._chunk_size = np.array(chunk_size)

    @property
    def client(self):
        return self._client

    @property
    def instance(self):
        return self._instance

    @property
    def table(self):
        return self._table

    @property
    def family_id(self):
        return "0"

    @property
    def fan_out(self):
        return self._fan_out

    @property
    def chunk_size(self):
        return self._chunk_size

    def get_cg_id_from_rg_id(self, atomic_id):
        """ Extracts ChunkedGraph id from RegionGraph id

        :param atomic_id: int
        :return: int
        """
        # There might be multiple chunk ids for a single rag id because
        # rag supervoxels get split at chunk boundaries. Here, only one
        # chunk id needs to be traced to the top to retrieve the
        # agglomeration id that they both belong to
        r = self.table.read_row(serialize_node_id(atomic_id))
        return np.frombuffer(r.cells[self.family_id][serialize_key("cg_id")][0].value,
                             dtype=np.uint64)[0]

    def get_rg_id_from_cg_id(self, atomic_id):
        """ Extracts RegionGraph id from ChunkedGraph id

        :param atomic_id: int
        :return: int
        """
        return self.read_row(atomic_id, "rg_id")[0]

    def find_unique_node_id(self, chunk_id):
        """ Finds a unique node id for the given chunk

        :param chunk_id: uint32
        :return: uint64
        """

        chunk_id = np.frombuffer(chunk_id.astype(np.uint32), dtype=np.uint8)
        node_id = np.frombuffer(np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint64), dtype=np.uint8).copy()
        node_id[4:] = chunk_id
        node_id = np.frombuffer(node_id, dtype=np.uint64)

        while self.table.read_row(serialize_node_id(node_id)) is not None:
            node_id = np.frombuffer(np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint64), dtype=np.uint8).copy()
            node_id[4:] = chunk_id
            node_id = np.frombuffer(node_id, dtype=np.uint64)

        return node_id

    def read_row(self, node_id, key, idx=0, dtype=np.uint64):
        row = self.table.read_row(serialize_node_id(node_id))
        return np.frombuffer(row.cells[self.family_id][serialize_key(key)][idx].value, dtype=dtype)

    def read_rows(self, node_ids, key, dtype=np.uint64):
        results = []

        for node_id in node_ids:
            results.append(np.frombuffer(self.table.read_row(
                serialize_node_id(node_id).cells[self.family_id][
                serialize_key(key)]), dtype=dtype))

        return results

    def add_atomic_edges_in_chunks(self, edge_ids, cross_edge_ids, edge_affs,
                                   cross_edge_affs, cg2rg_dict, rg2cg_dict,
                                   time_stamp=None):
        """ Creates atomic edges between supervoxels and first
            abstraction layer """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Catch trivial case
        if len(edge_ids) == 0:
            return 0

        # Write rg2cg mapping to table
        rows = []
        for rg_id in rg2cg_dict.keys():
            # Create node
            val_dict = {"cg_id": np.array([rg2cg_dict[rg_id]]).tobytes()}

            rows.append(mutate_row(self.table, serialize_node_id(rg_id),
                                   self.family_id, val_dict))
        status = self.table.mutate_rows(rows)

        # Make parent id creation easier
        z, y, x, l = get_chunk_id_from_node_id(edge_ids[0, 0])
        parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, l+1], dtype=np.uint8), dtype=np.uint32)

        # Get connected component within the chunk
        chunk_g = nx.from_edgelist(edge_ids)
        chunk_g.add_nodes_from(np.unique(cross_edge_ids[:, 0]))
        ccs = list(nx.connected_components(chunk_g))

        # print("%d ccs detected" % (len(ccs)))

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the print / speed measurement
        time_start = time.time()
        for i_cc, cc in enumerate(ccs):
            if node_c > 0:
                dt = time.time() - time_start
                print("%5d at %5d - %.5fs             " %
                      (i_cc, node_c, dt / node_c), end="\r")

            rows = []

            node_ids = np.array(list(cc))

            # Create parent id
            parent_id = parent_id_base.copy()
            parent_id[0] = i_cc
            parent_id = np.frombuffer(parent_id, dtype=np.uint64)
            parent_id_b = parent_id.tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # print("Node:", node_id)
                # Extract edges relevant to this node
                edge_col1_mask = edge_ids[:, 0] == node_id
                edge_col2_mask = edge_ids[:, 1] == node_id

                # Cross edges are ordered to always point OUT of the chunk
                cross_edge_mask = cross_edge_ids[:, 0] == node_id

                parent_cross_edges = np.concatenate([parent_cross_edges,
                                                     cross_edge_ids[cross_edge_mask]])

                connected_partner_ids = np.concatenate([edge_ids[edge_col1_mask][:, 1],
                                                        edge_ids[edge_col2_mask][:, 0],
                                                        cross_edge_ids[cross_edge_mask][:, 1]]).tobytes()

                connected_partner_affs = np.concatenate([edge_affs[np.logical_or(edge_col1_mask, edge_col2_mask)],
                                                         cross_edge_affs[cross_edge_mask]]).tobytes()

                # Create node
                val_dict = {"atomic_partners": connected_partner_ids,
                            "atomic_affinities": connected_partner_affs,
                            "parents": parent_id_b,
                            "rg_id": np.array([cg2rg_dict[node_id]]).tobytes()}

                rows.append(mutate_row(self.table, serialize_node_id(node_id),
                                       self.family_id, val_dict))
                node_c += 1

            # Create parent node
            val_dict = {"children": node_ids.tobytes(),
                        "atomic_cross_edges": parent_cross_edges.tobytes()}

            rows.append(mutate_row(self.table, serialize_node_id(parent_id),
                                   self.family_id, val_dict))

            node_c += 1

            status = self.table.mutate_rows(rows)

        try:
            dt = time.time() - time_start
            print("Average time: %.5fs / node; %.5fs / edge - Number of edges: %6d, %6d" %
                  (dt / node_c, dt / len(edge_ids), len(edge_ids), len(cross_edge_ids)))
        except:
            print("WARNING: NOTHING HAPPENED")

    def add_layer(self, layer_id, child_chunk_coords, time_stamp=None):
        """ Creates all hierarchy layers above the first abstract layer """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # 1 ----------
        # The first part is concerned with reading data from the child nodes
        # of this layer and pre-processing it for the second part

        atomic_child_ids = np.array([], dtype=np.uint64)    # ids in lowest layer
        child_ids = np.array([], dtype=np.uint64)   # ids in layer one below this one
        atomic_partner_id_dict = {}
        atomic_child_id_dict = {}

        leftover_atomic_edges = {}

        for chunk_coord in child_chunk_coords:
            # Get start and end key
            x, y, z = chunk_coord
            node_id_base = np.array([0, 0, 0, 0, z, y, x, layer_id - 1], dtype=np.uint8)
            node_id_base_next = node_id_base.copy()

            step = self.fan_out ** (layer_id - 3)
            node_id_base_next[-4] += step

            start_key = serialize_node_id(np.frombuffer(node_id_base, dtype=np.uint64)[0])
            end_key = serialize_node_id(np.frombuffer(node_id_base_next, dtype=np.uint64)[0])

            # Set up read
            range_read = self.table.read_rows(start_key=start_key, end_key=end_key, end_inclusive=False)
            # Execute read
            range_read.consume_all()

            # Loop through nodes from this chunk
            for row_key, row_data in range_read.rows.items():
                atomic_edges = np.frombuffer(row_data.cells[self.family_id][serialize_key("atomic_cross_edges")][0].value, dtype=np.uint64).reshape(-1, 2)
                atomic_partner_id_dict[int(row_key)] = atomic_edges[:, 1]
                atomic_child_id_dict[int(row_key)] = atomic_edges[:, 0]

                atomic_child_ids = np.concatenate([atomic_child_ids, atomic_edges[:, 0]])
                child_ids = np.concatenate([child_ids, np.array([row_key] * len(atomic_edges[:, 0]), dtype=np.uint64)])

            # print(chunk_coord, start_key, end_key, np.unique(get_chunk_ids_from_node_ids(atomic_child_ids, dtype=np.uint32)))

        # Extract edges from remaining cross chunk edges
        # and maintain unused cross chunk edges
        edge_ids = np.array([], np.uint64).reshape(0, 2)

        u_atomic_child_ids = np.unique(atomic_child_ids)
        atomic_partner_id_dict_keys = np.array(list(atomic_partner_id_dict.keys()), dtype=np.uint64)
        time_start = time.time()

        time_segs = [[], [], []]
        for i_child_key, child_key in enumerate(atomic_partner_id_dict_keys):
            if i_child_key % 20 == 1:
                dt = time.time() - time_start
                eta = dt / i_child_key * len(atomic_partner_id_dict_keys) - dt
                print("%5d - dt: %.3fs - eta: %.3fs - %.4fs - %.4fs - %.4fs           " %
                      (i_child_key, dt, eta, np.mean(time_segs[0]), np.mean(time_segs[1]), np.mean(time_segs[2])), end="\r")

            this_atomic_partner_ids = atomic_partner_id_dict[child_key]
            this_atomic_child_ids = atomic_child_id_dict[child_key]

            time_seg = time.time()

            leftover_mask = ~np.in1d(this_atomic_partner_ids, u_atomic_child_ids)

            time_segs[0].append(time.time() - time_seg)
            time_seg = time.time()
            leftover_atomic_edges[child_key] = np.concatenate([this_atomic_child_ids[leftover_mask, None],
                                                               this_atomic_partner_ids[leftover_mask, None]], axis=1)

            time_segs[1].append(time.time() - time_seg)
            time_seg = time.time()

            partners = np.unique(child_ids[np.in1d(atomic_child_ids, this_atomic_partner_ids)])
            these_edges = np.concatenate([np.array([child_key] * len(partners), dtype=np.uint64)[:, None], partners[:, None]], axis=1)

            edge_ids = np.concatenate([edge_ids, these_edges])

            time_segs[2].append(time.time() - time_seg)

            # if child_key == 288230376151711873:
            #     raise()

        # 2 ----------
        # The second part finds connected components, writes the parents to
        # BigTable and updates the childs

        # Make parent id creation easier
        x, y, z = np.min(child_chunk_coords, axis=0)
        parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, layer_id], dtype=np.uint8), dtype=np.uint32)

        # Extract connected components
        chunk_g = nx.from_edgelist(edge_ids)
        # chunk_g.add_nodes_from(atomic_partner_id_dict_keys)

        # Add single node objects that have no edges
        add_ccs = []
        for node_id in atomic_partner_id_dict_keys[~np.in1d(atomic_partner_id_dict_keys, np.unique(edge_ids))]:
            add_ccs.append([node_id])

        ccs = list(nx.connected_components(chunk_g)) + add_ccs

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the print / speed measurement
        time_start = time.time()
        for i_cc, cc in enumerate(ccs):
            if node_c > 0:
                dt = time.time() - time_start
                print("%5d at %5d - %.5fs             " %
                      (i_cc, node_c, dt / node_c), end="\r")

            rows = []

            node_ids = np.array(list(cc))

            # Create parent id
            parent_id = parent_id_base.copy()
            parent_id[0] = i_cc
            parent_id = np.frombuffer(parent_id, dtype=np.uint64)
            parent_id_b = parent_id.tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # Extract edges relevant to this node
                parent_cross_edges = np.concatenate([parent_cross_edges,
                                                     leftover_atomic_edges[node_id]])

                # Create node
                val_dict = {"parents": parent_id_b}

                rows.append(mutate_row(self.table, serialize_node_id(node_id),
                                       self.family_id, val_dict))
                node_c += 1

            # Create parent node
            val_dict = {"children": node_ids.tobytes(),
                        "atomic_cross_edges": parent_cross_edges.tobytes()}

            rows.append(mutate_row(self.table, serialize_node_id(parent_id),
                                   self.family_id, val_dict))

            node_c += 1

            status = self.table.mutate_rows(rows)

        try:
            dt = time.time() - time_start
            print("Average time: %.5fs / node; %.5fs / edge - Number of edges: %6d" %
                  (dt / node_c, dt / len(edge_ids), len(edge_ids)))
        except:
            print("WARNING: NOTHING HAPPENED")

    def get_parent(self, node_id, time_stamp=None):
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        parent_key = serialize_key("parents")

        row = self.table.read_row(serialize_node_id(node_id))

        # if parent_key in row.cells[self.family_id]:
        #     for parent_entry in row.cells[self.family_id][parent_key]:
        #         print(parent_entry.timestamp)

        if parent_key in row.cells[self.family_id]:
            for parent_entry in row.cells[self.family_id][parent_key]:
                if parent_entry.timestamp > time_stamp:
                    continue
                else:
                    return np.frombuffer(parent_entry.value, dtype=np.uint64)[0]
        else:
            return None

        raise Exception("Did not find a valid parent for %d with"
                        " the given time stamp" % node_id)

    def get_children(self, node_id):
        return self.read_row(node_id, "children", dtype=np.uint64)

    def get_root(self, atomic_id, collect_all_parents=False,
                 time_stamp=None, is_cg_id=False):
        """ Takes an atomic id and returns the associated agglomeration ids

        :param atomic_id: int
        :param collect_all_parents: bool
        :param time_stamp: None or datetime
        :return: int
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        if not is_cg_id:
            atomic_id = self.get_cg_id_from_rg_id(atomic_id)

        parent_id = atomic_id

        parent_ids = []
        while True:
            # print(parent_id)
            temp_parent_id = self.get_parent(parent_id, time_stamp)
            if temp_parent_id is None:
                break
            else:
                parent_id = temp_parent_id
                parent_ids.append(parent_id)

        if collect_all_parents:
            return parent_ids
        else:
            return parent_id

    def read_agglomeration_id_history(self, agglomeration_id, time_stamp=None):
        """ Returns all agglomeration ids agglomeration_id was part of

        :param agglomeration_id: int
        :param time_stamp: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole history
        :return: array of int
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.min

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        id_working_set = np.array([agglomeration_id], dtype=np.uint64)
        visited_ids = []
        id_history = [agglomeration_id]

        former_parent_key = serialize_key("former_parents")
        new_parent_key = serialize_key("new_parents")

        i = 0
        while len(id_working_set) > 0:
            i += 1

            next_id = id_working_set[0]
            visited_ids.append(id_working_set[0])

            # Get current row
            r = self.table.read_row(serialize_node_id(next_id))

            # Check if there is a newer parent and append
            if new_parent_key in r.cells[self.family_id]:
                new_parent_ids = np.frombuffer(r.cells[self.family_id][new_parent_key][0].value, dtype=np.uint64)

                id_working_set = np.concatenate([id_working_set, new_parent_ids])
                id_history.extend(new_parent_ids)

            # Check if there is an older parent and append if not too old
            if former_parent_key in r.cells[self.family_id]:
                if time_stamp < r.cells[self.family_id][former_parent_key][0].timestamp:
                    former_parent_ids = np.frombuffer(r.cells[self.family_id][former_parent_key][0].value, dtype=np.uint64)

                    id_working_set = np.concatenate([id_working_set, former_parent_ids])
                    id_history.extend(former_parent_ids)

            id_working_set = id_working_set[~np.in1d(id_working_set, visited_ids)]

        return np.unique(id_history)

    def get_subgraph(self, agglomeration_id, bounding_box=None,
                     bb_is_coordinate=False, return_rg_ids=False,
                     time_stamp=None):
        """ Returns all edges between supervoxels belonging to the specified
            agglomeration id within the defined bouning box

        :param agglomeration_id: int
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param time_stamp: datetime or None
        :return: edge list
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        if bb_is_coordinate:
            bounding_box = np.array(bounding_box,
                                    dtype=np.float32) / self.chunk_size
            bounding_box[0] = np.floor(bounding_box[0])
            bounding_box[1] = np.ceil(bounding_box[1])

        # bounding_box = np.array(bounding_box, dtype=np.int)

        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        affinities = np.array([], dtype=np.float32)
        child_ids = [agglomeration_id]

        while len(child_ids) > 0:
            new_childs = []
            layer = get_chunk_id_from_node_id(child_ids[0])[-1]
            # print(layer, get_chunk_id_from_node_id(child_ids[0]))
            for child_id in child_ids:
                if layer == 2:
                    this_edges, this_affinities = self.get_subgraph_chunk(
                        child_id, time_stamp=time_stamp)

                    affinities = np.concatenate([affinities, this_affinities])
                    edges = np.concatenate([edges, this_edges])
                else:
                    this_children = self.get_children(child_id)

                    # cids_min = np.frombuffer(this_children, dtype=np.uint8).reshape(-1, 8)[:, 4:-1][:, ::-1] * self.fan_out ** np.max([0, (layer - 2)])
                    # cids_max = cids_min + self.fan_out * np.max([0, (layer - 2)])
                    #
                    # child_id_mask_min_upper = np.all(cids_min <= bounding_box[1], axis=1)
                    # child_id_mask_max_lower = np.all(cids_max > bounding_box[0], axis=1)
                    #
                    # m = np.logical_and(child_id_mask_min_upper, child_id_mask_max_lower)
                    # this_children = this_children[m]

                    new_childs.extend(this_children)

            child_ids = new_childs

        if return_rg_ids:
            rg_edges = np.zeros_like(edges, dtype=np.uint64)

            for u_id in np.unique(edges):
                rg_edges[edges == u_id] = self.get_rg_id_from_cg_id(u_id)

            return np.array(rg_edges), affinities
        else:
            return edges, affinities

    def get_subgraph_chunk(self, parent_id, time_stamp=None):
        """ Takes an atomic id and returns the associated agglomeration ids

        :param parent_id: int
        :param time_stamp: None or datetime
        :return: edge list
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        child_ids = self.get_children(parent_id)
        edge_key = serialize_key("atomic_partners")
        affinity_key = serialize_key("atomic_affinities")

        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        affinities = np.array([], dtype=np.float32)
        for child_id in child_ids:
            node_edges = np.array([], dtype=np.uint64)
            node_affinities = np.array([], dtype=np.float32)

            r = self.table.read_row(serialize_node_id(child_id))
            for i_edgelist in range(len(r.cells[self.family_id][edge_key])):
                if time_stamp > r.cells[self.family_id][edge_key][i_edgelist].timestamp:
                    edge_batch = np.frombuffer(
                        r.cells[self.family_id][edge_key][i_edgelist].value,
                        dtype=np.uint64)
                    affinity_batch = np.frombuffer(
                        r.cells[self.family_id][affinity_key][i_edgelist].value,
                        dtype=np.float32)
                    edge_batch_m = ~np.in1d(edge_batch, node_edges)

                    affinity_batch = affinity_batch[
                                     :len(edge_batch)]  # TEMPORARY HACK

                    node_edges = np.concatenate(
                        [node_edges, edge_batch[edge_batch_m]])
                    node_affinities = np.concatenate([node_affinities,
                                                      affinity_batch[
                                                          edge_batch_m]])

            node_edge_m = node_affinities > 0
            node_edges = node_edges[node_edge_m]
            node_affinities = node_affinities[node_edge_m]

            if len(node_edges) > 0:
                node_edges = np.concatenate(
                    [np.ones((len(node_edges), 1), dtype=np.uint64) * child_id,
                     node_edges[:, None]], axis=1)

                edges = np.concatenate([edges, node_edges])
                affinities = np.concatenate([affinities, node_affinities])

        return edges, affinities

    def add_edge(self, atomic_edge, affinity=None, is_cg_id=False):
        """ Adds an atomic edge to the ChunkedGraph

        :param atomic_edge: list of two ints
        :param affinity: float
        :param is_cg_id: bool
        """
        time_stamp = datetime.datetime.now()
        time_stamp = UTC.localize(time_stamp)

        if affinity is None:
            affinity = 1

        rows = []

        if not is_cg_id:
            atomic_edge = [self.get_cg_id_from_rg_id(atomic_edge[0]),
                           self.get_cg_id_from_rg_id(atomic_edge[1])]

        # Walk up the hierarchy until a parent in the same chunk is found
        original_parent_ids = [self.get_root(atomic_edge[0], is_cg_id=True,
                                             collect_all_parents=True),
                               self.get_root(atomic_edge[1], is_cg_id=True,
                                             collect_all_parents=True)]

        original_parent_ids = np.array(original_parent_ids).T

        merge_layer = None
        for i_layer in range(len(original_parent_ids)):
            if test_if_nodes_are_in_same_chunk(original_parent_ids[i_layer]):
                merge_layer = i_layer
                break

        if merge_layer is None:
            raise Exception("No parents found. Did you set is_cg_id correctly?")

        original_root = original_parent_ids[-1]

        # Find a new node id and update all children
        # circumvented_nodes = current_parent_ids.copy()
        chunk_id = get_chunk_id_from_node_id(original_parent_ids[merge_layer][0],
                                             dtype=np.uint32)
        new_parent_id = self.find_unique_node_id(chunk_id)
        new_parent_id_b = np.array(new_parent_id).tobytes()
        current_node_id = None

        for i_layer in range(merge_layer, len(original_parent_ids)):
            current_parent_ids = original_parent_ids[i_layer]

            # Collect child ids of all nodes --> childs of new node
            if current_node_id is None:
                combined_child_ids = np.array([], dtype=np.uint64)
            else:
                combined_child_ids = current_node_id

            for prior_parent_id in current_parent_ids:
                child_ids = self.get_children(prior_parent_id)

                # Exclude parent nodes from old hierarchy path
                child_ids = child_ids[~np.in1d(child_ids, original_parent_ids)]
                combined_child_ids = np.concatenate([combined_child_ids,
                                                     child_ids])

                # Append new parent entry for all children
                for child_id in child_ids:
                    val_dict = {"parents": new_parent_id_b}
                    rows.append(mutate_row(self.table,
                                           serialize_node_id(child_id),
                                           self.family_id,
                                           val_dict, time_stamp))

            # Create new parent node
            val_dict = {"children": combined_child_ids.tobytes()}
            current_node_id = new_parent_id.copy()  # Store for later

            if i_layer < len(original_parent_ids) - 1:
                chunk_id = get_chunk_id_from_node_id(original_parent_ids[i_layer + 1][0],
                                                     dtype=np.uint32)

                new_parent_id = self.find_unique_node_id(chunk_id)
                new_parent_id_b = np.array(new_parent_id).tobytes()

                val_dict["parents"] = new_parent_id_b
            else:
                val_dict["former_parents"] = np.array(original_root).tobytes()

                rows.append(mutate_row(self.table,
                                       serialize_node_id(original_root[0]),
                                       self.family_id,
                                       {"new_parents": new_parent_id_b}))

                rows.append(mutate_row(self.table,
                                       serialize_node_id(original_root[1]),
                                       self.family_id,
                                       {"new_parents": new_parent_id_b}))

            # Read original cross chunk edges
            atomic_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)
            for original_parent_id in original_parent_ids[i_layer]:
                this_atomic_cross_edges = self.read_row(original_parent_id,
                                                        "atomic_cross_edges").reshape(-1, 2)
                atomic_cross_edges = np.concatenate([atomic_cross_edges,
                                                     this_atomic_cross_edges])

            val_dict["atomic_cross_edges"] = atomic_cross_edges.tobytes()

            rows.append(mutate_row(self.table,
                                   serialize_node_id(current_node_id),
                                   self.family_id, val_dict))

        # Atomic edge
        for i_atomic_id in range(2):
            val_dict = {"atomic_partners": np.array([atomic_edge[(i_atomic_id + 1) % 2]]).tobytes(),
                        "atomic_affinities": np.array([affinity]).tobytes()}
            rows.append(mutate_row(self.table, serialize_node_id(atomic_edge[i_atomic_id]),
                                   self.family_id, val_dict, time_stamp))

        status = self.table.mutate_rows(rows)

        return new_parent_id[0]

    def remove_edge(self, atomic_edges, is_cg_id=False):
        time_stamp = datetime.datetime.now()
        time_stamp = UTC.localize(time_stamp)

        # Make sure that we have a list of edges
        if isinstance(atomic_edges[0], np.uint64):
            atomic_edges = [atomic_edges]

        if not is_cg_id:
            for i_atomic_edge in range(len(atomic_edges)):
                atomic_edges[i_atomic_edge] = [self.get_cg_id_from_rg_id(atomic_edges[i_atomic_edge][0]),
                                               self.get_cg_id_from_rg_id(atomic_edges[i_atomic_edge][1])]

        atomic_edges = np.array(atomic_edges)
        u_atomic_ids = np.unique(atomic_edges)

        original_parent_ids = self.get_root(atomic_edges[0, 0], is_cg_id=True,
                                            collect_all_parents=True)
        original_root = original_parent_ids[-1]

        # Find lowest level chunks that might have changed
        chunk_ids = get_chunk_ids_from_node_ids(u_atomic_ids, dtype=np.uint32)[:, 0]
        u_chunk_ids, u_chunk_ids_idx = np.unique(chunk_ids, return_index=True)

        involved_chunk_id_dict = dict(zip(u_chunk_ids, u_atomic_ids[u_chunk_ids_idx]))

        # Note: After removing the atomic edges, we basically need to build the
        # ChunkedGraph for these chunks from the ground up.
        # involved_chunk_id_dict stores a representative for each chunk that we
        # can use to acquire the parent that knows about all atomic nodes in the
        # chunk.

        # Remove atomic edges
        rows = []
        for atomic_edge in atomic_edges:
            for i_atomic_id in range(2):
                atomic_id = atomic_edge[i_atomic_id]

                val_dict = {"atomic_partners": np.array([atomic_edge[(i_atomic_id + 1) % 2]]).tobytes(),
                            "atomic_affinities": np.zeros(1, dtype=np.float32).tobytes()}
                rows.append(mutate_row(self.table, serialize_node_id(atomic_id),
                                       self.family_id, val_dict, time_stamp))

        # Execute the removal of the atomic edges - we cannot wait for that
        # until the end because we want to compute connected components on the
        # subgraph

        self.table.mutate_rows(rows)
        rows = []

        # For each involved chunk we need to compute connected components
        new_layer_parent_dict = {}
        cross_edge_dict = {}
        old_id_dict = collections.defaultdict(list)
        for chunk_id in involved_chunk_id_dict.keys():
            # Get the local subgraph
            node_id = involved_chunk_id_dict[chunk_id]
            old_parent_id = self.get_parent(node_id)
            edges, affinities = self.get_subgraph_chunk(old_parent_id)

            z, y, x, l = get_chunk_id_from_node_id(old_parent_id)
            parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, l],
                                                    dtype=np.uint8),
                                           dtype=np.uint32)[1]

            # The cross chunk edges are passed on to the parent to compute
            # connected components in higher layers.
            cross_edge_mask = get_chunk_ids_from_node_ids(edges[:, 1], dtype=np.uint32)[:, 0] != get_chunk_id_from_node_id(node_id, dtype=np.uint32)
            cross_edges = edges[cross_edge_mask]

            g = nx.from_edgelist(edges)
            ccs = nx.connected_components(g)

            # For each connected component we create one new parent
            for cc in ccs:
                cc_node_ids = np.array(list(cc), dtype=np.uint64)
                cc_cross_edges = cross_edges[np.in1d(cross_edges[:, 0],
                                                     cc_node_ids)]

                new_parent_id = self.find_unique_node_id(parent_id_base)
                new_parent_id_b = np.array(new_parent_id).tobytes()
                new_parent_id = new_parent_id[0]

                new_layer_parent_dict[new_parent_id] = old_parent_id
                cross_edge_dict[new_parent_id] = cc_cross_edges
                old_id_dict[old_parent_id].append(new_parent_id)

                val_dict = {"children": cc_node_ids.tobytes(),
                            "atomic_cross_edges": cc_cross_edges.tobytes()}

                rows.append(mutate_row(self.table,
                                       serialize_node_id(new_parent_id),
                                       self.family_id, val_dict))

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(mutate_row(self.table,
                                           serialize_node_id(cc_node_id),
                                           self.family_id, val_dict))

        new_roots = []
        for i_layer in range(len(original_parent_ids) - 1):
            parent_cc_list = []
            parent_cc_old_parent_list = []
            parent_cc_mapping = {}
            leftover_edges = {}
            parent_id_base_dict = {}

            for new_layer_parent in new_layer_parent_dict.keys():

                old_parent_id = new_layer_parent_dict[new_layer_parent]
                cross_edges = cross_edge_dict[new_layer_parent]

                if i_layer == 4:
                    raise()

                old_next_layer_parent = self.get_parent(old_parent_id)
                old_chunk_neighbors = self.get_children(old_next_layer_parent)
                old_chunk_neighbors = old_chunk_neighbors[old_chunk_neighbors != old_parent_id]

                z, y, x, l = get_chunk_id_from_node_id(old_next_layer_parent)
                parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, l],
                                                        dtype=np.uint8),
                                               dtype=np.uint32)[1]

                parent_id_base_dict[new_layer_parent] = parent_id_base

                atomic_children = cross_edges[:, 0]
                atomic_id_map = np.ones(len(cross_edges), dtype=np.uint64) * new_layer_parent
                partner_cross_edges = {new_layer_parent: cross_edges}

                # print(old_chunk_neighbors)
                for old_chunk_neighbor in old_chunk_neighbors:
                    if old_chunk_neighbor in old_id_dict:
                        for new_neighbor in old_id_dict[old_chunk_neighbor]:
                            neigh_cross_edges = cross_edge_dict[new_neighbor]
                            atomic_children = np.concatenate([atomic_children, neigh_cross_edges[:, 0]])
                            partner_cross_edges[new_neighbor] = neigh_cross_edges
                            atomic_id_map = np.concatenate([atomic_id_map, np.ones(len(neigh_cross_edges), dtype=np.uint64) * new_neighbor])
                    else:
                        neigh_cross_edges = self.read_row(old_chunk_neighbor, "atomic_cross_edges").reshape(-1, 2)
                        atomic_children = np.concatenate([atomic_children, neigh_cross_edges[:, 0]])

                        partner_cross_edges[old_chunk_neighbor] = neigh_cross_edges
                        atomic_id_map = np.concatenate([atomic_id_map, np.ones(len(neigh_cross_edges), dtype=np.uint64) * old_chunk_neighbor])

                edge_ids = np.array([], dtype=np.uint64).reshape(-1, 2)

                for pot_partner in partner_cross_edges.keys():
                    this_atomic_partner_ids = partner_cross_edges[pot_partner][:, 1]
                    this_atomic_child_ids = partner_cross_edges[pot_partner][:, 0]
                    u_atomic_child_ids = np.unique(this_atomic_child_ids)

                    leftover_mask = ~np.in1d(this_atomic_partner_ids,
                                             u_atomic_child_ids)

                    leftover_edges[pot_partner] = np.concatenate(
                        [this_atomic_child_ids[leftover_mask, None],
                         this_atomic_partner_ids[leftover_mask, None]], axis=1)

                    partners = np.unique(atomic_id_map[np.in1d(atomic_children, this_atomic_partner_ids)])
                    these_edges = np.concatenate([np.array(
                        [pot_partner] * len(partners), dtype=np.uint64)[:, None],
                                                  partners[:, None]], axis=1)

                    edge_ids = np.concatenate([edge_ids, these_edges])

                chunk_g = nx.from_edgelist(edge_ids)
                chunk_g.add_nodes_from(np.array([new_layer_parent], dtype=np.uint64))
                ccs = list(nx.connected_components(chunk_g))

                partners = []
                for cc in ccs:
                    if new_layer_parent in cc:
                        partners = cc
                        break

                if new_layer_parent in parent_cc_mapping:
                    parent_cc_id = parent_cc_mapping[new_layer_parent]
                    parent_cc_list[parent_cc_id].extend(partners)
                    parent_cc_list[parent_cc_id].append(new_layer_parent)
                else:
                    parent_cc_id = len(parent_cc_list)
                    parent_cc_list.append(list(partners))
                    parent_cc_list[parent_cc_id].append(new_layer_parent)
                    parent_cc_old_parent_list.append(old_next_layer_parent)

                for partner_id in partners:
                    parent_cc_mapping[partner_id] = parent_cc_id

            new_layer_parent_dict = {}
            for i_cc, parent_cc in enumerate(parent_cc_list):
                next_parent_id_base = None
                for parent_id in parent_cc:
                    if parent_id in parent_id_base_dict:
                        next_parent_id_base = parent_id_base_dict[parent_id]

                assert next_parent_id_base is not None

                cc_node_ids = np.array(list(parent_cc), dtype=np.uint64)
                cc_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

                for parent_id in parent_cc:
                    cc_cross_edges = np.concatenate([cc_cross_edges,
                                                     leftover_edges[parent_id]])

                new_parent_id = self.find_unique_node_id(next_parent_id_base)
                new_parent_id_b = np.array(new_parent_id).tobytes()
                new_parent_id = new_parent_id[0]

                new_layer_parent_dict[new_parent_id] = parent_cc_old_parent_list[i_cc]
                cross_edge_dict[new_parent_id] = cc_cross_edges
                old_id_dict[old_parent_id].append(new_parent_id)

                for cc_node_id in cc_node_ids:
                    val_dict = {"parents": new_parent_id_b}

                    rows.append(mutate_row(self.table,
                                           serialize_node_id(cc_node_id),
                                           self.family_id, val_dict))

                val_dict = {"children": cc_node_ids.tobytes(),
                            "atomic_cross_edges": cc_cross_edges.tobytes()}

                if i_layer == len(original_parent_ids) - 2:
                    new_roots.append(new_parent_id)
                    val_dict["former_parents"] = np.array(original_root).tobytes()

                rows.append(mutate_row(self.table,
                                       serialize_node_id(new_parent_id),
                                       self.family_id, val_dict))

            if i_layer == len(original_parent_ids) - 2:
                rows.append(mutate_row(self.table,
                                       serialize_node_id(original_root),
                                       self.family_id,
                                       {"new_parents": np.array(new_roots, dtype=np.uint64).tobytes()}))

        status = self.table.mutate_rows(rows)
        return new_roots