import numpy as np
import time
import os
import networkx as nx

from google.cloud import bigtable

# global variables
home = os.path.expanduser("~")
N_DIGITS_UINT64 = len(str(np.iinfo(np.uint64).max))

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = home + "/.cloudvolume/secrets/google-secret.json"


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


class ChunkedGraph(object):
    def __init__(self, instance_id="pychunkedgraph",
                 project_id="neuromancer-seung-import",
                 chunk_size=(512, 512, 64), dev_mode=False):

        self._client = bigtable.Client(project=project_id, admin=True)
        self._instance = self.client.instance(instance_id)

        if dev_mode:
            self._table = self.instance.table("pychgtable_dev")
        else:
            self._table = self.instance.table("pychgtable")

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

    def add_atomic_edges_in_chunks(self, edge_ids, cross_edge_ids, edge_affs,
                                   cross_edge_affs, cg2rg_dict, rg2cg_dict,
                                   time_stamp=None):
        """ Creates atomic edges between supervoxels and first
            abstraction layer """
        if time_stamp is None:
            time_stamp = time.time()
        time_stamp = int(time_stamp)

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
        z, y, x, l = np.frombuffer(edge_ids[0, 0], dtype=np.uint8)[4:]
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
            parent_id = np.frombuffer(parent_id, dtype=np.uint64)[0]

            parent_ids = np.array([[parent_id, time_stamp]], dtype=np.uint64).tobytes()

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
                            "parents": parent_ids,
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
            time_stamp = time.time()
        time_stamp = int(time_stamp)

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
            node_id_base_next[-2] += 1

            start_key = serialize_node_id(np.frombuffer(node_id_base, dtype=np.uint64)[0])
            end_key = serialize_node_id(np.frombuffer(node_id_base_next, dtype=np.uint64)[0])

            print(start_key, end_key)

            # Set up read
            range_read = self.table.read_rows(start_key=start_key,
                                              end_key=end_key,
                                              end_inclusive=False)
            # Execute read
            range_read.consume_all()

            # Loop through nodes from this chunk
            for row_key, row_data in range_read.rows.items():
                atomic_edges = np.frombuffer(row_data.cells[self.family_id]["atomic_cross_edges".encode("utf-8")][0].value, dtype=np.uint64).reshape(-1, 2)
                atomic_partner_id_dict[int(row_key)] = atomic_edges[:, 1]
                atomic_child_id_dict[int(row_key)] = atomic_edges[:, 0]

                atomic_child_ids = np.concatenate([atomic_child_ids, atomic_edges[:, 0]])
                child_ids = np.concatenate([child_ids, np.array([row_key] * len(atomic_edges[:, 0]), dtype=np.uint64)])

        # Extract edges from remaining cross chunk edges
        # and maintain unused cross chunk edges
        edge_ids = np.array([], np.uint64).reshape(0, 2)

        u_atomic_child_ids = np.unique(atomic_child_ids)
        atomic_partner_id_dict_keys = list(atomic_partner_id_dict.keys())
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

        # 2 ----------
        # The second part finds connected components, writes the parents to
        # BigTable and updates the childs

        # Make parent id creation easier
        x, y, z = np.min(child_chunk_coords, axis=0)
        parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, layer_id], dtype=np.uint8), dtype=np.uint32)

        # Extract connected components
        chunk_g = nx.from_edgelist(edge_ids)
        chunk_g.add_nodes_from(atomic_partner_id_dict_keys)

        ccs = list(nx.connected_components(chunk_g))

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
            parent_id = np.frombuffer(parent_id, dtype=np.uint64)[0]

            parent_ids = np.array([[parent_id, time_stamp]], dtype=np.uint64).tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # Extract edges relevant to this node
                parent_cross_edges = np.concatenate([parent_cross_edges,
                                                     leftover_atomic_edges[node_id]])

                # Create node
                val_dict = {"parents": parent_ids}

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

    def get_root(self, atomic_id, time_stamp=None, is_cg_id=False):
        """ Takes an atomic id and returns the associated agglomeration ids

        :param atomic_id: int
        :param time_stamp: int
            None = time.time()
        :return: int
        """
        if time_stamp is None:
            time_stamp = time.time()

        if not is_cg_id:
            # There might be multiple chunk ids for a single rag id because
            # rag supervoxels get split at chunk boundaries. Here, only one
            # chunk id needs to be traced to the top to retrieve the
            # agglomeration id that they both belong to
            r = self.table.read_row(serialize_node_id(atomic_id))
            atomic_id = np.frombuffer(r.cells[self.family_id][serialize_key("cg_id")][0].value, dtype=np.uint64)[0]

        parent_id = atomic_id
        parent_key = serialize_key("parents")

        while True:
            row = self.table.read_row(serialize_node_id(parent_id))

            # print(parent_id)
            if parent_key in row.cells[self.family_id]:
                parent_ids = np.frombuffer(row.cells[self.family_id][parent_key][0].value, dtype=np.uint64).reshape(-1, 2)
                m_parent_ids = parent_ids[:, 1] < time_stamp
                parent_id = parent_ids[m_parent_ids, 0][np.argmax(parent_ids[m_parent_ids, 1])]
                print(parent_id, parent_ids)
            else:
                break

        return parent_id

    def read_atomic_id_with_agglomeration_id(self, agglomeration_id):
        pass

    def read_agglomeration_id_history(self, agglomeration_id, time_stamp=None):
        """ Returns all agglomeration ids agglomeration_id was part of

        :param agglomeration_id: int
        :param time_stamp: int
            restrict search to ids created after this time_stamp
            None=search whole history
        :return: array of int
        """

        return np.array([agglomeration_id])

    def read_atomic_ids_with_agglomeration_id(self, agglomeration_id):

        child_ids = []


    def add_atomic_edges(self, atomic_edge_ids):
        pass


    def remove_atomic_edges(self, atomic_edge_ids):
        pass


