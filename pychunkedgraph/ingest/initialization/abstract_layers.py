"""
Functions for creating parents in level 3 and above
"""

import collections
import datetime
from typing import Optional, Sequence

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.backend import flatgraph_utils
from pychunkedgraph.backend.chunkedgraph_utils import get_valid_timestamp
from pychunkedgraph.backend.utils import serializers, column_keys


def add_layer(
    self,
    layer_id: int,
    child_chunk_coords: Sequence[Sequence[int]],
    time_stamp: Optional[datetime.datetime] = None,
    n_threads: int = 20,
) -> None:
    def _read_subchunks_thread(chunk_coord):
        # Get start and end key
        x, y, z = chunk_coord

        columns = [column_keys.Hierarchy.Child] + [
            column_keys.Connectivity.CrossChunkEdge[l]
            for l in range(layer_id - 1, self.n_layers)
        ]
        range_read = self.range_read_chunk(layer_id - 1, x, y, z, columns=columns)

        # Due to restarted jobs some nodes in the layer below might be
        # duplicated. We want to ignore the earlier created node(s) because
        # they belong to the failed job. We can find these duplicates only
        # by comparing their children because each node has a unique id.
        # However, we can use that more recently created nodes have higher
        # segment ids (not true on root layer but we do not have that here.
        # We are only interested in the latest version of any duplicated
        # parents.

        # Deserialize row keys and store child with highest id for
        # comparison
        row_cell_dict = {}
        segment_ids = []
        row_ids = []
        max_child_ids = []
        for row_id, row_data in range_read.items():
            segment_id = self.get_segment_id(row_id)

            cross_edge_columns = {
                k: v
                for (k, v) in row_data.items()
                if k.family_id == self.cross_edge_family_id
            }
            if cross_edge_columns:
                row_cell_dict[row_id] = cross_edge_columns

            node_child_ids = row_data[column_keys.Hierarchy.Child][0].value

            max_child_ids.append(np.max(node_child_ids))
            segment_ids.append(segment_id)
            row_ids.append(row_id)

        segment_ids = np.array(segment_ids, dtype=np.uint64)
        row_ids = np.array(row_ids)
        max_child_ids = np.array(max_child_ids, dtype=np.uint64)

        sorting = np.argsort(segment_ids)[::-1]
        row_ids = row_ids[sorting]
        max_child_ids = max_child_ids[sorting]

        counter = collections.defaultdict(int)
        max_child_ids_occ_so_far = np.zeros(len(max_child_ids), dtype=np.int)
        for i_row in range(len(max_child_ids)):
            max_child_ids_occ_so_far[i_row] = counter[max_child_ids[i_row]]
            counter[max_child_ids[i_row]] += 1

        # Filter last occurences (we inverted the list) of each node
        m = max_child_ids_occ_so_far == 0
        row_ids = row_ids[m]
        ll_node_ids.extend(row_ids)

        # Loop through nodes from this chunk
        for row_id in row_ids:
            if row_id in row_cell_dict:
                cross_edge_dict[row_id] = {}

                cell_family = row_cell_dict[row_id]

                for l in range(layer_id - 1, self.n_layers):
                    row_key = column_keys.Connectivity.CrossChunkEdge[l]
                    if row_key in cell_family:
                        cross_edge_dict[row_id][l] = cell_family[row_key][0].value

                if int(layer_id - 1) in cross_edge_dict[row_id]:
                    atomic_cross_edges = cross_edge_dict[row_id][layer_id - 1]

                    if len(atomic_cross_edges) > 0:
                        atomic_partner_id_dict[row_id] = atomic_cross_edges[:, 1]

                        new_pairs = zip(
                            atomic_cross_edges[:, 0], [row_id] * len(atomic_cross_edges)
                        )
                        atomic_child_id_dict_pairs.extend(new_pairs)

    def _resolve_cross_chunk_edges_thread(args) -> None:
        start, end = args

        for i_child_key, child_key in enumerate(atomic_partner_id_dict_keys[start:end]):
            this_atomic_partner_ids = atomic_partner_id_dict[child_key]

            partners = {
                atomic_child_id_dict[atomic_cross_id]
                for atomic_cross_id in this_atomic_partner_ids
                if atomic_child_id_dict[atomic_cross_id] != 0
            }

            if len(partners) > 0:
                partners = np.array(list(partners), dtype=np.uint64)[:, None]

                this_ids = np.array([child_key] * len(partners), dtype=np.uint64)[
                    :, None
                ]
                these_edges = np.concatenate([this_ids, partners], axis=1)

                edge_ids.extend(these_edges)

    def _write_out_connected_components(args) -> None:
        start, end = args

        # Collect cc info
        parent_layer_ids = range(layer_id, self.n_layers + 1)
        cc_connections = {l: [] for l in parent_layer_ids}
        for i_cc, cc in enumerate(ccs[start:end]):
            node_ids = unique_graph_ids[cc]

            parent_cross_edges = collections.defaultdict(list)

            # Collect row info for nodes that are in this chunk
            for node_id in node_ids:
                if node_id in cross_edge_dict:
                    # Extract edges relevant to this node
                    for l in range(layer_id, self.n_layers):
                        if (
                            l in cross_edge_dict[node_id]
                            and len(cross_edge_dict[node_id][l]) > 0
                        ):
                            parent_cross_edges[l].append(cross_edge_dict[node_id][l])

            if self.use_skip_connections and len(node_ids) == 1:
                for l in parent_layer_ids:
                    if l == self.n_layers or len(parent_cross_edges[l]) > 0:
                        cc_connections[l].append([node_ids, parent_cross_edges])
                        break
            else:
                cc_connections[layer_id].append([node_ids, parent_cross_edges])

        # Write out cc info
        rows = []

        # Iterate through layers
        for parent_layer_id in parent_layer_ids:
            if len(cc_connections[parent_layer_id]) == 0:
                continue

            parent_chunk_id = parent_chunk_id_dict[parent_layer_id]
            reserved_parent_ids = self.get_unique_node_id_range(
                parent_chunk_id, step=len(cc_connections[parent_layer_id])
            )

            for i_cc, cc_info in enumerate(cc_connections[parent_layer_id]):
                node_ids, parent_cross_edges = cc_info

                parent_id = reserved_parent_ids[i_cc]
                val_dict = {column_keys.Hierarchy.Parent: parent_id}

                for node_id in node_ids:
                    rows.append(
                        self.mutate_row(
                            serializers.serialize_uint64(node_id),
                            val_dict,
                            time_stamp=time_stamp,
                        )
                    )

                val_dict = {column_keys.Hierarchy.Child: node_ids}
                for l in range(parent_layer_id, self.n_layers):
                    if l in parent_cross_edges and len(parent_cross_edges[l]) > 0:
                        val_dict[
                            column_keys.Connectivity.CrossChunkEdge[l]
                        ] = np.concatenate(parent_cross_edges[l])

                rows.append(
                    self.mutate_row(
                        serializers.serialize_uint64(parent_id),
                        val_dict,
                        time_stamp=time_stamp,
                    )
                )

                if len(rows) > 100000:
                    self.bulk_write(rows)
                    rows = []

        if len(rows) > 0:
            self.bulk_write(rows)

    time_stamp = get_valid_timestamp(time_stamp)

    # 1 --------------------------------------------------------------------
    # The first part is concerned with reading data from the child nodes
    # of this layer and pre-processing it for the second part
    atomic_partner_id_dict = {}
    cross_edge_dict = {}
    atomic_child_id_dict_pairs = []
    ll_node_ids = []

    multi_args = child_chunk_coords
    n_jobs = np.min([n_threads, len(multi_args)])

    if n_jobs > 0:
        mu.multithread_func(_read_subchunks_thread, multi_args, n_threads=n_jobs)

    d = dict(atomic_child_id_dict_pairs)
    atomic_child_id_dict = collections.defaultdict(np.uint64, d)
    ll_node_ids = np.array(ll_node_ids, dtype=np.uint64)

    # Extract edges from remaining cross chunk edges
    # and maintain unused cross chunk edges
    edge_ids = []
    # u_atomic_child_ids = np.unique(atomic_child_ids)
    atomic_partner_id_dict_keys = np.array(
        list(atomic_partner_id_dict.keys()), dtype=np.uint64
    )

    n_jobs = np.min([n_threads * 3 if n_threads > 1 else 1, len(atomic_partner_id_dict_keys)])
    if n_jobs > 0:
        spacing = np.linspace(0, len(atomic_partner_id_dict_keys), n_jobs + 1).astype(
            np.int
        )
        starts = spacing[:-1]
        ends = spacing[1:]
        multi_args = list(zip(starts, ends))
        mu.multithread_func(
            _resolve_cross_chunk_edges_thread, multi_args, n_threads=n_threads
        )

    # 2 --------------------------------------------------------------------
    # The second part finds connected components, writes the parents to
    # BigTable and updates the childs

    # Make parent id creation easier
    x, y, z = np.min(child_chunk_coords, axis=0) // self.fan_out
    chunk_id = self.get_chunk_id(layer=layer_id, x=x, y=y, z=z)

    parent_chunk_id_dict = self.get_parent_chunk_id_dict(chunk_id)

    # Extract connected components
    isolated_node_mask = ~np.in1d(ll_node_ids, np.unique(edge_ids))
    add_node_ids = ll_node_ids[isolated_node_mask].squeeze()
    add_edge_ids = np.vstack([add_node_ids, add_node_ids]).T
    edge_ids.extend(add_edge_ids)

    graph, _, _, unique_graph_ids = flatgraph_utils.build_gt_graph(
        edge_ids, make_directed=True
    )

    ccs = flatgraph_utils.connected_components(graph)

    # Add rows for nodes that are in this chunk
    # a connected component at a time
    n_jobs = np.min([n_threads * 3 if n_threads > 1 else 1, len(ccs)])

    spacing = np.linspace(0, len(ccs), n_jobs + 1).astype(np.int)
    starts = spacing[:-1]
    ends = spacing[1:]
    multi_args = list(zip(starts, ends))
    mu.multithread_func(
        _write_out_connected_components, multi_args, n_threads=n_threads
    )
    # to track worker completion
    return str(layer_id)