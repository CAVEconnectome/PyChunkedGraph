import itertools
import numpy as np
import time

import pychunkedgraph.backend.chunkedgraph_utils
from pychunkedgraph.backend.utils import column_keys
from pychunkedgraph.backend import chunkedgraph
from multiwrapper import multiprocessing_utils as mu


def _family_consistency_test_thread(args):
    """ Helper to test family consistency """

    table_id, coord, layer_id = args

    x, y, z = coord

    cg = chunkedgraph.ChunkedGraph(table_id)

    rows = cg.range_read_chunk(layer_id, x, y, z)
    parent_column = column_keys.Hierarchy.Parent

    failed_node_ids = []

    time_start = time.time()
    for i_k, node_id in enumerate(rows.keys()):
        if i_k % 100 == 1:
            dt = time.time() - time_start
            eta = dt / i_k * len(rows) - dt
            print("%d / %d - %.3fs -> %.3fs      " % (i_k, len(rows), dt, eta),
                  end="\r")

        parent_id = rows[node_id][parent_column][0].value

        if node_id not in cg.get_children(parent_id):
            failed_node_ids.append([node_id, parent_id])

    return failed_node_ids


def family_consistency_test(table_id, n_threads=64):
    """ Runs a simple test on the WHOLE graph

    tests: id in children(parent(id))

    :param table_id: str
    :param n_threads: int
    :return: dict
        n x 2 per layer
        each failed pair: (node_id, parent_id)
    """

    cg = chunkedgraph.ChunkedGraph(table_id)

    failed_node_id_dict = {}
    for layer_id in range(1, cg.n_layers):
        print("\n\n Layer %d \n\n" % layer_id)

        step = int(cg.fan_out ** np.max([0, layer_id - 2]))
        coords = list(itertools.product(range(0, 8, step),
                                        range(0, 8, step),
                                        range(0, 4, step)))

        multi_args = []
        for coord in coords:
            multi_args.append([table_id, coord, layer_id])

        collected_failed_node_ids = mu.multisubprocess_func(
            _family_consistency_test_thread, multi_args, n_threads=n_threads)

        failed_node_ids = []
        for _failed_node_ids in collected_failed_node_ids:
            failed_node_ids.extend(_failed_node_ids)

        failed_node_id_dict[layer_id] = np.array(failed_node_ids)

        print("\n%d nodes rows failed\n" % len(failed_node_ids))

    return failed_node_id_dict


def children_test(table_id, layer, coord_list):

    cg = chunkedgraph.ChunkedGraph(table_id)
    child_column = column_keys.Hierarchy.Child

    for coords in coord_list:
        x, y, z = coords

        node_ids = cg.range_read_chunk(layer, x, y, z, columns=child_column)
        all_children = []
        children_chunks = []
        for children in node_ids.values():
            children = children[0].value
            for child in children:
                all_children.append(child)
                children_chunks.append(cg.get_chunk_id(child))

        u_children_chunks, c_children_chunks = np.unique(children_chunks,
                                                         return_counts=True)
        u_chunk_coords = [cg.get_chunk_coordinates(c) for c in u_children_chunks]

        print("\n--- Layer %d ---- [%d, %d, %d] ---" % (layer, x, y, z))
        print("N(all children): %d" % len(all_children))
        print("N(unique children): %d" % len(np.unique(all_children)))
        print("N(unique children chunks): %d" % len(u_children_chunks))
        print("Unique children chunk coords", u_chunk_coords)
        print("N(ids per unique children chunk):", c_children_chunks)


def root_cross_edge_test(node_id, table_id=None, cg=None):
    if cg is None:
        assert isinstance(table_id, str)
        cg = chunkedgraph.ChunkedGraph(table_id)

    cross_edge_dict_layers = {}
    cross_edge_dict_children = {}
    for layer in range(2, cg.n_layers):
        child_ids = cg.get_subgraph_nodes(node_id, return_layers=[layer])

        cross_edge_dict = {}
        child_reference_ids = []
        for child_id in child_ids:
            cross_edge_dict = pychunkedgraph.backend.chunkedgraph_utils.combine_cross_chunk_edge_dicts(cross_edge_dict, cg.read_cross_chunk_edges(child_id))

        cross_edge_dict_layers[layer] = cross_edge_dict

    for layer in cross_edge_dict_layers.keys():
        print("\n--------\n")
        for i_layer in cross_edge_dict_layers[layer].keys():
            print(layer, i_layer, len(cross_edge_dict_layers[layer][i_layer]))

    return cross_edge_dict_layers
