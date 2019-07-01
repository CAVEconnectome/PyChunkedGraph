import numpy as np
import graph_tool
from graph_tool import topology


def build_gt_graph(edges, weights=None, is_directed=True, make_directed=False,
                   hashed=False):
    """ Builds a graph_tool graph

    :param edges: n x 2 numpy array
    :param weights: numpy array of length n
    :param is_directed: bool
    :param make_directed: bool
    :param hashed: bool
    :return: graph, capacities
    """
    if weights is not None:
        assert len(weights) == len(edges)
        weights = np.array(weights)

    unique_ids, edges = np.unique(edges, return_inverse=True)
    edges = edges.reshape(-1, 2)

    edges = np.array(edges)

    if make_directed:
        is_directed = True
        edges = np.concatenate([edges, edges[:, [1, 0]]])

        if weights is not None:
            weights = np.concatenate([weights, weights])

    weighted_graph = graph_tool.Graph(directed=is_directed)
    weighted_graph.add_edge_list(edge_list=edges, hashed=hashed)

    if weights is not None:
        cap = weighted_graph.new_edge_property("float", vals=weights)
    else:
        cap = None

    return weighted_graph, cap, edges, unique_ids


def remap_ids_from_graph(graph_ids, unique_ids):
    return unique_ids[graph_ids]


def connected_components(graph):
    """ Computes connected components of graph_tool graph

    :param graph: graph_tool.Graph
    :return: np.array of len == number of nodes
    """
    assert isinstance(graph, graph_tool.Graph)

    cc_labels = topology.label_components(graph)[0].a

    if len(cc_labels) == 0:
        return []

    idx_sort = np.argsort(cc_labels)
    vals, idx_start, count = np.unique(cc_labels[idx_sort], return_counts=True,
                                       return_index=True)

    res = np.split(idx_sort, idx_start[1:])

    return res