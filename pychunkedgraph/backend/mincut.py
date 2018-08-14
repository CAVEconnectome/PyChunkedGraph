import numpy as np
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import time


from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union



def merge_cross_chunk_edges(edges: Iterable[Sequence[np.uint64]],
                            affs: Sequence[np.uint64]):
    """ Merges cross chunk edges

    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :return:
    """

    cross_chunk_edge_mask = np.isinf(affs)

    cross_chunk_graph = nx.Graph()
    cross_chunk_graph.add_edges_from(edges[cross_chunk_edge_mask])

    ccs = nx.connected_components(cross_chunk_graph)

    remapping = {}
    mapping = np.array([], dtype=np.int).reshape(-1, 2)

    for cc in ccs:
        nodes = np.array(list(cc))
        rep_node = np.min(nodes)

        remapping[rep_node] = nodes

        rep_nodes = np.ones(len(nodes), dtype=np.int).reshape(-1, 1) * rep_node
        m = np.concatenate([nodes.reshape(-1, 1), rep_nodes], axis=1)

        mapping = np.concatenate([mapping, m], axis=0)

    u_nodes = np.unique(edges)
    u_unmapped_nodes = u_nodes[~np.in1d(u_nodes, mapping)]

    unmapped_mapping = np.concatenate([u_unmapped_nodes.reshape(-1, 1),
                                       u_unmapped_nodes.reshape(-1, 1)], axis=1)
    mapping = np.concatenate([mapping, unmapped_mapping], axis=0)

    sort_idx = np.argsort(mapping[:, 0])
    idx = np.searchsorted(mapping[:, 0], edges, sorter=sort_idx)
    remapped_edges = np.asarray(mapping[:, 1])[sort_idx][idx]

    remapped_edges = remapped_edges[~cross_chunk_edge_mask]
    remapped_affs = affs[~cross_chunk_edge_mask]

    return remapped_edges, remapped_affs, remapping


def mincut(edges: Iterable[Sequence[np.uint64]], affs: Sequence[np.uint64],
           source: np.uint64, sink: np.uint64) -> np.ndarray:
    """ Computes the min cut on a local graph

    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :param source: uint64
    :param sink: uint64
    :return: m x 2 array of uint64s
        edges that should be removed
    """

    time_start = time.time()

    # edges, affs, remapping = merge_cross_chunk_edges(edges, affs)

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = affs[i_edge]

    dt = time.time() - time_start
    print("Graph creation: %.2fms" % (dt * 1000))
    time_start = time.time()

    ccs = list(nx.connected_components(weighted_graph))
    for cc in ccs:
        if not (source in cc and sink in cc):
            weighted_graph.remove_nodes_from(cc)

    # cutset = nx.minimum_edge_cut(weighted_graph, source, sink)
    cutset = nx.minimum_edge_cut(weighted_graph, source, sink,
                                 flow_func=shortest_augmenting_path)

    dt = time.time() - time_start
    print("Mincut: %.2fms" % (dt * 1000))

    if cutset is None:
        return np.array([], dtype=np.uint64)

    time_start = time.time()

    weighted_graph.remove_edges_from(cutset)
    ccs = list(nx.connected_components(weighted_graph))
    print("Graph split up in %d parts" % (len(ccs)))

    for cc in ccs:
        print("CC size = %d" % len(cc))

    dt = time.time() - time_start
    print("Test: %.2fms" % (dt * 1000))

    return np.array(list(cutset), dtype=np.uint64)