import numpy as np
import networkx as nx
import itertools
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp, preflow_push
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
    mapping = np.array([], dtype=np.uint64).reshape(-1, 2)

    for cc in ccs:
        nodes = np.array(list(cc))
        rep_node = np.min(nodes)

        remapping[rep_node] = nodes

        rep_nodes = np.ones(len(nodes), dtype=np.uint64).reshape(-1, 1) * rep_node
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

    return remapped_edges, remapped_affs, mapping, remapping


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

    original_edges = edges.copy()
    original_affs = affs.copy()

    edges, affs, mapping, remapping = merge_cross_chunk_edges(edges.copy(),
                                                              affs.copy())

    if len(edges) == 0:
        return []

    sink_map = np.where(mapping[:, 0] == sink)[0]
    source_map = np.where(mapping[:, 0] == source)[0]

    if len(sink_map) == 0:
        pass
    elif len(sink_map) == 1:
        sink = mapping[sink_map[0]][1]
    else:
        raise Exception("Sink appears to be overmerged")

    if len(source_map) == 0:
        pass
    elif len(source_map) == 1:
        source = mapping[source_map[0]][1]
    else:
        raise Exception("Source appears to be overmerged")

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = affs[i_edge]
        weighted_graph[edge[0]][edge[1]]['weight'] = affs[i_edge]
        weighted_graph[edge[1]][edge[0]]['capacity'] = affs[i_edge]
        weighted_graph[edge[1]][edge[0]]['weight'] = affs[i_edge]

    # sink_neighbors = weighted_graph.neighbors(sink)
    # source_neighbors = weighted_graph.neighbors(source)
    #
    # if np.all(~np.in1d(sink_neighbors, source_neighbors)):
    #     print(sink_neighbors)
    #     print(source_neighbors)
    #     for sink_neighbor in sink_neighbors:
    #         weighted_graph[sink_neighbor][sink]['capacity'] = 1e9
    #         weighted_graph[sink][sink_neighbor]['capacity'] = 1e9
    #         weighted_graph[sink_neighbor][sink]['weight'] = 1e9
    #         weighted_graph[sink][sink_neighbor]['weight'] = 1e9
    #     for source_neighbor in source_neighbors:
    #         weighted_graph[source_neighbor][source]['capacity'] = 1e9
    #         weighted_graph[source][source_neighbor]['capacity'] = 1e9
    #         weighted_graph[source_neighbor][source]['weight'] = 1e9
    #         weighted_graph[source][source_neighbor]['weight'] = 1e9

    dt = time.time() - time_start
    print("Graph creation: %.2fms" % (dt * 1000))
    time_start = time.time()

    ccs = list(nx.connected_components(weighted_graph))
    for cc in ccs:
        if not (source in cc or sink in cc):
            weighted_graph.remove_nodes_from(cc)
        else:
            if not (source in cc and sink in cc):
                print("source and sink are in different "
                      "connected components")
                return []

    cutset = nx.minimum_edge_cut(weighted_graph, source, sink,
                                 flow_func=edmonds_karp)

    dt = time.time() - time_start
    print("Mincut comp: %.2fms" % (dt * 1000))

    if cutset is None:
        return []

    time_start = time.time()

    edge_cut = list(list(cutset)[0])

    weighted_graph.remove_edges_from([edge_cut])
    ccs = list(nx.connected_components(weighted_graph))

    for cc in ccs:
        print("CC size = %d" % len(cc))

    dt = time.time() - time_start
    print("Splitting: %.2fms" % (dt * 1000))

    remapped_cutset = []
    for cut in cutset:
        if cut[0] in remapping:
            pre_cut = remapping[cut[0]]
        else:
            pre_cut = [cut[0]]

        if cut[1] in remapping:
            post_cut = remapping[cut[1]]
        else:
            post_cut = [cut[1]]

        remapped_cutset.extend(list(itertools.product(pre_cut, post_cut)))
        remapped_cutset.extend(list(itertools.product(post_cut, pre_cut)))

    remapped_cutset = np.array(remapped_cutset, dtype=np.uint64)

    remapped_cutset_flattened_view = remapped_cutset.view(dtype='u8,u8')
    edges_flattened_view = original_edges.view(dtype='u8,u8')

    cutset_mask = np.in1d(remapped_cutset_flattened_view, edges_flattened_view)

    return remapped_cutset[cutset_mask]
