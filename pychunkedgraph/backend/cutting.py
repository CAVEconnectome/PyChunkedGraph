import numpy as np
import networkx as nx
import itertools
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
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

    edges, affs, mapping, remapping = merge_cross_chunk_edges(edges.copy(), affs.copy())

    sink_map = np.where(mapping[:, 0] == sink)[0]
    source_map = np.where(mapping[:, 0] == source)[0]

    print(sink, source)

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

    print(sink, source)

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = affs[i_edge]
        weighted_graph[edge[0]][edge[1]]['weight'] = affs[i_edge]

    # mst_weighted_graph = nx.minimum_spanning_tree(weighted_graph, weight="weight")

    dt = time.time() - time_start
    print("Graph creation: %.2fms" % (dt * 1000))
    time_start = time.time()

    ccs = list(nx.connected_components(weighted_graph))
    for cc in ccs:
        if not (source in cc or sink in cc):
            weighted_graph.remove_nodes_from(cc)
        else:
            if not (source in cc and sink in cc):
                raise Exception("source and sink are in different "
                                "connected components")

    # cutset = nx.minimum_edge_cut(weighted_graph, source, sink)
    cutset = nx.minimum_edge_cut(weighted_graph, source, sink,
                                 flow_func=edmonds_karp)

    dt = time.time() - time_start
    print("Mincut comp: %.2fms" % (dt * 1000))

    if cutset is None:
        return []

    # if len(cutset) != 1:
    #     raise Exception("Too many or too few cuts: %d" %
    #                     len(min_cut_set))

    time_start = time.time()

    edge_cut = list(list(cutset)[0])

    print(edge_cut)

    weighted_graph.remove_edges_from([edge_cut])
    # mst_weighted_graph.add_nodes_from(edge_cut)
    ccs = list(nx.connected_components(weighted_graph))

    for cc in ccs:
        print("CC size = %d" % len(cc))

    # if len(ccs) != 2:
    #     raise  Exception("Too many or too few connected components: %d" %
    #                      len(ccs))
    #
    # flat_edges = edges.flatten()
    #
    # cc0 = np.array(list(ccs[0]), dtype=np.uint64)
    # cc0_edge_mask = np.sum(np.in1d(flat_edges, cc0).reshape(-1, 2), axis=1) == 1
    #
    # cc1 = np.array(list(ccs[1]), dtype=np.uint64)
    # cc1_edge_mask = np.sum(np.in1d(flat_edges, cc1).reshape(-1, 2), axis=1) == 1
    #
    # cutset = edges[np.where(np.logical_and(cc0_edge_mask, cc1_edge_mask))]

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
