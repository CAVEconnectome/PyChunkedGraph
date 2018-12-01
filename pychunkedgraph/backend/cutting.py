import numpy as np
import networkx as nx
import itertools
import logging
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp, preflow_push
from networkx.algorithms.connectivity import minimum_st_edge_cut
import time


from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

float_max = np.finfo(np.float32).max


def merge_cross_chunk_edges(edges: Iterable[Sequence[np.uint64]],
                            affs: Sequence[np.uint64],
                            logger: Optional[logging.Logger] = None):
    """ Merges cross chunk edges
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :return:
    """

    # mask for edges that have to be merged
    cross_chunk_edge_mask = np.isinf(affs)

    # graph with edges that have to be merged
    cross_chunk_graph = nx.Graph()
    cross_chunk_graph.add_edges_from(edges[cross_chunk_edge_mask])

    # connected components in this graph will be combined in one component
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
           sources: Sequence[np.uint64], sinks: Sequence[np.uint64],
           logger: Optional[logging.Logger] = None) -> np.ndarray:
    """ Computes the min cut on a local graph
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :param sources: uint64
    :param sinks: uint64
    :return: m x 2 array of uint64s
        edges that should be removed
    """

    time_start = time.time()

    original_edges = edges.copy()

    edges, affs, mapping, remapping = merge_cross_chunk_edges(edges.copy(),
                                                              affs.copy())

    if len(edges) == 0:
        return []

    assert np.unique(mapping[:, 0], return_counts=True)[1].max() == 1

    mapping_dict = dict(mapping)

    remapped_sinks = []
    remapped_sources = []

    for sink in sinks:
        remapped_sinks.append(mapping_dict[sink])

    for source in sources:
        remapped_sources.append(mapping_dict[source])

    sinks = remapped_sinks
    sources = remapped_sources

    sink_connections = np.array(list(itertools.product(sinks, sinks)))
    source_connections = np.array(list(itertools.product(sources, sources)))

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)
    weighted_graph.add_edges_from(sink_connections)
    weighted_graph.add_edges_from(source_connections)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = affs[i_edge]
        weighted_graph[edge[1]][edge[0]]['capacity'] = affs[i_edge]

    # Add infinity edges for multicut
    for sink_i in sinks:
        for sink_j in sinks:
            weighted_graph[sink_i][sink_j]['capacity'] = float_max

    for source_i in sources:
        for source_j in sources:
            weighted_graph[source_i][source_j]['capacity'] = float_max


    dt = time.time() - time_start
    if logger is not None:
        logger.debug("Graph creation: %.2fms" % (dt * 1000))
    time_start = time.time()

    ccs = list(nx.connected_components(weighted_graph))
    for cc in ccs:
        cc_list = list(cc)

        # If connected component contains no sources and/or no sinks,
        # remove its nodes from the mincut computation
        if not np.any(np.in1d(sources, cc_list)) or \
                not np.any(np.in1d(sinks, cc_list)):
            weighted_graph.remove_nodes_from(cc)

    r_flow = edmonds_karp(weighted_graph, sinks[0], sources[0])
    cutset = minimum_st_edge_cut(weighted_graph, sources[0], sinks[0],
                                 residual=r_flow)

    # cutset = nx.minimum_edge_cut(weighted_graph, sources[0], sinks[0], flow_func=edmonds_karp)

    dt = time.time() - time_start
    if logger is not None:
        logger.debug("Mincut comp: %.2fms" % (dt * 1000))

    if cutset is None:
        return []

    time_start = time.time()

    edge_cut = list(list(cutset))

    weighted_graph.remove_edges_from(edge_cut)
    ccs = list(nx.connected_components(weighted_graph))

    # assert len(ccs) == 2

    for cc in ccs:
        cc_list = list(cc)
        if logger is not None:
            logger.debug("CC size = %d" % len(cc_list))

        if np.any(np.in1d(sources, cc_list)):
            assert np.all(np.in1d(sources, cc_list))
            assert ~np.any(np.in1d(sinks, cc_list))

        if np.any(np.in1d(sinks, cc_list)):
            assert np.all(np.in1d(sinks, cc_list))
            assert ~np.any(np.in1d(sources, cc_list))

    dt = time.time() - time_start
    if logger is not None:
        logger.debug("Splitting local graph: %.2fms" % (dt * 1000))

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
