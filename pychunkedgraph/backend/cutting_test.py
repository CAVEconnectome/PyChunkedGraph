import networkx as nx
import numpy as np
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp, \
    preflow_push

def ex_graph1():
    w_n = .8
    w_c = .5
    w_l = .2
    w_h = 1.
    inf = np.finfo(np.float32).max

    edgelist = [
        [1, 2, w_n],
        [1, 4, w_l],
        [1, 7, w_c],
        [2, 4, w_l],
        [2, 5, w_l],
        [2, 3, w_n],
        [2, 8, w_c],
        [3, 5, w_l],
        [3, 9, w_c],
        [4, 6, w_l],
        [5, 6, w_l],
        [7, 8, w_n],
        [7, 10, w_n],
        [8, 10, w_n],
        [8, 11, w_n],
        [9, 11, w_n],
        [10, 12, w_n],
        [11, 12, w_n],
        # [4, 5, inf]
    ]

    edgelist = np.array(edgelist)
    edges = edgelist[:, :2].astype(np.int)
    weights = edgelist[:, 2].astype(np.float)

    weighted_graph = nx.from_edgelist(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = weights[i_edge]
        # weighted_graph[edge[1]][edge[0]]['capacity'] = weights[i_edge]

    return weighted_graph

