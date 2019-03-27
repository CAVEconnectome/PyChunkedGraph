import networkx as nx
import numpy as np
import graph_tool.all, graph_tool.flow, graph_tool.topology
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from networkx.algorithms.connectivity import minimum_st_edge_cut


def ex_graph():
    w_n = .8
    w_c = .5
    w_l = .2
    w_h = 1.
    inf = np.finfo(np.float32).max

    edgelist = [
        [1, 2, w_n],
        [1, 3, w_l],
        [4, 7, w_l],
        [6, 9, w_l],
        [2, 4, w_l],
        [2, 5, w_l],
        [2, 3, w_n],
        [8, 9, w_c],
        [3, 5, w_l],
        [3, 6, w_c],
        [4, 5, w_l],
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

    r_flow = edmonds_karp(weighted_graph, 1 , 12)
    cutset = minimum_st_edge_cut(weighted_graph, 1, 12, residual=r_flow)

    print("NETWORKX:", cutset)

    g = graph_tool.all.Graph(directed=True)
    g.add_vertex(100)
    g.add_edge_list(edge_list=np.concatenate([edgelist[:, [0, 1]], edgelist[:, [1, 0]]]), hashed=False)
    cap = g.new_edge_property("float", vals=np.concatenate([edgelist[:, 2], edgelist[:, 2]]))
    src, tgt = g.vertex(1), g.vertex(12)

    res = graph_tool.flow.boykov_kolmogorov_max_flow(g, src, tgt, cap)

    part = graph_tool.all.min_st_cut(g, src, cap, res)
    rm_edges = [e for e in g.edges() if part[e.source()] != part[e.target()]]

    s = rm_edges[0].source()
    print(s)
    print(rm_edges)

    raise()


    return edgelist