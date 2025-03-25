import networkx as nx
import numpy as np
import graph_tool.all, graph_tool.flow, graph_tool.topology
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from networkx.algorithms.connectivity import minimum_st_edge_cut
import logging
import sys
import time

from pychunkedgraph.backend import cutting


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
        # [4, 5, w_l],
        [5, 6, w_l],
        [7, 8, w_n],
        [7, 10, w_n],
        [8, 10, w_n],
        [8, 11, w_n],
        [9, 11, w_n],
        [10, 12, w_n],
        [11, 12, w_n],
        [4, 5, inf]
    ]

    edgelist = np.array(edgelist)

    edges = edgelist[:, :2].astype(int) - 1
    weights = edgelist[:, 2].astype(np.float)

    n_nodes = 100000
    edges = np.unique(np.sort(np.random.randint(0, n_nodes, n_nodes*5).reshape(-1, 2), axis=1), axis=0)
    weights = np.random.rand(len(edges))

    if not len(np.unique(edges) == 12):
        edges, weights = ex_graph()

    edges += 100

    return edges.astype(np.uint64), weights


def test_raw():
    edges, weights = ex_graph()

    weighted_graph = nx.from_edgelist(edges)
    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['capacity'] = weights[i_edge]

    r_flow = edmonds_karp(weighted_graph, 0 , 11)
    cutset = minimum_st_edge_cut(weighted_graph, 0, 11, residual=r_flow)

    weighted_graph.remove_edges_from(cutset)
    ccs = list(nx.connected_components(weighted_graph))

    print("NETWORKX:", cutset)
    print("NETWORKX:", ccs)

    g = graph_tool.all.Graph(directed=True)
    g.add_edge_list(edge_list=np.concatenate([edges, edges[:, [1, 0]]]), hashed=False)
    cap = g.new_edge_property("float", vals=np.concatenate([weights, weights]))
    src, tgt = g.vertex(0), g.vertex(11)

    res = graph_tool.flow.boykov_kolmogorov_max_flow(g, src, tgt, cap)

    part = graph_tool.all.min_st_cut(g, src, cap, res)
    rm_edges = [e for e in g.edges() if part[e.source()] != part[e.target()]]

    print("GRAPHTOOL:", [(rm_edge.source().__str__(), rm_edge.target().__str__()) for rm_edge in rm_edges])

    ccs = []
    for i_cc in np.unique(part.a):
        ccs.append(np.where(part.a == i_cc)[0])

    print("GRAPHTOOL:", ccs)

    return edges, weights


def test_imp():
    edges, weights = ex_graph()

    logger = logging.getLogger("%d" % np.random.randint(0, 100000000))
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    # print(edges)
    # print(weights)

    sources = np.unique(edges)[:10]
    sinks = np.unique(edges)[-10:]

    time_start = time.time()
    out_gt = cutting.mincut_graph_tool(edges, weights, sources, sinks, logger=logger)
    time_gt = time.time() - time_start

    # print(out_gt)

    print("----------------")

    time_start = time.time()
    out_nx = cutting.mincut_nx(edges, weights, sources, sinks, logger=logger)
    time_nx = time.time() - time_start

    # print(out_nx)

    print("Time networkx: %.3fs" % (time_nx))
    print("Time graph_tool: %.3fs" % (time_gt))

    return np.array_equal(np.unique(out_nx, axis=0), np.unique(out_gt, axis=0))

