import fastremap
import numpy as np
from itertools import combinations, chain
from graph_tool import Graph, GraphView
from graph_tool import topology, search


def build_gt_graph(
    edges, weights=None, is_directed=True, make_directed=False, hashed=False
):
    """Builds a graph_tool graph
    :param edges: n x 2 numpy array
    :param weights: numpy array of length n
    :param is_directed: bool
    :param make_directed: bool
    :param hashed: bool
    :return: graph, capacities
    """
    edges = np.array(edges, np.uint64)
    if weights is not None:
        assert len(weights) == len(edges)
        weights = np.array(weights)

    unique_ids, edges = np.unique(edges, return_inverse=True)
    edges = edges.reshape(-1, 2)

    edges = np.array(edges)

    str1 = f"type: {type(edges)}, dtype:{edges.dtype}, shape:{edges.shape}"
    if make_directed:
        is_directed = True
        edges = np.concatenate([edges, edges[:, [1, 0]]])

        if weights is not None:
            weights = np.concatenate([weights, weights])


    str2 = f"type: {type(edges)}, dtype:{edges.dtype}, shape:{edges.shape}"

    weighted_graph = Graph(directed=is_directed)
    try:
        weighted_graph.add_edge_list(edge_list=edges, hashed=hashed)
    except:
        raise ValueError(str1+'\n'+str2)

    if weights is not None:
        cap = weighted_graph.new_edge_property("float", vals=weights)
    else:
        cap = None
    return weighted_graph, cap, edges, unique_ids


def remap_ids_from_graph(graph_ids, unique_ids):
    return unique_ids[graph_ids]


def connected_components(graph):
    """Computes connected components of graph_tool graph
    :param graph: graph_tool.Graph
    :return: np.array of len == number of nodes
    """
    assert isinstance(graph, Graph)

    cc_labels = topology.label_components(graph)[0].a

    if len(cc_labels) == 0:
        return []

    idx_sort = np.argsort(cc_labels)
    _, idx_start = np.unique(cc_labels[idx_sort], return_index=True)

    return np.split(idx_sort, idx_start[1:])


def team_paths_all_to_all(graph, capacity, team_vertex_ids):
    dprop = capacity.copy()
    # Use inverse affinity as the distance between vertices.
    dprop.a = 1 / (dprop.a + np.finfo(np.float64).eps)

    paths_v = []
    paths_e = []
    path_affinities = []
    for i1, i2 in combinations(team_vertex_ids, 2):
        v_list, e_list = topology.shortest_path(
            graph,
            source=graph.vertex(i1),
            target=graph.vertex(i2),
            weights=dprop,
        )
        paths_v.append(v_list)
        paths_e.append(e_list)
        path_affinities.append(np.sum([dprop[e] for e in e_list]))

    return paths_v, paths_e, path_affinities


def neighboring_edges(graph, vertex_id):
    """Returns vertex and edge lists of a seed vertex, in the same format as team_paths_all_to_all."""
    add_v = []
    add_e = []
    v0 = graph.vertex(vertex_id)
    neibs = v0.out_neighbors()
    for v in neibs:
        add_v.append(v)
        add_e.append(graph.edge(v, v0))
    return [add_v], [add_e], [1]


def intersect_nodes(paths_v_s, paths_v_y):
    inds_s = np.unique([int(v) for v in chain.from_iterable(paths_v_s)])
    inds_y = np.unique([int(v) for v in chain.from_iterable(paths_v_y)])
    return np.intersect1d(inds_s, inds_y)


def harmonic_mean_paths(x):
    return np.power(np.product(x), 1 / len(x))


def compute_filtered_paths(
    graph,
    capacity,
    team_vertex_ids,
    intersect_vertices,
):
    """Make a filtered GraphView that excludes intersect vertices and recompute shortest paths"""
    intersection_filter = np.full(graph.num_vertices(), True)
    intersection_filter[intersect_vertices] = False
    vfilt = graph.new_vertex_property("bool", vals=intersection_filter)
    gfilt = GraphView(graph, vfilt=vfilt)
    paths_v, paths_e, path_affinities = team_paths_all_to_all(
        gfilt, capacity, team_vertex_ids
    )

    # graph-tool will invalidate the vertex and edge properties if I don't rebase them on the main graph
    # before tearing down the GraphView
    new_paths_e = []
    for pth in paths_e:
        # An empty path means vertices are not connected, which is disallowed
        assert len(pth) > 0
        new_path = []
        for e in pth:
            new_path.append(graph.edge(int(e.source()), int(e.target())))
        new_paths_e.append(new_path)

    new_paths_v = []
    for pth in paths_v:
        new_path = []
        for v in pth:
            new_path.append(graph.vertex(int(v)))
        new_paths_v.append(new_path)
    return new_paths_v, new_paths_e, path_affinities


def remove_overlapping_edges(paths_v_s, paths_e_s, paths_v_y, paths_e_y):
    """Remove vertices that are in the paths from both teams"""
    iverts = intersect_nodes(paths_v_s, paths_v_y)
    if len(iverts) == 0:
        return paths_e_s, paths_e_y, False
    else:
        path_e_s_out = [
            [
                e
                for e in chain.from_iterable(paths_e_s)
                if not np.any(np.isin([int(e.source()), int(e.target())], iverts))
            ]
        ]
        path_e_y_out = [
            [
                e
                for e in chain.from_iterable(paths_e_y)
                if not np.any(np.isin([int(e.source()), int(e.target())], iverts))
            ]
        ]
        return path_e_s_out, path_e_y_out, True


def check_connectedness(vertices, edges, expected_number=1):
    """Returns True if the augmenting edges still form a single connected component"""
    paths_inds = np.unique([int(v) for v in chain.from_iterable(vertices)])
    edge_list_inds = np.array(
        [[int(e.source()), int(e.target())] for e in chain.from_iterable(edges)]
    )

    rmap = {v: ii for ii, v in enumerate(paths_inds)}
    edge_list_remap = fastremap.remap(edge_list_inds, rmap)

    g2 = Graph(directed=False)
    g2.add_vertex(n=len(paths_inds))
    if len(edge_list_remap) > 0:
        g2.add_edge_list(np.atleast_2d(edge_list_remap))

    _, count = topology.label_components(g2)
    return len(count) == expected_number


def reverse_edge(graph, edge):
    """Returns the complementary edge"""
    return graph.edge(edge.target(), edge.source())


def adjust_affinities(graph, capacity, paths_e, value=np.finfo(np.float32).max):
    """Set affinity of a subset of paths to a particular value (typically the largest double)."""
    capacity = capacity.copy()

    e_array = np.array(
        [(int(e.source()), int(e.target())) for e in chain.from_iterable(paths_e)]
    )
    if len(e_array) > 0:
        e_array = np.sort(e_array, axis=1)
        e_array = np.unique(e_array, axis=0)
        e_list = [graph.edge(e[0], e[1]) for e in e_array]
    else:
        e_list = []

    for edge in e_list:
        capacity[edge] = value
        # Capacity is a symmetric directed network
        capacity[reverse_edge(graph, edge)] = value
    return capacity


def flatten_edge_list(paths_e):
    return np.unique(
        [
            (int(e.source()), int(e.target()))
            for e in chain.from_iterable(x for x in paths_e)
        ]
    )
