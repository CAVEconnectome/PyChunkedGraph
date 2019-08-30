import collections
import numpy as np
import networkx as nx
import itertools
import logging
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp, preflow_push
from networkx.algorithms.connectivity import minimum_st_edge_cut
import time
import graph_tool
import graph_tool.flow

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pychunkedgraph.backend import flatgraph_utils
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

float_max = np.finfo(np.float32).max
DEBUG_MODE = False

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

    # Build mapping
    # For each connected component the smallest node id is chosen to be the
    # representative.
    remapping = {}
    mapping_ks = []
    mapping_vs = []

    for cc in ccs:
        nodes = np.array(list(cc))
        rep_node = np.min(nodes)

        remapping[rep_node] = nodes
        mapping_ks.extend(nodes)
        mapping_vs.extend([rep_node] * len(nodes))

    # Initialize mapping with a each node mapping to itself, then update
    # those edges merged to one across chunk boundaries.
    # u_nodes = np.unique(edges)
    # mapping = dict(zip(u_nodes, u_nodes))
    # mapping.update(dict(zip(mapping_ks, mapping_vs)))
    mapping = dict(zip(mapping_ks, mapping_vs))

    # Vectorize remapping
    mapping_vec = np.vectorize(lambda a : mapping[a] if a in mapping else a)
    remapped_edges = mapping_vec(edges)

    # Remove cross chunk edges
    remapped_edges = remapped_edges[~cross_chunk_edge_mask]
    remapped_affs = affs[~cross_chunk_edge_mask]

    return remapped_edges, remapped_affs, mapping, remapping


def merge_cross_chunk_edges_graph_tool(edges: Iterable[Sequence[np.uint64]],
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
    graph, _, _, unique_ids = flatgraph_utils.build_gt_graph(
        edges[cross_chunk_edge_mask], make_directed=True)

    # connected components in this graph will be combined in one component
    ccs = flatgraph_utils.connected_components(graph)

    remapping = {}
    mapping = np.array([], dtype=np.uint64).reshape(-1, 2)

    for cc in ccs:
        nodes = unique_ids[cc]
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


def mincut_nx(edges: Iterable[Sequence[np.uint64]], affs: Sequence[np.uint64],
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
    mapping_vec = np.vectorize(lambda a: mapping[a] if a in mapping else a)

    if len(edges) == 0:
        return []

    if len(mapping) > 0:
        assert np.unique(list(mapping.keys()), return_counts=True)[1].max() == 1

    remapped_sinks = mapping_vec(sinks)
    remapped_sources = mapping_vec(sources)

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


class LocalMincutGraph:
    """
    Helper class for mincut computation. Used by the mincut_graph_tool function to: 
    (1) set up a local graph-tool graph, (2) compute a mincut, (3) ensure required conditions hold, 
    and (4) return the ChunkedGraph edges to be removed. 
    """

    def __init__(
        self, edges, affs, sources, sinks, gt_to_sv_mapping, split_preview=False, logger=None
    ):
        self.sources = sources
        self.sinks = sinks
        self.split_preview = split_preview
        self.gt_to_sv_mapping = gt_to_sv_mapping

        self.source_edges = list(itertools.product(sources, sources))
        self.sink_edges = list(itertools.product(sinks, sinks))

        # Assemble edges: Edges after remapping combined with fake infinite affinity
        # edges between sinks and sources
        comb_edges = np.concatenate([edges, self.source_edges, self.sink_edges])
        comb_affs = np.concatenate(
            [affs, [float_max] * (len(self.source_edges) + len(self.sink_edges))]
        )

        # To make things easier for everyone involved, we map the ids to
        # [0, ..., len(unique_ids) - 1]
        # Generate weighted graph with graph_tool
        self.weighted_graph, self.capacities, self.gt_edges, self.unique_ids = flatgraph_utils.build_gt_graph(
            comb_edges, comb_affs, make_directed=True
        )

        self.source_graph_ids = np.where(np.in1d(self.unique_ids, sources))[0]
        self.sink_graph_ids = np.where(np.in1d(self.unique_ids, sinks))[0]

        self.logger = logger

        if logger is not None:
            logger.debug(f"{sinks}, {self.sink_graph_ids}")
            logger.debug(f"{sources}, {self.source_graph_ids}")

        self._create_fake_edge_property(affs)


    def compute_mincut(self, cg_edges):
        """
        Compute mincut and return the supervoxel cut edge set
        """
        self._filter_graph_connected_components()
        time_start = time.time()
        src, tgt = (
            self.weighted_graph.vertex(self.source_graph_ids[0]),
            self.weighted_graph.vertex(self.sink_graph_ids[0]),
        )

        residuals = graph_tool.flow.push_relabel_max_flow(
            self.weighted_graph, src, tgt, self.capacities
        )
        partition = graph_tool.flow.min_st_cut(
            self.weighted_graph, src, self.capacities, residuals
        )

        dt = time.time() - time_start
        if self.logger is not None:
            self.logger.debug("Mincut comp: %.2fms" % (dt * 1000))

        if DEBUG_MODE:
            self._gt_mincut_sanity_check(partition)

        labeled_edges = partition.a[self.gt_edges]
        cut_edge_set = self.gt_edges[labeled_edges[:, 0] != labeled_edges[:, 1]]

        if self.split_preview:
            return self._get_split_preview_connected_components(cut_edge_set)

        self._sink_and_source_connectivity_sanity_check(cut_edge_set)
        return self._remap_cut_edge_set(cut_edge_set, cg_edges)


    def _remap_cut_edge_set(self, cut_edge_set, cg_edges):
        """
        Remap the cut edge set from graph ids to supervoxel ids and return it
        """
        remapped_cutset = []
        for s, t in flatgraph_utils.remap_ids_from_graph(cut_edge_set, self.unique_ids):

            if s in self.gt_to_sv_mapping:
                s = self.gt_to_sv_mapping[s]
            else:
                s = [s]

            if t in self.gt_to_sv_mapping:
                t = self.gt_to_sv_mapping[t]
            else:
                t = [t]

            remapped_cutset.extend(list(itertools.product(s, t)))
            remapped_cutset.extend(list(itertools.product(t, s)))

        remapped_cutset = np.array(remapped_cutset, dtype=np.uint64)

        remapped_cutset_flattened_view = remapped_cutset.view(dtype="u8,u8")
        edges_flattened_view = cg_edges.view(dtype="u8,u8")

        cutset_mask = np.in1d(remapped_cutset_flattened_view, edges_flattened_view)

        return remapped_cutset[cutset_mask]


    def _get_split_preview_connected_components(self, cut_edge_set):
        """
        Return the connected components of the local graph (in terms of supervoxels)
        when doing a split preview
        """
        ccs_test_post_cut, illegal_split = self._sink_and_source_connectivity_sanity_check(
            cut_edge_set
        )
        supervoxel_ccs = [None] * len(ccs_test_post_cut)
        # Return a list of connected components where the first component always contains
        # the most sources and the second always contains the most sinks (to make life easier for Neuroglancer)
        max_source_index = -1
        max_sources = 0
        max_sink_index = -1
        max_sinks = 0
        i = 0
        for cc in ccs_test_post_cut:
            num_sources = np.count_nonzero(np.in1d(self.source_graph_ids, cc))
            num_sinks = np.count_nonzero(np.in1d(self.sink_graph_ids, cc))
            if num_sources > max_sources:
                max_sources = num_sources
                max_source_index = i
            if num_sinks > max_sinks:
                max_sinks = num_sinks
                max_sink_index = i
            i += 1
        supervoxel_ccs[0] = self.unique_ids[ccs_test_post_cut[max_source_index]]
        supervoxel_ccs[1] = self.unique_ids[ccs_test_post_cut[max_sink_index]]
        i = 0
        j = 2
        for cc in ccs_test_post_cut:
            if i != max_source_index and i != max_sink_index:
                supervoxel_ccs[j] = self.unique_ids[cc]
                j += 1
            i += 1
        return (supervoxel_ccs, illegal_split)


    def _create_fake_edge_property(self, affs):
        """
        Create an edge property to remove fake edges later
        (will be used to test whether split valid)
        """
        is_fake_edge = np.concatenate(
            [
                [False] * len(affs),
                [True] * (len(self.source_edges) + len(self.sink_edges)),
            ]
        )
        remove_edges_later = np.concatenate([is_fake_edge, is_fake_edge])
        self.edges_to_remove = self.weighted_graph.new_edge_property(
            "bool", vals=remove_edges_later
        )


    def _filter_graph_connected_components(self):
        """
        Filter out connected components in the graph 
        that are not involved in the local mincut
        """
        ccs = flatgraph_utils.connected_components(self.weighted_graph)

        removed = self.weighted_graph.new_vertex_property("bool")
        removed.a = False
        if len(ccs) > 1:
            for cc in ccs:
                # If connected component contains no sources or no sinks,
                # remove its nodes from the mincut computation
                if not (
                    np.any(np.in1d(self.source_graph_ids, cc))
                    and np.any(np.in1d(self.sink_graph_ids, cc))
                ):
                    for node_id in cc:
                        removed[node_id] = True

        self.weighted_graph.set_vertex_filter(removed, inverted=True)
        pruned_graph = graph_tool.Graph(self.weighted_graph, prune=True)
        # Test that there is only one connected component left
        ccs = flatgraph_utils.connected_components(pruned_graph)

        if len(ccs) > 1:
            if self.logger is not None:
                self.logger.warning(
                    "Not all sinks and sources are within the same (local)"
                    "connected component"
                )
            raise cg_exceptions.PreconditionError(
                "Not all sinks and sources are within the same (local)"
                "connected component"
            )
        elif len(ccs) == 0:
            raise cg_exceptions.PreconditionError(
                "Sinks and sources are not connected through the local graph. "
                "Please try a different set of vertices to perform the mincut."
            )


    def _gt_mincut_sanity_check(self, partition):
        """
        After the mincut has been computed, assert that: the sources are within
        one connected component, and the sinks are within another separate one.
        These assertions should not fail. If they do, 
        then something went wrong with the graph_tool mincut computation
        """
        for i_cc in np.unique(partition.a):
            # Make sure to read real ids and not graph ids
            cc_list = self.unique_ids[
                np.array(np.where(partition.a == i_cc)[0], dtype=np.int)
            ]

            if np.any(np.in1d(self.sources, cc_list)):
                assert np.all(np.in1d(self.sources, cc_list))
                assert ~np.any(np.in1d(self.sinks, cc_list))

            if np.any(np.in1d(self.sinks, cc_list)):
                assert np.all(np.in1d(self.sinks, cc_list))
                assert ~np.any(np.in1d(self.sources, cc_list))


    def _sink_and_source_connectivity_sanity_check(self, cut_edge_set):
        """
        Similar to _gt_mincut_sanity_check, except we do the check again *after*
        removing the fake infinite affinity edges.
        """
        time_start = time.time()
        for cut_edge in cut_edge_set:
            # May be more than one edge from vertex cut_edge[0] to vertex cut_edge[1], remove them all
            parallel_edges = self.weighted_graph.edge(
                cut_edge[0], cut_edge[1], all_edges=True
            )
            for edge_to_remove in parallel_edges:
                self.edges_to_remove[edge_to_remove] = True

        self.weighted_graph.set_edge_filter(self.edges_to_remove, True)
        ccs_test_post_cut = flatgraph_utils.connected_components(self.weighted_graph)

        # Make sure sinks and sources are among each other and not in different sets
        # after removing the cut edges and the fake infinity edges
        illegal_split = False
        try:
            for cc in ccs_test_post_cut:
                if np.any(np.in1d(self.source_graph_ids, cc)):
                    assert np.all(np.in1d(self.source_graph_ids, cc))
                    assert ~np.any(np.in1d(self.sink_graph_ids, cc))

                if np.any(np.in1d(self.sink_graph_ids, cc)):
                    assert np.all(np.in1d(self.sink_graph_ids, cc))
                    assert ~np.any(np.in1d(self.source_graph_ids, cc))
        except AssertionError:
            if self.split_preview:
                # If we are performing a split preview, we allow these illegal splits,
                # but return a flag to return a message to the user
                illegal_split = True
            else:
                raise cg_exceptions.PreconditionError(
                    "Failed to find a cut that separated the sources from the sinks. "
                    "Please try another cut that partitions the sets cleanly if possible. "
                    "If there is a clear path between all the supervoxels in each set, "
                    "that helps the mincut algorithm."
                )

        dt = time.time() - time_start
        if self.logger is not None:
            self.logger.debug("Verifying local graph: %.2fms" % (dt * 1000))

        return ccs_test_post_cut, illegal_split


def mincut_graph_tool(cg_edges: Iterable[Sequence[np.uint64]],
                      cg_affs: Sequence[np.uint64],
                      cg_sources: Sequence[np.uint64],
                      cg_sinks: Sequence[np.uint64],
                      logger: Optional[logging.Logger] = None,
                      split_preview: bool = False) -> np.ndarray:
    """ Computes the min cut on a local graph
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :param sources: uint64
    :param sinks: uint64
    :return: m x 2 array of uint64s
        edges that should be removed
    """
    time_start = time.time()

    # Stitch supervoxels across chunk boundaries and represent those that are
    # connected with a cross chunk edge with a single id. This may cause id
    # changes among sinks and sources that need to be taken care of.
    edges, affs, sv_to_gt_mapping, gt_to_sv_mapping = merge_cross_chunk_edges(cg_edges.copy(),cg_affs.copy())

    dt = time.time() - time_start
    if logger is not None:
        logger.debug("Cross edge merging: %.2fms" % (dt * 1000))
    time_start = time.time()

    mapping_vec = np.vectorize(lambda a: sv_to_gt_mapping[a] if a in sv_to_gt_mapping else a)

    if len(edges) == 0:
        return []

    if len(sv_to_gt_mapping) > 0:
        assert np.unique(list(sv_to_gt_mapping.keys()), return_counts=True)[1].max() == 1

    sources = mapping_vec(cg_sources)
    sinks = mapping_vec(cg_sinks)

    local_mincut_graph = LocalMincutGraph(edges, affs, sources, sinks, gt_to_sv_mapping, split_preview, logger)

    dt = time.time() - time_start
    if logger is not None:
        logger.debug("Graph creation: %.2fms" % (dt * 1000))

    mincut = local_mincut_graph.compute_mincut(cg_edges)
    if len(mincut) == 0:
        return []

    return mincut


def mincut(edges: Iterable[Sequence[np.uint64]],
           affs: Sequence[np.uint64],
           sources: Sequence[np.uint64],
           sinks: Sequence[np.uint64],
           logger: Optional[logging.Logger] = None,
           split_preview: bool = False) -> np.ndarray:
    """ Computes the min cut on a local graph
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :param sources: uint64
    :param sinks: uint64
    :return: m x 2 array of uint64s
        edges that should be removed
    """

    return mincut_graph_tool(cg_edges=edges, cg_affs=affs, cg_sources=sources,
                             cg_sinks=sinks, logger=logger, split_preview=split_preview)

