import collections
import fastremap
import numpy as np
import itertools
import logging
import time
import graph_tool
import graph_tool.flow

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pychunkedgraph.backend import flatgraph_utils
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

float_max = np.finfo(np.float32).max
DEBUG_MODE = False


def merge_cross_chunk_edges_graph_tool(
    edges: Iterable[Sequence[np.uint64]], affs: Sequence[np.uint64]
):
    """ Merges cross chunk edges
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :return:
    """

    # mask for edges that have to be merged
    cross_chunk_edge_mask = np.isinf(affs)

    # graph with edges that have to be merged
    graph, _, _, unique_supervoxel_ids = flatgraph_utils.build_gt_graph(
        edges[cross_chunk_edge_mask], make_directed=True
    )

    # connected components in this graph will be combined in one component
    ccs = flatgraph_utils.connected_components(graph)

    remapping = {}
    mapping = []

    for cc in ccs:
        nodes = unique_supervoxel_ids[cc]
        rep_node = np.min(nodes)

        remapping[rep_node] = nodes

        rep_nodes = np.ones(len(nodes), dtype=np.uint64).reshape(-1, 1) * rep_node
        m = np.concatenate([nodes.reshape(-1, 1), rep_nodes], axis=1)

        mapping.append(m)

    if len(mapping) > 0:
        mapping = np.concatenate(mapping)
    u_nodes = np.unique(edges)
    u_unmapped_nodes = u_nodes[~np.in1d(u_nodes, mapping)]

    unmapped_mapping = np.concatenate(
        [u_unmapped_nodes.reshape(-1, 1), u_unmapped_nodes.reshape(-1, 1)], axis=1
    )
    if len(mapping) > 0:
        complete_mapping = np.concatenate([mapping, unmapped_mapping], axis=0)
    else:
        complete_mapping = unmapped_mapping

    sort_idx = np.argsort(complete_mapping[:, 0])
    idx = np.searchsorted(complete_mapping[:, 0], edges, sorter=sort_idx)
    mapped_edges = np.asarray(complete_mapping[:, 1])[sort_idx][idx]

    mapped_edges = mapped_edges[~cross_chunk_edge_mask]
    mapped_affs = affs[~cross_chunk_edge_mask]

    return mapped_edges, mapped_affs, mapping, complete_mapping, remapping


class LocalMincutGraph:
    """
    Helper class for mincut computation. Used by the mincut_graph_tool function to:
    (1) set up a local graph-tool graph, (2) compute a mincut, (3) ensure required conditions hold,
    and (4) return the ChunkedGraph edges to be removed.
    """

    def __init__(
        self, cg_edges, cg_affs, cg_sources, cg_sinks, split_preview=False, logger=None
    ):
        self.cg_edges = cg_edges
        self.split_preview = split_preview
        self.logger = logger
        time_start = time.time()

        # Stitch supervoxels across chunk boundaries and represent those that are
        # connected with a cross chunk edge with a single id. This may cause id
        # changes among sinks and sources that need to be taken care of.
        mapped_edges, mapped_affs, cross_chunk_edge_mapping, complete_mapping, self.cross_chunk_edge_remapping = merge_cross_chunk_edges_graph_tool(
            cg_edges, cg_affs
        )

        dt = time.time() - time_start
        if logger is not None:
            logger.debug("Cross edge merging: %.2fms" % (dt * 1000))
        time_start = time.time()

        if len(mapped_edges) == 0:
            raise cg_exceptions.PostconditionError(
                f"Local graph somehow only contains cross chunk edges"
            )

        if len(cross_chunk_edge_mapping) > 0:
            assert (
                np.unique(cross_chunk_edge_mapping[:, 0], return_counts=True)[1].max()
                == 1
            )

        # Map cg sources and sinks with the cross chunk edge mapping
        self.sources = fastremap.remap_from_array_kv(
            np.array(cg_sources), complete_mapping[:, 0], complete_mapping[:, 1]
        )
        self.sinks = fastremap.remap_from_array_kv(
            np.array(cg_sinks), complete_mapping[:, 0], complete_mapping[:, 1]
        )

        self._build_gt_graph(mapped_edges, mapped_affs)

        dt = time.time() - time_start
        if logger is not None:
            logger.debug("Graph creation: %.2fms" % (dt * 1000))

        self._create_fake_edge_property(mapped_affs)

    def _build_gt_graph(self, edges, affs):
        """
        Create the graph that will be used to compute the mincut.
        """
        self.source_edges = list(itertools.product(self.sources, self.sources))
        self.sink_edges = list(itertools.product(self.sinks, self.sinks))

        # Assemble edges: Edges after remapping combined with fake infinite affinity
        # edges between sinks and sources
        comb_edges = np.concatenate([edges, self.source_edges, self.sink_edges])
        comb_affs = np.concatenate(
            [affs, [float_max] * (len(self.source_edges) + len(self.sink_edges))]
        )

        # To make things easier for everyone involved, we map the ids to
        # [0, ..., len(unique_supervoxel_ids) - 1]
        # Generate weighted graph with graph_tool
        self.weighted_graph, self.capacities, self.gt_edges, self.unique_supervoxel_ids = flatgraph_utils.build_gt_graph(
            comb_edges, comb_affs, make_directed=True
        )

        self.source_graph_ids = np.where(
            np.in1d(self.unique_supervoxel_ids, self.sources)
        )[0]
        self.sink_graph_ids = np.where(np.in1d(self.unique_supervoxel_ids, self.sinks))[
            0
        ]

        if self.logger is not None:
            self.logger.debug(f"{self.sinks}, {self.sink_graph_ids}")
            self.logger.debug(f"{self.sources}, {self.source_graph_ids}")

    def compute_mincut(self):
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
        return self._remap_cut_edge_set(cut_edge_set)

    def _remap_cut_edge_set(self, cut_edge_set):
        """
        Remap the cut edge set from graph ids to supervoxel ids and return it
        """
        remapped_cutset = []
        for s, t in flatgraph_utils.remap_ids_from_graph(
            cut_edge_set, self.unique_supervoxel_ids
        ):

            if s in self.cross_chunk_edge_remapping:
                s = self.cross_chunk_edge_remapping[s]
            else:
                s = [s]

            if t in self.cross_chunk_edge_remapping:
                t = self.cross_chunk_edge_remapping[t]
            else:
                t = [t]

            remapped_cutset.extend(list(itertools.product(s, t)))
            remapped_cutset.extend(list(itertools.product(t, s)))

        remapped_cutset = np.array(remapped_cutset, dtype=np.uint64)

        remapped_cutset_flattened_view = remapped_cutset.view(dtype="u8,u8")
        edges_flattened_view = self.cg_edges.view(dtype="u8,u8")

        cutset_mask = np.in1d(remapped_cutset_flattened_view, edges_flattened_view)

        return remapped_cutset[cutset_mask]

    def _remap_graph_ids_to_cg_supervoxels(self, graph_ids):
        supervoxel_list = []
        # Supervoxels that were passed into graph
        mapped_supervoxels = self.unique_supervoxel_ids[graph_ids]
        # Now need to remap these using the cross_chunk_edge_remapping
        for sv in mapped_supervoxels:
            if sv in self.cross_chunk_edge_remapping:
                supervoxel_list.extend(self.cross_chunk_edge_remapping[sv])
            else:
                supervoxel_list.append(sv)
        return np.array(supervoxel_list)

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
        supervoxel_ccs[0] = self._remap_graph_ids_to_cg_supervoxels(
            ccs_test_post_cut[max_source_index]
        )
        supervoxel_ccs[1] = self._remap_graph_ids_to_cg_supervoxels(
            ccs_test_post_cut[max_sink_index]
        )
        i = 0
        j = 2
        for cc in ccs_test_post_cut:
            if i != max_source_index and i != max_sink_index:
                supervoxel_ccs[j] = self._remap_graph_ids_to_cg_supervoxels(cc)
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
            cc_list = self.unique_supervoxel_ids[
                np.array(np.where(partition.a == i_cc)[0], dtype=int)
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


def mincut(
    edges: Iterable[Sequence[np.uint64]],
    affs: Sequence[np.uint64],
    sources: Sequence[np.uint64],
    sinks: Sequence[np.uint64],
    logger: Optional[logging.Logger] = None,
    split_preview: bool = False,
) -> np.ndarray:
    """ Computes the min cut on a local graph
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :param sources: uint64
    :param sinks: uint64
    :return: m x 2 array of uint64s
        edges that should be removed
    """

    local_mincut_graph = LocalMincutGraph(
        edges, affs, sources, sinks, split_preview, logger
    )

    mincut = local_mincut_graph.compute_mincut()
    if len(mincut) == 0:
        return []

    return mincut
