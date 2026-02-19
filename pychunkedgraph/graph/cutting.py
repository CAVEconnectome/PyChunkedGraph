import collections
import fastremap
import numpy as np
import itertools
import logging
import time
import graph_tool
import graph_tool.flow

from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Sequence
from typing import Iterable

from .utils import flatgraph
from .utils import basetypes
from .utils.generic import get_bounding_box
from .edges import Edges
from .exceptions import PreconditionError
from .exceptions import PostconditionError

DEBUG_MODE = False


class IsolatingCutException(Exception):
    """Raised when mincut would split off one of the labeled supervoxel exactly.
    This is used to trigger a PostconditionError with a custom message.
    """

    pass


def merge_cross_chunk_edges_graph_tool(
    edges: Iterable[Sequence[np.uint64]], affs: Sequence[np.uint64]
):
    """Merges cross chunk edges
    :param edges: n x 2 array of uint64s
    :param affs: float array of length n
    :return:
    """
    # mask for edges that have to be merged
    cross_chunk_edge_mask = np.isinf(affs)
    # graph with edges that have to be merged
    graph, _, _, unique_supervoxel_ids = flatgraph.build_gt_graph(
        edges[cross_chunk_edge_mask], make_directed=True
    )

    # connected components in this graph will be combined in one component
    ccs = flatgraph.connected_components(graph)
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
    u_unmapped_nodes = u_nodes[~np.isin(u_nodes, mapping)]
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
        self,
        cg_edges,
        cg_affs,
        cg_sources,
        cg_sinks,
        split_preview=False,
        path_augment=True,
        disallow_isolating_cut=True,
        logger=None,
    ):
        self.cg_edges = cg_edges
        self.split_preview = split_preview
        self.logger = logger
        self.path_augment = path_augment
        self.disallow_isolating_cut = disallow_isolating_cut

        time_start = time.time()

        # Stitch supervoxels across chunk boundaries and represent those that are
        # connected with a cross chunk edge with a single id. This may cause id
        # changes among sinks and sources that need to be taken care of.
        (
            mapped_edges,
            mapped_affs,
            cross_chunk_edge_mapping,
            complete_mapping,
            self.cross_chunk_edge_remapping,
        ) = merge_cross_chunk_edges_graph_tool(cg_edges, cg_affs)

        dt = time.time() - time_start
        if logger is not None:
            logger.debug("Cross edge merging: %.2fms" % (dt * 1000))
        time_start = time.time()

        if len(mapped_edges) == 0:
            raise PostconditionError(
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

        self.source_path_vertices = self.source_graph_ids
        self.sink_path_vertices = self.sink_graph_ids

        dt = time.time() - time_start
        if logger is not None:
            logger.debug("Graph creation: %.2fms" % (dt * 1000))

        self._create_fake_edge_property(mapped_affs)

    def _build_gt_graph(self, edges, affs):
        """
        Create the graphs that will be used to compute the mincut.
        """

        # Assemble graph without infinite-affinity edges
        (
            self.weighted_graph_raw,
            self.capacities_raw,
            self.gt_edges_raw,
            _,
        ) = flatgraph.build_gt_graph(edges, affs, make_directed=True)

        self.source_edges = list(itertools.product(self.sources, self.sources))
        self.sink_edges = list(itertools.product(self.sinks, self.sinks))

        # Assemble edges: Edges after remapping combined with fake infinite affinity
        # edges between sinks and sources
        comb_edges = np.concatenate([edges, self.source_edges, self.sink_edges])
        comb_affs = np.concatenate(
            [
                affs,
                [np.finfo(np.float32).max]
                * (len(self.source_edges) + len(self.sink_edges)),
            ]
        )

        # To make things easier for everyone involved, we map the ids to
        # [0, ..., len(unique_supervoxel_ids) - 1]
        # Generate weighted graph with graph_tool
        (
            self.weighted_graph,
            self.capacities,
            self.gt_edges,
            self.unique_supervoxel_ids,
        ) = flatgraph.build_gt_graph(comb_edges, comb_affs, make_directed=True)

        self.source_graph_ids = np.where(
            np.isin(self.unique_supervoxel_ids, self.sources)
        )[0]
        self.sink_graph_ids = np.where(np.isin(self.unique_supervoxel_ids, self.sinks))[
            0
        ]

        if self.logger is not None:
            self.logger.debug(f"{self.sinks}, {self.sink_graph_ids}")
            self.logger.debug(f"{self.sources}, {self.source_graph_ids}")

    def _compute_mincut_direct(self):
        """Uses additional edges directly between source/sink points."""
        self._filter_graph_connected_components()
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
        return partition

    def _augment_mincut_capacity(self):
        """Increase affinities along all pairs shortest paths between sources/sinks
        in the supervoxel graph.
        """
        try:
            paths_v_s, paths_e_s, invaff_s = flatgraph.compute_filtered_paths(
                self.weighted_graph_raw,
                self.capacities_raw,
                self.source_graph_ids,
                self.sink_graph_ids,
            )
            paths_v_y, paths_e_y, invaff_y = flatgraph.compute_filtered_paths(
                self.weighted_graph_raw,
                self.capacities_raw,
                self.sink_graph_ids,
                self.source_graph_ids,
            )
        except AssertionError:
            raise PreconditionError(
                "Paths between source or sink points irreparably overlap other labels from other side. "
                "Check that labels are correct and consider spreading points out farther."
            )

        paths_e_s_no, paths_e_y_no, do_check = flatgraph.remove_overlapping_edges(
            paths_v_s, paths_e_s, paths_v_y, paths_e_y
        )
        if do_check:
            e_connected = flatgraph.check_connectedness(paths_v_s, paths_e_s_no)
            y_connected = flatgraph.check_connectedness(paths_v_y, paths_e_y_no)
            if e_connected is False or y_connected is False:
                try:
                    paths_e_s_no, paths_e_y_no = self.rerun_paths_without_overlap(
                        paths_v_s,
                        paths_e_s,
                        invaff_s,
                        paths_v_y,
                        paths_e_y,
                        invaff_y,
                    )
                except AssertionError:
                    raise PreconditionError(
                        "Paths between source point pairs and sink point pairs overlapped irreparably. "
                        "Consider doing cut in multiple parts."
                    )

        self.source_path_vertices = flatgraph.flatten_edge_list(paths_e_s_no)
        self.sink_path_vertices = flatgraph.flatten_edge_list(paths_e_y_no)

        adj_capacity = flatgraph.adjust_affinities(
            self.weighted_graph_raw, self.capacities_raw, paths_e_s_no + paths_e_y_no
        )
        return adj_capacity

    def rerun_paths_without_overlap(
        self,
        paths_v_s,
        paths_e_s,
        invaff_s,
        paths_v_y,
        paths_e_y,
        invaff_y,
        invert_winner=False,
    ):

        # smaller distance means larger affinity
        s_wins = flatgraph.harmonic_mean_paths(
            invaff_s
        ) < flatgraph.harmonic_mean_paths(invaff_y)
        if invert_winner:
            s_wins = not s_wins

        # Omit winning team vertices from graph
        try:
            if s_wins:
                paths_e_s_no = paths_e_s
                omit_verts = [int(v) for v in itertools.chain.from_iterable(paths_v_s)]
                _, paths_e_y_no, _ = flatgraph.compute_filtered_paths(
                    self.weighted_graph_raw,
                    self.capacities_raw,
                    self.sink_graph_ids,
                    omit_verts,
                )

            else:
                omit_verts = [int(v) for v in itertools.chain.from_iterable(paths_v_y)]
                _, paths_e_s_no, _ = flatgraph.compute_filtered_paths(
                    self.weighted_graph_raw,
                    self.capacities_raw,
                    self.source_graph_ids,
                    omit_verts,
                )
                paths_e_y_no = paths_e_y
        except AssertionError:
            # If no path is found and this hasn't been tried before, try giving the overlap to the other team and finding paths
            if not invert_winner:
                paths_e_s_no, paths_e_y_no = self.rerun_paths_without_overlap(
                    paths_v_s,
                    paths_e_s,
                    invaff_s,
                    paths_v_y,
                    paths_e_y,
                    invaff_y,
                    invert_winner=True,
                )
            else:
                # Otherwise propagate the AssertionError back up
                raise AssertionError
        return paths_e_s_no, paths_e_y_no

    def _compute_mincut_path_augmented(self):
        """Compute mincut using edges found from a shortest-path search."""
        adj_capacity = self._augment_mincut_capacity()

        gr = self.weighted_graph_raw
        src, tgt = gr.vertex(self.source_graph_ids[0]), gr.vertex(
            self.sink_graph_ids[0]
        )

        residuals = graph_tool.flow.boykov_kolmogorov_max_flow(
            gr, src, tgt, adj_capacity
        )

        partition = graph_tool.flow.min_st_cut(gr, src, adj_capacity, residuals)
        return partition

    def compute_mincut(self):
        """
        Compute mincut and return the supervoxel cut edge set
        """

        time_start = time.time()

        if self.path_augment:
            partition = self._compute_mincut_path_augmented()
        else:
            partition = self._compute_mincut_direct()

        dt = time.time() - time_start
        if self.logger is not None:
            self.logger.debug("Mincut comp: %.2fms" % (dt * 1000))

        if DEBUG_MODE:
            self._gt_mincut_sanity_check(partition)

        if self.path_augment:
            labeled_edges = partition.a[self.gt_edges_raw]
            cut_edge_set = self.gt_edges_raw[labeled_edges[:, 0] != labeled_edges[:, 1]]
        else:
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
        for s, t in flatgraph.remap_ids_from_graph(
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

        cutset_mask = np.isin(remapped_cutset_flattened_view, edges_flattened_view).ravel()

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
        (
            ccs_test_post_cut,
            illegal_split,
        ) = self._sink_and_source_connectivity_sanity_check(cut_edge_set)
        supervoxel_ccs = [None] * len(ccs_test_post_cut)
        # Return a list of connected components where the first component always contains
        # the most sources and the second always contains the most sinks (to make life easier for Neuroglancer)
        max_source_index = -1
        max_sources = 0
        max_sink_index = -1
        max_sinks = 0
        i = 0
        for cc in ccs_test_post_cut:
            num_sources = np.count_nonzero(np.isin(self.source_graph_ids, cc))
            num_sinks = np.count_nonzero(np.isin(self.sink_graph_ids, cc))
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
        ccs = flatgraph.connected_components(self.weighted_graph)

        removed = self.weighted_graph.new_vertex_property("bool")
        removed.a = False
        if len(ccs) > 1:
            for cc in ccs:
                # If connected component contains no sources or no sinks,
                # remove its nodes from the mincut computation
                if not (
                    np.any(np.isin(self.source_graph_ids, cc))
                    and np.any(np.isin(self.sink_graph_ids, cc))
                ):
                    for node_id in cc:
                        removed[node_id] = True

        keep = self.weighted_graph.new_vertex_property("bool")
        keep.a = ~removed.a.astype(bool)
        self.weighted_graph.set_vertex_filter(keep)
        pruned_graph = graph_tool.Graph(self.weighted_graph, prune=True)
        # Test that there is only one connected component left
        ccs = flatgraph.connected_components(pruned_graph)
        if len(ccs) > 1:
            if self.logger is not None:
                self.logger.warning(
                    "Not all sinks and sources are within the same (local)"
                    "connected component"
                )
            raise PreconditionError(
                "Not all sinks and sources are within the same (local)"
                "connected component"
            )
        elif len(ccs) == 0:
            raise PreconditionError(
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

            if np.any(np.isin(self.sources, cc_list)):
                assert np.all(np.isin(self.sources, cc_list))
                assert ~np.any(np.isin(self.sinks, cc_list))

            if np.any(np.isin(self.sinks, cc_list)):
                assert np.all(np.isin(self.sinks, cc_list))
                assert ~np.any(np.isin(self.sources, cc_list))

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

        self.edges_to_remove.a = ~self.edges_to_remove.a.astype(bool)
        self.weighted_graph.set_edge_filter(self.edges_to_remove)
        ccs_test_post_cut = flatgraph.connected_components(self.weighted_graph)

        # Make sure sinks and sources are among each other and not in different sets
        # after removing the cut edges and the fake infinity edges
        illegal_split = False
        try:
            for cc in ccs_test_post_cut:
                if np.any(np.isin(self.source_graph_ids, cc)):
                    assert np.all(np.isin(self.source_graph_ids, cc))
                    assert ~np.any(np.isin(self.sink_graph_ids, cc))
                    if (
                        len(self.source_path_vertices) == len(cc)
                        and self.disallow_isolating_cut
                    ):
                        if not self.partition_edges_within_label(cc):
                            raise IsolatingCutException("Source")

                if np.any(np.isin(self.sink_graph_ids, cc)):
                    assert np.all(np.isin(self.sink_graph_ids, cc))
                    assert ~np.any(np.isin(self.source_graph_ids, cc))
                    if (
                        len(self.sink_path_vertices) == len(cc)
                        and self.disallow_isolating_cut
                    ):
                        if not self.partition_edges_within_label(cc):
                            raise IsolatingCutException("Sink")

        except AssertionError:
            if self.split_preview:
                # If we are performing a split preview, we allow these illegal splits,
                # but return a flag to return a message to the user
                illegal_split = True
            else:
                raise PreconditionError(
                    "Failed to find a cut that separated the sources from the sinks. "
                    "Please try another cut that partitions the sets cleanly if possible. "
                    "If there is a clear path between all the supervoxels in each set, "
                    "that helps the mincut algorithm."
                )
        except IsolatingCutException as e:
            if self.split_preview:
                illegal_split = True
            else:
                raise PreconditionError(
                    f"Split cut off only the labeled points on the {e} side. Please additional points to other parts of the merge error on that side."
                )

        dt = time.time() - time_start
        if self.logger is not None:
            self.logger.debug("Verifying local graph: %.2fms" % (dt * 1000))
        return ccs_test_post_cut, illegal_split

    def partition_edges_within_label(self, cc):
        """Test is an isolated component has out-edges only within the original
        labeled points of the cut
        """
        label_graph_ids = np.concatenate((self.source_graph_ids, self.sink_graph_ids))

        for vind in cc:
            v = self.weighted_graph_raw.vertex(vind)
            out_vinds = [int(x) for x in v.out_neighbors()]
            if not np.all(np.isin(out_vinds, label_graph_ids)):
                return False
        else:
            return True


def run_multicut(
    edges: Edges,
    source_ids: Sequence[np.uint64],
    sink_ids: Sequence[np.uint64],
    *,
    split_preview: bool = False,
    path_augment: bool = True,
    disallow_isolating_cut: bool = True,
):
    local_mincut_graph = LocalMincutGraph(
        edges.get_pairs(),
        edges.affinities,
        source_ids,
        sink_ids,
        split_preview,
        path_augment,
        disallow_isolating_cut=disallow_isolating_cut,
    )
    atomic_edges = local_mincut_graph.compute_mincut()
    if len(atomic_edges) == 0:
        raise PostconditionError(f"Mincut failed. Try with a different set of points.")
    return atomic_edges


def run_split_preview(
    cg,
    source_ids: Sequence[np.uint64],
    sink_ids: Sequence[np.uint64],
    source_coords: Sequence[Sequence[int]],
    sink_coords: Sequence[Sequence[int]],
    bb_offset: Tuple[int, int, int] = (120, 120, 12),
    path_augment: bool = True,
    disallow_isolating_cut: bool = True,
):
    root_ids = set(
        cg.get_roots(np.concatenate([source_ids, sink_ids]), assert_roots=True)
    )
    if len(root_ids) > 1:
        raise PreconditionError("Supervoxels must belong to the same object.")

    bbox = get_bounding_box(source_coords, sink_coords, bb_offset)
    l2id_agglomeration_d, edges = cg.get_subgraph(
        root_ids.pop(), bbox=bbox, bbox_is_coordinate=True
    )
    in_edges, out_edges, cross_edges = edges
    edges = in_edges + out_edges + cross_edges
    supervoxels = np.concatenate(
        [agg.supervoxels for agg in l2id_agglomeration_d.values()]
    )
    mask0 = np.isin(edges.node_ids1, supervoxels)
    mask1 = np.isin(edges.node_ids2, supervoxels)
    edges = edges[mask0 & mask1]
    edges_to_remove, illegal_split = run_multicut(
        edges,
        source_ids,
        sink_ids,
        split_preview=True,
        path_augment=path_augment,
        disallow_isolating_cut=disallow_isolating_cut,
    )

    if len(edges_to_remove) == 0:
        raise PostconditionError("Mincut could not find any edges to remove.")

    return edges_to_remove, illegal_split
