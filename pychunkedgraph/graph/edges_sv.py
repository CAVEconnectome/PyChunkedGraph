"""
Edge routing logic for supervoxel splits.

When a supervoxel (SV) is split into multiple fragments, all edges that
connected the original SV to its neighbors must be reassigned to the
appropriate new fragment(s). This module handles that reassignment.

Edge classification:
    Active edges: partner SV shares the same root as the split SV.
        These edges are routed based on affinity type:
        - Inf-affinity (cross-chunk) to a split partner: matched by split label,
          connecting fragments that received the same label during the split.
        - Inf-affinity (cross-chunk) to an unsplit partner: assigned to the
          closest fragment only. Broadcasting to all fragments would create an
          uncuttable bridge between source/sink sides of the split.
        - Finite-affinity: assigned to fragments within a distance threshold
          of the partner, or the closest fragment if none are within threshold.

    Inactive edges: partner SV has a different root.
        These are edges to neighboring objects. All fragments inherit the edge
        since any fragment could border the neighbor.

Distance computation:
    For partners within the segmentation bbox, distances are precomputed via
    kdtree pairwise distances. For active partners outside the bbox (e.g.
    cross-chunk fragments excluded by _get_whole_sv's bbox clipping), distances
    are computed from each new fragment's kdtree to the partner's chunk boundary.
"""

from __future__ import annotations

import time
from functools import reduce
from typing import TYPE_CHECKING
from datetime import datetime

import fastremap
import numpy as np

from pychunkedgraph import get_logger
from pychunkedgraph.graph import attributes, basetypes, serializers
from pychunkedgraph.graph.exceptions import PostconditionError
from scipy.spatial import cKDTree
from pychunkedgraph.graph.cutting_sv import build_coords_by_label
from pychunkedgraph.graph.edges import Edges

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph

logger = get_logger(__name__)


def _match_by_label(new_ids, partner, aff, area, new_id_label_map, distances_row):
    """For inf-affinity (cross-chunk) edges: connect fragments with matching split label."""
    partner_label = new_id_label_map[partner]
    matching = np.array(
        [nid for nid in new_ids if new_id_label_map.get(nid) == partner_label],
        dtype=basetypes.NODE_ID,
    )
    if len(matching):
        edges = np.column_stack(
            [matching, np.full(len(matching), partner, dtype=np.uint64)]
        )
        affs = np.full(len(matching), aff, dtype=basetypes.EDGE_AFFINITY)
        areas = np.full(len(matching), area, dtype=basetypes.EDGE_AREA)
        return edges, affs, areas
    # fallback: closest fragment
    close = new_ids[np.argmin(distances_row)]
    return (
        np.array([[close, partner]], dtype=np.uint64),
        np.array([aff], dtype=basetypes.EDGE_AFFINITY),
        np.array([area], dtype=basetypes.EDGE_AREA),
    )


def _match_by_proximity(new_ids, partner, aff, area, distances_row, threshold):
    """For regular edges: connect fragments within distance threshold."""
    close_mask = distances_row < threshold
    nearby = new_ids[close_mask]
    if len(nearby):
        edges = np.column_stack(
            [nearby, np.full(len(nearby), partner, dtype=np.uint64)]
        )
        affs = np.full(len(nearby), aff, dtype=basetypes.EDGE_AFFINITY)
        areas = np.full(len(nearby), area, dtype=basetypes.EDGE_AREA)
        return edges, affs, areas
    close = new_ids[np.argmin(distances_row)]
    return (
        np.array([[close, partner]], dtype=np.uint64),
        np.array([aff], dtype=basetypes.EDGE_AFFINITY),
        np.array([area], dtype=basetypes.EDGE_AREA),
    )


def _match_inf_unsplit(new_ids, partner, aff, area, distances_row):
    """Inf-affinity edge to an unsplit partner: assign to closest fragment only.
    Connecting all fragments would create an uncuttable bridge between source/sink sides.
    """
    closest = new_ids[np.argmin(distances_row)]
    return (
        np.array([[closest, partner]], dtype=np.uint64),
        np.array([aff], dtype=basetypes.EDGE_AFFINITY),
        np.array([area], dtype=basetypes.EDGE_AREA),
    )


def _match_partner(
    new_ids, partner, aff, area, distances_row, new_id_label_map, threshold
):
    """Route a single old edge to the appropriate new fragment(s)."""
    if np.isinf(aff):
        if new_id_label_map and partner in new_id_label_map:
            return _match_by_label(
                new_ids, partner, aff, area, new_id_label_map, distances_row
            )
        return _match_inf_unsplit(new_ids, partner, aff, area, distances_row)
    return _match_by_proximity(new_ids, partner, aff, area, distances_row, threshold)


def _expand_partners(active_partners, active_affs, active_areas, old_new_map):
    """If a partner was also split, expand it to its new fragment IDs."""
    remapped_lists = [
        np.asarray(list(old_new_map.get(p, {p})), dtype=np.uint64)
        for p in active_partners
    ]
    if not remapped_lists:
        return (
            [],
            np.array([], dtype=basetypes.EDGE_AFFINITY),
            np.array([], dtype=basetypes.EDGE_AREA),
        )
    counts = np.array([len(r) for r in remapped_lists])
    partners = np.concatenate(remapped_lists)
    affs = np.repeat(active_affs, counts)
    areas = np.repeat(active_areas, counts)
    return partners, affs, areas


def _compute_partner_distances(new_kdtrees, partner_coords):
    """Compute min distance from each new fragment kdtree to a partner's voxel coords."""
    partner_tree = cKDTree(partner_coords)
    distances = np.empty(len(new_kdtrees), dtype=float)
    for i, kt in enumerate(new_kdtrees):
        if kt.n <= partner_tree.n:
            d, _ = partner_tree.query(kt.data, k=1, workers=-1)
        else:
            d, _ = kt.query(partner_tree.data, k=1, workers=-1)
        distances[i] = float(np.min(d))
    return distances


def _compute_boundary_distances(cg, new_kdtrees, partner, old_chunk, chunk_size):
    """Compute distance from each new fragment to a partner's chunk boundary.
    Used for active partners outside the bbox that have no kdtree entry.
    old_chunk and chunk_size should be precomputed by the caller.
    """
    partner_chunk = cg.get_chunk_coordinates(partner)
    diff = partner_chunk.astype(int) - old_chunk.astype(int)
    axis = np.argmax(np.abs(diff))
    if diff[axis] > 0:
        boundary = (old_chunk[axis] + 1) * chunk_size[axis]
    else:
        boundary = old_chunk[axis] * chunk_size[axis]
    return np.array([np.min(np.abs(kt.data[:, axis] - boundary)) for kt in new_kdtrees])


def _get_new_edges(
    edges_info: tuple,
    old_new_map: dict,
    coords_by_label: dict,
    root_id: basetypes.NODE_ID,
    sv_root_map: dict,
    cg: "ChunkedGraph",
    new_kdtrees: list,
    new_ids_arr: np.ndarray,
    new_id_label_map: dict = None,
    threshold: int = 10,
):
    edge_batches, aff_batches, area_batches = [], [], []
    edges, affinities, areas = edges_info

    for old, new in old_new_map.items():
        new_ids = np.array(list(new), dtype=basetypes.NODE_ID)
        edges_m = np.any(edges == old, axis=1)
        selected_edges = edges[edges_m]
        sel_m = selected_edges != old
        assert np.all(np.sum(sel_m, axis=1) == 1)

        partners = selected_edges[sel_m]
        edge_affs = affinities[edges_m]
        edge_areas = areas[edges_m]
        partner_roots = np.array(
            [sv_root_map.get(p, 0) for p in partners], dtype=np.uint64
        )
        active_m = partner_roots == root_id

        # Inactive partners (different root): broadcast to all fragments
        inactive_idx = np.where(~active_m)[0]
        if len(inactive_idx) > 0:
            inactive_partners = partners[inactive_idx]
            n_frag = len(new_ids)
            broadcast_edges = np.column_stack(
                [
                    np.repeat(new_ids, len(inactive_partners)),
                    np.tile(inactive_partners, n_frag),
                ]
            )
            edge_batches.append(broadcast_edges)
            aff_batches.append(np.tile(edge_affs[inactive_idx], n_frag))
            area_batches.append(np.tile(edge_areas[inactive_idx], n_frag))

        # Active partners (same root): route based on affinity type
        active_partners, act_affs, act_areas = _expand_partners(
            partners[active_m], edge_affs[active_m], edge_areas[active_m], old_new_map
        )
        if len(active_partners) > 0:
            # Build kdtrees for this old SV's fragments only
            frag_kdtrees = [cKDTree(coords_by_label[int(nid)]) for nid in new_ids]
            old_chunk = cg.get_chunk_coordinates(new_ids[0]) if cg else None
            chunk_size = cg.meta.graph_config.CHUNK_SIZE if cg else None
            for k, partner in enumerate(active_partners):
                partner_coords = coords_by_label.get(int(partner))
                if partner_coords is not None:
                    act_dist_row = _compute_partner_distances(
                        frag_kdtrees, partner_coords
                    )
                else:
                    act_dist_row = _compute_boundary_distances(
                        cg, frag_kdtrees, partner, old_chunk, chunk_size
                    )
                e, a, ar = _match_partner(
                    new_ids,
                    partner,
                    act_affs[k],
                    act_areas[k],
                    act_dist_row,
                    new_id_label_map,
                    threshold,
                )
                edge_batches.append(e)
                aff_batches.append(a)
                area_batches.append(ar)

        # Low-affinity edges between split fragments (cuttable by mincut)
        if len(new_ids) > 1:
            i_idx, j_idx = np.triu_indices(len(new_ids), k=1)
            pairs = np.column_stack([new_ids[i_idx], new_ids[j_idx]])
            edge_batches.append(pairs)
            n_pairs = len(pairs)
            aff_batches.append(np.full(n_pairs, 0.001, dtype=basetypes.EDGE_AFFINITY))
            area_batches.append(np.zeros(n_pairs, dtype=basetypes.EDGE_AREA))

    if len(edge_batches) == 0:
        return (
            np.array([], dtype=basetypes.NODE_ID).reshape(0, 2),
            np.array([], dtype=basetypes.EDGE_AFFINITY),
            np.array([], dtype=basetypes.EDGE_AREA),
        )
    all_edges = np.concatenate(edge_batches)
    all_affs = np.concatenate(aff_batches)
    all_areas = np.concatenate(area_batches)
    edges_ = np.sort(all_edges.astype(basetypes.NODE_ID), axis=1)
    edges_, idx = np.unique(edges_, return_index=True, axis=0)
    return edges_, all_affs[idx], all_areas[idx]


def validate_split_edges(edges, affinities, old_new_map, new_id_label_map=None):
    """Validate edge routing results before writing to prevent graph corruption.

    Checks:
    A. No cross-label inf bridges — if an unsplit partner connects via inf edges
       to fragments with different labels (different sides of the split), that
       creates an uncuttable bridge through mincut.
    B. No self-loops.
    C. All old SVs have replacement edges from their fragments.
    D. Inter-fragment edges exist between all fragment pairs.

    Raises PostconditionError on any violation.
    """
    if len(edges) == 0:
        return

    all_new_ids_arr = np.array(
        [nid for ids in old_new_map.values() for nid in ids], dtype=np.uint64
    )

    # B. No self-loops (cheapest check first)
    self_loops = edges[:, 0] == edges[:, 1]
    if self_loops.any():
        raise PostconditionError(f"Self-loop edges detected: {edges[self_loops]}")

    # A. No cross-label inf bridges to unsplit partners
    if new_id_label_map:
        inf_mask = np.isinf(affinities)
        if inf_mask.any():
            inf_edges = edges[inf_mask]
            is_frag_0 = np.isin(inf_edges[:, 0], all_new_ids_arr)
            is_frag_1 = np.isin(inf_edges[:, 1], all_new_ids_arr)
            mixed_mask = is_frag_0 ^ is_frag_1
            if mixed_mask.any():
                mixed = inf_edges[mixed_mask]
                mixed_frag0 = is_frag_0[mixed_mask]
                partners = np.where(mixed_frag0, mixed[:, 1], mixed[:, 0])
                fragments = np.where(mixed_frag0, mixed[:, 0], mixed[:, 1])
                unsplit_mask = ~np.isin(partners, all_new_ids_arr)
                if unsplit_mask.any():
                    unsplit_partners = partners[unsplit_mask]
                    unsplit_fragments = fragments[unsplit_mask]
                    for p in np.unique(unsplit_partners):
                        p_frags = unsplit_fragments[unsplit_partners == p]
                        labels = {
                            new_id_label_map[int(f)]
                            for f in p_frags
                            if int(f) in new_id_label_map
                        }
                        if len(labels) > 1:
                            raise PostconditionError(
                                f"Inf-affinity edge to unsplit partner {p} bridges "
                                f"fragments with different labels {labels}. "
                                f"This creates an uncuttable bridge in mincut."
                            )

    # C. All old SVs have replacement edges
    edge_svs = np.unique(edges.ravel())
    for old_id, new_ids in old_new_map.items():
        new_arr = np.array(list(new_ids), dtype=np.uint64)
        if not np.any(np.isin(new_arr, edge_svs)):
            raise PostconditionError(
                f"Old SV {old_id} has no replacement edges from fragments {new_ids}"
            )

    # D. Inter-fragment edges exist
    # edges are already sorted (col0 < col1), so check sorted pairs directly
    edge_set = set(map(tuple, edges.tolist()))
    for new_ids in old_new_map.values():
        ids = sorted(new_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if (ids[i], ids[j]) not in edge_set:
                    raise PostconditionError(
                        f"Missing inter-fragment edge between {ids[i]} and {ids[j]}"
                    )


def update_edges(
    cg: "ChunkedGraph",
    root_id: basetypes.NODE_ID,
    bbox: np.ndarray,
    new_seg: np.ndarray,
    old_new_map: dict,
    new_id_label_map: dict = None,
):
    old_new_map = dict(old_new_map)
    t0 = time.time()
    coords_by_label = build_coords_by_label(new_seg)
    new_ids = np.array(list(set.union(*old_new_map.values())), dtype=basetypes.NODE_ID)
    new_kdtrees = [cKDTree(coords_by_label[int(k)]) for k in new_ids]
    logger.note(
        f"build_coords {len(coords_by_label)} labels, {len(new_ids)} fragment trees ({time.time() - t0:.2f}s)"
    )

    t0 = time.time()
    _, edges_tuple = cg.get_subgraph(root_id, bbox, bbox_is_coordinate=True)
    edges_ = reduce(lambda x, y: x + y, edges_tuple, Edges([], []))
    logger.note(
        f"get_subgraph {len(edges_.get_pairs())} edges ({time.time() - t0:.2f}s)"
    )

    edges = edges_.get_pairs()
    affinities = edges_.affinities
    areas = edges_.areas

    edges = np.sort(edges, axis=1)
    _, edges_idx = np.unique(edges, axis=0, return_index=True)
    edges_idx = edges_idx[edges[edges_idx, 0] != edges[edges_idx, 1]]

    edges = edges[edges_idx]
    affinities = affinities[edges_idx]
    areas = areas[edges_idx]

    t0 = time.time()
    all_edge_svs = np.unique(edges)
    all_roots = cg.get_roots(all_edge_svs)
    sv_root_map = dict(zip(all_edge_svs, all_roots))
    logger.note(f"get_roots {len(all_edge_svs)} svs ({time.time() - t0:.2f}s)")

    t0 = time.time()
    result = _get_new_edges(
        (edges, affinities, areas),
        old_new_map,
        coords_by_label,
        root_id,
        sv_root_map,
        cg,
        new_kdtrees,
        new_ids,
        new_id_label_map,
        threshold=cg.meta.sv_split_threshold,
    )
    logger.note(f"_get_new_edges {result[0].shape} ({time.time() - t0:.2f}s)")

    validate_split_edges(result[0], result[1], old_new_map, new_id_label_map)
    return result


def add_new_edges(cg: "ChunkedGraph", edges_tuple: tuple, time_stamp: datetime = None):
    edges_, affinites_, areas_ = edges_tuple
    logger.note(f"new edges: {edges_.shape}")

    nodes = fastremap.unique(edges_)
    chunks = cg.get_chunk_ids_from_node_ids(cg.get_parents(nodes))
    node_chunks = dict(zip(nodes, chunks))

    edges = np.r_[edges_, edges_[:, ::-1]]
    affinites = np.r_[affinites_, affinites_]
    areas = np.r_[areas_, areas_]

    rows = []
    chunks_arr = fastremap.remap(edges, node_chunks)
    for chunk_id in np.unique(chunks):
        val_dict = {}
        mask = chunks_arr[:, 0] == chunk_id
        val_dict[attributes.Connectivity.SplitEdges] = edges[mask]
        val_dict[attributes.Connectivity.Affinity] = affinites[mask]
        val_dict[attributes.Connectivity.Area] = areas[mask]
        rows.append(
            cg.client.mutate_row(
                serializers.serialize_uint64(chunk_id, fake_edges=True),
                val_dict=val_dict,
                time_stamp=time_stamp,
            )
        )
        # logger.note(f"writing {edges[mask].shape} edges to {chunk_id}")
    return rows
