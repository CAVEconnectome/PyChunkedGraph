"""
Manage new supervoxels after a supervoxel split.
"""

from functools import reduce
import logging
import multiprocessing as mp
from typing import Callable
from datetime import datetime
from collections import defaultdict, deque

import fastremap
import numpy as np
from tqdm import tqdm
from pychunkedgraph.graph import attributes, ChunkedGraph, cache as cache_utils
from pychunkedgraph.graph.chunks.utils import chunks_overlapping_bbox, get_neighbors
from pychunkedgraph.graph.cutting_sv import (
    build_kdtrees_by_label,
    pairwise_min_distance_two_sets,
    split_supervoxel_helper,
)
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.types import empty_2d
from pychunkedgraph.graph import basetypes
from pychunkedgraph.graph import serializers
from pychunkedgraph.graph.utils import get_local_segmentation
from pychunkedgraph.io.edges import get_chunk_edges


def _get_whole_sv(
    cg: ChunkedGraph, node: basetypes.NODE_ID, min_coord, max_coord
) -> set:
    all_chunks = [
        (x, y, z)
        for x in range(min_coord[0], max_coord[0])
        for y in range(min_coord[1], max_coord[1])
        for z in range(min_coord[2], max_coord[2])
    ]
    edges = get_chunk_edges(cg.meta.data_source.EDGES, all_chunks)
    cx_edges = edges["cross"].get_pairs()
    if len(cx_edges) == 0:
        return {node}

    explored_nodes = set([node])
    queue = deque([node])
    while queue:
        vertex = queue.popleft()
        mask = cx_edges[:, 0] == vertex
        neighbors = cx_edges[mask][:, 1]

        if len(neighbors) > 0:
            neighbor_coords = cg.get_chunk_coordinates_multiple(neighbors)
            min_mask = (neighbor_coords >= min_coord).all(axis=1)
            max_mask = (neighbor_coords < max_coord).all(axis=1)
            neighbors = neighbors[min_mask & max_mask]

        for neighbor in neighbors:
            if neighbor not in explored_nodes:
                explored_nodes.add(neighbor)
                queue.append(neighbor)
    return explored_nodes


def _update_chunk(args):
    """
    For a chunk that overlaps bounding box for supervoxel split,
      If chunk contains mask for the split supervoxel,
      return indices of mask, old and new supervoxel IDs from this chunk.
    """
    graph_id, chunk_coord, chunk_bbox, seg, result_seg, bb_start = args
    cg = ChunkedGraph(graph_id=graph_id)
    x, y, z = chunk_coord
    chunk_id = cg.get_chunk_id(layer=1, x=x, y=y, z=z)

    _s, _e = chunk_bbox - bb_start
    og_chunk_seg = seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]
    chunk_seg = result_seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]

    labels = fastremap.unique(chunk_seg[chunk_seg != 0])
    if labels.size < 2:
        return None

    _indices = []
    _old_values = []
    _new_values = []
    _label_id_map = {}
    for _id in labels:
        _mask = chunk_seg == _id
        _idx = np.unravel_index(np.flatnonzero(_mask)[0], og_chunk_seg.shape)
        _og_value = og_chunk_seg[_idx]
        _index = np.argwhere(_mask)
        _indices.append(_index)
        _ones = np.ones(len(_index), dtype=basetypes.NODE_ID)
        _old_values.append(_ones * _og_value)
        new_id = cg.id_client.create_node_id(chunk_id)
        _new_values.append(_ones * new_id)
        _label_id_map[int(_id)] = new_id

    _indices = np.concatenate(_indices) + (chunk_bbox[0] - bb_start)
    _old_values = np.concatenate(_old_values)
    _new_values = np.concatenate(_new_values)
    return (_indices, _old_values, _new_values, _label_id_map)


def _voxel_crop(bbs, bbe, bbs_, bbe_):
    xS, yS, zS = bbs - bbs_
    xE, yE, zE = (None if i == 0 else -1 for i in bbe_ - bbe)
    voxel_overlap_crop = np.s_[xS:xE, yS:yE, zS:zE]
    return voxel_overlap_crop


def _parse_results(results, seg, bbs, bbe):
    old_new_map = defaultdict(set)
    new_id_label_map = {}
    for result in results:
        if result:
            indexer, old_values, new_values, label_id_map = result
            seg[tuple(indexer.T)] = new_values
            for old_sv, new_sv in zip(old_values, new_values):
                old_new_map[old_sv].add(new_sv)
            for label, new_id in label_id_map.items():
                new_id_label_map[new_id] = label

    assert np.all(seg.shape == bbe - bbs), f"{seg.shape} != {bbe - bbs}"
    slices = tuple(slice(start, end) for start, end in zip(bbs, bbe)) + (slice(None),)
    return seg, old_new_map, slices, new_id_label_map


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
    partners, affs, areas = [], [], []
    for i in range(len(active_partners)):
        remapped = old_new_map.get(active_partners[i], [active_partners[i]])
        partners.extend(remapped)
        affs.extend([active_affs[i]] * len(remapped))
        areas.extend([active_areas[i]] * len(remapped))
    return partners, affs, areas


def _get_new_edges(
    edges_info: tuple,
    sv_ids: np.ndarray,
    old_new_map: dict,
    distances: np.ndarray,
    dist_vec: Callable,
    new_dist_vec: Callable,
    new_id_label_map: dict = None,
    threshold: int = 10,
):
    new_edges, new_affs, new_areas = [], [], []
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
        active_m = np.isin(partners, sv_ids)

        # Inactive partners (different root, outside distance map): all fragments get the edge
        for k in np.where(~active_m)[0]:
            for new_id in new_ids:
                new_edges.append(np.array([new_id, partners[k]], dtype=np.uint64))
                new_affs.append(edge_affs[k])
                new_areas.append(edge_areas[k])

        # Active partners (same root): route based on affinity type
        active_partners, act_affs, act_areas = _expand_partners(
            partners[active_m], edge_affs[active_m], edge_areas[active_m], old_new_map
        )
        new_id_rows = new_dist_vec(new_ids)
        act_dists = distances[new_id_rows][:, dist_vec(active_partners)].T
        for k, partner in enumerate(active_partners):
            e, a, ar = _match_partner(
                new_ids,
                partner,
                act_affs[k],
                act_areas[k],
                act_dists[k],
                new_id_label_map,
                threshold,
            )
            new_edges.extend(e)
            new_affs.extend(a)
            new_areas.extend(ar)

        # Low-affinity edges between split fragments (cuttable by mincut)
        for i in range(len(new_ids)):
            for j in range(i + 1, len(new_ids)):
                new_edges.append(np.array([new_ids[i], new_ids[j]], dtype=np.uint64))
                new_affs.append(0.001)
                new_areas.append(0)

    if len(new_edges) == 0:
        return (
            np.array([], dtype=basetypes.NODE_ID),
            np.array([], dtype=basetypes.EDGE_AFFINITY),
            np.array([], dtype=basetypes.EDGE_AREA),
        )
    affinities_ = np.array(new_affs, dtype=basetypes.EDGE_AFFINITY)
    areas_ = np.array(new_areas, dtype=basetypes.EDGE_AREA)
    edges_ = np.sort(np.array(new_edges, dtype=basetypes.NODE_ID), axis=1)
    edges_, idx = np.unique(edges_, return_index=True, axis=0)
    return edges_, affinities_[idx], areas_[idx]


def _update_edges(
    cg: ChunkedGraph,
    sv_ids: np.ndarray,
    root_id: basetypes.NODE_ID,
    bbox: np.ndarray,
    new_seg: np.ndarray,
    old_new_map: dict,
    new_id_label_map: dict = None,
):
    old_new_map = dict(old_new_map)
    kdtrees, _ = build_kdtrees_by_label(new_seg)
    distance_map = dict(zip(kdtrees.keys(), np.arange(len(kdtrees))))
    dist_vec = np.vectorize(distance_map.get)

    _, edges_tuple = cg.get_subgraph(root_id, bbox, bbox_is_coordinate=True)
    edges_ = reduce(lambda x, y: x + y, edges_tuple, Edges([], []))

    edges = edges_.get_pairs()
    affinities = edges_.affinities
    areas = edges_.areas

    edges = np.sort(edges, axis=1)
    _, edges_idx = np.unique(edges, axis=0, return_index=True)
    edges_idx = edges_idx[edges[edges_idx, 0] != edges[edges_idx, 1]]

    edges = edges[edges_idx]
    affinities = affinities[edges_idx]
    areas = areas[edges_idx]
    new_ids = np.array(list(set.union(*old_new_map.values())), dtype=basetypes.NODE_ID)
    new_kdtrees = [kdtrees[k] for k in new_ids]
    new_disance_map = dict(zip(new_ids, np.arange(len(new_ids))))
    new_dist_vec = np.vectorize(new_disance_map.get)
    distances = pairwise_min_distance_two_sets(new_kdtrees, list(kdtrees.values()))
    return _get_new_edges(
        (edges, affinities, areas),
        sv_ids,
        old_new_map,
        distances,
        dist_vec,
        new_dist_vec,
        new_id_label_map,
        threshold=cg.meta.sv_split_threshold,
    )


def _add_new_edges(cg: ChunkedGraph, edges_tuple: tuple, time_stamp: datetime = None):
    edges_, affinites_, areas_ = edges_tuple
    logging.info(f"new edges: {edges_.shape}")

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
        logging.info(f"writing {edges[mask].shape} edges to {chunk_id}")
    return rows


def split_supervoxel(
    cg: ChunkedGraph,
    sv_id: basetypes.NODE_ID,
    source_coords: np.ndarray,
    sink_coords: np.ndarray,
    operation_id: int,
    verbose: bool = False,
    time_stamp: datetime = None,
) -> dict[int, set]:
    """
    Lookups coordinates of given supervoxel in segmentation.
    Finds its counterparts split by chunk boundaries and splits them as a whole.
    Updates the segmentation with new IDs.
    """
    vol_start = cg.meta.voxel_bounds[:, 0]
    vol_end = cg.meta.voxel_bounds[:, 1]
    chunk_size = cg.meta.graph_config.CHUNK_SIZE
    _coords = np.concatenate([source_coords, sink_coords])
    _padding = np.array([cg.meta.resolution[-1] * 2] * 3) / cg.meta.resolution

    bbs = np.clip((np.min(_coords, 0) - _padding).astype(int), vol_start, vol_end)
    bbe = np.clip((np.max(_coords, 0) + _padding).astype(int), vol_start, vol_end)
    chunk_min, chunk_max = bbs // chunk_size, np.ceil(bbe / chunk_size).astype(int)
    bbs, bbe = chunk_min * chunk_size, chunk_max * chunk_size
    logging.info(
        f"cg.meta.ws_ocdbt: {cg.meta.ws_ocdbt.shape}; res {cg.meta.resolution}"
    )
    logging.info(f"chunk and padding {chunk_size}; {_padding}")
    logging.info(f"bbox and chunk min max {(bbs, bbe)}; {(chunk_min, chunk_max)}")

    cut_supervoxels = _get_whole_sv(cg, sv_id, min_coord=chunk_min, max_coord=chunk_max)
    supervoxel_ids = np.array(list(cut_supervoxels), dtype=basetypes.NODE_ID)
    logging.info(f"whole sv {sv_id} -> {supervoxel_ids.tolist()}")

    # one voxel overlap for neighbors
    bbs_ = np.clip(bbs - 1, vol_start, vol_end)
    bbe_ = np.clip(bbe + 1, vol_start, vol_end)
    seg = get_local_segmentation(cg.meta, bbs_, bbe_).squeeze()
    binary_seg = np.isin(seg, supervoxel_ids)
    voxel_overlap_crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
    split_result = split_supervoxel_helper(
        binary_seg[voxel_overlap_crop],
        source_coords - bbs,
        sink_coords - bbs,
        cg.meta.resolution,
        verbose=verbose,
    )
    logging.info(f"split_result: {split_result.shape}")

    chunks_bbox_map = chunks_overlapping_bbox(bbs, bbe, cg.meta.graph_config.CHUNK_SIZE)
    tasks = [
        (cg.graph_id, *item, seg[voxel_overlap_crop], split_result, bbs)
        for item in chunks_bbox_map.items()
    ]
    logging.info(f"tasks count: {len(tasks)}")
    with mp.Pool() as pool:
        results = [*tqdm(pool.imap_unordered(_update_chunk, tasks), total=len(tasks))]
    seg_cropped = seg[voxel_overlap_crop].copy()
    new_seg, old_new_map, slices, new_id_label_map = _parse_results(
        results, seg_cropped, bbs, bbe
    )

    sv_ids = fastremap.unique(seg)
    roots = cg.get_roots(sv_ids)
    sv_root_map = dict(zip(sv_ids, roots))
    root = sv_root_map[sv_id]
    logging.info(f"{sv_id} -> {root}")

    root_mask = fastremap.remap(seg, sv_root_map, in_place=False) == root
    seg[~root_mask] = 0
    sv_ids = fastremap.unique(seg)
    seg[voxel_overlap_crop] = new_seg
    edges_tuple = _update_edges(
        cg,
        sv_ids,
        root,
        np.array([bbs, bbe]),
        seg,
        old_new_map,
        new_id_label_map,
    )

    rows0 = copy_parents_and_add_lineage(cg, operation_id, old_new_map)
    rows1 = _add_new_edges(cg, edges_tuple, time_stamp=time_stamp)
    rows = rows0 + rows1
    logging.info(f"{operation_id}: writing {len(rows)} new rows")

    cg.meta.ws_ocdbt[slices] = new_seg[..., np.newaxis]
    cg.client.write(rows)
    return old_new_map, edges_tuple


def copy_parents_and_add_lineage(
    cg: ChunkedGraph,
    operation_id: int,
    old_new_map: dict,
) -> list:
    """
    Copy parents column from `old_id` to each of `new_ids`.
      This makes it easy to get old hierarchy with `new_ids` using an older timestamp.
    Link `old_id` and `new_ids` to create a lineage at supervoxel layer.
    Returns a list of mutations to be persisted.
    """
    result = []
    parents = set()
    old_new_map = {k: list(v) for k, v in old_new_map.items()}
    parent_cells_map = cg.client.read_nodes(
        node_ids=list(old_new_map.keys()), properties=attributes.Hierarchy.Parent
    )
    for old_id, new_ids in old_new_map.items():
        for new_id in new_ids:
            val_dict = {
                attributes.Hierarchy.FormerIdentity: np.array(
                    [old_id], dtype=basetypes.NODE_ID
                ),
                attributes.OperationLogs.OperationID: operation_id,
            }
            result.append(
                cg.client.mutate_row(serializers.serialize_uint64(new_id), val_dict)
            )
            for cell in parent_cells_map[old_id]:
                cache_utils.update(cg.cache.parents_cache, [new_id], cell.value)
                parents.add(cell.value)
                result.append(
                    cg.client.mutate_row(
                        serializers.serialize_uint64(new_id),
                        {attributes.Hierarchy.Parent: cell.value},
                        time_stamp=cell.timestamp,
                    )
                )
        val_dict = {
            attributes.Hierarchy.NewIdentity: np.array(new_ids, dtype=basetypes.NODE_ID)
        }
        result.append(
            cg.client.mutate_row(serializers.serialize_uint64(old_id), val_dict)
        )

    children_cells_map = cg.client.read_nodes(
        node_ids=list(parents), properties=attributes.Hierarchy.Child
    )
    for parent, children_cells in children_cells_map.items():
        assert len(children_cells) == 1, children_cells
        for cell in children_cells:
            mask = np.isin(cell.value, list(old_new_map.keys()))
            replace = np.concatenate([old_new_map[x] for x in cell.value[mask]])
            children = np.concatenate([cell.value[~mask], replace])
            cg.cache.children_cache[parent] = children
            result.append(
                cg.client.mutate_row(
                    serializers.serialize_uint64(parent),
                    {attributes.Hierarchy.Child: children},
                    time_stamp=cell.timestamp,
                )
            )
    return result
