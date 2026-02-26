"""
Manage new supervoxels after a supervoxel split.
"""

from functools import reduce
import logging
import multiprocessing as mp
from typing import Callable, Iterable
from datetime import datetime
from collections import defaultdict, deque

import fastremap
import numpy as np
from tqdm import tqdm
from pychunkedgraph.graph import ChunkedGraph, cache as cache_utils
from pychunkedgraph.graph.attributes import Connectivity
from pychunkedgraph.graph.chunks.utils import chunks_overlapping_bbox, get_neighbors
from pychunkedgraph.graph.cutting_sv import (
    build_kdtrees_by_label,
    pairwise_min_distance_two_sets,
    split_supervoxel_helper,
)
from pychunkedgraph.graph.attributes import Hierarchy, OperationLogs
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.types import empty_2d
from pychunkedgraph.graph.utils import basetypes
from pychunkedgraph.graph.utils import get_local_segmentation
from pychunkedgraph.graph.utils.serializers import serialize_uint64
from pychunkedgraph.io.edges import get_chunk_edges


def _get_whole_sv(
    cg: ChunkedGraph, node: basetypes.NODE_ID, min_coord, max_coord
) -> set:
    cx_edges = [empty_2d]
    explored_chunks = set()
    explored_nodes = set([node])
    queue = deque([node])

    while len(queue) > 0:
        vertex = queue.popleft()
        chunk = cg.get_chunk_coordinates(vertex)
        chunks = get_neighbors(chunk, min_coord=min_coord, max_coord=max_coord)

        unexplored_chunks = []
        for _chunk in chunks:
            if tuple(_chunk) not in explored_chunks:
                unexplored_chunks.append(tuple(_chunk))

        edges = get_chunk_edges(cg.meta.data_source.EDGES, unexplored_chunks)
        explored_chunks.update(unexplored_chunks)
        _cx_edges = edges["cross"].get_pairs()
        cx_edges.append(_cx_edges)
        _cx_edges = np.concatenate(cx_edges)

        mask = _cx_edges[:, 0] == vertex
        neighbors = _cx_edges[mask][:, 1]

        if len(neighbors) > 0:
            neighbor_coords = cg.get_chunk_coordinates_multiple(neighbors)
            min_mask = (neighbor_coords >= min_coord).all(axis=1)
            max_mask = (neighbor_coords < max_coord).all(axis=1)
            neighbors = neighbors[min_mask & max_mask]

        for neighbor in neighbors:
            if neighbor in explored_nodes:
                continue
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

    # TODO: remove these 3 lines, testing only
    rr = cg.range_read_chunk(chunk_id)
    max_node_id = max(rr.keys())
    cg.id_client.set_max_node_id(chunk_id, max_node_id)

    _s, _e = chunk_bbox - bb_start
    og_chunk_seg = seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]
    chunk_seg = result_seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]

    labels = fastremap.unique(chunk_seg[chunk_seg != 0])
    if labels.size < 2:
        return None

    _indices = []
    _old_values = []
    _new_values = []
    for _id in labels:
        _mask = chunk_seg == _id
        if np.any(_mask):
            _idx = np.unravel_index(np.flatnonzero(_mask)[0], og_chunk_seg.shape)
            _og_value = og_chunk_seg[_idx]
            _index = np.argwhere(_mask)
            _indices.append(_index)
            _ones = np.ones(len(_index), dtype=basetypes.NODE_ID)
            _old_values.append(_ones * _og_value)
            _new_values.append(_ones * cg.id_client.create_node_id(chunk_id))

    _indices = np.concatenate(_indices) + (chunk_bbox[0] - bb_start)
    _old_values = np.concatenate(_old_values)
    _new_values = np.concatenate(_new_values)
    return (_indices, _old_values, _new_values)


def _voxel_crop(bbs, bbe, bbs_, bbe_):
    xS, yS, zS = bbs - bbs_
    xE, yE, zE = (None if i == 0 else -1 for i in bbe_ - bbe)
    voxel_overlap_crop = np.s_[xS:xE, yS:yE, zS:zE]
    logging.info(f"voxel_overlap_crop: {voxel_overlap_crop}")
    return voxel_overlap_crop


def _parse_results(results, seg, bbs, bbe):
    old_new_map = defaultdict(set)
    for result in results:
        if result:
            indexer, old_values, new_values = result
            seg[tuple(indexer.T)] = new_values
            for old_sv, new_sv in zip(old_values, new_values):
                old_new_map[old_sv].add(new_sv)

    assert np.all(seg.shape == bbe - bbs), f"{seg.shape} != {bbe - bbs}"
    slices = tuple(slice(start, end) for start, end in zip(bbs, bbe)) + (slice(None),)
    logging.info(f"slices {slices}")
    return seg, old_new_map, slices


def _get_new_edges(
    edges_info: tuple,
    sv_ids: np.ndarray,
    old_new_map: dict,
    distances: np.ndarray,
    dist_vec: Callable,
    new_dist_vec: Callable,
):
    THRESHOLD = 10
    new_edges, new_affs, new_areas = [], [], []
    edges, affinities, areas = edges_info

    for old, new in old_new_map.items():
        logging.info(f"old and new {old, new}")
        new_ids = np.array(list(new), dtype=basetypes.NODE_ID)
        edges_m = np.any(edges == old, axis=1)
        selected_edges = edges[edges_m]
        sel_m = selected_edges != old
        assert np.all(np.sum(sel_m, axis=1) == 1)

        partners = selected_edges[sel_m]
        active_m = np.isin(partners, sv_ids)

        logging.info(f"sv_ids: {np.sum(sv_ids > 0)}")
        logging.info(f"edges: {edges.shape} {np.sum(edges_m)} {np.sum(sel_m)}")
        logging.info(f"selected_edges: {selected_edges.shape}")

        # inactive
        for new_id in new_ids:
            _a = [[new_id] * np.sum(~active_m), partners[~active_m]]
            new_edges.extend(np.array(_a, dtype=np.uint64).T)
            new_affs.extend(affinities[edges_m][np.any(sel_m, axis=1)][~active_m])
            new_areas.extend(areas[edges_m][np.any(sel_m, axis=1)][~active_m])

        # active
        active_partners_ = partners[active_m]
        active_affs_ = affinities[edges_m][np.any(sel_m, axis=1)][active_m]
        active_areas_ = areas[edges_m][np.any(sel_m, axis=1)][active_m]

        logging.info(f"partners: {partners.shape} {active_partners_.shape}")

        active_partners = []
        active_affs = []
        active_areas = []
        for i in range(len(active_partners_)):
            remapped_ = old_new_map.get(active_partners_[i], [active_partners_[i]])
            active_partners.extend(remapped_)
            active_affs.extend([active_affs_[i]] * len(remapped_))
            active_areas.extend([active_areas_[i]] * len(remapped_))

        logging.info(f"new_ids, active_partners: {new_ids, len(active_partners)}")
        logging.info(f"new_dist_vec(new_ids): {new_dist_vec(new_ids)}")
        logging.info(f"dist_vec(active_partners): {dist_vec(active_partners)}")
        distances_ = distances[new_dist_vec(new_ids)][:, dist_vec(active_partners)].T
        for i, _ in enumerate(active_partners):
            new_ids_ = new_ids[distances_[i] < THRESHOLD]
            if len(new_ids_):
                _a = [new_ids_, [active_partners[i]] * len(new_ids_)]
                new_edges.extend(np.array(_a, dtype=np.uint64).T)
                new_affs.extend([active_affs[i]] * len(new_ids_))
                new_areas.extend([active_areas[i]] * len(new_ids_))
            else:
                close_new_sv_id = new_ids[np.argmin(distances_[i])]
                _a = [close_new_sv_id, active_partners[i]]
                new_edges.append(np.array(_a, dtype=np.uint64))
                new_affs.append(active_affs[i])
                new_areas.append(active_areas[i])

        # edges between split fragments
        for i in range(len(new_ids)):
            for j in range(i + 1, len(new_ids)):  # includes no selfedges
                _a = [new_ids[i], new_ids[j]]
                new_edges.append(np.array(_a, dtype=np.uint64))
                new_affs.append(0.001)
                new_areas.append(0)

    affinites = np.array(new_affs, dtype=basetypes.EDGE_AFFINITY)
    areas = np.array(new_areas, dtype=basetypes.EDGE_AREA)
    edges = np.array(new_edges, dtype=basetypes.NODE_ID)
    edges, idx = np.unique(edges, return_index=True, axis=0)
    return edges, affinites[idx], areas[idx]


def _update_edges(
    cg: ChunkedGraph,
    sv_ids: np.ndarray,
    root_id: basetypes.NODE_ID,
    bbox: np.ndarray,
    new_seg: np.ndarray,
    old_new_map: dict,
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
    logging.info(f"edges.shape, affinities.shape {edges.shape, affinities.shape}")

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
        val_dict[Connectivity.SplitEdges] = edges[mask]
        val_dict[Connectivity.Affinity] = affinites[mask]
        val_dict[Connectivity.Area] = areas[mask]
        rows.append(
            cg.client.mutate_row(
                serialize_uint64(chunk_id, fake_edges=True),
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
    verbose: bool = True,
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
    _padding = np.array([64] * 3) / cg.meta.resolution

    bbs = np.clip((np.min(_coords, 0) - _padding).astype(int), vol_start, vol_end)
    bbe = np.clip((np.max(_coords, 0) + _padding).astype(int), vol_start, vol_end)
    chunk_min, chunk_max = bbs // chunk_size, np.ceil(bbe / chunk_size).astype(int)
    bbs, bbe = chunk_min * chunk_size, chunk_max * chunk_size
    logging.info(f"cg.meta.ws_ocdbt: {cg.meta.ws_ocdbt.shape}")
    logging.info(f"{chunk_size}; {_padding}; {(bbs, bbe)}; {(chunk_min, chunk_max)}")

    cut_supervoxels = _get_whole_sv(cg, sv_id, min_coord=chunk_min, max_coord=chunk_max)
    supervoxel_ids = np.array(list(cut_supervoxels), dtype=basetypes.NODE_ID)
    logging.info(f"{sv_id} -> {cut_supervoxels}")

    # one voxel overlap for neighbors
    bbs_ = np.clip(bbs - 1, vol_start, vol_end)
    bbe_ = np.clip(bbe + 1, vol_start, vol_end)
    seg = get_local_segmentation(cg.meta, bbs_, bbe_).squeeze()
    binary_seg = np.isin(seg, supervoxel_ids)
    logging.info(f"{seg.shape}; {binary_seg.shape}; {bbs, bbe}; {bbs_, bbe_}")

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
    new_seg, old_new_map, slices = _parse_results(results, seg_cropped, bbs, bbe)

    seg_roots = seg.copy()
    sv_ids = fastremap.unique(seg)
    roots = cg.get_roots(sv_ids)
    seg_roots = fastremap.remap(seg_roots, dict(zip(sv_ids, roots)), in_place=True)

    root = cg.get_root(sv_id)
    logging.info(f"root {root}")

    seg_masked = seg.copy()
    seg_masked[seg_roots != root] = 0
    sv_ids = fastremap.unique(seg_masked)

    seg_masked[voxel_overlap_crop] = new_seg
    edges_tuple = _update_edges(
        cg, sv_ids, root, np.array([bbs, bbe]), seg_masked, old_new_map
    )

    rows0 = copy_parents_and_add_lineage(cg, operation_id, old_new_map)
    rows1 = _add_new_edges(cg, edges_tuple, time_stamp=time_stamp)
    rows = rows0 + rows1
    logging.info(f"{operation_id}: writing {len(rows)} new rows")

    cg.client.write(rows)
    cg.meta.ws_ocdbt[slices] = new_seg[..., np.newaxis]
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
        node_ids=list(old_new_map.keys()), properties=Hierarchy.Parent
    )
    for old_id, new_ids in old_new_map.items():
        for new_id in new_ids:
            val_dict = {
                Hierarchy.FormerIdentity: np.array([old_id], dtype=basetypes.NODE_ID),
                OperationLogs.OperationID: operation_id,
            }
            result.append(cg.client.mutate_row(serialize_uint64(new_id), val_dict))
            for cell in parent_cells_map[old_id]:
                cache_utils.update(cg.cache.parents_cache, [new_id], cell.value)
                parents.add(cell.value)
                result.append(
                    cg.client.mutate_row(
                        serialize_uint64(new_id),
                        {Hierarchy.Parent: cell.value},
                        time_stamp=cell.timestamp,
                    )
                )
        val_dict = {Hierarchy.NewIdentity: np.array(new_ids, dtype=basetypes.NODE_ID)}
        result.append(cg.client.mutate_row(serialize_uint64(old_id), val_dict))

    children_cells_map = cg.client.read_nodes(
        node_ids=list(parents), properties=Hierarchy.Child
    )
    for parent, children_cells in children_cells_map.items():
        assert len(children_cells) == 1, children_cells
        for cell in children_cells:
            logging.info(f"{parent}: {cell.value}")
            mask = np.isin(cell.value, list(old_new_map.keys()))
            replace = np.concatenate([old_new_map[x] for x in cell.value[mask]])
            children = np.concatenate([cell.value[~mask], replace])
            logging.info(f"{parent}: {children}")
            cg.cache.children_cache[parent] = children
            result.append(
                cg.client.mutate_row(
                    serialize_uint64(parent),
                    {Hierarchy.Child: children},
                    time_stamp=cell.timestamp,
                )
            )
    return result
