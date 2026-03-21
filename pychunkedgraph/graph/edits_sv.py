"""
Manage new supervoxels after a supervoxel split.
"""

import time
from datetime import datetime
from collections import defaultdict, deque

import fastremap
import numpy as np

from pychunkedgraph import get_logger
from pychunkedgraph.graph import (
    attributes,
    ChunkedGraph,
    cache as cache_utils,
    basetypes,
    serializers,
)
from pychunkedgraph.graph.chunks.utils import chunks_overlapping_bbox
from pychunkedgraph.graph.cutting_sv import split_supervoxel_helper
from pychunkedgraph.graph.edges_sv import update_edges, add_new_edges
from pychunkedgraph.graph.utils import get_local_segmentation
from pychunkedgraph.io.edges import get_chunk_edges

logger = get_logger(__name__)


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


def _update_chunks(cg, chunks_bbox_map, seg, result_seg, bb_start):
    """Process all chunks in a single pass: assign new SV IDs to split fragments.

    For each chunk overlapping the split bbox, finds split labels and
    batch-allocates new IDs. No multiprocessing needed.
    """
    results = []
    for chunk_coord, chunk_bbox in chunks_bbox_map.items():
        x, y, z = chunk_coord
        chunk_id = cg.get_chunk_id(layer=1, x=x, y=y, z=z)

        _s, _e = chunk_bbox - bb_start
        og_chunk_seg = seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]
        chunk_seg = result_seg[_s[0] : _e[0], _s[1] : _e[1], _s[2] : _e[2]]

        labels = fastremap.unique(chunk_seg[chunk_seg != 0])
        if labels.size < 2:
            continue

        new_ids = cg.id_client.create_node_ids(chunk_id, size=len(labels))
        _indices = []
        _old_values = []
        _new_values = []
        _label_id_map = {}
        for _id, new_id in zip(labels, new_ids):
            _mask = chunk_seg == _id
            voxel_locs = np.where(_mask)
            _og_value = og_chunk_seg[
                voxel_locs[0][0], voxel_locs[1][0], voxel_locs[2][0]
            ]
            _index = np.column_stack(voxel_locs)
            n = len(_index)
            _indices.append(_index)
            _old_values.append(np.full(n, _og_value, dtype=basetypes.NODE_ID))
            _new_values.append(np.full(n, new_id, dtype=basetypes.NODE_ID))
            _label_id_map[int(_id)] = new_id

        _indices = np.concatenate(_indices) + (chunk_bbox[0] - bb_start)
        _old_values = np.concatenate(_old_values)
        _new_values = np.concatenate(_new_values)
        results.append((_indices, _old_values, _new_values, _label_id_map))
    return results


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
    logger.note(f"cg.meta.ws_ocdbt: {cg.meta.ws_ocdbt.shape}; res {cg.meta.resolution}")
    logger.note(f"chunk and padding {chunk_size}; {_padding}")
    logger.note(f"bbox and chunk min max {(bbs, bbe)}; {(chunk_min, chunk_max)}")

    t0 = time.time()
    cut_supervoxels = _get_whole_sv(cg, sv_id, min_coord=chunk_min, max_coord=chunk_max)
    supervoxel_ids = np.array(list(cut_supervoxels), dtype=basetypes.NODE_ID)
    logger.note(
        f"whole sv {sv_id} -> {supervoxel_ids.tolist()} ({time.time() - t0:.2f}s)"
    )

    # one voxel overlap for neighbors
    bbs_ = np.clip(bbs - 1, vol_start, vol_end)
    bbe_ = np.clip(bbe + 1, vol_start, vol_end)
    t0 = time.time()
    seg = get_local_segmentation(cg.meta, bbs_, bbe_).squeeze()
    logger.note(f"segmentation read {seg.shape} ({time.time() - t0:.2f}s)")

    binary_seg = np.isin(seg, supervoxel_ids)
    voxel_overlap_crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
    t0 = time.time()
    split_result = split_supervoxel_helper(
        binary_seg[voxel_overlap_crop],
        source_coords - bbs,
        sink_coords - bbs,
        cg.meta.resolution,
        verbose=verbose,
    )
    logger.note(f"split computation {split_result.shape} ({time.time() - t0:.2f}s)")

    chunks_bbox_map = chunks_overlapping_bbox(bbs, bbe, cg.meta.graph_config.CHUNK_SIZE)
    t0 = time.time()
    results = _update_chunks(
        cg, chunks_bbox_map, seg[voxel_overlap_crop], split_result, bbs
    )
    logger.note(
        f"chunk updates {len(chunks_bbox_map)} chunks, {len(results)} with splits ({time.time() - t0:.2f}s)"
    )

    seg_cropped = seg[voxel_overlap_crop].copy()
    new_seg, old_new_map, slices, new_id_label_map = _parse_results(
        results, seg_cropped, bbs, bbe
    )

    sv_ids = fastremap.unique(seg)
    roots = cg.get_roots(sv_ids)
    sv_root_map = dict(zip(sv_ids, roots))
    root = sv_root_map[sv_id]
    logger.note(f"{sv_id} -> {root}")

    root_mask = fastremap.remap(seg, sv_root_map, in_place=False) == root
    seg[~root_mask] = 0
    sv_ids = fastremap.unique(seg)
    seg[voxel_overlap_crop] = new_seg
    t0 = time.time()
    edges_tuple = update_edges(
        cg,
        root,
        np.array([bbs, bbe]),
        seg,
        old_new_map,
        new_id_label_map,
    )
    logger.note(f"edge update ({time.time() - t0:.2f}s)")

    rows0 = copy_parents_and_add_lineage(cg, operation_id, old_new_map)
    rows1 = add_new_edges(cg, edges_tuple, time_stamp=time_stamp)
    rows = rows0 + rows1

    t0 = time.time()
    cg.meta.ws_ocdbt[slices] = new_seg[..., np.newaxis]
    cg.client.write(rows)
    logger.note(f"write seg + {len(rows)} rows ({time.time() - t0:.2f}s)")
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
