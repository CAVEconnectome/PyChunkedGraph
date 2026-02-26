"""
Utils functions for node and segment IDs.
"""

from typing import Optional
from typing import Sequence
from typing import Callable
from datetime import datetime

import numpy as np

from . import basetypes
from .generic import get_local_segmentation
from ..meta import ChunkedGraphMeta
from ..chunks import utils as chunk_utils


def get_segment_id_limit(
    meta: ChunkedGraphMeta, node_or_chunk_id: basetypes.CHUNK_ID
) -> basetypes.SEGMENT_ID:
    """Get maximum possible Segment ID for given Node ID or Chunk ID."""
    layer = chunk_utils.get_chunk_layer(meta, node_or_chunk_id)
    chunk_offset = 64 - meta.graph_config.LAYER_ID_BITS - 3 * meta.bitmasks[layer]
    return np.uint64(2 ** chunk_offset - 1)


def get_segment_id(
    meta: ChunkedGraphMeta, node_id: basetypes.NODE_ID
) -> basetypes.SEGMENT_ID:
    """Extract Segment ID from Node ID."""
    return node_id & get_segment_id_limit(meta, node_id)


def get_node_id(
    meta: ChunkedGraphMeta,
    segment_id: basetypes.SEGMENT_ID,
    chunk_id: basetypes.CHUNK_ID = None,
    layer: int = None,
    x: int = None,
    y: int = None,
    z: int = None,
) -> basetypes.NODE_ID:
    """
    (1) Build Node ID from Segment ID and Chunk ID
    (2) Build Node ID from Segment ID, Layer, X, Y and Z components
    """
    if chunk_id is not None:
        return chunk_id | segment_id
    else:
        return chunk_utils.get_chunk_id(meta, layer=layer, x=x, y=y, z=z) | segment_id


def get_atomic_id_from_coord(
    meta: ChunkedGraphMeta,
    get_root: callable,
    x: int,
    y: int,
    z: int,
    parent_id: np.uint64,
    n_tries: int = 5,
    time_stamp: Optional[datetime] = None,
) -> np.uint64:
    """Determines atomic id given a coordinate."""
    x = int(x / 2 ** meta.data_source.CV_MIP)
    y = int(y / 2 ** meta.data_source.CV_MIP)
    z = int(z)

    checked = []
    atomic_id = None
    root_id = get_root(parent_id, time_stamp=time_stamp)

    for i_try in range(n_tries):
        # Define block size -- increase by one each try
        x_l = x - (i_try - 1) ** 2
        y_l = y - (i_try - 1) ** 2
        z_l = z - (i_try - 1) ** 2

        x_h = x + 1 + (i_try - 1) ** 2
        y_h = y + 1 + (i_try - 1) ** 2
        z_h = z + 1 + (i_try - 1) ** 2

        x_l = 0 if x_l < 0 else x_l
        y_l = 0 if y_l < 0 else y_l
        z_l = 0 if z_l < 0 else z_l

        # Get atomic ids from cloudvolume
        atomic_id_block = meta.cv[x_l:x_h, y_l:y_h, z_l:z_h]
        atomic_ids, atomic_id_count = np.unique(atomic_id_block, return_counts=True)

        # sort by frequency and discard those ids that have been checked
        # previously
        sorted_atomic_ids = atomic_ids[np.argsort(atomic_id_count)]
        sorted_atomic_ids = sorted_atomic_ids[~np.isin(sorted_atomic_ids, checked)]

        # For each candidate id check whether its root id corresponds to the
        # given root id
        for candidate_atomic_id in sorted_atomic_ids:
            if candidate_atomic_id != 0:
                ass_root_id = get_root(candidate_atomic_id, time_stamp=time_stamp)
                if ass_root_id == root_id:
                    # atomic_id is not None will be our indicator that the
                    # search was successful
                    atomic_id = candidate_atomic_id
                    break
                else:
                    checked.append(candidate_atomic_id)
        if atomic_id is not None:
            break
    # Returns None if unsuccessful
    return atomic_id


def get_atomic_ids_from_coords(
    meta: ChunkedGraphMeta,
    coordinates: Sequence[Sequence[int]],
    parent_id: np.uint64,
    parent_id_layer: int,
    parent_ts: datetime,
    get_roots: Callable,
    max_dist_nm: int = 150,
) -> Sequence[np.uint64]:
    """Retrieves supervoxel ids for multiple coords.

    :param coordinates: n x 3 np.ndarray of locations in voxel space
    :param parent_id: parent id common to all coordinates at any layer
    :param max_dist_nm: max distance explored
    :return: supervoxel ids; returns None if no solution was found
    """
    import fastremap

    if parent_id_layer == 1:
        return np.array([parent_id] * len(coordinates), dtype=np.uint64)

    coordinates_nm = coordinates * np.array(meta.resolution)
    # Define bounding box to be explored
    max_dist_vx = np.ceil(max_dist_nm / meta.resolution).astype(dtype=np.int32)
    bbox = np.array(
        [
            np.min(coordinates, axis=0) - max_dist_vx,
            np.max(coordinates, axis=0) + max_dist_vx + 1,
        ]
    )

    local_sv_seg = get_local_segmentation(meta, bbox[0], bbox[1]).squeeze()
    # limit get_roots calls to the relevant areas of the data
    lower_bs = np.floor(
        (np.array(coordinates_nm) - max_dist_nm) / np.array(meta.resolution) - bbox[0]
    ).astype(np.int32)
    upper_bs = np.ceil(
        (np.array(coordinates_nm) + max_dist_nm) / np.array(meta.resolution) - bbox[0]
    ).astype(np.int32)
    local_sv_ids = []
    for lb, ub in zip(lower_bs, upper_bs):
        local_sv_ids.extend(
            fastremap.unique(local_sv_seg[lb[0] : ub[0], lb[1] : ub[1], lb[2] : ub[2]])
        )
    local_sv_ids = fastremap.unique(np.array(local_sv_ids, dtype=np.uint64))
    local_parent_ids = get_roots(
        local_sv_ids,
        time_stamp=parent_ts,
        stop_layer=parent_id_layer,
        fail_to_zero=True
    )

    local_parent_seg = fastremap.remap(
        local_sv_seg,
        dict(zip(local_sv_ids, local_parent_ids)),
        preserve_missing_labels=True,
    )

    parent_id_locs_vx = np.array(np.where(local_parent_seg == parent_id)).T
    if len(parent_id_locs_vx) == 0:
        return None

    parent_id_locs_nm = (parent_id_locs_vx + bbox[0]) * np.array(meta.resolution)
    # find closest supervoxel ids and check that they are closer than the limit
    dist_mat = np.sqrt(
        np.sum((parent_id_locs_nm[:, None] - coordinates_nm) ** 2, axis=-1)
    )
    match_ids = np.argmin(dist_mat, axis=0)
    matched_dists = np.array([dist_mat[idx, i] for i, idx in enumerate(match_ids)])
    if np.any(matched_dists > max_dist_nm):
        return None

    local_coords = parent_id_locs_vx[match_ids]
    matched_sv_ids = [local_sv_seg[tuple(c)] for c in local_coords]
    return matched_sv_ids
