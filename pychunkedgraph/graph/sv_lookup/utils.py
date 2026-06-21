"""Low-level segmentation reads and parent-constrained nearest-SV search.

See ``README.md`` for the overall contract. These primitives are
segmentation-agnostic: they go through ``get_local_segmentation`` which
dispatches to OCDBT or CloudVolume transparently.
"""

from typing import Optional
from typing import Sequence
from typing import Callable
from datetime import datetime

import numpy as np
import fastremap

from ..meta import ChunkedGraphMeta
from ..utils.generic import get_local_segmentation


def lookup_svs_from_seg(meta: ChunkedGraphMeta, coordinates) -> np.ndarray:
    """Read SV IDs at the given voxel coordinates.

    One batched seg read over the coords' bounding box; returns the
    literal SV at each coord (0 for background).
    """
    coordinates = np.asarray(coordinates, dtype=int)
    bbox_start = coordinates.min(axis=0)
    bbox_end = coordinates.max(axis=0) + 1
    seg = get_local_segmentation(meta, bbox_start, bbox_end)[..., 0]
    local = coordinates - bbox_start
    return seg[local[:, 0], local[:, 1], local[:, 2]].astype(np.uint64)


def get_atomic_id_from_coord(
    meta: ChunkedGraphMeta,
    get_root: Callable,
    x: int,
    y: int,
    z: int,
    parent_id: np.uint64,
    n_tries: int = 5,
    time_stamp: Optional[datetime] = None,
) -> np.uint64:
    """Determines atomic id given a coordinate."""
    x = int(x / 2**meta.data_source.CV_MIP)
    y = int(y / 2**meta.data_source.CV_MIP)
    z = int(z)
    xyz = np.array([x, y, z])

    checked = []
    atomic_id = None
    root_id = get_root(parent_id, time_stamp=time_stamp)

    for i_try in range(n_tries):
        r = (i_try - 1) ** 2
        lo = np.maximum(xyz - r, 0)
        hi = xyz + r + 1
        atomic_id_block = (
            meta.ws_ts[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].read().result()
        )
        atomic_ids, atomic_id_count = np.unique(atomic_id_block, return_counts=True)

        sorted_atomic_ids = atomic_ids[np.argsort(atomic_id_count)]
        sorted_atomic_ids = sorted_atomic_ids[~np.isin(sorted_atomic_ids, checked)]

        for candidate_atomic_id in sorted_atomic_ids:
            if candidate_atomic_id != 0:
                ass_root_id = get_root(candidate_atomic_id, time_stamp=time_stamp)
                if ass_root_id == root_id:
                    atomic_id = candidate_atomic_id
                    break
                else:
                    checked.append(candidate_atomic_id)
        if atomic_id is not None:
            break
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
    """Parent-constrained nearest-SV search over multiple coords.

    Reads one bbox-sized seg block around the coords, masks every voxel
    to its root via ``get_roots``, then picks the nm-closest voxel whose
    root matches ``parent_id`` for each input coord. Returns ``None`` if
    no voxel within ``max_dist_nm`` matches.

    :param coordinates: n x 3 np.ndarray of locations in voxel space
    :param parent_id: parent id common to all coordinates at any layer
    :param max_dist_nm: max distance explored
    """
    if parent_id_layer == 1:
        return np.array([parent_id] * len(coordinates), dtype=np.uint64)

    coordinates_nm = coordinates * np.array(meta.resolution)
    max_dist_vx = np.ceil(max_dist_nm / meta.resolution).astype(dtype=np.int32)
    bbox = np.array(
        [
            np.min(coordinates, axis=0) - max_dist_vx,
            np.max(coordinates, axis=0) + max_dist_vx + 1,
        ]
    )

    local_sv_seg = get_local_segmentation(meta, bbox[0], bbox[1]).squeeze()
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
        fail_to_zero=True,
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
