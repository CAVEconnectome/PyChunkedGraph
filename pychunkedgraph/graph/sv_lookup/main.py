"""Public SV-lookup orchestrator.

See ``README.md`` for the segmentation-agnostic 2D/3D contract this
function enforces.
"""

from typing import Optional
from typing import Sequence

import numpy as np

from .. import exceptions as cg_exceptions
from .utils import lookup_svs_from_seg


def _resolve_3d_with_root(
    cg,
    coords: np.ndarray,
    root_id: np.uint64,
    max_dist_steps: Sequence[float],
) -> np.ndarray:
    """Per-root growing-radius parent-constrained search.

    Used for 3D click coords whose literal seg SV's root != ``root_id``.
    One ``cg.get_atomic_ids_from_coords`` call per radius; breaks on
    first success. Raises ``BadRequest`` if the largest radius fails.
    """
    for max_dist_nm in max_dist_steps:
        resolved = cg.get_atomic_ids_from_coords(
            coords, parent_id=root_id, max_dist_nm=max_dist_nm
        )
        if resolved is not None:
            return np.asarray(resolved, dtype=np.uint64)
    raise cg_exceptions.BadRequest(
        f"Could not determine supervoxel ID for coordinates "
        f"{coords.tolist()} - Lookup stage."
    )


def resolve_supervoxels_at_coords(
    cg,
    coordinates: Sequence[Sequence[int]],
    node_ids: Sequence[np.uint64],
    max_dist_steps: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Resolve voxel coordinates to current supervoxel ids.

    Segmentation-agnostic (OCDBT or precomputed CV; dispatched by
    ``get_local_segmentation``). The constraint never depends on
    caller-supplied node ids beyond their layer:

    - layer 1 (2D slice click) -> accept the literal seg SV.
    - layer >= 2 (3D mesh click, ``node_id`` interpreted as root) ->
      accept the literal SV when its current root matches; otherwise
      run a per-root parent-constrained nearest-SV search.

    Returns ``(N,)`` uint64. Raises ``cg_exceptions.BadRequest`` on
    invalid input or unresolvable coords.
    """
    coordinates = np.asarray(coordinates, dtype=int)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise cg_exceptions.BadRequest(
            f"Could not determine supervoxel ID for coordinates "
            f"{coordinates} - Validation stage."
        )
    node_ids = np.asarray(node_ids, dtype=np.uint64)
    if max_dist_steps is None:
        max_dist_steps = np.array([4, 8, 14, 28], dtype=float) * np.mean(
            cg.meta.resolution
        )

    lit_svs = lookup_svs_from_seg(cg.meta, coordinates)
    layers = cg.get_chunk_layers(node_ids)
    is_3d = layers >= 2
    out = lit_svs.astype(np.uint64, copy=True)

    if not is_3d.any():
        return out

    three_d_idx = np.where(is_3d)[0]
    three_d_lit = lit_svs[three_d_idx]
    nz_in_3d = three_d_lit != 0
    lit_roots_3d = np.zeros(len(three_d_idx), dtype=np.uint64)
    if nz_in_3d.any():
        lit_roots_3d[nz_in_3d] = cg.get_roots(three_d_lit[nz_in_3d], fail_to_zero=True)

    needs_search = lit_roots_3d != node_ids[three_d_idx]
    if not needs_search.any():
        return out

    search_idx_in_3d = np.where(needs_search)[0]
    search_node_ids = node_ids[three_d_idx][search_idx_in_3d]
    search_coords = coordinates[three_d_idx][search_idx_in_3d]

    for root_id in np.unique(search_node_ids):
        root_mask = search_node_ids == root_id
        sub_search_coords = search_coords[root_mask]
        resolved = _resolve_3d_with_root(cg, sub_search_coords, root_id, max_dist_steps)
        out_positions = three_d_idx[search_idx_in_3d[root_mask]]
        out[out_positions] = resolved
    return out
