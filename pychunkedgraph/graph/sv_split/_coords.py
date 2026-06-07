"""Per-label coordinate utilities used by edge routing after a SV split."""

from typing import Dict, Iterable, Optional

import fastremap
import numpy as np

from pychunkedgraph.profiler import get_profiler

_prof = get_profiler()


def _label_boundary_mask(vol: np.ndarray) -> np.ndarray:
    """6-conn neighbor-differs mask. Background voxels may also be True
    where adjacent to foreground; the consumer multiplies vol in place
    and background is already 0, so the extra True bits are no-ops.
    """
    diff = np.zeros(vol.shape, dtype=bool)
    dz = vol[1:] != vol[:-1]
    diff[1:] |= dz
    diff[:-1] |= dz
    del dz
    dy = vol[:, 1:] != vol[:, :-1]
    diff[:, 1:] |= dy
    diff[:, :-1] |= dy
    del dy
    dx = vol[:, :, 1:] != vol[:, :, :-1]
    diff[:, :, 1:] |= dx
    diff[:, :, :-1] |= dx
    del dx
    return diff


def build_coords_by_label(
    vol: np.ndarray,
    *,
    labels: Optional[Iterable[int]] = None,
    background: int = 0,
    min_points: int = 1,
    dtype: np.dtype = np.float32,
    boundary_only: bool = False,
) -> Dict[int, np.ndarray]:
    """Group voxel coords by label via ``fastremap.point_cloud``.

    Returns ``{label: (M_label, 3) coords in (z, y, x)}`` cast to
    ``dtype``. ``fastremap.point_cloud`` is a C++ single-pass
    implementation that emits ``uint16`` coords grouped by label,
    treating ``0`` as background.

    ``labels`` restricts the output dict; the underlying C++ scan
    visits every voxel regardless (faster and lighter than a
    label-filtered Python scan). ``min_points`` drops labels with
    fewer than that many voxels. ``background != 0`` removes that
    label from the result after the call.

    ``boundary_only=True`` returns only 6-conn boundary voxels per
    label and **zeros non-boundary entries in `vol` in place** to
    avoid a full-size copy. min-distance between any two labels'
    boundary point sets equals min-distance between their interior
    point sets, so this is correctness-preserving for nearest-neighbor
    consumers.
    """
    if vol.ndim != 3:
        raise ValueError("`vol` must be a 3D array.")
    if boundary_only:
        with _prof.profile("boundary_mask"):
            mask = _label_boundary_mask(vol)
        with _prof.profile("apply_mask"):
            vol *= mask
            del mask
    with _prof.profile("point_cloud"):
        raw = fastremap.point_cloud(vol)
    if background != 0:
        raw.pop(background, None)

    if labels is not None:
        wanted = {int(x) for x in labels}
        if not wanted:
            return {}
        items = ((k, v) for k, v in raw.items() if int(k) in wanted)
    else:
        items = raw.items()

    result: Dict[int, np.ndarray] = {}
    for k, coords in items:
        if coords.shape[0] < min_points:
            continue
        result[int(k)] = coords.astype(dtype, copy=False)
    return result
