"""Per-label coordinate utilities used by edge routing after a SV split."""

from typing import Dict, Iterable, Optional

import fastremap
import numpy as np


def build_coords_by_label(
    vol: np.ndarray,
    *,
    labels: Optional[Iterable[int]] = None,
    background: int = 0,
    min_points: int = 1,
    dtype: np.dtype = np.float32,
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
    """
    if vol.ndim != 3:
        raise ValueError("`vol` must be a 3D array.")
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
