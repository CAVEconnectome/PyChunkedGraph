"""Per-label coordinate utilities used by edge routing after a SV split."""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import fastremap
import numpy as np
from scipy.spatial import cKDTree


def build_kdtrees_by_label(
    vol: np.ndarray,
    *,
    background: int = 0,
    leafsize: int = 16,
    balanced_tree: bool = True,
    compact_nodes: bool = True,
    min_points: int = 1,
    dtype: np.dtype = np.float32,
) -> Tuple[Dict[int, cKDTree], Dict[int, int]]:
    """
    Build a cKDTree of voxel coordinates for every unique (non-background) label in a 3D volume.

    Parameters
    ----------
    vol : np.ndarray
        3D label volume (e.g., shape (Z, Y, X)). Can be any integer dtype (incl. uint64).
    background : int, default 0
        Label treated as background and skipped.
    leafsize : int, default 16
        Passed to cKDTree (larger can be faster for queries on large trees).
    balanced_tree : bool, default True
        Passed to cKDTree.
    compact_nodes : bool, default True
        Passed to cKDTree.
    min_points : int, default 1
        Skip labels with fewer than this many voxels.
    dtype : np.dtype, default np.float32
        Coordinate dtype used to build the trees (lower memory than float64).

    Returns
    -------
    trees : Dict[int, cKDTree]
        Mapping label -> cKDTree built
        from the (z, y, x) coordinates of that label’s voxels.
    counts : Dict[int, int]
        Mapping label -> number of voxels used to build the tree.

    Notes
    -----
    - This runs in O(N log N) due to a single sort over N foreground voxels.
    - Uses one pass over non-background voxels; avoids per-label boolean masking.
    - Coordinates are (z, y, x) in voxel units.
    """
    if vol.ndim != 3:
        raise ValueError("`vol` must be a 3D array.")
    Z, Y, X = vol.shape

    # Flatten once and select foreground voxels
    flat = vol.ravel()
    if background == 0:
        nz = np.flatnonzero(flat)  # fast path when background is 0
    else:
        nz = np.flatnonzero(flat != background)

    if nz.size == 0:
        return {}, {}

    # Labels of foreground voxels (kept as integer/uint64)
    labels = flat[nz]

    # Coordinates for those voxels (computed once)
    z, y, x = np.unravel_index(nz, (Z, Y, X))
    coords = np.column_stack((z, y, x)).astype(dtype, copy=False)

    # Group by label via sort (stable to preserve any incidental ordering)
    order = np.argsort(labels, kind="mergesort")
    labels_sorted = labels[order]

    # Find group boundaries (run-length encoding over sorted labels)
    starts = np.flatnonzero(np.r_[True, labels_sorted[1:] != labels_sorted[:-1]])
    ends = np.r_[starts[1:], labels_sorted.size]

    trees: Dict[int, cKDTree] = {}
    counts: Dict[int, int] = {}

    for s, e in zip(starts, ends):
        lab = int(labels_sorted[s])  # Python int key (handles uint64 safely)
        block = coords[order[s:e]]
        n = block.shape[0]
        if n < min_points:
            continue
        # cKDTree copies data into its own memory; no need to keep `block` afterwards.
        trees[lab] = cKDTree(
            block,
            leafsize=leafsize,
            balanced_tree=balanced_tree,
            compact_nodes=compact_nodes,
        )
        counts[lab] = n

    return trees, counts


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


def pairwise_min_distance_two_sets(
    trees_a: Sequence[cKDTree],
    trees_b: Sequence[cKDTree],
    *,
    max_distance: Optional[float] = None,
    workers: int = -1,
) -> np.ndarray:
    """
    Compute pairwise shortest distances between point sets represented by two lists
    of cKDTrees. Result has shape (len(trees_a), len(trees_b)).

    Parameters
    ----------
    trees_a, trees_b : sequences of cKDTree
        Each tree encodes the (z,y,x) points for one segment.
    max_distance : float or None
        If None (default): compute exact min distances (dense, finite).
        If set: compute within this cutoff using sparse_distance_matrix; pairs with
        no neighbors within cutoff are set to np.inf.
    workers : int
        Parallelism for cKDTree.query (SciPy >= 1.6). -1 uses all cores.

    Returns
    -------
    D : ndarray, shape (len(trees_a), len(trees_b))
        D[i,j] = min distance between any point in trees_a[i] and trees_b[j].
        If max_distance is not None, entries may be np.inf.
    """
    A, B = len(trees_a), len(trees_b)
    if A == 0 or B == 0:
        return np.zeros((A, B), dtype=float)

    D = np.zeros((A, B), dtype=float)

    if max_distance is not None:
        # Cutoff mode: faster when many pairs are far apart.
        D.fill(np.inf)
        for i in range(A):
            ti = trees_a[i]
            for j in range(B):
                tj = trees_b[j]
                s = ti.sparse_distance_matrix(
                    tj, max_distance, output_type="coo_matrix"
                )
                if s.nnz > 0:
                    D[i, j] = float(s.data.min())
        return D

    # Exact mode: query points of the smaller tree into the larger tree (k=1) and take min.
    for i in range(A):
        ti = trees_a[i]
        ni = ti.n
        for j in range(B):
            tj = trees_b[j]
            nj = tj.n
            if ni <= nj:
                d, _ = tj.query(ti.data, k=1, workers=workers)
            else:
                d, _ = ti.query(tj.data, k=1, workers=workers)
            # d can be scalar if one tree has 1 point; np.min handles both
            D[i, j] = float(np.min(d))
    return D
