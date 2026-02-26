from time import perf_counter

import numpy as np
from typing import Dict, Tuple, Optional, Sequence
from scipy.spatial import cKDTree


# EDT backends: prefer Seung-Lab edt, fallback to scipy.ndimage
try:
    from edt import edt as _edt_fast

    _HAVE_EDT_FAST = True
except Exception:
    _HAVE_EDT_FAST = False

from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage.morphology import (
    ball,
)  # keep only ball; use ndi.binary_dilation everywhere

# ---------- Fast CC wrappers ----------
try:
    import cc3d

    _HAVE_CC3D = True
except Exception:
    _HAVE_CC3D = False
    from skimage.measure import label as _sk_label

try:
    import fastremap as _fr

    _HAVE_FASTREMAP = True
except Exception:
    _HAVE_FASTREMAP = False


def _cc_label_26(mask: np.ndarray):
    """
    Fast 3D connected components (26-connectivity).
    Returns (labels:int32, n_components:int).
    """
    if _HAVE_CC3D:
        lbl = cc3d.connected_components(
            mask.astype(np.uint8, copy=False), connectivity=26, out_dtype=np.uint32
        )
        return lbl, int(lbl.max())
    # Fallback: skimage (connectivity=3 ~ 26-neighborhood)
    lbl = _sk_label(mask, connectivity=3).astype(np.int32, copy=False)
    return lbl, int(lbl.max())


def _largest_component_id(lbl: np.ndarray):
    """
    Return the label ID (>=1) of the largest component in 'lbl'.
    lbl should already be a CC label image where 0=background.
    """
    if _HAVE_FASTREMAP:
        u, counts = _fr.unique(lbl, return_counts=True)
        if u.size:
            bg = np.where(u == 0)[0]
            if bg.size:
                counts[bg[0]] = 0
            return int(u[np.argmax(counts)])
        return 0
    cnt = np.bincount(lbl.ravel())
    if cnt.size:
        cnt[0] = 0
    return int(np.argmax(cnt)) if cnt.size else 0


# =========================
# Order / utility helpers
# =========================
def _to_zyx_sampling(vs, vox_order):
    vs = tuple(map(float, vs))
    if vox_order.lower() == "xyz":  # (x,y,z) -> (z,y,x)
        return (vs[2], vs[1], vs[0])
    if vox_order.lower() == "zyx":
        return vs
    raise ValueError("vox_order must be 'xyz' or 'zyx'")


def _to_internal_zyx_volume(vol, vol_order):
    if vol_order.lower() == "zyx":
        return vol, False
    if vol_order.lower() == "xyz":  # (x,y,z) -> (z,y,x)
        return np.transpose(vol, (2, 1, 0)), True
    raise ValueError("vol_order must be 'xyz' or 'zyx'")


def _from_internal_zyx_volume(vol_zyx, vol_order):
    if vol_order.lower() == "zyx":
        return vol_zyx
    if vol_order.lower() == "xyz":  # (z,y,x) -> (x,y,z)
        return np.transpose(vol_zyx, (2, 1, 0))
    raise ValueError("vol_order must be 'xyz' or 'zyx'")


def _seeds_to_zyx(seeds, seed_order):
    arr = np.asarray(seeds, dtype=float).reshape(-1, 3)
    if seed_order.lower() == "xyz":
        arr = arr[:, [2, 1, 0]]  # (x,y,z) -> (z,y,x)
    elif seed_order.lower() != "zyx":
        raise ValueError("seed_order must be 'xyz' or 'zyx'")
    return np.round(arr).astype(int)


def _seeds_from_zyx(seeds_zyx, seed_order):
    arr = np.asarray(seeds_zyx, dtype=int).reshape(-1, 3)
    if seed_order.lower() == "xyz":
        return arr[:, [2, 1, 0]]  # (z,y,x) -> (x,y,z)
    elif seed_order.lower() == "zyx":
        return arr
    else:
        raise ValueError("seed_order must be 'xyz' or 'zyx'")


# =========================
# Snapping (KDTree-based)
# =========================
def _extract_mask_boundary(mask, erosion_iters=1):
    """
    Extract boundary voxels of a 3D mask using binary erosion.
    Boundary = mask & (~eroded(mask))

    Parameters:
        mask           : 3D boolean array
        erosion_iters  : number of erosion iterations (higher removes thicker border)

    Returns:
        boundary_mask  : 3D boolean array of the same shape
    """
    if erosion_iters < 1:
        # No erosion => boundary = mask (not recommended unless extremely thin structures)
        return mask.copy()

    structure = np.ones((3, 3, 3), dtype=bool)
    interior = ndi.binary_erosion(
        mask, structure=structure, iterations=erosion_iters, border_value=0
    )
    boundary = mask & (~interior)
    return boundary


def _downsample_points(points, mode="stride", stride=2, target=None, rng=None):
    """
    Downsample a set of points (N,3) by either:
      - 'stride': take one every 'stride' points (fast, deterministic),
      - 'random': keep ~target points uniformly at random.

    Args:
        points : (N, 3) int or float array of coordinates
        mode   : 'stride' or 'random'
        stride : int >= 1 (for 'stride' mode)
        target : number of points to keep (for 'random' mode); if None, default is 50k
        rng    : np.random.Generator for reproducible random sampling

    Returns:
        (M, 3) array with M <= N
    """
    n = points.shape[0]
    if n == 0:
        return points

    if mode == "stride":
        stride = max(1, int(stride))
        return points[::stride]

    elif mode == "random":
        if target is None:
            target = min(n, 50_000)  # default target
        target = max(1, int(target))
        if target >= n:
            return points
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(n, size=target, replace=False)
        return points[idx]

    else:
        raise ValueError("downsample mode must be 'stride' or 'random'")


def snap_seeds_to_segment(
    seeds_xyz,
    mask,
    mask_order="zyx",
    voxel_size=(1.0, 1.0, 1.0),
    use_boundary=True,
    erosion_iters=1,
    downsample=True,
    downsample_mode="stride",  # 'stride' or 'random'
    downsample_stride=2,  # used if mode='stride'
    downsample_target=None,  # used if mode='random'
    rng=None,
    return_index=False,
    leafsize=16,
    log=lambda x: None,
    tag="snap",
    method="kdtree",  # accepted for compatibility; only 'kdtree' currently
):
    """
    Snap seeds (in XYZ) to the closest True voxel of a 3D mask using cKDTree over
    a *reduced* set of candidate voxels:
      - boundary-only (mask & ~eroded(mask)), if use_boundary=True
      - optionally downsampled (stride or random)

    This approach works well for speed while retaining high accuracy for snapping.

    Parameters:
        seeds_xyz         : (N,3) float or int array in XYZ order.
        mask              : 3D boolean array; binary segment.
        mask_order        : 'zyx' (default) or 'xyz' indicating memory layout of mask.
        voxel_size        : (vx, vy, vz) in XYZ physical units (e.g., (8.0, 8.0, 40.0)).
        use_boundary      : If True, only use boundary voxels for KDTree.
        erosion_iters     : Number of erosion iterations for boundary extraction.
        downsample        : If True, further reduce boundary points (stride or random).
        downsample_mode   : 'stride' or 'random' for boundary sampling.
        downsample_stride : If stride mode, use every Nth boundary voxel.
        downsample_target : If random mode, target number of boundary points to keep.
        rng               : Optional np.random.Generator for reproducible random sampling.
        return_index      : If True, also return indices of nearest boundary points.
        leafsize          : cKDTree leafsize parameter.
        log               : callable for logging
        tag               : string to prefix timings
        method            : currently only 'kdtree' supported. Present for backward compatibility.

    Returns:
        snapped_xyz : (N,3) int array in XYZ order, coordinates within volume bounds.
        match_idx   : (optional) indices into the candidate points array, if return_index=True.

    Notes:
        - Seeds outside the volume are supported; they will snap to the nearest segment voxel.
        - If use_boundary=True yields no boundary (thin segment), we fall back to the full mask.
        - If the mask is empty, we raise ValueError.
    """
    t0 = perf_counter()
    if method != "kdtree":
        log(f"[{tag}] Warning: 'method={method}' not supported; using 'kdtree'.")

    # Validate mask
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D boolean array")
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if mask_order not in ("zyx", "xyz"):
        raise ValueError("mask_order must be 'zyx' or 'xyz'")

    # Optional boundary extraction for speed
    tb = perf_counter()
    if use_boundary:
        candidate_mask = _extract_mask_boundary(mask, erosion_iters=erosion_iters)
        # Fallback to full mask if boundary is empty
        if not candidate_mask.any():
            candidate_mask = mask
            log(f"[{tag}] boundary empty → fallback to full mask")
    else:
        candidate_mask = mask
    log(f"[{tag}] candidate extraction | {perf_counter()-tb:.3f}s")

    # Obtain candidate voxel coordinates in XYZ order
    tc = perf_counter()
    if mask_order == "zyx":
        # mask shape is (Z, Y, X), np.where -> (z, y, x)
        zc, yc, xc = np.where(candidate_mask)
        points_xyz = np.stack([xc, yc, zc], axis=1)
        max_x, max_y, max_z = mask.shape[2] - 1, mask.shape[1] - 1, mask.shape[0] - 1
    else:
        # mask shape is (X, Y, Z), np.where -> (x, y, z)
        xc, yc, zc = np.where(candidate_mask)
        points_xyz = np.stack([xc, yc, zc], axis=1)
        max_x, max_y, max_z = mask.shape[0] - 1, mask.shape[1] - 1, mask.shape[2] - 1
    log(
        f"[{tag}] candidate coordinates | {perf_counter()-tc:.3f}s (n={len(points_xyz)})"
    )

    if points_xyz.shape[0] == 0:
        raise ValueError(
            "The mask (or boundary) contains no True voxels (empty segment)."
        )

    # Optional: further downsample candidate points
    td = perf_counter()
    if downsample:
        before = len(points_xyz)
        points_xyz = _downsample_points(
            points_xyz,
            mode=downsample_mode,
            stride=downsample_stride,
            target=downsample_target,
            rng=rng,
        )
        after = len(points_xyz)
        log(f"[{tag}] downsample points {before} → {after} | {perf_counter()-td:.3f}s")

    # Prepare seeds array
    seeds_xyz = np.asarray(seeds_xyz, dtype=np.float64)
    if seeds_xyz.ndim == 1:
        seeds_xyz = seeds_xyz[None, :]
    if seeds_xyz.shape[1] != 3:
        raise ValueError("seeds_xyz must be shape (N, 3)")

    # Scale coordinates to physical space to respect anisotropy
    vx, vy, vz = voxel_size
    scale = np.array([vx, vy, vz], dtype=np.float64)

    points_scaled = points_xyz * scale[None, :]
    seeds_scaled = seeds_xyz * scale[None, :]

    # cKDTree nearest neighbor lookup
    te = perf_counter()
    tree = cKDTree(points_scaled, leafsize=leafsize)
    _, nn_indices = tree.query(seeds_scaled, k=1, workers=-1)
    log(f"[{tag}] KDTree build+query | {perf_counter()-te:.3f}s")

    # Map back to integer voxel coords (XYZ)
    snapped_xyz = points_xyz[nn_indices].astype(np.int64)

    # Ensure snapped coords are valid (should already be in bounds)
    snapped_xyz[:, 0] = np.clip(snapped_xyz[:, 0], 0, max_x)
    snapped_xyz[:, 1] = np.clip(snapped_xyz[:, 1], 0, max_y)
    snapped_xyz[:, 2] = np.clip(snapped_xyz[:, 2], 0, max_z)

    log(f"[{tag}] snapped {len(seeds_xyz)} seeds | total {perf_counter()-t0:.3f}s")
    if return_index:
        return snapped_xyz, nn_indices
    else:
        return snapped_xyz


# ============================================================
# EDT wrapper (Seung-Lab edt preferred, fallback to scipy)
# ============================================================
def _compute_edt(mask: np.ndarray, sampling_zyx, log=lambda x: None, tag="edt"):
    """
    Compute Euclidean distance transform using Seung-Lab edt if available,
    otherwise fallback to scipy.ndimage.distance_transform_edt.

    - mask: boolean array in ZYX order
    - sampling_zyx: anisotropy tuple in ZYX (float)
    """
    t0 = perf_counter()
    if _HAVE_EDT_FAST:
        dist = _edt_fast(mask.astype(np.uint8, copy=False), anisotropy=sampling_zyx)
        log(f"[{tag}] Seung-Lab edt | {perf_counter()-t0:.3f}s")
        return dist
    else:
        dist = ndi.distance_transform_edt(mask, sampling=sampling_zyx)
        log(f"[{tag}] SciPy EDT | {perf_counter()-t0:.3f}s")
        return dist


# ------------------------------------------------------------
# Helpers for upsampling
# ------------------------------------------------------------
def _upsample_bool(mask_ds, steps, target_shape):
    up = mask_ds.repeat(steps[0], 0).repeat(steps[1], 1).repeat(steps[2], 2)
    return up[: target_shape[0], : target_shape[1], : target_shape[2]]


def _upsample_labels(lbl_ds, steps, target_shape):
    up = lbl_ds.repeat(steps[0], 0).repeat(steps[1], 1).repeat(steps[2], 2)
    return up[: target_shape[0], : target_shape[1], : target_shape[2]]


# ============================================================
# Combined connector (ROI + DS + MST paths) — uses snapping + fast EDT
# ============================================================
def connect_both_seeds_via_ridge(
    binary_sv: np.ndarray,
    seeds_a,
    seeds_b,
    voxel_size=(1.0, 1.0, 1.0),
    *,
    vol_order: str = "xyz",
    vox_order: str = "xyz",
    seed_order: str = "xyz",
    ridge_power: float = 2.0,
    roi_pad_zyx=(24, 48, 48),
    downsample=(2, 2, 1),
    refine_fullres_when_fail: bool = True,
    snap_method: str = "kdtree",
    snap_kwargs: dict | None = None,
    verbose: bool = True,
):
    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    def _bbox_pad_zyx(points_zyx, shape, pad=(24, 48, 48)):
        pts = np.asarray(points_zyx, int)
        if pts.size == 0:
            return (0, 0, 0, shape[0], shape[1], shape[2])
        z0, y0, x0 = pts.min(0)
        z1, y1, x1 = pts.max(0) + 1
        z0 = max(0, z0 - pad[0])
        y0 = max(0, y0 - pad[1])
        x0 = max(0, x0 - pad[2])
        z1 = min(shape[0], z1 + pad[0])
        y1 = min(shape[1], y1 + pad[1])
        x1 = min(shape[2], x1 + pad[2])
        return (z0, y0, x0, z1, y1, x1)

    def _mst_edges_phys(pts_zyx, sampling):
        P = np.asarray(pts_zyx, float)
        if len(P) <= 1:
            return []
        S = np.array(sampling, float)[None, :]
        phys = P * S
        n = len(P)
        in_tree = np.zeros(n, bool)
        in_tree[0] = True
        best = np.full(n, np.inf)
        parent = np.full(n, -1, int)
        d0 = np.sqrt(((phys - phys[0]) ** 2).sum(1))
        best[:] = d0
        best[0] = np.inf
        parent[:] = 0
        edges = []
        for _ in range(n - 1):
            i = int(np.argmin(best))
            if not np.isfinite(best[i]):
                break
            edges.append((int(parent[i]), i))
            in_tree[i] = True
            best[i] = np.inf
            di = np.sqrt(((phys - phys[i]) ** 2).sum(1))
            relax = (~in_tree) & (di < best)
            parent[relax] = i
            best[relax] = di[relax]
        return edges

    t0 = perf_counter()
    log(
        f"[connect] vol_order={vol_order}, vox_order={vox_order}, seed_order={seed_order}"
    )
    log(
        f"[connect] mask shape: {binary_sv.shape}, ridge_power={ridge_power}, ds={downsample}"
    )

    sv_zyx, _ = _to_internal_zyx_volume(binary_sv, vol_order)
    sampling = _to_zyx_sampling(voxel_size, vox_order)

    # SNAP seeds to mask
    A_in_zyx = _seeds_to_zyx(seeds_a, seed_order)
    B_in_zyx = _seeds_to_zyx(seeds_b, seed_order)

    # Default snapping config; override via snap_kwargs
    snap_cfg = dict(
        use_boundary=True,
        erosion_iters=1,
        downsample=True,
        downsample_mode="random",
        downsample_target=50_000,
        method=snap_method,  # allow pass-through compatibility
    )
    if snap_kwargs is not None:
        snap_cfg.update(snap_kwargs)

    def _snap(pts_zyx, name):
        if pts_zyx.size == 0:
            return np.empty((0, 3), dtype=int)
        # Convert ZYX -> XYZ for snapper
        pts_xyz = pts_zyx[:, [2, 1, 0]]
        # Use snapping over full 3D sv_zyx with ZYX mask
        snapped_xyz = snap_seeds_to_segment(
            pts_xyz,
            mask=sv_zyx,
            mask_order="zyx",
            voxel_size=(
                sampling[2],
                sampling[1],
                sampling[0],
            ),  # convert ZYX->XYZ spacing
            log=log,
            tag=f"{name}@snap",
            **snap_cfg,
        )
        # Back to ZYX
        return snapped_xyz[:, [2, 1, 0]]

    A_zyx = _snap(A_in_zyx, "A")
    B_zyx = _snap(B_in_zyx, "B")

    if len(A_zyx) == 0 or len(B_zyx) == 0:
        log("[connect] after snapping, one side has no seeds; skipping connection")
        return (
            _seeds_from_zyx(A_zyx, seed_order),
            _seeds_from_zyx(B_zyx, seed_order),
            (len(A_zyx) > 0),
            (len(B_zyx) > 0),
        )

    # ROI for speed
    z0, y0, x0, z1, y1, x1 = _bbox_pad_zyx(
        np.vstack([A_zyx, B_zyx]), sv_zyx.shape, pad=roi_pad_zyx
    )
    roi = sv_zyx[z0:z1, y0:y1, x0:x1]
    log(f"[connect] ROI: z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] → shape {roi.shape}")

    # Downsample ROI
    sz, sy, sx = map(int, downsample)
    ti_ds = perf_counter()
    if (sz, sy, sx) != (1, 1, 1):
        roi_ds = roi[::sz, ::sy, ::sx]
    else:
        roi_ds = roi
    sampling_ds = (sampling[0] * sz, sampling[1] * sy, sampling[2] * sx)
    log(
        f"[connect] ROI downsampled {roi.shape} -> {roi_ds.shape} | {perf_counter()-ti_ds:.3f}s"
    )

    # Robust seed placement on the downsampled grid:
    # (1) Map to ROI-local coords
    # (2) Divide by (sz,sy,sx) to approximate DS coords
    # (3) SNAP them to the nearest True voxel in roi_ds using KDTree
    def _to_roi_ds_snapped(pts_zyx, name="seedDS"):
        if pts_zyx.size == 0:
            return np.empty((0, 3), dtype=int)
        local = np.asarray(pts_zyx, int) - np.array([z0, y0, x0])  # roi-local
        seeds_ds = local / np.array(
            [sz, sy, sx], dtype=float
        )  # DS coordinates (float OK)
        # Convert to XYZ for snapper
        seeds_ds_xyz = seeds_ds[:, [2, 1, 0]]
        try:
            snapped_ds_xyz = snap_seeds_to_segment(
                seeds_ds_xyz,
                mask=roi_ds,
                mask_order="zyx",
                voxel_size=(sampling_ds[2], sampling_ds[1], sampling_ds[0]),
                log=log,
                tag=f"{name}@roi_ds",
                use_boundary=False,
                downsample=False,
                method="kdtree",
            )
            snapped_ds_zyx = snapped_ds_xyz[:, [2, 1, 0]]
            return snapped_ds_zyx.astype(int)
        except ValueError as e:
            # If roi_ds is empty or degenerate, bail out gracefully:
            log(
                f"[{name}@roi_ds] snapping failed ({e}); falling back to nearest-int grid & mask check."
            )
            approx = np.floor(seeds_ds + 0.5).astype(int)
            Z, Y, X = roi_ds.shape
            approx[:, 0] = np.clip(approx[:, 0], 0, Z - 1)
            approx[:, 1] = np.clip(approx[:, 1], 0, Y - 1)
            approx[:, 2] = np.clip(approx[:, 2], 0, X - 1)
            # Keep only those approx coords that are inside mask
            valid = [tuple(p) for p in approx if roi_ds[tuple(p)]]
            return np.array(valid, dtype=int)

    A_ds = _to_roi_ds_snapped(A_zyx, "A")
    B_ds = _to_roi_ds_snapped(B_zyx, "B")

    okA = len(A_ds) >= 1
    okB = len(B_ds) >= 1
    if not (okA and okB):
        log(
            "[connect] seeds disappeared or failed to map on DS grid; consider smaller ds or use_boundary=False/downsample=False in snapping."
        )
        return (
            _seeds_from_zyx(A_zyx, seed_order),
            _seeds_from_zyx(B_zyx, seed_order),
            okA,
            okB,
        )

    # EDT and cost on DS ROI (Seung-Lab edt if available)
    t1 = perf_counter()
    dist = _compute_edt(roi_ds, sampling_ds, log=log, tag="connect:EDT")
    if dist.max() <= 0:
        log("[connect] empty EDT in ROI; skipping connection")
        return (
            _seeds_from_zyx(A_zyx, seed_order),
            _seeds_from_zyx(B_zyx, seed_order),
            False,
            False,
        )
    dn = dist / dist.max()
    eps = 1e-6
    cost = np.full_like(dn, 1e12, dtype=float)
    cost[roi_ds] = 1.0 / (eps + np.clip(dn[roi_ds], 0, 1) ** max(0.0, ridge_power))
    log(f"[connect] EDT/cost ready on DS-ROI  | {perf_counter()-t1:.3f}s")

    # Shortest paths via MST
    def _path_mask_ds(start, end):
        tmcp = perf_counter()
        mcp = MCP_Geometric(cost, sampling=sampling_ds)
        costs, _ = mcp.find_costs([tuple(start)], find_all_ends=False)
        mid = perf_counter()
        v = costs[tuple(end)]
        if not np.isfinite(v):
            log(
                f"[MCP] start={tuple(start)} -> end={tuple(end)} FAILED | setup+run={mid-tmcp:.3f}s"
            )
            return None
        path = np.asarray(mcp.traceback(tuple(end)), int)
        m = np.zeros_like(roi_ds, bool)
        m[tuple(path.T)] = True
        log(
            f"[MCP] start={tuple(start)} -> end={tuple(end)} OK | total={perf_counter()-tmcp:.3f}s"
        )
        return m

    def _augment_team_ds(team_name, pts_ds):
        if len(pts_ds) <= 1:
            return np.zeros_like(roi_ds, bool), True
        edges = _mst_edges_phys(pts_ds, sampling_ds)
        pmask = np.zeros_like(roi_ds, bool)
        ok = True
        for i, j in edges:
            m = _path_mask_ds(pts_ds[i], pts_ds[j])
            if m is None:
                log(f"[connect:{team_name}] DS path FAILED for edge {i}-{j}")
                ok = False
                if refine_fullres_when_fail:
                    # fallback full-res EDT and path
                    tfr = perf_counter()
                    dist_fr = _compute_edt(
                        roi, sampling, log=log, tag="connect:EDT(fullres)"
                    )
                    dnm = dist_fr / (dist_fr.max() if dist_fr.max() > 0 else 1.0)
                    cost_fr = np.full_like(dist_fr, 1e12, dtype=float)
                    cost_fr[roi] = 1.0 / (
                        eps + np.clip(dnm[roi], 0, 1) ** max(0.0, ridge_power)
                    )
                    s = np.array(pts_ds[i]) * np.array([sz, sy, sx])
                    e = np.array(pts_ds[j]) * np.array([sz, sy, sx])
                    mcp_fr = MCP_Geometric(cost_fr, sampling=sampling)
                    costs_fr, _ = mcp_fr.find_costs([tuple(s)], find_all_ends=False)
                    if np.isfinite(costs_fr[tuple(e)]):
                        path_fr = np.asarray(mcp_fr.traceback(tuple(e)), int)
                        m_fr = np.zeros_like(roi, bool)
                        m_fr[tuple(path_fr.T)] = True
                        m = m_fr[::sz, ::sy, ::sx]
                        ok = True
                        log(
                            f"[connect:{team_name}] fallback full-res path OK | {perf_counter()-tfr:.3f}s"
                        )
                    else:
                        log(
                            f"[connect:{team_name}] Full-res ROI path also FAILED for edge {i}-{j}"
                        )
                        m = None
            if m is not None:
                pmask |= m
        return pmask, ok

    t_aug = perf_counter()
    pA_ds, okA2 = _augment_team_ds("A", A_ds)
    pB_ds, okB2 = _augment_team_ds("B", B_ds)
    okA &= okA2
    okB &= okB2
    log(f"[connect] MST+paths built | {perf_counter()-t_aug:.3f}s")

    if not (okA and okB):
        log(
            "[connect] connection failed for at least one team — consider smaller downsample or refine_fullres_when_fail."
        )
        return (
            _seeds_from_zyx(A_zyx, seed_order),
            _seeds_from_zyx(B_zyx, seed_order),
            okA,
            okB,
        )

    # Up-project to full resolution and dilate
    pA = _upsample_bool(pA_ds, (sz, sy, sx), roi.shape) & roi
    pB = _upsample_bool(pB_ds, (sz, sy, sx), roi.shape) & roi
    struc = ball(1)
    tpost = perf_counter()
    pA = ndi.binary_dilation(pA, structure=struc) & roi
    pB = ndi.binary_dilation(pB, structure=struc) & roi
    log(f"[connect] postproc dilation on paths | {perf_counter()-tpost:.3f}s")

    A_aug = set(map(tuple, A_zyx))
    B_aug = set(map(tuple, B_zyx))
    Az, Ay, Ax = np.nonzero(pA)
    Bz, By, Bx = np.nonzero(pB)
    for z, y, x in zip(Az, Ay, Ax):
        A_aug.add((z0 + z, y0 + y, x0 + x))
    for z, y, x in zip(Bz, By, Bx):
        B_aug.add((z0 + z, y0 + y, x0 + x))

    A_aug = _seeds_from_zyx(np.array(sorted(list(A_aug)), int), seed_order)
    B_aug = _seeds_from_zyx(np.array(sorted(list(B_aug)), int), seed_order)
    log(
        f"[connect] done; +{len(A_aug)-len(seeds_a)} vox for A, +{len(B_aug)-len(seeds_b)} for B  | total {perf_counter()-t0:.3f}s"
    )
    return A_aug, B_aug, True, True


def split_supervoxel_growing(
    binary_sv: np.ndarray,
    seeds_a,
    seeds_b,
    voxel_size=(1.0, 1.0, 1.0),
    *,
    # conventions / orders
    vol_order: str = "xyz",
    vox_order: str = "xyz",
    seed_order: str = "xyz",
    # geometry / cost
    halo: int = 1,
    gamma_neck: float = 1.6,  # boundary slowdown
    k_prox: float = 2.0,  # proximity boost strength
    lambda_prox: float = 1.0,  # proximity decay
    narrow_band_rel: float = 0.08,  # relative difference threshold
    nb_dilate: int = 1,  # dilate band to stabilize
    # optional: compute TA/TB on a downsampled grid
    downsample_geodesic: tuple | None = None,  # e.g. (1,2,2)
    # post-processing / guarantees
    allow_third_label: bool = True,
    enforce_single_cc: bool = True,
    # final validation
    check_seeds_same_cc: bool = True,
    raise_if_seed_split: bool = True,
    raise_if_multi_cc: bool = False,
    # snapping control (NEW)
    snap_method: str = "kdtree",
    snap_kwargs: dict | None = None,
    # logging
    verbose: bool = True,
):
    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    # Helpers reused from the module: _cc_label_26, _largest_component_id, _to_internal_zyx_volume, _from_internal_zyx_volume
    # _seeds_to_zyx, _compute_edt, etc. are assumed available.

    # ---------- helpers ----------
    def _enforce_single_component(out_labels, lab, seed_pts_global, allow3=True):
        t = perf_counter()
        mask = out_labels == lab
        if not np.any(mask):
            return 0, 0
        comp, ncomp = _cc_label_26(mask)
        if ncomp <= 1:
            log(f"[single-cc:{lab}] ncomp=1  | {perf_counter()-t:.3f}s")
            return 1, 0

        keep_ids = set()
        for z, y, x in seed_pts_global:
            if (
                0 <= z < out_labels.shape[0]
                and 0 <= y < out_labels.shape[1]
                and 0 <= x < out_labels.shape[2]
            ):
                if out_labels[z, y, x] == lab:
                    cid = comp[z, y, x]
                    if cid > 0:
                        keep_ids.add(int(cid))

        if not keep_ids:
            keep_ids = {_largest_component_id(comp)}

        lut = np.zeros(ncomp + 1, dtype=np.bool_)
        lut[list(keep_ids)] = True
        bad_mask = (comp > 0) & (~lut[comp])
        moved = int(bad_mask.sum())
        if allow3 and moved:
            out_labels[bad_mask] = 3
        log(
            f"[single-cc:{lab}] kept={len(keep_ids)}, moved_to_3={moved}  | {perf_counter()-t:.3f}s"
        )
        return len(keep_ids), moved

    def _resolve_label3_touching_vectorized(
        out_labels, seedsA=None, seedsB=None, sampling=(1, 1, 1)
    ):
        t0 = perf_counter()
        comp3, n3 = _cc_label_26(out_labels == 3)
        n3_vox = int((out_labels == 3).sum())
        log(f"[touching] n3 comps={n3}, vox={n3_vox}")
        if n3 == 0:
            log(f"[touching] no label-3 components  | {perf_counter()-t0:.3f}s")
            return 0, 0

        t1 = perf_counter()
        struc = np.ones((3, 3, 3), bool)
        N1 = ndi.binary_dilation(out_labels == 1, structure=struc) & (comp3 > 0)
        N2 = ndi.binary_dilation(out_labels == 2, structure=struc) & (comp3 > 0)

        cnt1 = np.bincount(comp3[N1], minlength=n3 + 1)
        cnt2 = np.bincount(comp3[N2], minlength=n3 + 1)

        assign = np.zeros(n3 + 1, dtype=np.int16)  # 0=undecided, 1 or 2 otherwise
        assign[cnt1 > cnt2] = 1
        assign[cnt2 > cnt1] = 2
        undec = np.where(assign[1:] == 0)[0] + 1
        log(
            f"[touching] maj→1={int((assign==1).sum())}, maj→2={int((assign==2).sum())}, ties={len(undec)}  | {perf_counter()-t1:.3f}s"
        )

        if (
            len(undec) > 0
            and (seedsA is not None)
            and (seedsB is not None)
            and len(seedsA)
            and len(seedsB)
        ):
            t2 = perf_counter()
            sA = np.zeros_like(out_labels, bool)
            sA[tuple(np.array(seedsA).T)] = True
            sB = np.zeros_like(out_labels, bool)
            sB[tuple(np.array(seedsB).T)] = True
            dA = _compute_edt(~sA, sampling, log=log, tag="split:EDT(dA)")
            dB = _compute_edt(~sB, sampling, log=log, tag="split:EDT(dB)")
            closer2 = (dB < dA) & (comp3 > 0)

            pref2 = np.bincount(comp3[closer2], minlength=n3 + 1)
            total = np.bincount(comp3[comp3 > 0], minlength=n3 + 1)

            tie_ids = np.array(undec, dtype=int)
            choose2 = pref2[tie_ids] > (total[tie_ids] - pref2[tie_ids])
            assign[tie_ids[choose2]] = 2
            assign[tie_ids[~choose2]] = 1
            log(
                f"[touching] tie-break EDT done: to2={int(choose2.sum())}, to1={int((~choose2).sum())}  | {perf_counter()-t2:.3f}s"
            )

        moved1 = moved2 = 0
        if (assign == 1).any():
            mask1 = assign[comp3] == 1
            moved1 = int(mask1.sum())
            out_labels[mask1] = 1
        if (assign == 2).any():
            mask2 = assign[comp3] == 2
            moved2 = int(mask2.sum())
            out_labels[mask2] = 2

        log(
            f"[touching] reassigned 3→1: {moved1}, 3→2: {moved2}  | total {perf_counter()-t0:.3f}s"
        )
        return moved1, moved2

    # ---------- begin ----------
    t0 = perf_counter()
    log(f"[init] vol_order={vol_order}, vox_order={vox_order}, seed_order={seed_order}")
    log(f"[init] input volume shape: {binary_sv.shape}")

    # Convert input volumes and sampling into internal ZYX
    sv_zyx, _ = _to_internal_zyx_volume(binary_sv, vol_order)
    sampling = _to_zyx_sampling(voxel_size, vox_order)
    log(f"[init] internal shape (z,y,x): {sv_zyx.shape}")
    log(f"[init] sampling (z,y,x): {sampling}")

    # SNAP seeds to mask using the same KDTree-based method
    A_all = _seeds_to_zyx(seeds_a, seed_order)
    B_all = _seeds_to_zyx(seeds_b, seed_order)
    log("[snap] snapping seeds to segment mask...")

    snap_cfg = dict(
        use_boundary=True,
        erosion_iters=1,
        downsample=True,
        downsample_mode="random",
        downsample_target=50_000,
        method=snap_method,  # compatibility key
    )
    if snap_kwargs is not None:
        snap_cfg.update(snap_kwargs)

    def _snap_ZYX(pts_zyx, tagname):
        if pts_zyx.size == 0:
            return np.empty((0, 3), dtype=int)
        # Convert ZYX -> XYZ for snapper
        pts_xyz = pts_zyx[:, [2, 1, 0]]
        snapped_xyz = snap_seeds_to_segment(
            pts_xyz,
            mask=sv_zyx,
            mask_order="zyx",
            voxel_size=(
                sampling[2],
                sampling[1],
                sampling[0],
            ),  # convert ZYX→XYZ spacing
            log=log,
            tag=tagname,
            **snap_cfg,
        )
        return snapped_xyz[:, [2, 1, 0]]

    A = _snap_ZYX(A_all, "A@snap")
    B = _snap_ZYX(B_all, "B@snap")
    log(f"[seeds] A={len(A)}, B={len(B)}")

    out_zyx = np.zeros_like(sv_zyx, dtype=np.int16)
    if A.size == 0 or B.size == 0 or not np.any(sv_zyx):
        log("[seeds] missing seeds or empty SV; returning label=1 for entire SV")
        out_zyx[sv_zyx] = 1
        return _from_internal_zyx_volume(out_zyx, vol_order)

    # Tight bbox ROI around mask with halo
    t_bbox = perf_counter()
    Z, Y, X = sv_zyx.shape
    coords = np.argwhere(sv_zyx)
    z0, y0, x0 = coords.min(0)
    z1, y1, x1 = coords.max(0) + 1
    z0h = max(z0 - halo, 0)
    y0h = max(y0 - halo, 0)
    x0h = max(x0 - halo, 0)
    z1h = min(z1 + halo, Z)
    y1h = min(y1 + halo, Y)
    x1h = min(x1 + halo, X)
    sv = sv_zyx[z0h:z1h, y0h:y1h, x0h:x1h]
    A_roi = A - np.array([z0h, y0h, x0h])
    B_roi = B - np.array([z0h, y0h, x0h])
    log(
        f"[crop] ROI shape (internal): {sv.shape} (halo {halo})  | {perf_counter()-t_bbox:.3f}s"
    )

    # Build travel cost via EDT (Seung-Lab edt if available)
    t1 = perf_counter()
    dist = _compute_edt(sv, sampling, log=log, tag="split:EDT(mask)")
    distn = dist / dist.max() if dist.max() > 0 else dist
    eps = 1e-6
    speed = np.clip(distn ** max(gamma_neck, 0.0), eps, 1.0)
    travel_cost = np.full_like(speed, 1e12, dtype=float)
    travel_cost[sv] = 1.0 / speed[sv]
    log(
        f"[speed] EDT + speed map  | {perf_counter()-t1:.3f}s  (total {perf_counter()-t0:.3f}s)"
    )

    # Optional downsample for geodesic
    use_ds = downsample_geodesic is not None
    if use_ds:
        dz, dy, dx = map(int, downsample_geodesic)
        log(f"[geodesic] downsample grid: {downsample_geodesic}")
        cost_ds = travel_cost[::dz, ::dy, ::dx]
        mask_ds = sv[::dz, ::dy, ::dx]
        sampling_ds = (sampling[0] * dz, sampling[1] * dy, sampling[2] * dx)

        def _to_ds(pts):
            pts = (np.asarray(pts, int) // np.array([dz, dy, dx])).astype(int)
            Zs, Ys, Xs = mask_ds.shape
            keep = []
            for z, y, x in pts:
                if 0 <= z < Zs and 0 <= y < Ys and 0 <= x < Xs and mask_ds[z, y, x]:
                    keep.append((z, y, x))
            return keep

        A_sub = _to_ds(A_roi)
        B_sub = _to_ds(B_roi)
        log(f"[geodesic] seeds on DS grid: A={len(A_sub)}, B={len(B_sub)}")
        if len(A_sub) == 0 or len(B_sub) == 0:
            log("[geodesic] DS removed all seeds; falling back to full-res")
            use_ds = False
    if not use_ds:
        cost_ds = travel_cost
        mask_ds = sv
        sampling_ds = sampling
        A_sub = [tuple(p) for p in A_roi.tolist()]
        B_sub = [tuple(p) for p in B_roi.tolist()]

    # Geodesic arrival times
    t2 = perf_counter()
    mcpA = MCP_Geometric(cost_ds, sampling=sampling_ds)
    TA, _ = mcpA.find_costs(A_sub, find_all_ends=False)
    mcpB = MCP_Geometric(cost_ds, sampling=sampling_ds)
    TB, _ = mcpB.find_costs(B_sub, find_all_ends=False)
    TA = np.where(mask_ds, TA, np.inf)
    TB = np.where(mask_ds, TB, np.inf)
    log(
        f"[geodesic] TA/TB computed  | {perf_counter()-t2:.3f}s  (total {perf_counter()-t0:.3f}s)"
    )

    # Narrow band
    t3 = perf_counter()
    finite = np.isfinite(TA) & np.isfinite(TB) & mask_ds
    denom = TA + TB + 1e-12
    reldiff = np.zeros_like(TA)
    reldiff[finite] = np.abs(TA[finite] - TB[finite]) / denom[finite]
    band = finite & (reldiff <= narrow_band_rel)
    if nb_dilate > 0:
        band = ndi.binary_dilation(band, structure=ball(nb_dilate)) & mask_ds
    if band.sum() < 64:
        band = mask_ds.copy()
        log("[band] tiny band -> using full ROI on current grid")
    log(
        f"[band] voxels: {int(band.sum())}  | {perf_counter()-t3:.3f}s  (total {perf_counter()-t0:.3f}s)"
    )

    # Proximity-boosted labeling
    t4 = perf_counter()
    denomA = 1.0 + k_prox * np.exp(-lambda_prox * np.clip(TB, 0, np.inf))
    denomB = 1.0 + k_prox * np.exp(-lambda_prox * np.clip(TA, 0, np.inf))
    CA = TA / denomA
    CB = TB / denomB
    sub_labels_ds = np.zeros_like(mask_ds, dtype=np.int16)
    sub_labels_ds[(CA <= CB) & band] = 1
    sub_labels_ds[(CB < CA) & band] = 2
    outer = mask_ds & (sub_labels_ds == 0)
    sub_labels_ds[(TA <= TB) & outer] = 1
    sub_labels_ds[(TB < TA) & outer] = 2
    for z, y, x in A_sub:
        sub_labels_ds[z, y, x] = 1
    for z, y, x in B_sub:
        sub_labels_ds[z, y, x] = 2
    log(
        f"[label] DS labeling done  | {perf_counter()-t4:.3f}s  (total {perf_counter()-t0:.3f}s)"
    )

    # Upsample if needed
    if use_ds:
        sub_labels = _upsample_labels(sub_labels_ds, (dz, dy, dx), sv.shape)
        sub_labels[~sv] = 0
        for z, y, x in A_roi:
            sub_labels[z, y, x] = 1
        for z, y, x in B_roi:
            sub_labels[z, y, x] = 2
        log(f"[label] upsampled DS→full ROI")
    else:
        sub_labels = sub_labels_ds

    # Writeback
    out_zyx[sv_zyx] = 1
    out_zyx[z0h:z1h, y0h:y1h, x0h:x1h][sub_labels == 1] = 1
    out_zyx[z0h:z1h, y0h:y1h, x0h:x1h][sub_labels == 2] = 2
    log("[writeback] labels written to full volume")

    # Enforce single CC per label
    if enforce_single_cc:
        keptA, movedA = _enforce_single_component(
            out_zyx, 1, A, allow3=allow_third_label
        )
        keptB, movedB = _enforce_single_component(
            out_zyx, 2, B, allow3=allow_third_label
        )
        log(
            f"[single-cc] label1 kept {keptA}, moved {movedA} -> 3; label2 kept {keptB}, moved {movedB} -> 3"
        )

    # Resolve 3-touching
    moved1, moved2 = _resolve_label3_touching_vectorized(out_zyx, A, B, sampling)
    if moved1 or moved2:
        if enforce_single_cc:
            keptA, movedA = _enforce_single_component(
                out_zyx, 1, A, allow3=allow_third_label
            )
            keptB, movedB = _enforce_single_component(
                out_zyx, 2, B, allow3=allow_third_label
            )
            log(
                f"[single-cc 2nd] label1 kept {keptA}, moved {movedA}; label2 kept {keptB}, moved {movedB}"
            )

    # Final check
    for lab in (1, 2):
        _, ncomp = _cc_label_26(out_zyx == lab)
        if ncomp > 1:
            msg = f"[check] label {lab} has {ncomp} connected components"
            if raise_if_multi_cc:
                raise ValueError(msg)
            else:
                log(msg)

    log(f"[done] total elapsed {perf_counter()-t0:.3f}s")
    return _from_internal_zyx_volume(out_zyx, vol_order)


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


def split_supervoxel_helper(
    binary_seg: np.ndarray,
    source_coords: np.ndarray,
    sink_coords: np.ndarray,
    voxel_size: tuple,
    verbose: bool = False,
):
    voxel_size = np.array(voxel_size)
    downsample = voxel_size.max() // voxel_size

    # 1) Connect seed teams first
    A_aug, B_aug, okA, okB = connect_both_seeds_via_ridge(
        binary_seg,
        source_coords,
        sink_coords,
        voxel_size=voxel_size,
        downsample=downsample,
        vol_order="xyz",
        vox_order="xyz",
        seed_order="xyz",
        snap_method="kdtree",
        snap_kwargs=dict(
            use_boundary=False,  # disables boundary-only snapping for maximum safety
            downsample=False,  # avoids losing candidates
            method="kdtree",
        ),
        verbose=verbose,
    )
    if not (okA and okB):
        raise RuntimeError(
            "In-mask connection failed for at least one team; skipping split."
        )

    # 2) Run the corridor-free splitter with same snapping settings
    return split_supervoxel_growing(
        binary_seg,
        A_aug,
        B_aug,
        voxel_size=voxel_size,
        vol_order="xyz",
        vox_order="xyz",
        seed_order="xyz",
        halo=1,
        gamma_neck=1.6,
        narrow_band_rel=0.08,
        nb_dilate=1,
        downsample_geodesic=(1, 2, 2),
        enforce_single_cc=True,
        raise_if_seed_split=True,
        raise_if_multi_cc=True,
        verbose=verbose,
        snap_method="kdtree",
        snap_kwargs=dict(
            use_boundary=False,  # match the connector for consistency
            downsample=False,
            method="kdtree",
        ),
    )
