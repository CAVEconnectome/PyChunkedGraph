"""OCDBT-backed neuroglancer_precomputed segmentation store — public API.

Architecture: one immutable base OCDBT per watershed + one delta OCDBT per
ChunkedGraph. Reads merge base + delta via tensorstore's kvstack driver.
Writes land in the delta via OCDBT's ``*_data_prefix`` options.

Multi-scale (MIP pyramid) is supported: the source watershed's info JSON
drives the scale layout. All scales share one OCDBT kvstore; the precomputed
driver prefixes keys by scale key automatically.

Versioned reads
---------------
Every OCDBT commit gets a monotonically-increasing ``generation_number`` and
an ``absl::Now()``-stamped ``commit_time`` (nanoseconds since epoch). The
tensorstore OCDBT driver lets callers pin a read-only open to a prior version
via the ``version`` spec field; accepts either an integer generation number
or an ISO-8601 UTC timestamp string. The timestamp form requires a ``Z``
suffix (not ``+00:00``) and is interpreted as ``commit_time <= T`` — the open
returns the latest version at or before the pinned time.

The commit_time itself cannot be overridden by the caller: OCDBT stamps each
commit from the writer's local clock (``absl::Now()`` in
``btree_writer_commit_operation.cc``). This means we can't make OCDBT commits
align exactly with a caller-provided operation timestamp. What the L2 chunk
lock guarantees instead: no other writer can commit to our chunks while we
hold the lock, so any timestamp captured under the lock before our first
commit is a valid pin for "pre-op state of our chunks."

Retention: the OCDBT spec exposes no pruning fields. All versions are
retained by default.
"""

from os import environ

import numpy as np
import tensorstore as ts

from pychunkedgraph import get_logger

from .debug import bbox_failure_payload, dump_failure_to_gcs
from .meta import OcdbtConfig
from ..dry_run import is_dry_run
from .utils import (
    _base_ocdbt_path,
    _ensure_trailing_slash,
    _open_precomputed_scale,
    _read_source_scales,
    _schema_from_src,
    base_exists,
    fork_exists,
)

logger = get_logger(__name__)


def create_base_ocdbt(ws_path: str, config: OcdbtConfig):
    """One-time bootstrap: create the shared base OCDBT at ``<ws>/ocdbt/base/``.

    Wipes any existing base first, then opens each scale with create=True
    so the info JSON is built from the source. Populating the base with
    actual chunk data happens separately via ``copy_ws_chunk_multiscale``
    or ``copy_ws_bbox_multiscale`` during the per-chunk ingest tasks.

    Returns (src_list, dst_list, resolutions) for the caller to use with
    the copy helpers.
    """
    base = _base_ocdbt_path(ws_path)
    # Wipe via the underlying GCS/file driver, NOT through the ocdbt
    # driver. Opening as ocdbt on an empty dir creates a default-config
    # `manifest.ocdbt` stub (max_inline_value_bytes=100); on a dir with
    # an existing manifest it only clears the B+tree, leaving the
    # manifest's config in place. Either way the subsequent open with a
    # different config mismatches.
    try:
        kvs = ts.KvStore.open(base).result()
        kvs.delete_range(ts.KvStore.KeyRange()).result()
    except Exception:
        pass

    scales = _read_source_scales(ws_path)
    resolutions = [s["resolution"] for s in scales]
    base_kvstore = {"driver": "ocdbt", "base": base, "config": config.ts_config()}

    src_list, dst_list = [], []
    for i in range(len(scales)):
        src_i = ts.open(
            {"driver": "neuroglancer_precomputed", "kvstore": ws_path, "scale_index": i}
        ).result()
        dst_i = _open_precomputed_scale(
            base_kvstore, i, create=True, **_schema_from_src(src_i)
        )
        src_list.append(src_i)
        dst_list.append(dst_i)
    return src_list, dst_list, resolutions


def wipe_base_ocdbt(ws_path: str):
    """Wipe the base OCDBT entirely (for --reset-ocdbt)."""
    base = _base_ocdbt_path(ws_path)
    # Wipe via the underlying GCS/file driver so the manifest file is
    # deleted too. Opening as ocdbt only clears the B+tree.
    try:
        kvs = ts.KvStore.open(base).result()
        kvs.delete_range(ts.KvStore.KeyRange()).result()
    except Exception:
        pass


def open_base_ocdbt(
    ws_path: str, config: OcdbtConfig, coordinator_address: str | None = None
):
    """Open the existing base OCDBT (read/write) for populating during ingest.

    Used by per-chunk ingest tasks that copy precomputed data into the shared
    base. NOT used at runtime — runtime always goes through the per-CG fork
    spec via ``get_seg_source_and_destination_ocdbt``.

    ``coordinator_address`` (``"host:port"``) routes every OCDBT commit
    through a ``DistributedCoordinatorServer`` so parallel workers don't
    race the shared manifest's CAS — the only thing that prevents the
    orphan ``d/`` file explosion. Required for any concurrent writer; the
    arg is optional so single-process callers (e.g. tests, notebooks) can
    skip it.

    Returns (src_list, dst_list, resolutions).
    """
    base = _base_ocdbt_path(ws_path)
    scales = _read_source_scales(ws_path)
    resolutions = [s["resolution"] for s in scales]
    # No `config` block: the base already exists (created by
    # `create_base_ocdbt`), so its on-disk manifest is authoritative.
    # Embedding `config.ts_config()` here would assert our in-code
    # defaults against whatever is persisted and raise
    # FAILED_PRECONDITION on any drift.
    base_kvstore = {"driver": "ocdbt", "base": base}
    if coordinator_address:
        base_kvstore["coordinator"] = {"address": coordinator_address}

    src_list, dst_list = [], []
    for i in range(len(scales)):
        src_i = ts.open(
            {"driver": "neuroglancer_precomputed", "kvstore": ws_path, "scale_index": i}
        ).result()
        dst_i = _open_precomputed_scale(base_kvstore, i, **_schema_from_src(src_i))
        src_list.append(src_i)
        dst_list.append(dst_i)
    return src_list, dst_list, resolutions


def build_cg_ocdbt_spec(
    ws_path: str,
    graph_id: str,
    config: OcdbtConfig,
    *,
    pinned_at: "int | str | None" = None,
) -> dict:
    """Open-time kvstore spec for a CG's OCDBT, backed by a shared immutable base.

    This function is a pure spec-constructor — it doesn't materialize
    the fork. The fork's ``manifest.ocdbt`` must exist before ``ts.open``
    on this spec will succeed; it's created by ``fork_base_manifest``
    (invoked from the ingest CLI's OCDBT path or the ``seg_ocdbt``
    notebook). ``ChunkedGraphMeta.ws_ocdbt_scales`` asserts presence via
    ``fork_exists`` so callers get a clear error instead of a tensorstore
    internal failure.

    All three kvstack layers below AND all three ``*_data_prefix`` options
    are load-bearing; removing any of them causes fork writes to leak
    into the immutable base (verified empirically).

    When ``pinned_at`` is set, the opened kvstore is read-only and returns
    state as of the specified version. Accepts an integer generation
    number (exact) or an ISO-8601 UTC timestamp string with ``Z`` suffix
    (interpreted as ``commit_time <= T``).
    """
    base = _base_ocdbt_path(ws_path)
    fork_dir = _ensure_trailing_slash(f"{ws_path.rstrip('/')}/ocdbt/{graph_id}")
    data_prefix = f"{graph_id}_d/"

    # Catch-all. Lets the fork READ base's B+tree (manifest + d/<hash>
    # data files) via fall-through. Must be first so later layers can
    # override sub-ranges.
    base_layer = {"base": base}

    # Single-key override. Routes the fork's manifest file so new
    # commits by this CG are visible only to this CG. Without this layer
    # manifest writes silently clobber base's manifest.
    fork_manifest_layer = {
        "exact": "manifest.ocdbt",
        "base": fork_dir + "manifest.ocdbt",
    }

    # Catches OCDBT's new data-file writes for the fork. Pairs with the
    # *_data_prefix options: OCDBT would otherwise write under the
    # default `d/` prefix — no later layer claims `d/`, so kvstack
    # falls through to the base catch-all and the writes corrupt base.
    fork_data_layer = {
        "prefix": data_prefix,
        "base": _ensure_trailing_slash(fork_dir + data_prefix),
    }

    # No `config` block: this spec opens an existing OCDBT (the shared
    # base + this fork's manifest+data layers). Tensorstore validates
    # every field of `config` against the on-disk manifest and raises
    # FAILED_PRECONDITION on mismatch, so embedding our in-code defaults
    # here would break any base that was created with different values
    # (e.g. an older default for `max_inline_value_bytes`). On-disk wins.
    spec = {
        "driver": "ocdbt",
        "base": {
            "driver": "kvstack",
            "layers": [base_layer, fork_manifest_layer, fork_data_layer],
        },
        # Steer every kind of OCDBT write under `<graph_id>_d/` so the
        # fork_data_layer catches them.
        "value_data_prefix": data_prefix,
        "btree_node_data_prefix": data_prefix,
        "version_tree_node_data_prefix": data_prefix,
    }
    if pinned_at is not None:
        spec["version"] = pinned_at
    return spec


def fork_base_manifest(ws_path: str, graph_id: str, wipe_existing: bool = False):
    """Initialize a CG's delta directory by copying the base manifest.

    If wipe_existing=True, deletes the existing fork directory first (for
    --retry when a prior ingest failed and left partial delta state).
    """
    assert base_exists(ws_path), "base OCDBT must exist before forking"
    base = _base_ocdbt_path(ws_path)
    fork_dir = _ensure_trailing_slash(f"{ws_path.rstrip('/')}/ocdbt/{graph_id}")

    if wipe_existing:
        try:
            kvs = ts.KvStore.open(fork_dir).result()
            kvs.delete_range(ts.KvStore.KeyRange()).result()
        except Exception:
            pass

    base_kvs = ts.KvStore.open(base).result()
    fork_kvs = ts.KvStore.open(fork_dir).result()
    manifest = base_kvs.read("manifest.ocdbt").result().value
    fork_kvs.write("manifest.ocdbt", manifest).result()


def ensure_fork_synced(ws_path: str, graph_id: str) -> bool:
    """Sync fork manifest to base — but only before the fork's first edit.

    Invariant we enforce: a fresh, edit-free fork must reflect base's
    *current* manifest at open time. ``setup_base`` calls
    ``fork_base_manifest`` once at graph creation, possibly before
    populate has committed most of its writes; any subsequent populate
    commit to base would otherwise be invisible through the fork
    (symptom: meshing reads return zeros). We close that window by
    re-snapshotting on the first runtime open before any edit lands.

    Once the fork has any edit (anything under ``<graph_id>_d/``), the
    function has no work to do: base is immutable post-setup, so the
    fork manifest cannot fall behind in any way that matters — its
    divergence from base is just the fork's own forward progress.
    Edit files are stable, so listing the prefix is a sufficient
    short-circuit and skips reading both manifests on every runtime
    open.

    Returns True iff the fork manifest was refreshed.
    """
    if not fork_exists(ws_path, graph_id):
        return False
    fork_dir = _ensure_trailing_slash(f"{ws_path.rstrip('/')}/ocdbt/{graph_id}")
    fork_kvs = ts.KvStore.open(fork_dir).result()
    data_prefix = f"{graph_id}_d/"
    edit_files = fork_kvs.list(
        ts.KvStore.KeyRange(data_prefix, data_prefix[:-1] + chr(ord("/") + 1))
    ).result()
    if len(edit_files) > 0:
        # Steady state — fork has progressed forward by design.
        return False
    base = _base_ocdbt_path(ws_path)
    base_kvs = ts.KvStore.open(base).result()
    base_manifest = base_kvs.read("manifest.ocdbt").result().value
    fork_manifest = fork_kvs.read("manifest.ocdbt").result().value
    if base_manifest == fork_manifest:
        return False
    fork_kvs.write("manifest.ocdbt", base_manifest).result()
    logger.note(f"refreshed fork manifest at {fork_dir} from base (no edits)")
    return True


def get_seg_source_and_destination_ocdbt(
    ws_path: str,
    graph_id: str,
    config: OcdbtConfig,
    *,
    pinned_at: "int | str | None" = None,
) -> tuple:
    """Open source watershed + CG's delta OCDBT destination (all scales).

    Always uses the fork-based kvstack spec. Requires the base to exist and
    the fork's manifest to be present (set up at ingest time).

    When ``pinned_at`` is set, the destination OCDBT handles are opened
    read-only at that version — used by the recovery path to read
    pre-op seg values via ``ChunkedGraphMeta.pinned_seg_reads``.

    Returns:
        (src_list, dst_list, resolutions): per-scale TensorStore handles
        and [x,y,z] resolutions.
    """
    scales = _read_source_scales(ws_path)
    resolutions = [s["resolution"] for s in scales]
    cg_kvstore = build_cg_ocdbt_spec(ws_path, graph_id, config, pinned_at=pinned_at)

    src_list, dst_list = [], []
    for i in range(len(scales)):
        src_i = ts.open(
            {"driver": "neuroglancer_precomputed", "kvstore": ws_path, "scale_index": i}
        ).result()
        dst_i = _open_precomputed_scale(cg_kvstore, i, **_schema_from_src(src_i))
        src_list.append(src_i)
        dst_list.append(dst_i)
    return src_list, dst_list, resolutions


def copy_ws_chunk(
    source,
    destination,
    chunk_size: tuple,
    coords: list,
    voxel_bounds: np.ndarray,
):
    """Copy one chunk from source watershed to OCDBT destination at the same scale.

    Coordinates are interpreted at the source/destination's native scale —
    callers must pre-scale them when copying coarser MIP levels.
    """
    coords = np.array(coords, dtype=int)
    chunk_size = np.array(chunk_size, dtype=int)
    vx_start = coords * chunk_size + voxel_bounds[:, 0]
    vx_end = vx_start + chunk_size
    xE, yE, zE = voxel_bounds[:, 1]

    x0, y0, z0 = vx_start
    x1, y1, z1 = vx_end
    x1 = min(x1, xE)
    y1 = min(y1, yE)
    z1 = min(z1, zE)

    data = source[x0:x1, y0:y1, z0:z1].read().result()
    destination[x0:x1, y0:y1, z0:z1].write(data).result()


def copy_ws_chunk_multiscale(
    src_list,
    dst_list,
    resolutions,
    chunk_size: tuple,
    coords: list,
    voxel_bounds: np.ndarray,
):
    """Copy a base-resolution chunk's physical region across all MIP scales.

    The graph's chunk grid is defined at base resolution. For each coarser
    scale we copy the SAME physical region — voxel coordinates are divided
    by the cumulative downsample factor (derived from resolution ratios).
    Source already has correct data at every scale, so this is a pure copy
    with no recomputation.
    """
    assert len(src_list) == len(dst_list) == len(resolutions)
    coords = np.array(coords, dtype=int)
    chunk_size_arr = np.array(chunk_size, dtype=int)
    base_res = np.array(resolutions[0])

    # Physical region at base resolution.
    vx_start_base = coords * chunk_size_arr + voxel_bounds[:, 0]
    vx_end_base = np.minimum(vx_start_base + chunk_size_arr, voxel_bounds[:, 1])

    for i, (src, dst) in enumerate(zip(src_list, dst_list)):
        # Cumulative factor from base to this scale (e.g. [2,2,1] per level).
        factor = (np.array(resolutions[i]) / base_res).astype(int)
        x0, y0, z0 = vx_start_base // factor
        x1, y1, z1 = vx_end_base // factor
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            logger.debug(f"skipping empty region at scale {i}")
            continue
        data = src[x0:x1, y0:y1, z0:z1].read().result()
        dst[x0:x1, y0:y1, z0:z1].write(data).result()


def copy_ws_bbox_multiscale(
    src_list,
    dst_list,
    resolutions,
    bbox_lo: np.ndarray,
    bbox_hi: np.ndarray,
    dump_tag: str | None = None,
):
    """Copy a base-resolution voxel bbox across all MIP scales under one
    transaction so the whole multi-scale write lands as a single OCDBT commit.

    The transaction (not ``atomic=True``) is what's load-bearing: it batches
    every per-chunk underlying-kvstore write across every scale into one
    commit, so the d/ file count for one call is constant in bbox size and
    grows only with scale count. ``atomic=True`` would add cross-key
    isolation but is rejected by tensorstore's distributed-OCDBT path —
    when the kvstore is opened with a ``coordinator``, atomic transactions
    cannot span multiple keys (verified empirically). Non-atomic still
    batches; the coordinator handles concurrency by serializing the commit
    on the wire.

    Passing the source TensorStore directly into ``write(...)`` lets
    tensorstore stream the copy without materializing an intermediate
    numpy array in Python — peak RSS drops by roughly one scale's
    worth versus the read-into-numpy-then-write pattern.
    """
    assert len(src_list) == len(dst_list) == len(resolutions)
    dump_enabled = bool(environ.get("ERROR_DUMP"))
    base_res = np.array(resolutions[0])
    txn = ts.Transaction()
    # per_scale rows are only populated when dump_enabled, so the failure
    # path has enough context for the structured GCS dump without paying any
    # bookkeeping cost on the happy path.
    per_scale: list = []
    for i, (src, dst) in enumerate(zip(src_list, dst_list)):
        factor = (np.array(resolutions[i]) / base_res).astype(int)
        x0, y0, z0 = bbox_lo // factor
        x1, y1, z1 = bbox_hi // factor
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        if dump_enabled:
            dims = (int(x1 - x0), int(y1 - y0), int(z1 - z0))
            nvox = dims[0] * dims[1] * dims[2]
            bpv = int(np.dtype(dst.dtype.numpy_dtype).itemsize)
            # The precomputed driver's read_chunk shape includes a channel
            # axis; the spatial chunk shape is the first three dims.
            chunk_shape = tuple(int(s) for s in dst.chunk_layout.read_chunk.shape[:3])
            n_keys = int(
                np.prod(
                    [int(np.ceil(d / c)) if c else 0 for d, c in zip(dims, chunk_shape)]
                )
            )
            max_raw_per_key = int(np.prod(chunk_shape)) * bpv
            per_scale.append(
                (i, dims, nvox, nvox * bpv, chunk_shape, n_keys, max_raw_per_key)
            )
        dst.with_transaction(txn)[x0:x1, y0:y1, z0:z1].write(
            src[x0:x1, y0:y1, z0:z1]
        ).result()
    try:
        txn.commit_async().result()
    except Exception as exc:
        if dump_enabled:
            payload = bbox_failure_payload(
                exc,
                dump_tag,
                bbox_lo,
                bbox_hi,
                resolutions,
                per_scale,
                dst_list[0],
                src_list[0],
            )
            path = dump_failure_to_gcs(payload, dump_tag)
            if path:
                logger.note(f"OCDBT commit failure dump → {path}")
        raise


def _mode_downsample(data: np.ndarray, factors: tuple) -> np.ndarray:
    """Mode downsample 4D segmentation array [X,Y,Z,C] by per-axis factors.

    Mode (most-frequent label) is the correct downsampling for segmentation:
    it preserves exact label IDs (no interpolation) and biases toward the
    dominant label in each block.

    Fast path for 2x2x1: uses a vectorized 4-element pairwise comparison.
    Among 4 voxels {a,b,c,d}, if any value appears at least twice it is the
    mode. Order of comparisons biases ties toward the top-left corner, which
    is the standard convention for segmentation downsampling.
    """
    fx, fy, fz = factors
    X, Y, Z, C = data.shape

    # Pad with edge values so dimensions are divisible by the factor.
    # Using 'edge' (not zeros) avoids introducing a phantom background label.
    pad = [(0, (-X % fx) % fx), (0, (-Y % fy) % fy), (0, (-Z % fz) % fz), (0, 0)]
    if any(p[1] > 0 for p in pad):
        data = np.pad(data, pad, mode="edge")
    X, Y, Z, C = data.shape

    if fx == 2 and fy == 2 and fz == 1:
        # Fast vectorized path for the common 2x2x1 case.
        reshaped = data.reshape(X // 2, 2, Y // 2, 2, Z, C)
        a = reshaped[:, 0, :, 0]
        b = reshaped[:, 0, :, 1]
        c = reshaped[:, 1, :, 0]
        d = reshaped[:, 1, :, 1]
        return np.where(
            (a == b) | (a == c) | (a == d),
            a,
            np.where((b == c) | (b == d), b, np.where(c == d, c, a)),
        )

    if fx == 2 and fy == 2 and fz == 2:
        # 2x2x2 (8-element mode) — strided subsample is fast and label-safe
        # for typical segmentation where adjacent voxels share labels.
        return data[::2, ::2, ::2]

    # Generic factor: reshape into blocks, take strided first element.
    # This is label-safe but loses the mode property; downsampling factor
    # ratios in production are 2x2x1 or 2x2x2 so the fast paths cover them.
    reshaped = data.reshape(X // fx, fx, Y // fy, fy, Z // fz, fz, C)
    return reshaped[:, 0, :, 0, :, 0]


def propagate_to_coarser_scales(dst_scales, resolutions, base_slices):
    """Cascade-downsample data from base scale through all coarser scales.

    Called after writing to the base scale (e.g. after an SV split). Each
    coarser scale reads from the level below it (not from base directly),
    so total downsample cost shrinks geometrically — each level processes
    1/N the data of the previous one.

    Args:
        dst_scales: TensorStore handles, one per MIP level.
        resolutions: [x,y,z] resolution arrays per scale, used to derive
            per-axis downsample factors from consecutive resolution ratios.
        base_slices: tuple of 3 slices (x, y, z) covering the region written
            at base resolution.
    """
    prev_slices = base_slices
    for i in range(1, len(dst_scales)):
        # Per-axis downsample factor from actual resolution ratio.
        # Never hardcoded — different datasets may have different ratios.
        factor = (np.array(resolutions[i]) / np.array(resolutions[i - 1])).astype(int)

        # Map prev-level slices to this level's coordinates.
        # Ceil division on stop ensures we cover any partial block.
        target_slices = tuple(
            slice(s.start // f, -(-s.stop // f)) for s, f in zip(prev_slices, factor)
        )

        data = dst_scales[i - 1][prev_slices + (slice(None),)].read().result()
        downsampled = _mode_downsample(data, tuple(int(f) for f in factor))
        dst_scales[i][target_slices + (slice(None),)].write(downsampled).result()

        prev_slices = target_slices


def write_seg_chunks(meta, seg_writes):
    """Write a flat batch of pre-sliced L2 chunks to OCDBT in parallel.

    ``seg_writes`` is the aggregated output of ``edits_sv.split_supervoxels``
    across every rep in an operation — each pair is one L2 chunk's worth
    of ``(voxel_slices, data)``. Flattening across reps matters: one
    ``write_seg_chunks`` call fires every chunk write in one parallel
    tensorstore batch instead of serializing rep-by-rep.

    Only chunks that actually received new SV IDs appear here; gap
    chunks between cross-chunk-connected pieces and neighbor chunks the
    overlap read touched are skipped by the split planner.

    Coarser MIP levels stay the downsample worker's job — it picks up
    the pubsub message ``publish_edit`` sends after this returns.

    Args:
        meta: ChunkedGraphMeta with ``ws_ocdbt`` (base-scale handle).
        seg_writes: iterable of ``(voxel_slices, data)`` pairs, where
            ``voxel_slices`` is a 3-tuple of ``slice`` objects covering one
            L2 chunk's x/y/z extent and ``data`` is the 3D label block
            (shape matches the slice extents).
    """
    if is_dry_run():
        return
    futures = [
        meta.ws_ocdbt[voxel_slices + (slice(None),)].write(data[..., np.newaxis])
        for voxel_slices, data in seg_writes
    ]
    for f in futures:
        f.result()
