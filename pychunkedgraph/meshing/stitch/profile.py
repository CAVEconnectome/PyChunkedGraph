"""Profile + correctness harness for the parallel sharded stitching task.

Drives ``stitch.task.chunk_initial_sharded_stitching_task_mp`` on a real chunk,
writing to a ``test_ref`` / ``test_par`` output prefix (never production), and
compares the produced shard against production for mesh-equivalence. Each entry
point is a single call returning a report, for interactive use.
"""

import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from tqdm import tqdm
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume

from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing import meshgen_utils
from pychunkedgraph.meshing.stitch import task as stitch_task
from pychunkedgraph.meshing.stitch import worker as stitch_worker
from pychunkedgraph.profiler import HierarchicalProfiler, get_profiler

_METRICS_ROOT = Path(tempfile.gettempdir()) / "pcg_mesh_profile"


def make_cv(cg):
    return CloudVolume(
        "graphene://https://localhost/segmentation/table/dummy",
        info=meshgen_utils.get_json_info(cg),
    )


def resolve_chunk_id(cg, layer, shard_id, cv=None):
    """Find the layer-``layer`` chunk_id whose shard filename is
    ``{shard_id}-0.shard`` (inverts ``decode_chunk_position_number``).
    Returns the single matching ``chunk_id`` (raises if 0 or >1 match)."""
    if cv is None:
        cv = make_cv(cg)
    bounds = cg.meta.layer_chunk_bounds[layer]
    matches = []
    for x in range(int(bounds[0])):
        for y in range(int(bounds[1])):
            for z in range(int(bounds[2])):
                cid = cg.get_chunk_id(layer=layer, x=x, y=y, z=z)
                if int(cv.meta.decode_chunk_position_number(cid)) == int(shard_id):
                    matches.append(int(cid))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 layer-{layer} chunk for shard {shard_id}, "
            f"got {len(matches)}: {[hex(m) for m in matches]}"
        )
    return np.uint64(matches[0])


def _table(title, headers, rows):
    """Render an aligned monospace table (no dict/json/wall-of-text)."""
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(col[i]) for col in cols) for i in range(len(headers))]
    sep = "  "
    out = [
        title,
        sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        sep.join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in rows:
        out.append(sep.join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def _profiler_text(ret):
    """Capture the cross-process rollup + the per-batch percentile breakdown
    (the straggler view) into a string for the metrics file."""
    buf = StringIO()
    with redirect_stdout(buf):
        ret["profiler"].metrics_report()
        HierarchicalProfiler.percentile_report(ret["stage_blocks"])
    return buf.getvalue()


def _run(cg, chunk_id, mip, out_subdir, n_processes, high_padding, max_parents=None):
    """Run the stitch task, build the bottleneck table, write it to
    ``metrics.txt`` and print it. Returns ``ret`` + ``wall`` + ``metrics_path``.
    ``max_parents`` caps parents for a quick full-pipeline + report preview."""
    target = _METRICS_ROOT / cg.graph_id / out_subdir / str(int(chunk_id))
    target.mkdir(parents=True, exist_ok=True)
    tb_path = target / "traceback.txt"
    if tb_path.exists():
        tb_path.unlink()

    # Enable the shared profiler for this run only (default off in production).
    # mp.Pool forks inside the task AFTER this, so workers inherit the enabled
    # instance. with_memory off: tracemalloc is the heavy path. with_rss off:
    # the default RSS path spawns a sampler thread per block, which would fire on
    # every per-fragment substage; RSS is taken once per batch via the explicit
    # sampled_rss "stitch" block instead, keeping the hot substages timing-only.
    prof = get_profiler()
    prof.enabled = True
    prof.with_memory_default = False
    prof.with_rss_default = False

    t = time.time()
    try:
        ret = stitch_task.chunk_initial_sharded_stitching_task_mp(
            cg.graph_id,
            chunk_id,
            mip,
            cg=cg,
            high_padding=high_padding,
            cache=False,
            n_processes=n_processes,
            out_subdir=out_subdir,
            max_parents=max_parents,
        )
    except Exception:
        tb_path.write_text(traceback.format_exc())
        print(f"[stitch_profile] run FAILED; traceback saved to {tb_path}")
        raise
    finally:
        prof.enabled = False
    ret["wall"] = time.time() - t

    report = (
        _table(
            "run",
            [
                "wall_s",
                "total_s",
                "n_parents",
                "bad",
                "biggest_frag_vx_ct",
                "join_s",
                "synth_s",
                "upload_s",
                "peak_rss_gb",
            ],
            [
                [
                    f"{ret['wall']:.1f}",
                    f"{ret['total_time']:.1f}",
                    ret["number_frag"],
                    len(ret["bad meshes"]),
                    ret["biggest_frag_vx_ct"],
                    f"{ret['join_s']:.1f}",
                    f"{ret['synth_s']:.1f}",
                    f"{ret['upload_s']:.1f}",
                    (
                        "n/a"
                        if ret["peak_rss_bytes"] is None
                        else f"{ret['peak_rss_bytes'] / 1e9:.2f}"
                    ),
                ]
            ],
        )
        + "\n\n"
        + _profiler_text(ret)
        + "\n"
    )

    (target / "metrics.txt").write_text(report)
    ret["metrics_path"] = str(target / "metrics.txt")
    print(report)
    return ret


def load_traceback(cg, chunk_id, out_subdir="test_par"):
    """Return the saved traceback for the last failed run under ``out_subdir``,
    or ``None`` if the last run succeeded."""
    tb_path = (
        _METRICS_ROOT / cg.graph_id / out_subdir / str(int(chunk_id)) / "traceback.txt"
    )
    return tb_path.read_text() if tb_path.exists() else None


def run_parallel(
    cg, chunk_id, mip, out_subdir, high_padding=1, n_processes=None, max_parents=None
):
    """Parallel run -> writes shard under ``out_subdir``. The caller picks the
    location so a partial ``max_parents`` preview and a full run don't collide."""
    return _run(
        cg,
        chunk_id,
        mip,
        out_subdir,
        n_processes,
        high_padding,
        max_parents=max_parents,
    )


def _shard_label_set(cg, chunk_id, subdir, cv=None):
    """The set of labels in the shard under ``subdir``, read from the minishard
    index only (no full-shard download). ``None`` if the shard is absent."""
    if cv is None:
        cv = make_cv(cg)
    layer = cg.get_chunk_layer(chunk_id)
    reader = cv.mesh.readers[layer]
    shard_filename = reader.get_filename(chunk_id)
    path = os.path.join(cv.mesh.meta.mesh_path, subdir, str(layer))
    if not CloudFiles(os.path.join(cv.cloudpath, path)).exists(shard_filename):
        return None
    return set(int(x) for x in reader.list_labels(shard_filename, path=path))


def _shard_head(cg, chunk_id, subdir, cv=None):
    """``(crc32c, size)`` of the shard under ``subdir`` without downloading it,
    or ``None`` if absent."""
    if cv is None:
        cv = make_cv(cg)
    layer = cg.get_chunk_layer(chunk_id)
    shard_filename = cv.mesh.readers[layer].get_filename(chunk_id)
    cf = CloudFiles(
        os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, subdir, str(layer))
    )
    if not cf.exists(shard_filename):
        return None
    h = cf.head(shard_filename)
    return h["Content-Crc32c"], h["Content-Length"]


def _canonical(mesh):
    """Sort a decoded mesh's vertices so order-differences don't matter."""
    v = np.asarray(mesh["vertices"]).reshape(-1, 3)
    order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
    return v[order], int(len(mesh["faces"]))


def _decode_label_pair(args):
    label, bytes_a, bytes_b = args
    va, fa = _canonical(meshgen.decode_draco_mesh_buffer(bytes_a))
    vb, fb = _canonical(meshgen.decode_draco_mesh_buffer(bytes_b))
    return label, (va.shape == vb.shape and fa == fb and np.array_equal(va, vb))


def compare_shards(
    cg,
    chunk_id,
    subdir_a="test_ref",
    subdir_b="test_par",
    cv=None,
    n_processes=None,
    subset=False,
    fail_fast=True,
):
    """Gate: mesh-equivalence + label-set parity between two shards.

    Fast path: equal crc32c -> byte-identical, returned instantly (no download).
    Otherwise: read both shards' label sets from the minishard index (no
    full-shard download), check label-set parity (the prior failure mode =
    dropped fragments) + count, then byte-range fetch only the labels to compare
    (reusing the stitch worker's fetch) and decode them in parallel, asserting
    identical vertices/faces. ``fail_fast`` (default) raises on the first
    mismatched label without decoding the rest; ``fail_fast=False`` scans every
    label and raises once with the full mismatch count + a sample.

    ``subset=True`` verifies ``subdir_b`` is a correct SUBSET of ``subdir_a``
    (for a partial ``max_parents`` run vs full production): every ``b`` label
    must exist in ``a`` and decode-match; ``a`` may have labels ``b`` doesn't.
    A label in ``b`` but not ``a`` is still a failure (a stray/wrong id). The
    crc32c fast path is skipped since a subset can't be byte-identical."""
    if cv is None:
        cv = make_cv(cg)

    head_a = _shard_head(cg, chunk_id, subdir_a, cv)
    head_b = _shard_head(cg, chunk_id, subdir_b, cv)
    assert head_a is not None, f"{subdir_a} shard missing"
    assert head_b is not None, f"{subdir_b} shard missing"
    if not subset and head_a[0] is not None and head_a == head_b:
        return {
            "subdir_a": subdir_a,
            "subdir_b": subdir_b,
            "crc32c": head_a[0],
            "size": head_a[1],
            "result": "BYTE-IDENTICAL",
        }

    la = _shard_label_set(cg, chunk_id, subdir_a, cv)
    lb = _shard_label_set(cg, chunk_id, subdir_b, cv)
    only_b = lb - la
    assert not only_b, (
        f"{len(only_b)} labels in {subdir_b} not in {subdir_a} (stray/wrong id): "
        f"{list(only_b)[:5]}..."
    )
    if not subset:
        only_a = la - lb
        assert not only_a, (
            f"LABEL SET MISMATCH (dropped fragments): "
            f"{len(only_a)} only in {subdir_a} ({list(only_a)[:5]}...)"
        )
        assert len(la) == len(lb), f"count mismatch {len(la)} vs {len(lb)}"

    # Byte-range fetch only the labels under comparison (lb) from each shard's
    # mesh dir — the same targeted fetch the worker uses, not a whole-shard
    # download + disassemble.
    a = stitch_worker._fetch_fragments(cv, cg.meta, lb, mesh_subdir=subdir_a)
    b = stitch_worker._fetch_fragments(cv, cg.meta, lb, mesh_subdir=subdir_b)
    work = [(lbl, a[lbl], b[lbl]) for lbl in lb]
    mismatched = []
    with ProcessPoolExecutor(max_workers=n_processes or os.cpu_count()) as ex:
        for label, equal in tqdm(
            ex.map(_decode_label_pair, work, chunksize=16),
            total=len(work),
            desc="compare",
            unit="label",
        ):
            if not equal:
                if fail_fast:
                    raise AssertionError(f"label {label} differs in decoded mesh")
                mismatched.append(label)
    assert (
        not mismatched
    ), f"{len(mismatched)} labels differ in decoded mesh: {mismatched[:10]}"
    return {
        "n_labels": len(b),
        "subdir_a": subdir_a,
        "subdir_b": subdir_b,
        "result": "SUBSET-EQUIVALENT" if subset else "MESH-EQUIVALENT",
    }


def check_parent_resolution(cg, chunk_id):
    """Gate: the new parent-side filter + worker-side descendant resolution
    yields the SAME ``{parent: set(descendants)}`` as the original
    ``meshgen.get_multi_child_nodes``. Catches a regression in the reshape that
    moved descendant resolution from the (whole-flat) parent call to per-parent
    worker calls. Returns a report; raises on mismatch."""
    ref, _ = meshgen.get_multi_child_nodes(cg, chunk_id)
    ref = {int(k): set(int(c) for c in v) for k, v in ref.items()}

    immediate = stitch_task._multi_child_parents(cg, chunk_id)
    got = {}
    for parent_id, child_ids in immediate.items():
        desc = meshgen_utils.get_downstream_multi_child_nodes(
            cg, np.array(child_ids, dtype=np.uint64)
        )
        got[int(parent_id)] = set(int(c) for c in desc)

    only_ref, only_got = set(ref) - set(got), set(got) - set(ref)
    assert not only_ref and not only_got, (
        f"PARENT SET MISMATCH: {len(only_ref)} only in meshgen, "
        f"{len(only_got)} only in new ({list(only_ref)[:5]} / {list(only_got)[:5]})"
    )
    bad = [p for p in ref if ref[p] != got[p]]
    assert not bad, f"{len(bad)} parents have different descendants: {bad[:5]}"
    return {"n_parents": len(ref), "result": "PARENT RESOLUTION MATCHES meshgen"}


def compare_against_production(
    cg, chunk_id, subdir, cv=None, subset=False, fail_fast=True
):
    """Full correctness gate against the production ``initial`` shard on GCS:
    label-set parity + per-label decoded mesh equality (crc32c fast path first).

    ``subset=True`` for a partial ``max_parents`` run: verify every produced
    label exists in production and decode-matches (a subset can't be byte- or
    set-identical, but each mesh it did produce must still match production).
    ``fail_fast`` (default) raises on the first mismatched label; pass
    ``fail_fast=False`` to scan all and report the full mismatch count."""
    return compare_shards(
        cg,
        chunk_id,
        subdir_a="initial",
        subdir_b=subdir,
        cv=cv,
        subset=subset,
        fail_fast=fail_fast,
    )
