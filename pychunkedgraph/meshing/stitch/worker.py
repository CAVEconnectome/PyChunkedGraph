# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import DracoPy
from cloudvolume import CloudVolume
from cloudfiles.exceptions import DecompressionError, IntegrityError
from cloudvolume.exceptions import SpecViolation
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from pychunkedgraph.graph.chunkedgraph import ChunkedGraph
from pychunkedgraph.graph.chunks import utils as chunk_utils
from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing import meshgen_utils
from pychunkedgraph.meshing.stitch import utils as su
from pychunkedgraph.profiler import get_profiler

# Per-process state, built once per worker and reused across batches. The cg is
# constructed inside the worker (from the picklable graph_id) rather than passed
# from the parent — a live cg carries tensorstore/bigtable handles that don't
# survive fork; rebuilding per process is the fork-safe pattern used by ingest.
_CG = None
_CV = None


def _get_cg(cg_info):
    global _CG
    if _CG is None:
        _CG = ChunkedGraph(**cg_info)
    return _CG


def _get_cv(info):
    global _CV
    if _CV is None:
        _CV = CloudVolume(
            "graphene://https://localhost/segmentation/table/dummy", info=info
        )
    return _CV


@retry(
    retry=retry_if_exception_type((DecompressionError, IntegrityError, SpecViolation)),
    stop=stop_after_attempt(7),
    wait=wait_random_exponential(0.5, 60.0),
    reraise=True,
)
def _fetch_fragments(cv, meta, labels, mesh_subdir=None):
    """Byte-range fetch of just ``labels``' draco bytes from a sharded mesh dir,
    grouped by layer. Goes straight to the sharded reader (the inner call of
    get_meshes_on_bypass) to skip its per-label probe of the nonexistent
    ``dynamic`` path. ``mesh_subdir`` selects the dir under ``mesh_path`` (default
    is the production sharded mesh dir); the compare harness passes its
    ``test_*`` / ``initial`` prefix. Returns ``{label:int -> raw draco bytes}``;
    coalesces adjacent byte ranges and amortizes minishard-index reads across
    all labels in the batch."""
    labels_arr = np.fromiter(labels, dtype=np.uint64, count=len(labels))
    by_layer = defaultdict(list)
    for label, layer_id in zip(
        labels_arr, chunk_utils.get_chunk_layers(meta, labels_arr)
    ):
        by_layer[int(layer_id)].append(int(label))

    mesh_meta = cv.mesh.meta
    if mesh_subdir is None:
        mesh_subdir = mesh_meta.sharded_mesh_dir
    out = {}
    for layer_id, layer_labels in by_layer.items():
        subdir = mesh_meta.join(mesh_meta.mesh_path, mesh_subdir, str(layer_id))
        for label, raw in (
            cv.mesh.readers[layer_id].get_data(layer_labels, path=subdir).items()
        ):
            if raw is not None:
                out[int(label)] = raw
    return out


def _stitch_one(
    prof, meta, fragment_bytes, parent_id, descendants, boundary, layer, mip_info
):
    """Stitch one parent from prefetched ``fragment_bytes`` ({label: raw draco
    bytes}). Descendants are pre-resolved and their bytes batch-fetched by the
    caller; this does decode -> transform -> merge -> encode. Each substage is a
    ``prof.profile`` block (timing only — RSS is sampled once per batch by the
    enclosing ``stitch`` block); ``profile`` is a no-op when profiling is off."""
    n_missing = sum(1 for d in descendants if int(d) not in fragment_bytes)

    old_fragments = []
    first_options = None
    for child_node_id in descendants:
        raw = fragment_bytes.get(int(child_node_id))
        if raw is None:
            continue
        with prof.profile("decode"):
            mesh = meshgen.decode_draco_mesh_buffer(raw)
            eo = mesh["encoding_options"]
            mesh["encoding_options_qr"] = eo.quantization_range
            mesh["encoding_options_qb"] = eo.quantization_bits
            fragment_layer = chunk_utils.get_chunk_layer(meta, np.uint64(child_node_id))
        with prof.profile("transform"):
            geom = su.compute_layer_geometry(
                meta, np.uint64(child_node_id), layer, mip_info
            )
            options = su.apply_draco_transform(mesh, geom, fragment_layer, layer)
        if first_options is None:
            first_options = options
        old_fragments.append({"mesh": mesh, "node_id": np.uint64(child_node_id)})

    if not old_fragments:
        return (int(parent_id), None, 0, False, n_missing)

    with prof.profile("merge"):
        new_fragment = su.merge_draco_meshes_across_boundaries_pure(
            old_fragments, boundary
        )
    vx_ct = len(new_fragment["vertices"])
    try:
        with prof.profile("encode"):
            new_fragment_b = DracoPy.encode_mesh_to_buffer(
                new_fragment["vertices"], new_fragment["faces"], **first_options
            )
    except Exception:
        print(f"failed to merge {parent_id}")
        return (int(parent_id), None, vx_ct, True, n_missing)

    return (int(parent_id), new_fragment_b, vx_ct, False, n_missing)


def _split_in_two(items):
    """Partition ``items`` into two contiguous halves for prefetch overlap.
    Returns one non-empty group when there are fewer than two items (never an
    empty group, so the fetch/stitch loop has nothing to skip)."""
    if len(items) < 2:
        return [items]
    mid = len(items) // 2
    return [items[:mid], items[mid:]]


def _group_labels(group):
    """Flat set of descendant labels for one group of ``(parent_id, descendants)``."""
    return {int(d) for _, descs in group for d in descs}


def _resolve_descendants(cg, parents):
    """``[(parent_id, descendant_ids), ...]`` for ``parents``
    (``[(parent_id, [immediate_child_id,...]), ...]``).

    ``get_downstream_multi_child_nodes`` maps each input node to its descendant
    independently and preserves input order, so resolving all parents' children
    in ONE call and slicing the result back per parent gives the identical
    per-parent descendant lists as a call-per-parent — with far fewer bigtable
    round-trips. Order is preserved (concatenate, resolve, slice are all
    order-preserving), which matters: the stitch encodes with the first surviving
    fragment's options, so a reorder would change output bytes."""
    all_children = [np.uint64(c) for _, child_ids in parents for c in child_ids]
    if not all_children:
        return [(parent_id, np.array([], dtype=np.uint64)) for parent_id, _ in parents]
    resolved = meshgen_utils.get_downstream_multi_child_nodes(
        cg, np.array(all_children, dtype=np.uint64)
    )
    out = []
    offset = 0
    for parent_id, child_ids in parents:
        out.append((parent_id, resolved[offset : offset + len(child_ids)]))
        offset += len(child_ids)
    return out


def stitch_parents_batch(batch):
    """Stitch a batch of parents in one worker process. ``batch`` =
    ``{cg_info, info, mip_info, layer, boundary, parents}`` where ``parents`` is
    ``[(parent_id, [child_node_id,...]), ...]``, ``cg_info`` is
    ``cg.get_serialized_info()`` ({graph_id}), ``info`` is the graphene CV info
    dict, and ``mip_info`` is a ``MipInfo``. No ``meta.cv``/tensorstore is
    touched in the worker (it is fork-hostile). Returns ``(results, blocks)``
    where ``blocks`` is this batch's per-stage ``BlockMetrics`` (a no-op empty
    list when profiling is disabled), rolled up across batches by the parent."""
    cg = _get_cg(batch["cg_info"])
    cv = _get_cv(batch["info"])
    meta = cg.meta
    boundary = batch["boundary"]
    layer = batch["layer"]
    mip_info = batch["mip_info"]

    prof = get_profiler()
    prof.reset()

    # Resolve every parent's descendants in one batched graph read; each half's
    # fragments are then byte-range fetched together so adjacent ranges coalesce
    # and the minishard index is read once per half.
    with prof.profile("graph"):
        descendants_per_parent = _resolve_descendants(cg, batch["parents"])

    # Split the batch into two halves and prefetch the second half's fragments
    # while the first half stitches: read is IO (GIL-released) and stitch is CPU,
    # so they overlap. A single fetch thread runs at a time, so get_data is never
    # called concurrently and at most two halves' fragment bytes are in flight
    # (lower peak than fetching the whole batch up front). The "read" block times
    # only the blocking wait on each half's fetch — the read NOT hidden behind
    # stitch — so it trends toward 0 when the overlap fully covers it.
    groups = _split_in_two(descendants_per_parent)

    results = []
    with ThreadPoolExecutor(max_workers=1) as fetch_pool:
        pending = fetch_pool.submit(
            _fetch_fragments, cv, meta, _group_labels(groups[0])
        )
        for i, group in enumerate(groups):
            with prof.profile("read"):
                fragment_bytes = pending.result()
            if i + 1 < len(groups):
                pending = fetch_pool.submit(
                    _fetch_fragments, cv, meta, _group_labels(groups[i + 1])
                )
            # one RSS sample pair per batch wraps the substages (light); the
            # substages themselves are timed individually inside _stitch_one.
            with prof.profile("stitch", sampled_rss=True):
                for parent_id, descendants in group:
                    results.append(
                        _stitch_one(
                            prof,
                            meta,
                            fragment_bytes,
                            parent_id,
                            descendants,
                            boundary,
                            layer,
                            mip_info,
                        )
                    )
    return results, prof.blocks
