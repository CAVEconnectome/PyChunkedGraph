# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import os
import math
import time
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

from pychunkedgraph import get_logger
from pychunkedgraph.graph import attributes
from pychunkedgraph.graph.chunkedgraph import ChunkedGraph
from pychunkedgraph.graph.utils.generic import filter_failed_node_ids
from pychunkedgraph.meshing import meshgen_utils
from pychunkedgraph.meshing.stitch import utils as su
from pychunkedgraph.meshing.stitch.worker import stitch_parents_batch
from pychunkedgraph.profiler import HierarchicalProfiler, cgroup_peak_bytes
from pychunkedgraph.scheduling import WorkItem, lpt_partition, n_bins_for

logger = get_logger(__name__)


def _multi_child_parents(cg, chunk_id):
    """Whole-chunk scan -> ``{parent_id: [immediate_child_id,...]}`` for valid
    parents with >1 child. Failed/incomplete-ingest nodes are dropped via
    ``filter_failed_node_ids`` (the ingest-canonical filter; local dedup on the
    range-read, no extra bigtable). Descendant resolution is left to workers."""
    child_col = attributes.Hierarchy.Child
    range_read = cg.range_read_chunk(chunk_id, properties=child_col)

    row_ids = []
    children = []
    max_children_ids = []
    for row_id, row_data in range_read.items():
        child_ids = row_data[0].value
        row_ids.append(row_id)
        children.append(child_ids)
        max_children_ids.append(np.max(child_ids))

    row_ids = np.array(row_ids, dtype=np.uint64)
    segment_ids = np.array([cg.get_segment_id(r) for r in row_ids])
    valid = set(
        int(x) for x in filter_failed_node_ids(row_ids, segment_ids, max_children_ids)
    )

    return {
        int(row_id): [int(c) for c in child_ids]
        for row_id, child_ids in zip(row_ids, children)
        if int(row_id) in valid and len(child_ids) > 1
    }


class _Results:
    """Collects per-parent worker results into the shard's label->bytes map and
    reporting stats, order-independent. Gathers every batch's profiler blocks
    for a cross-process rollup."""

    def __init__(self):
        self.merged_meshes = {}
        self.bad_meshes = []
        self.biggest_frag = 0
        self.biggest_frag_vx_ct = 0
        self.n_processed = 0
        self.n_missing_fragments = 0
        self.blocks = []

    def record_batch(self, results, blocks):
        self.blocks.extend(blocks)
        for parent_id, frag_bytes, vx_ct, is_bad, n_missing in results:
            self.n_processed += 1
            self.n_missing_fragments += n_missing
            if vx_ct > self.biggest_frag_vx_ct:
                self.biggest_frag = parent_id
                self.biggest_frag_vx_ct = vx_ct
            if is_bad:
                self.bad_meshes.append(parent_id)
            elif frag_bytes is not None:
                self.merged_meshes[parent_id] = frag_bytes


def _make_batches(
    multi_child_nodes, cg_info, info, mip_info, boundary, layer, batch_size
):
    """Yield worker batches. Each batch carries only IDs + the picklable
    cg_info ({graph_id}), the graphene info dict, the MipInfo, and the shared
    boundary — no live cg/meta, no tensorstore, no mesh bytes in the parent.

    Parents are LPT-partitioned (``scheduling.lpt_partition``) into bins balanced
    by immediate-child count (a free proxy for encode cost), so a giant parent
    seeds its own bin and runs solo on one worker while the rest pack the others
    — the straggler fix. Bins come back heaviest-first, so the giant's bin is
    dispatched first under the pull-based pool."""
    items = [
        WorkItem(
            payload=(int(parent_id), [int(c) for c in multi_child_nodes[parent_id]]),
            weight=len(multi_child_nodes[parent_id]),
        )
        for parent_id in multi_child_nodes
    ]
    for bin_items in lpt_partition(items, n_bins_for(len(items), batch_size)):
        yield {
            "cg_info": cg_info,
            "info": info,
            "mip_info": mip_info,
            "layer": layer,
            "boundary": boundary,
            "parents": [it.payload for it in bin_items],
        }


def _assert_no_fragment_loss(acc, n_parents, multi_child_nodes):
    assert (
        acc.n_processed == n_parents
    ), f"result/parent mismatch: {acc.n_processed} results vs {n_parents} parents"
    assert len(acc.merged_meshes) + len(acc.bad_meshes) <= acc.n_processed, (
        f"accounting: {len(acc.merged_meshes)} merged + {len(acc.bad_meshes)} bad "
        f"> {acc.n_processed} processed"
    )
    parent_keys = {int(k) for k in multi_child_nodes}
    assert (
        set(acc.merged_meshes) | set(acc.bad_meshes) <= parent_keys
    ), "produced a mesh for an id that is not a multi_child_nodes parent"


def _write_shard(
    cv, sharding_spec, layer, chunk_id, merged_meshes, out_subdir, cache_string
):
    t = time.time()
    shard_binary = sharding_spec.synthesize_shard(merged_meshes)
    synth_s = time.time() - t
    shard_filename = cv.mesh.readers[layer].get_filename(chunk_id)
    cf = CloudFiles(
        os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, out_subdir, str(layer))
    )
    logger.note(
        "synthesized shard (%.2f GB, %s labels) in %.1fs; uploading %s/%s/%s",
        len(shard_binary) / 1e9,
        len(merged_meshes),
        synth_s,
        out_subdir,
        int(layer),
        shard_filename,
    )
    t = time.time()
    cf.put(
        shard_filename,
        shard_binary,
        content_type="application/octet-stream",
        compress=False,
        cache_control=cache_string,
    )
    return len(shard_binary), synth_s, time.time() - t


def chunk_initial_sharded_stitching_task_mp(
    cg_name,
    chunk_id,
    mip,
    cg=None,
    high_padding=1,
    cache=True,
    n_processes=None,
    out_subdir="initial",
    max_parents=None,
):
    """Parallel sharded stitch. Main does only the serial work (one graph query,
    fan-out, collect, synthesize, upload); each worker gets ``cg.meta`` + the
    JSON info once and fetches/stitches its own child shards. Output is
    mesh-equivalent to ``meshgen.chunk_initial_sharded_stitching_task``.

    ``out_subdir`` overrides the write location (default ``"initial"``).
    ``max_parents`` caps the number of parents stitched (debug preview of the
    full pipeline + report); they are sampled from the smaller half (medium down
    to tiny), skipping the giant stragglers so the preview stays fast. The shard
    it writes is partial — never use it as ``initial``.
    """
    start_time = time.time()
    if n_processes is None:
        n_processes = (
            int(os.environ.get("PCG_MESH_STITCH_WORKERS", 0)) or os.cpu_count()
        )
    if max_parents is not None:
        # The pool must fork before any network I/O (s2n atfork), so cap workers
        # to the batch count up front — workers beyond n_batches never get a
        # batch. n_parents <= max_parents, so this batch count is the worst case.
        capped_batch_size = su.derive_batch_size(max_parents, max(n_processes, 1))
        n_batches = max(1, math.ceil(max_parents / capped_batch_size))
        n_processes = min(n_processes, n_batches)

    # Fork the worker pool FIRST, before the parent touches the network. The
    # forked children re-init their own TLS/cg; forking AFTER the parent has
    # done cloud I/O trips s2n's pthread_atfork guard ("fork() detected") and
    # kills the child. mp.Pool forks eagerly at construction (unlike
    # ProcessPoolExecutor, which forks lazily on first submit).
    pool = None if n_processes == 1 else mp.Pool(n_processes)

    try:
        if cg is None:
            cg = ChunkedGraph(graph_id=cg_name)
        cache_string = "public" if cache else "no-cache"

        layer = cg.get_chunk_layer(chunk_id)
        info = meshgen_utils.get_json_info(cg)
        cv = CloudVolume(
            "graphene://https://localhost/segmentation/table/dummy", info=info
        )
        sharding_spec = ShardingSpecification.from_dict(
            cv.mesh.meta.info["sharding"][str(layer)]
        )

        cg_info = cg.get_serialized_info()
        mip_info = su.MipInfo.from_meta(cg.meta, mip)
        multi_child_nodes = _multi_child_parents(cg, chunk_id)
        if max_parents is not None:
            multi_child_nodes = su.sample_debug(multi_child_nodes, max_parents)
        n_parents = len(multi_child_nodes)
        boundary = su.compute_merge_boundary(cg.meta, chunk_id, mip_info, high_padding)
        batch_size = su.derive_batch_size(n_parents, max(n_processes, 1))
        logger.note(
            "chunk %s layer %s: %s parents, %s workers, batch_size=%s, out=%s",
            int(chunk_id),
            int(layer),
            n_parents,
            n_processes,
            batch_size,
            out_subdir,
        )

        acc = _Results()
        progress = tqdm(total=n_parents, desc="stitch", unit="parent")
        batches = _make_batches(
            multi_child_nodes, cg_info, info, mip_info, boundary, layer, batch_size
        )

        if pool is None:
            for batch in batches:
                results, blocks = stitch_parents_batch(batch)
                acc.record_batch(results, blocks)
                progress.update(len(results))
        else:
            for results, blocks in pool.imap_unordered(stitch_parents_batch, batches):
                acc.record_batch(results, blocks)
                progress.update(len(results))
        progress.close()
        join_s = 0.0
        if pool is not None:
            # imap_unordered is fully consumed here, so every result is already
            # in acc. A graceful close()/join() blocks for minutes while each
            # worker's exit waits on grpc's non-daemon C-core threads to drain;
            # terminate() SIGTERMs them immediately and loses no results.
            t = time.time()
            pool.terminate()
            pool.join()
            join_s = time.time() - t
            logger.note(
                "all batches collected; worker pool terminated in %.1fs", join_s
            )
    finally:
        if pool is not None:
            pool.terminate()

    _assert_no_fragment_loss(acc, n_parents, multi_child_nodes)
    if acc.n_missing_fragments:
        logger.warning(
            "%s expected child fragments were missing (absent in initial shards)",
            acc.n_missing_fragments,
        )

    shard_bytes, synth_s, upload_s = _write_shard(
        cv,
        sharding_spec,
        layer,
        chunk_id,
        acc.merged_meshes,
        out_subdir,
        cache_string,
    )
    total_time = time.time() - start_time
    # Whole-cgroup peak: parent + all forked workers + the synthesize tail (the
    # parent-side high-water moment). This is what a k8s pod's memory limit
    # enforces, so it sizes the pod's memory request. None when not under a
    # memory cgroup (dev box / macOS).
    peak_rss_bytes = cgroup_peak_bytes()
    logger.note(
        "%s/%s/%s (%.2f GB); peak rss %s; total %.1fs",
        out_subdir,
        int(layer),
        cv.mesh.readers[layer].get_filename(chunk_id),
        shard_bytes / 1e9,
        "n/a" if peak_rss_bytes is None else f"{peak_rss_bytes / 1e9:.2f} GB",
        total_time,
    )

    profiler = HierarchicalProfiler.from_blocks(acc.blocks)

    return {
        "chunk_id": chunk_id,
        "total_time": total_time,
        "biggest_frag": acc.biggest_frag,
        "biggest_frag_vx_ct": acc.biggest_frag_vx_ct,
        "number_frag": acc.n_processed,
        "bad meshes": acc.bad_meshes,
        "missing_fragments": acc.n_missing_fragments,
        "peak_rss_bytes": peak_rss_bytes,
        "profiler": profiler,
        "stage_blocks": acc.blocks,
        "join_s": join_s,
        "synth_s": synth_s,
        "upload_s": upload_s,
    }
