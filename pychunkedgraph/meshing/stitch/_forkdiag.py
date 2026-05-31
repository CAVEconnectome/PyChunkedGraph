"""Pinpoint which parent-side op arms the s2n/grpc pthread_atfork guard that
aborts forked children doing TLS. Run in a FRESH kernel via debug.ipynb.

After each candidate parent op, a one-shot mp.Pool(1) worker does a real GCS
TLS op (the exact failure path). The first op after which the pool goes
OK -> FAIL is the one that makes the process fork-hostile."""

import os
import multiprocessing as mp

import numpy as np
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume

from pychunkedgraph.graph import attributes
from pychunkedgraph.graph.chunkedgraph import ChunkedGraph
from pychunkedgraph.meshing import meshgen_utils
from pychunkedgraph.meshing.stitch import utils as su
from pychunkedgraph.meshing.stitch import task as st
from pychunkedgraph.meshing.stitch.worker import stitch_parents_batch

_GCS = "gs://pcg_ws/pinky_sv_size_exp/20240920152649/graphene_meshes/initial/6"


def _worker_gcs(_):
    return CloudFiles(_GCS).exists("200-0.shard")


def _worker_rebuild_cg(cg_info):
    """What the real stitch worker does first: rebuild cg + touch meta.cv."""
    cg = ChunkedGraph(**cg_info)
    _ = meshgen_utils.get_json_info(cg)
    _ = cg.meta.cv.mip_resolution(0)
    return True


def _worker_descendants(args):
    """Child does ONLY the bigtable descendant resolution (get_children)."""
    cg_info, child_ids = args
    cg = ChunkedGraph(**cg_info)
    meshgen_utils.get_downstream_multi_child_nodes(
        cg, np.array(child_ids, dtype=np.uint64)
    )
    return True


def _worker_disassemble(args):
    """Child does ONLY graphene cv build + fetch + disassemble_shard."""
    cg_info, layer = args
    cg = ChunkedGraph(**cg_info)
    cv = CloudVolume(
        "graphene://https://localhost/segmentation/table/dummy",
        info=meshgen_utils.get_json_info(cg),
    )
    cf = CloudFiles(os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, "initial"))
    content = cf.get(f"{layer}/200-0.shard")
    if content is not None:
        cv.mesh.readers[layer].disassemble_shard(content)
    return True


def _pool_ok(fn, arg):
    try:
        with mp.Pool(1) as p:
            p.map(fn, [arg])
        return True
    except Exception:
        return False


def run(graph_id, chunk_id):
    """Localize the abort: is it the parent doing a bigtable/gRPC RPC, the child
    doing GCS/TLS (s2n), or the combination? Each line forks a fresh mp.Pool(1)
    GCS worker after a different amount of parent work. FAIL = the child
    hard-aborted (s2n/grpc atfork). No spawn, no env changes."""
    print("(a) parent did NOTHING -> child GCS:")
    print("   ", "OK" if _pool_ok(_worker_gcs, 0) else "FAIL")

    cg = ChunkedGraph(graph_id=graph_id)
    print("(b) parent built cg -> child GCS:")
    print("   ", "OK" if _pool_ok(_worker_gcs, 0) else "FAIL")

    cg.range_read_chunk(chunk_id, properties=attributes.Hierarchy.Child)
    print("(c) parent did bigtable range_read -> child GCS:")
    print("   ", "OK" if _pool_ok(_worker_gcs, 0) else "FAIL")

    cg_info = cg.get_serialized_info()
    print("(d) child rebuilds cg + get_json_info + meta.cv.mip_resolution:")
    print("   ", "OK" if _pool_ok(_worker_rebuild_cg, cg_info) else "FAIL")

    # (e) the REAL worker on one real batch (rebuild cg + graphene cv +
    # disassemble shard + descendant resolution + stitch).
    mip_info = su.MipInfo.from_meta(cg.meta, 0)
    mcn = st._multi_child_parents(cg, chunk_id)
    one_parent, one_children = next(iter(mcn.items()))
    boundary = su.compute_merge_boundary(cg.meta, chunk_id, mip_info, 1)
    batch = {
        "cg_info": cg_info,
        "info": meshgen_utils.get_json_info(cg),
        "mip_info": mip_info,
        "layer": int(cg.get_chunk_layer(chunk_id)),
        "boundary": boundary,
        "parents": [(one_parent, one_children)],
    }
    print("(e) REAL worker stitch_parents_batch on one parent:")
    print("   ", "OK" if _pool_ok(stitch_parents_batch, batch) else "FAIL")
