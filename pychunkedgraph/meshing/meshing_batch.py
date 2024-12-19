import argparse, os
import numpy as np
from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
from taskqueue import TaskQueue, LocalTaskQueue

from pychunkedgraph.graph.chunkedgraph import ChunkedGraph # noqa
from pychunkedgraph.meshing.meshing_sqs import MeshTask
from pychunkedgraph.meshing import meshgen_utils  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue_name', type=str, default=None)
    parser.add_argument('--chunk_start', nargs=3, type=int)
    parser.add_argument('--chunk_end', nargs=3, type=int)
    parser.add_argument('--cg_name', type=str)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--mip', type=int)
    parser.add_argument('--skip_cache', action='store_true')
    parser.add_argument('--overwrite', type=bool)

    args = parser.parse_args()
    cache = not args.skip_cache

    cg = ChunkedGraph(graph_id=args.cg_name)
    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        info=meshgen_utils.get_json_info(cg),
    )
    mesh_dst = os.path.join(
        cv.cloudpath, cv.mesh.meta.mesh_path, "initial", str(args.layer)
    )
    cf = CloudFiles(mesh_dst)
    if len(list(cf.list())) > 0 and not args.overwrite:
        raise ValueError("Mesh destination is not empty. Use `overwrite` to proceed anyway.")

    chunks_arr = []
    for x in range(args.chunk_start[0],args.chunk_end[0]):
        for y in range(args.chunk_start[1], args.chunk_end[1]):
            for z in range(args.chunk_start[2], args.chunk_end[2]):
                chunks_arr.append((x, y, z))

    np.random.shuffle(chunks_arr)

    class MeshTaskIterator(object):
        def __init__(self, chunks):
            self.chunks = chunks
        def __iter__(self):
            for chunk in self.chunks:
                chunk_id = cg.get_chunk_id(layer=args.layer, x=chunk[0], y=chunk[1], z=chunk[2])
                yield MeshTask(args.cg_name, args.layer, int(chunk_id), args.mip, cache)

    if args.queue_name is not None:
        with TaskQueue(args.queue_name) as tq:
            tq.insert_all(MeshTaskIterator(chunks_arr))
    else:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(MeshTaskIterator(chunks_arr))