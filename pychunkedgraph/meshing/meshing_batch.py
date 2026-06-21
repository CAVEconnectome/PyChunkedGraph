import argparse, os
import numpy as np
from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
from taskqueue import TaskQueue, LocalTaskQueue

from pychunkedgraph.graph.chunkedgraph import ChunkedGraph  # noqa
from pychunkedgraph.meshing.meshing_sqs import MeshTask
from pychunkedgraph.meshing import meshgen_utils  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_name", type=str, default=None)
    parser.add_argument("--cg_name", type=str)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--mip", type=int)
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument(
        "--skip",
        action="store_true",
        help="do not queue a chunk whose shard already exists",
    )

    args = parser.parse_args()
    cache = not args.skip_cache

    cg = ChunkedGraph(graph_id=args.cg_name)
    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        info=meshgen_utils.get_json_info(cg),
    )
    dst = os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, "initial", str(args.layer))
    cf = CloudFiles(dst)

    bounds = cg.meta.layer_chunk_bounds[args.layer]
    chunks_arr = np.indices(tuple(int(b) for b in bounds)).reshape(3, -1).T
    np.random.shuffle(chunks_arr)

    class MeshTaskIterator(object):
        def __init__(self, chunks):
            self.chunks = chunks

        def __iter__(self):
            meshed = set(cf.list()) if args.skip else set()
            for x, y, z in self.chunks:
                chunk_id = cg.get_chunk_id(
                    layer=args.layer, x=int(x), y=int(y), z=int(z)
                )
                shard_filename = cv.mesh.readers[args.layer].get_filename(chunk_id)
                if shard_filename in meshed:
                    continue
                yield MeshTask(args.cg_name, args.layer, int(chunk_id), args.mip, cache)

    if args.queue_name is not None:
        with TaskQueue(args.queue_name) as tq:
            tq.insert_all(MeshTaskIterator(chunks_arr))
    else:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(MeshTaskIterator(chunks_arr))
