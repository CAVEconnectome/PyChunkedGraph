from taskqueue import TaskQueue, LocalTaskQueue, MockTaskQueue
import argparse
from pychunkedgraph.graph.chunkedgraph import ChunkedGraph # noqa
import numpy as np
from pychunkedgraph.meshing.meshing_sqs import MeshTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue_name', type=str, default=None)
    parser.add_argument('--chunk_start', nargs=3, type=int)
    parser.add_argument('--chunk_end', nargs=3, type=int)
    parser.add_argument('--cg_name', type=str)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--mip', type=int)

    args = parser.parse_args()
    cv_graphene_path = 'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v0'
    cv_mesh_dir = 'graphene_meshes'

    cg = ChunkedGraph(graph_id=args.cg_name)

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
                yield MeshTask(args.cg_name, int(chunk_id), args.mip, cv_graphene_path, cv_mesh_dir)

    if args.queue_name is not None:
        with TaskQueue(queue_name=args.queue_name) as tq:
            tq.insert_all(MeshTaskIterator(chunks_arr))
    else:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(MeshTaskIterator(chunks_arr))