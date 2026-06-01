import traceback
import multiprocessing

import numpy as np
from taskqueue import RegisteredTask

from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing.stitch import chunk_initial_sharded_stitching_task_mp


def _mesh_chunk(cg_name, layer, chunk_id, mip, cache):
    """Mesh one chunk. Top-level (not a method/closure) so it is picklable as the
    forked-child target. Resets the inherited cloudfiles connection pool so the
    child's reads use fresh sockets, not the parent's."""
    from cloudfiles import reset_connection_pools

    reset_connection_pools()
    chunk_id = np.uint64(chunk_id)
    if layer == 2:
        return meshgen.chunk_initial_mesh_task(
            cg_name, chunk_id, None, mip=mip, sharded=True, cache=cache
        )
    return chunk_initial_sharded_stitching_task_mp(cg_name, chunk_id, mip, cache=cache)


def _run_in_child(conn, fn, args):
    """Forked-child entry: run ``fn``, send the outcome back, exit. Module-level so
    the forked child can import it as the Process target."""
    try:
        conn.send(("ok", fn(*args)))
    except BaseException:  # pylint: disable=broad-except
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class MeshTask(RegisteredTask):
    def __init__(self, cg_name, layer, chunk_id, mip, cache=True):
        super().__init__(cg_name, layer, chunk_id, mip, cache)

    def execute(self):
        """Mesh the chunk in a fresh forked child that exits when done. A long-lived
        poll process accumulates dirty network state (grpc/s2n/SSL/tensorstore +
        cloudfiles sockets) and unreleased heap across chunks; a worker pool forked
        from that dirty parent inherits both, which corrupts sharded reads and bloats
        memory. The child's exit returns all of it to the OS, so the next chunk forks
        from a clean parent. Re-raises the child's failure so the poll loop fails the
        task loudly (and restarts the pod)."""
        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        args = (self.cg_name, self.layer, int(self.chunk_id), self.mip, self.cache)
        proc = ctx.Process(target=_run_in_child, args=(child_conn, _mesh_chunk, args))
        proc.start()
        child_conn.close()
        try:
            status, payload = parent_conn.recv()
        except EOFError:
            proc.join()
            raise RuntimeError(
                f"mesh child died without a result (exitcode {proc.exitcode})"
            )
        finally:
            parent_conn.close()
        proc.join()
        if status == "error":
            raise RuntimeError(f"mesh child raised:\n{payload}")
        return payload
