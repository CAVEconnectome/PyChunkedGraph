"""Offline gates for the fork-per-chunk mesh task and the in-process heap reclaim.

No live cg / cloud: the fork test monkeypatches the chunk worker, the trim test
exercises the glibc-arena reclaim directly. Both target the failure modes the
change fixes: a chunk forked from a process that already meshed a prior chunk
inheriting its dirty state, and freed mesh-buffer arenas staying resident under
bare gc.
"""

import os
import gc

import pytest

from pychunkedgraph.meshing import meshing_sqs
from pychunkedgraph.meshing.stitch import task


def _rss_bytes():
    with open(f"/proc/{os.getpid()}/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def test_execute_runs_each_chunk_in_a_fresh_exited_child(monkeypatch):
    def fake(cg_name, layer, chunk_id, mip, cache):
        return {"pid": os.getpid(), "ppid": os.getppid()}

    monkeypatch.setattr(meshing_sqs, "_mesh_chunk", fake)
    parent = os.getpid()

    r1 = meshing_sqs.MeshTask("cg", 6, 111, 0, True).execute()
    r2 = meshing_sqs.MeshTask("cg", 6, 222, 0, True).execute()

    # work ran in a child forked from this process, not inline
    assert r1["pid"] != parent
    assert r1["ppid"] == parent
    # each task forks its own child that has already exited (pids not aliased)
    assert r2["pid"] != r1["pid"]
    assert os.getpid() == parent


def test_execute_reraises_child_failure(monkeypatch):
    def boom(*_args):
        raise ValueError("synthetic-child-failure")

    monkeypatch.setattr(meshing_sqs, "_mesh_chunk", boom)
    parent = os.getpid()

    with pytest.raises(RuntimeError) as exc:
        meshing_sqs.MeshTask("cg", 6, 333, 0, True).execute()

    # the child traceback crosses back so the poll loop fails loudly
    assert "synthetic-child-failure" in str(exc.value)
    assert "ValueError" in str(exc.value)
    assert os.getpid() == parent  # parent survived the child error


@pytest.mark.skipif(task._MALLOC_TRIM is None, reason="malloc_trim is glibc-only")
def test_malloc_trim_reclaims_what_gc_leaves_resident():
    import numpy as np

    base = _rss_bytes()
    # fragmenting pattern: keep every 10th allocation alive so freeing the rest
    # leaves holes glibc retains — bare gc cannot return these to the OS
    keep, tmp = [], []
    for i in range(400_000):
        a = np.empty(2048, dtype=np.uint8)
        (keep if i % 10 == 0 else tmp).append(a)
    del tmp
    gc.collect()
    after_gc = _rss_bytes()
    task._malloc_trim()
    after_trim = _rss_bytes()
    del keep

    held_by_gc = after_gc - base
    reclaimed = after_gc - after_trim
    # gc alone leaves a large resident footprint; malloc_trim returns most of it
    assert held_by_gc > 100 * 2**20
    assert reclaimed > held_by_gc / 2
