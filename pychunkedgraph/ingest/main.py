"""
Ingest / create chunkedgraph on a single machine / instance
"""

from typing import Dict
from typing import Optional
from typing import Iterable
from threading import Timer
from datetime import datetime
from itertools import product
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import cpu_count
from multiprocessing.synchronize import Lock
from multiprocessing.managers import SyncManager
from traceback import format_exc

import numpy as np

from .types import ChunkTask
from .manager import IngestionManager
from .ingestion import create_atomic_chunk_helper
from .ingestion import create_parent_chunk_helper


NUMBER_OF_PROCESSES = cpu_count() - 1
STOP_SENTINEL = "STOP"


def _display_progess(
    imanager: IngestionManager,
    layer_task_counts_d_shared: Dict,
    task_queue: Queue,
    interval: float,
):
    t = Timer(
        interval,
        _display_progess,
        args=((imanager, layer_task_counts_d_shared, task_queue, interval)),
    )
    t.daemon = True
    t.start()

    try:
        result = []
        for layer in range(2, imanager.cg_meta.layer_count + 1):
            layer_c = layer_task_counts_d_shared.get(f"{layer}c", 0)
            layer_q = layer_task_counts_d_shared.get(f"{layer}q", 0)
            result.append(f"{layer}: ({layer_c}, {layer_q})")
        print(f"status {' '.join(result)}")
    except:
        pass


def _enqueue_parent(
    manager: SyncManager,
    task_queue: Queue,
    im_info: dict,
    parent: ChunkTask,
    parent_children_count_d_shared: Dict,
    parent_children_count_d_locks: Lock,
    time_stamp: Optional[datetime] = None,
) -> None:
    with parent_children_count_d_locks[parent.id]:
        if not parent.id in parent_children_count_d_shared:
            # set initial number of child chunks
            parent_children_count_d_shared[parent.id] = len(parent.children_coords)

        # decrement child count by 1
        parent_children_count_d_shared[parent.id] -= 1
        # if zero, all dependents complete -> return parent
        if parent_children_count_d_shared[parent.id] == 0:
            del parent_children_count_d_shared[parent.id]
            parent_children_count_d_locks[parent.parent_task().id] = manager.Lock()
            task_queue.put((create_parent_chunk_helper, (im_info, parent, time_stamp),))
            return True
    return False


def _signal_end(task_queue: Queue):
    for _ in range(NUMBER_OF_PROCESSES):
        task_queue.put(STOP_SENTINEL)


def _work(
    func: callable,
    args: Iterable,
    task_queue: Queue,
    *,
    manager: SyncManager,
    parent_children_count_d_shared: Dict[str, int],
    parent_children_count_d_locks: Lock,
    layer_task_counts_d_shared: Dict[int, int],
    layer_task_counts_d_lock: Lock,
    build_graph: bool,
    im_info: dict,
    time_stamp: Optional[datetime] = None,
):
    retry = 1
    while retry:
        try:
            task = func(*args)
            retry = 0
        except:
            print(f"{retry}: {format_exc()}")
            retry += 1

    queued = False
    if build_graph:
        parent = task.parent_task()
        if parent.layer > parent.cg_meta.layer_count:
            _signal_end(task_queue)
            return

        queued = _enqueue_parent(
            manager,
            task_queue,
            im_info,
            parent,
            parent_children_count_d_shared,
            parent_children_count_d_locks,
            time_stamp,
        )
    with layer_task_counts_d_lock:
        layer_task_counts_d_shared[f"{task.layer}c"] += 1
        layer_task_counts_d_shared[f"{task.layer}q"] -= 1
        if queued:
            layer_task_counts_d_shared[f"{parent.layer}q"] += 1


def _worker(
    task_queue: Queue, **kwargs,
):
    for func, args in iter(task_queue.get, STOP_SENTINEL):
        try:
            _work(func, args, task_queue, **kwargs)
        except:
            # requeue task
            task_queue.put((func, args,))
            print(f"requeued: {format_exc()}")


def start_ingest(
    imanager: IngestionManager,
    *,
    time_stamp: Optional[datetime] = None,
    n_workers: int = NUMBER_OF_PROCESSES,
    progress_interval: float = 300.0,
    test_chunks=None,
):
    atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
    atomic_chunks = list(product(*[range(r) for r in atomic_chunk_bounds]))

    if test_chunks:
        atomic_chunks = test_chunks

    np.random.shuffle(atomic_chunks)
    manager = Manager()
    task_queue = Queue()
    parent_children_count_d_shared = manager.dict()
    parent_children_count_d_locks = manager.dict()
    layer_task_counts_d_shared = manager.dict()
    layer_task_counts_d_lock = manager.Lock()  # pylint: disable=no-member

    for layer in range(2, imanager.cg_meta.layer_count + 1):
        layer_task_counts_d_shared[f"{layer}c"] = 0
        layer_task_counts_d_shared[f"{layer}q"] = 0

    for coords in atomic_chunks:
        task = ChunkTask(imanager.cg_meta, np.array(coords, dtype=np.int))
        task_queue.put(
            (
                create_atomic_chunk_helper,
                (imanager.get_serialized_info(), task, time_stamp,),
            )
        )
        parent_children_count_d_locks[task.parent_task().id] = None
    layer_task_counts_d_shared["2q"] += task_queue.qsize()

    if not imanager.config.build_graph:
        _signal_end(task_queue)

    for parent_id in parent_children_count_d_locks:
        parent_children_count_d_locks[
            parent_id
        ] = manager.Lock()  # pylint: disable=no-member

    processes = []
    args = (task_queue,)
    kwargs = {
        "manager": manager,
        "parent_children_count_d_shared": parent_children_count_d_shared,
        "parent_children_count_d_locks": parent_children_count_d_locks,
        "layer_task_counts_d_shared": layer_task_counts_d_shared,
        "layer_task_counts_d_lock": layer_task_counts_d_lock,
        "build_graph": imanager.config.build_graph,
        "im_info": imanager.get_serialized_info(),
        "time_stamp": time_stamp,
    }
    for _ in range(n_workers):
        processes.append(Process(target=_worker, args=args, kwargs=kwargs))
        processes[-1].start()
    print(f"{n_workers} workers started.")

    _display_progess(
        imanager, layer_task_counts_d_shared, task_queue, progress_interval
    )
    for proc in processes:
        proc.join()

    print("Complete.")
    manager.shutdown()
    task_queue.close()
