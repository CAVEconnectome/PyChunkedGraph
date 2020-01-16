"""
Ingest / create chunkedgraph on a single machine / instance
"""

import threading
from typing import Dict
from itertools import product
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import cpu_count
from multiprocessing import current_process
from multiprocessing.synchronize import Lock

import numpy as np

from .types import ChunkTask
from .manager import IngestionManager
from .ingestion import create_atomic_chunk_helper
from .ingestion import create_parent_chunk_helper


NUMBER_OF_PROCESSES = cpu_count()


def _progress(
    imanager: IngestionManager, layer_task_counts_d_shared: Dict, task_queue: Queue
):
    t = threading.Timer(
        120.0, _progress, args=((imanager, layer_task_counts_d_shared, task_queue))
    )
    t.start()

    result = []
    for layer in range(2, imanager.cg_meta.layer_count):
        result.append(f"{layer} {layer_task_counts_d_shared.get(layer, 0)}")
    print(f"{' '.join(result)} | {task_queue.qsize()}")


def _enqueue_parent(
    task_queue: Queue,
    im_info: dict,
    parent: ChunkTask,
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: Lock,
) -> None:
    with parent_children_count_d_lock:
        if not parent.id in parent_children_count_d_shared:
            # set initial number of child chunks
            parent_children_count_d_shared[parent.id] = len(parent.children_coords)

        # decrement child count by 1
        parent_children_count_d_shared[parent.id] -= 1
        # if zero, all dependents complete -> return parent
        if parent_children_count_d_shared[parent.id] == 0:
            del parent_children_count_d_shared[parent.id]
            task_queue.put((create_parent_chunk_helper, (im_info, parent,),))


def worker(
    task_queue: Queue,
    parent_children_count_d_shared: Dict[str, int],
    parent_children_count_d_lock: Lock,
    layer_task_counts_d_shared: Dict[int, int],
    layer_task_counts_d_lock: Lock,
    im_info: dict,
):
    while not task_queue.empty():
        func, args = task_queue.get()
        task = func(*args)
        with layer_task_counts_d_lock:
            layer_task_counts_d_shared[task.layer] += 1

        parent = task.parent_task()
        if parent.layer > parent.cg_meta.layer_count:
            break

        _enqueue_parent(
            task_queue,
            im_info,
            parent,
            parent_children_count_d_shared,
            parent_children_count_d_lock,
        )


def start_ingest(imanager: IngestionManager, n_workers: int = NUMBER_OF_PROCESSES):
    atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
    atomic_chunks = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(atomic_chunks)

    dependency_mgr = Manager()
    lock_manager = Manager()
    stats_mgr = Manager()
    task_queue = Queue()
    parent_children_count_d_shared = dependency_mgr.dict()
    parent_children_count_d_lock = lock_manager.Lock()  # pylint: disable=no-member
    layer_task_counts_d_shared = stats_mgr.dict()
    layer_task_counts_d_lock = lock_manager.Lock()  # pylint: disable=no-member

    for layer in range(2, imanager.cg_meta.layer_count):
        layer_task_counts_d_shared[layer] = 0

    for coords in atomic_chunks:
        task = ChunkTask(imanager.cg.meta, np.array(coords, dtype=np.int))
        task_queue.put(
            (create_atomic_chunk_helper, (imanager.get_serialized_info(), task,),)
        )

    processes = []
    args = (
        task_queue,
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        layer_task_counts_d_shared,
        layer_task_counts_d_lock,
        imanager.get_serialized_info(),
    )
    for _ in range(n_workers):
        processes.append(Process(target=worker, args=args))
        processes[-1].start()

    _progress(imanager, layer_task_counts_d_shared, task_queue)
    print(f"{n_workers} workers started.")
    for proc in processes:
        proc.join()

    print("Complete.")
    dependency_mgr.shutdown()
    lock_manager.shutdown()
    stats_mgr.shutdown()
