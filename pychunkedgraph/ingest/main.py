"""
Ingest / create chunkedgraph on a single machine / instance
"""

from typing import Dict
from itertools import product
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import cpu_count
from multiprocessing import current_process
from multiprocessing.synchronize import RLock

import numpy as np

from .types import ChunkTask
from .manager import IngestionManager
from .ingestion import create_atomic_chunk_helper
from .ingestion import create_parent_chunk_helper

NUMBER_OF_PROCESSES = cpu_count()
STOP_SIGNAL = "stop"


def _stop_signal(task_queue):
    for _ in range(NUMBER_OF_PROCESSES):
        task_queue.put(STOP_SIGNAL)


def worker(
    manager: Manager,
    task_queue: Queue,
    parent_children_count_d_shared: Dict[str, int],
    parent_children_count_d_locks: Dict[str, RLock],
    im_info: dict,
):
    for func, args in iter(task_queue.get, STOP_SIGNAL):
        task = func(*args)
        if not task:
            continue
        if task.layer > task.cg_meta.layer_count:
            _stop_signal(task_queue)
            continue
        parent_children_count_d_shared.pop(task.id, None)
        parent_children_count_d_locks.pop(task.id, None)

        parent = task.parent_task()
        parent_children_count_d_locks[parent.id] = manager.RLock()
        task_queue.put(
            (
                create_parent_chunk_helper,
                (
                    parent_children_count_d_shared,
                    parent_children_count_d_locks,
                    im_info,
                    task,
                ),
            )
        )


def start_ingest(imanager: IngestionManager, n_workers: int = NUMBER_OF_PROCESSES):
    atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
    atomic_chunks = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(atomic_chunks)

    task_queue = Queue()
    with Manager() as manager:
        parent_children_count_d_shared = manager.dict()
        parent_children_count_d_locks = manager.dict()  # pylint: disable=no-member

        common_args = [
            parent_children_count_d_shared,
            parent_children_count_d_locks,
            imanager.get_serialized_info(),
        ]
        for coords in atomic_chunks:
            task = ChunkTask(imanager.cg.meta, np.array(coords, dtype=np.int))
            task_queue.put((create_atomic_chunk_helper, (*common_args, task,),))
            parent_children_count_d_locks[task.parent_task().id] = None

        for parent_id in parent_children_count_d_locks:
            parent_children_count_d_locks[parent_id] = manager.RLock()

        print("parent_children_count_d_locks", len(parent_children_count_d_locks))

        processes = []
        for _ in range(n_workers):
            processes.append(
                Process(target=worker, args=(manager, task_queue, *common_args))
            )
            processes[-1].start()

        print(f"{n_workers} started.")
        for proc in processes:
            proc.join()
