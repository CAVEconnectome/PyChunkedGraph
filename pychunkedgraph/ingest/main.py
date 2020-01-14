"""
Ingest / create chunkedgraph on a single machine / instance
"""

from typing import Dict
from itertools import product
from multiprocessing import RLock
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import cpu_count
from multiprocessing import current_process

import numpy as np

from .types import ChunkTask
from .manager import IngestionManager
from .ingestion import create_atomic_chunk_helper
from .ingestion import create_parent_chunk_helper

NUMBER_OF_PROCESSES = cpu_count()


def _stop_signal(task_queue):
    for _ in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")


def worker(
    task_queue: Queue,
    parent_children_count_d_shared: Dict,
    parent_children_count_d_lock: RLock,
    im_info: dict,
):
    for func, args in iter(task_queue.get, "STOP"):
        parent_task = func(*args)
        if not parent_task:
            continue
        if parent_task.layer > parent_task.cg_meta.layer_count:
            _stop_signal(task_queue)
            continue
        task_queue.put(
            (
                create_parent_chunk_helper,
                (
                    parent_children_count_d_shared,
                    parent_children_count_d_lock,
                    im_info,
                    parent_task,
                ),
            )
        )


def start_ingest(imanager: IngestionManager):
    atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
    atomic_chunks = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(atomic_chunks)

    task_queue = Queue()
    with Manager() as manager:
        parent_children_count_d_shared = manager.dict()
        parent_children_count_d_lock = manager.RLock()  # pylint: disable=no-member

        common_args = [
            parent_children_count_d_shared,
            parent_children_count_d_lock,
            imanager.get_serialized_info(),
        ]
        for coords in atomic_chunks:
            task_queue.put(
                (
                    create_atomic_chunk_helper,
                    (*common_args, np.array(coords, dtype=np.int),),
                )
            )

        processes = []
        for _ in range(NUMBER_OF_PROCESSES):
            processes.append(Process(target=worker, args=(task_queue, *common_args)))

        for proc in processes:
            proc.start()
            proc.join()
