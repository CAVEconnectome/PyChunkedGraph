"""
Ingest / create chunkedgraph on a single machine / instance
"""

import time
import random
from itertools import product
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import cpu_count
from multiprocessing import current_process

from numpy.random import shuffle

from .types import ChunkTask
from .manager import IngestionManager
from .ingestion import create_atomic_chunk_helper
from .ingestion import create_parent_chunk_helper


def worker(task_queue):
    for func, args in iter(task_queue.get, "STOP"):
        parent_task = func(*args)
        task_queue.put(
            (
                create_atomic_chunk_helper,
                (
                    parent_children_count_d_shared,
                    parent_children_count_d_lock,
                    imanager.get_serialized_info(),
                    coords,
                ),
            )
        )


def start_ingest(imanager: IngestionManager):
    NUMBER_OF_PROCESSES = cpu_count()
    atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
    chunks_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    shuffle(chunks_coords)

    task_queue = Queue()
    with Manager() as manager:
        parent_children_count_d_shared = manager.dict()
        parent_children_count_d_lock = manager.RLock()  # pylint: disable=no-member

        for coords in chunks_coords:
            task_queue.put(
                (
                    create_atomic_chunk_helper,
                    (
                        parent_children_count_d_shared,
                        parent_children_count_d_lock,
                        imanager.get_serialized_info(),
                        coords,
                    ),
                )
            )

        for _ in range(NUMBER_OF_PROCESSES):
            Process(target=worker, args=(task_queue)).start()
