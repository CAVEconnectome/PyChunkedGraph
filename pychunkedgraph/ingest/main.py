import time
import random

from multiprocessing import Process, Queue, current_process

#
# Function run by worker processes
#


def worker(input, output):
    for func, args in iter(input.get, "STOP"):
        result = calculate(func, args)
        output.put(result)


#
# Function used to calculate result
#


def calculate(func, args):
    result = func(*args)
    return "%s says that %s%s = %s" % (
        current_process().name,
        func.__name__,
        args,
        result,
    )


#
# Functions referenced by tasks
#


def mul(a, b):
    time.sleep(0.5 * random.random())
    return a * b


def plus(a, b):
    time.sleep(0.5 * random.random())
    return a + b


#
#
#


def test():
    NUMBER_OF_PROCESSES = 4
    TASKS1 = [(mul, (i, 7)) for i in range(20)]
    TASKS2 = [(plus, (i, 8)) for i in range(10)]

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in TASKS1:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print("Unordered results:")
    for i in range(len(TASKS1)):
        print("\t", done_queue.get())

    # Add more tasks using `put()`
    for task in TASKS2:
        task_queue.put(task)

    # Get and print some more results
    for i in range(len(TASKS2)):
        print("\t", done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")


def _create_atomic_chunks_helper(args):
    """ helper to start atomic tasks """
    (
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        im_info,
        chunk_coords,
    ) = args
    imanager = IngestionManager(**im_info)
    chunk_coords = np.array(list(chunk_coords), dtype=np.int)
    chunk_edges_all, mapping = _get_atomic_chunk_data(imanager, chunk_coords)

    ids, affs, areas, isolated = get_chunk_data_old_format(chunk_edges_all, mapping)
    imanager.cg.add_atomic_edges_in_chunks(ids, affs, areas, isolated)
    _post_task_completion(
        parent_children_count_d_shared,
        parent_children_count_d_lock,
        imanager,
        2,
        chunk_coords,
    )


def start_ingest(imanager: IngestionManager):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    with mp.Manager() as manager:
        parent_children_count_d_shared = manager.dict()
        parent_children_count_d_lock = manager.RLock()  # pylint: disable=no-member
        multi_args = []
        for chunk_coords in chunks_coords:
            multi_args.append(
                (
                    parent_children_count_d_shared,
                    parent_children_count_d_lock,
                    imanager.get_serialized_info(),
                    chunk_coords,
                )
            )
        mu.multiprocess_func(
            _create_atomic_chunks_helper,
            multi_args,
            n_threads=min(len(multi_args), mp.cpu_count()),
        )
