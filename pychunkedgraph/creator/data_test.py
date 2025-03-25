import glob
import collections
import numpy as np

from pychunkedgraph.creator import creator_utils
from multiwrapper import multiprocessing_utils as mu


def _test_unique_edge_assignment_thread(args):
    paths = args[0]

    id_dict = collections.Counter()
    for path in paths:
        try:
            ids = creator_utils.read_edge_file_h5(path)["edge_ids"]
        except:
            ids = creator_utils.read_edge_file_h5(path)["node_ids"]
        u_ids = np.unique(ids)

        u_id_d = dict(zip(u_ids, np.ones(len(u_ids), dtype=int)))
        add_counter = collections.Counter(u_id_d)

        id_dict += add_counter

    # return np.array(list(id_dict.items()))
    return id_dict


def test_unique_edge_assignment(dir, n_threads=128):
    file_paths = glob.glob(dir + "/*")

    file_chunks = np.array_split(file_paths, n_threads * 3)
    multi_args = []
    for i_file_chunk, file_chunk in enumerate(file_chunks):
        multi_args.append([file_chunk])

    # Run parallelizing
    if n_threads == 1:
        results = mu.multiprocess_func(_test_unique_edge_assignment_thread,
                                       multi_args, n_threads=n_threads,
                                       verbose=True, debug=n_threads==1)
    else:
        results = mu.multiprocess_func(_test_unique_edge_assignment_thread,
                                       multi_args, n_threads=n_threads)

    id_dict = collections.Counter()
    for result in results:
        # id_dict += collections.Counter(dict(result))
        id_dict += result

    return id_dict