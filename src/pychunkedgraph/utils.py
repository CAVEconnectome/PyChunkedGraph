import numpy as np
from multiprocessing import cpu_count, Process
from multiprocessing.pool import Pool
import time


def read_edge_file_cv(cv_st, path):
    """ Reads the edge ids and affinities from an edge file """

    dt = np.dtype('uint64')
    # dt = dt.newbyteorder('>')
    edge_buffer = np.frombuffer(cv_st.get_file(path), dtype=dt).reshape(-1, 3)
    edge_ids = edge_buffer[:, :2]
    edge_affs = np.frombuffer(edge_buffer[:, -1].tobytes(), dtype=np.float32)[::2]

    return edge_ids, edge_affs


def read_mapping(cv_st, path):
    """ Reads the mapping information from a file """

    return np.frombuffer(cv_st.get_file(path), dtype=np.uint64).reshape(-1, 2)


def start_multiprocess(func, params, debug=False, verbose=False, nb_cpus=None):

    if nb_cpus is None:
        nb_cpus = max(cpu_count(), 1)

    if debug:
        nb_cpus = 1

    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), nb_cpus))

    start = time.time()
    if not debug:
        pool = Pool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = []
        for p in params:
            result.append(func(p))

    if verbose:
        print("\nTime to compute grid: %.3fs" % (time.time() - start))

    return result