import os
import numpy as np
import time
import sys
import threading

# Hack the imports for now
sys.path.append("..")
import pychunkedgraph.chunkedgraph as chunkedgraph

HOME = os.path.expanduser("~")


def measure_command(func, kwargs):
    """ Measures the execution time of a function

    :param func: function
    :param kwargs: dict
        keyword arguments
    :return: float, result
        (time, result of fucntion)
    """
    time_start = time.time()
    r = func(**kwargs)
    dt = time.time() - time_start

    return dt, r


class ChunkedGraphSimulator(object):
    def __init__(self, table_id, dir=HOME + '/benchmarking/', n_clients=10):
        self.cg = chunkedgraph.ChunkedGraph(table_id=table_id)

        if not dir.strip("/").endswith(table_id):
            dir += "/%s/" % table_id
        self._dir = dir
        self._n_clients = n_clients

        if not os.path.exists(dir):
            os.makedirs(dir)

    @property
    def n_clients(self):
        return self._n_clients

    @property
    def dir(self):
        return self._dir

    def read_all_rows(self, layer=None):
        """ Reads all rows of the table or specific layer

        :param layer: int or None
        :return: dict
        """
        if layer is None:
            start_layer = 1
            end_layer = 1000
        else:
            start_layer = layer
            end_layer = layer + 1

        node_id_base = np.array([0, 0, 0, 0, 0, 0, 0, start_layer],
                                dtype=np.uint8)
        node_id_base_next = node_id_base.copy()
        node_id_base_next[-1] = end_layer

        start_key = chunkedgraph.serialize_node_id(np.frombuffer(node_id_base,
                                                                 dtype=np.uint64)[0])
        end_key = chunkedgraph.serialize_node_id(np.frombuffer(node_id_base_next,
                                                               dtype=np.uint64)[0])

        range_read = self.cg.table.read_rows(start_key=start_key,
                                             end_key=end_key,
                                             end_inclusive=False)
        range_read.consume_all()

        return range_read

    def run_root_benchmark(self, n_tests=5000):
        """ Measures time of get_root command

        :param n_tests: int or None
            number of tests
        :return: list of floats
            times
        """
        atomic_rows = self.read_all_rows(layer=1)
        atomic_ids = list(atomic_rows.rows.keys())
        times = []

        if n_tests is None:
            n_tests = len(atomic_ids)
        else:
            n_tests = np.min([n_tests, len(atomic_ids)])

        np.random.shuffle(atomic_ids)

        for atomic_id in atomic_ids[: n_tests]:
            dt, _ = measure_command(self.cg.get_root,
                                    {"atomic_id": int(atomic_id),
                                     "is_cg_id": True})

            times.append(dt)

            if len(times) % 100 == 0:
                print("%d / %d - %.3fms +- %.3fms" % (len(times),
                                                      n_tests,
                                                      np.mean(times) * 1000,
                                                      np.std(times) * 1000))

        np.save(self.dir + "/root_single_%d.npy" % (n_tests), times)
        return times

    def run_all_leaves_benchmark(self, n_tests=10000, get_edges=False):
        """ Measures time of get_root command

        :param n_tests: int or None
            number of tests
        :param get_edges: bool
        :return: list of floats
            times
        """
        root_rows = self.read_all_rows(layer=6)
        root_ids = list(root_rows.rows.keys())
        times = []

        if n_tests is None:
            n_tests = len(root_ids)
        else:
            n_tests = np.min([n_tests, len(root_ids)])

        np.random.shuffle(root_ids)

        for root_id in root_ids[: n_tests]:
            if get_edges:
                dt, _ = measure_command(self.cg.get_subgraph_edges,
                                        {"agglomeration_id": int(root_id)})
            else:
                dt, _ = measure_command(self.cg.get_subgraph_nodes,
                                        {"agglomeration_id": int(root_id)})

            times.append(dt)

            if len(times) % 100 == 0:
                print("%d / %d - %.3fms +- %.3fms" % (len(times),
                                                      n_tests,
                                                      np.mean(times) * 1000,
                                                      np.std(times) * 1000))

        if get_edges:
            np.save(self.dir + "/all_leave_edges_single_%d.npy" % (n_tests),
                    times)
        else:
            np.save(self.dir + "/all_leave_nodes_single_%d.npy" % (n_tests),
                    times)
        return times




# class Client(threading.Thread):
