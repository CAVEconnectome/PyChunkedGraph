import os
import threading


HOME = os.path.expanduser("~")


class NeuromniSimulator():
    def __init__(self, dir=HOME + '/benchmarking/',
                 n_clients=10):
        self._dir = dir
        self._n_clients = n_clients

    @property
    def n_clients(self):
        return self._n_clients

    @property
    def dir(self):
        return self._dir

    def run_leaves_benchmark(self, table_id):
        pass





# class Client(threading.Thread):
