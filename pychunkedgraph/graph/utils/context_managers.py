from time import time


class TimeIt:
    def __init__(self, message=""):
        self._message = message
        self._start = None

    def __enter__(self):
        self._start = time()

    def __exit__(self, *args):
        print(f"{self._message} {time()-self._start}")
