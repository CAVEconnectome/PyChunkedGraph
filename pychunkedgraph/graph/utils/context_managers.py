import builtins
import traceback
from time import time

indent = 0

builtin_print = builtins.print


def foo(*args, **kwargs):
    builtin_print(indent * " ", *args, **kwargs)


class TimeIt:
    def __init__(self, message="", tablen=2):
        self._message = message
        self._start = None
        self._tablen = tablen

        global indent
        indent += tablen

    def __enter__(self):
        print(f"start {self._message}")
        self._start = time()
        builtins.print = foo

    def __exit__(self, *args):
        print(f"end {self._message}: {time()-self._start}\n")
        global indent
        indent -= self._tablen
        builtins.print = builtin_print
