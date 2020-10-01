import builtins
import traceback
from time import time

indent = 0

builtin_print = builtins.print


def foo(*args, **kwargs):
    indent_str = ""
    for _ in range(2, indent+1, 2):
        indent_str += 2 * " "
        indent_str += "|"
    builtin_print(indent_str, *args, **kwargs)


class TimeIt:
    def __init__(self, message="", *args, **kwargs):
        self._message = message
        self._args = args
        self._kwargs = kwargs
        self._start = None

    def __enter__(self):
        builtins.print = foo
        print(f"start {self._message}")
        global indent
        indent += 2
        if self._args:
            args_str = " ".join(str(x) for x in self._args)
            print(args_str)
        if self._kwargs:
            kwargs_str = " ".join(f"{k}:{v}" for k, v in self._kwargs)
            print(kwargs_str)
        self._start = time()

    def __exit__(self, *args):
        builtins.print = foo
        global indent
        indent -= 2
        print(f"end {self._message} -- {time()-self._start}")
        builtins.print = builtin_print
