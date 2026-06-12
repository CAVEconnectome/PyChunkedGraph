"""Container entrypoint: ``python -m pychunkedgraph.pipeline.migrate``."""

import os
import sys

from .worker import main

if __name__ == "__main__":
    code = main()
    # os._exit, not sys.exit: the bigtable.data client leaves a non-daemon
    # channel-refresh thread that atexit join()s forever, so a normal exit would
    # hang until the pod's grace period SIGKILLs it (losing the real exit code).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
