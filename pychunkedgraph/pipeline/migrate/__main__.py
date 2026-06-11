"""Container entrypoint: ``python -m pychunkedgraph.pipeline.migrate``."""

import sys

from .worker import main

if __name__ == "__main__":
    sys.exit(main())
