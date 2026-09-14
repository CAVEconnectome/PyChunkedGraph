"""Container entrypoint: ``python -m pychunkedgraph.pipeline.ingest``."""

from .. import run_and_exit
from .worker import main

if __name__ == "__main__":
    run_and_exit(main)
