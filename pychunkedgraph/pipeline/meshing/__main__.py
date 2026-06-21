"""Container entrypoint: ``python -m pychunkedgraph.pipeline.meshing``."""

from cave_pipeline.distribution import run_and_exit
from .worker import main

if __name__ == "__main__":
    run_and_exit(main)
