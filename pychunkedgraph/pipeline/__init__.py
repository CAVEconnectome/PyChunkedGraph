"""Kubernetes-native chunk-batch pipeline: one Indexed Job per layer, no Redis/RQ.

Workload-agnostic core (``grid`` scatter, ``lock`` per-chunk CAS, ``exit_codes`` Job
contract, ``worker`` harness) shared by per-workload subpackages (``ingest``,
``meshing``). Each worker reads its JOB_COMPLETION_INDEX, maps it to a batch of
scattered chunk coords, and processes each chunk. Self-contained and branch-portable
(main + pcgv3). Run a workload as a module, e.g. ``python -m pychunkedgraph.pipeline.ingest``.
"""

import os
import sys
import traceback


def run_and_exit(main) -> None:
    """Run a pipeline entrypoint, then ``os._exit``; every container and one-shot entry goes
    through this. A normal return can hang joining a client's non-daemon thread at exit, and
    the pod then stalls until SIGKILL. The exit code is main()'s return (0 if None), the
    SystemExit code, or 1 with a printed traceback on any unhandled error."""
    try:
        code = main() or 0
    except SystemExit as exc:  # argparse / explicit exit
        code = exc.code
        if isinstance(code, str):
            print(code, file=sys.stderr)
            code = 1
        elif not isinstance(code, int):
            code = 0 if code is None else 1
    except BaseException:  # report and exit non-zero, never hang
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
