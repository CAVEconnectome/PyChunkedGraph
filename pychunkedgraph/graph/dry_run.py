import os
from contextlib import contextmanager

DRY_RUN_ENV = "PCG_DRY_RUN"


def is_dry_run() -> bool:
    """True iff ``PCG_DRY_RUN=1`` in the environment.

    When true, every write function in the edit flow
    (``operation._write``, the operation log writes, ``write_seg_chunks``,
    and the lock acquire/release paths) returns early without
    persisting. Used by debug tooling to re-run edits against
    production BT/OCDBT state without mutating it.

    Strict ``"1"`` match so unset / empty / ``"true"`` / typos do not
    accidentally trigger in production.
    """
    return os.environ.get(DRY_RUN_ENV) == "1"


@contextmanager
def dry_run_scope():
    """Set ``PCG_DRY_RUN=1`` for the duration of the block; restore on exit.

    Single point for set/restore of the env var. The caller's
    pre-existing value (including absence) is restored even if the
    block raises.
    """
    prev = os.environ.get(DRY_RUN_ENV)
    os.environ[DRY_RUN_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(DRY_RUN_ENV, None)
        else:
            os.environ[DRY_RUN_ENV] = prev
