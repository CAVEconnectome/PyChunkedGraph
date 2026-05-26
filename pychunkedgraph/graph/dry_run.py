import os

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
