from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

from . import exceptions as cg_exceptions

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph


class RootLock:
    """Attempts to lock the requested root IDs using a unique operation ID.

    :raises cg_exceptions.LockingError: throws when one or more root ID locks could not be
        acquired.
    :return: The RootLock context, including the locked root IDs and the linked operation ID
    :rtype: RootLock
    """

    __slots__ = ["cg", "locked_root_ids", "lock_acquired", "operation_id"]
    # FIXME: `locked_root_ids` is only required and exposed because `cg.lock_root_loop`
    #        currently might lock different (more recent) root IDs than requested.

    def __init__(
        self, cg: "ChunkedGraph", root_ids: Union[np.uint64, Sequence[np.uint64]]
    ) -> None:
        self.cg = cg
        self.locked_root_ids = np.atleast_1d(root_ids)
        self.lock_acquired = False
        self.operation_id = None

    def __enter__(self):
        self.operation_id = self.cg.get_unique_operation_id()
        self.lock_acquired, self.locked_root_ids = self.cg.lock_root_loop(
            root_ids=self.locked_root_ids, operation_id=self.operation_id, max_tries=7
        )
        if not self.lock_acquired:
            raise cg_exceptions.LockingError("Could not acquire root lock")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.lock_acquired:
            for locked_root_id in self.locked_root_ids:
                self.cg.unlock_root(locked_root_id, self.operation_id)
