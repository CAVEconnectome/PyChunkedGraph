from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

from . import exceptions as exceptions

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph


class RootLock:
    """Attempts to lock the requested root IDs using a unique operation ID.
    :raises exceptions.LockingError: throws when one or more root ID locks could not be
        acquired.
    :return: The RootLock context, including the locked root IDs and the linked operation ID
    :rtype: RootLock
    """

    __slots__ = ["cg", "locked_root_ids", "lock_acquired", "operation_id"]
    # FIXME: `locked_root_ids` is only required and exposed because `cg.client.lock_roots`
    #        currently might lock different (more recent) root IDs than requested.

    def __init__(
        self, cg: "ChunkedGraph", root_ids: Union[np.uint64, Sequence[np.uint64]]
    ) -> None:
        self.cg = cg
        self.locked_root_ids = np.atleast_1d(root_ids)
        self.lock_acquired = False
        self.operation_id = None

    def __enter__(self):
        self.operation_id = self.cg.id_client.create_operation_id()
        self.lock_acquired, self.locked_root_ids = self.cg.client.lock_roots(
            root_ids=self.locked_root_ids, operation_id=self.operation_id, max_tries=7
        )
        if not self.lock_acquired:
            raise exceptions.LockingError("Could not acquire root lock")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.lock_acquired:
            for locked_root_id in self.locked_root_ids:
                self.cg.client.unlock_root(locked_root_id, self.operation_id)
