from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union
from typing import Sequence
from collections import defaultdict
from warnings import warn

import networkx as nx
import numpy as np

from . import exceptions
from .types import empty_1d
from .lineage import lineage_graph


class RootLock:
    """Attempts to lock the requested root IDs using a unique operation ID.
    :raises exceptions.LockingError: throws when one or more root ID locks could not be
        acquired.
    """

    __slots__ = [
        "cg",
        "root_ids",
        "locked_root_ids",
        "lock_acquired",
        "operation_id",
        "privileged_mode",
    ]
    # FIXME: `locked_root_ids` is only required and exposed because `cg.client.lock_roots`
    #        currently might lock different (more recent) root IDs than requested.

    def __init__(
        self,
        cg,
        root_ids: Union[np.uint64, Sequence[np.uint64]],
        *,
        operation_id: np.uint64 = None,
        privileged_mode: bool = False,
    ) -> None:
        self.cg = cg
        self.root_ids = np.atleast_1d(root_ids)
        self.locked_root_ids = []
        self.lock_acquired = False
        self.operation_id = operation_id
        # `privileged_mode` if True, override locking.
        # This is intended to be used in extremely rare cases to fix errors
        # caused by failed writes. Must be used with `operation_id`,
        # meaning only existing failed operations can be run this way.
        self.privileged_mode = privileged_mode

    def __enter__(self):
        if not self.operation_id:
            self.operation_id = self.cg.id_client.create_operation_id()

        if self.privileged_mode:
            warn("Warning: Privileged mode without acquiring lock.")
            return self

        nodes_ts = self.cg.get_node_timestamps(self.root_ids, return_numpy=0)
        min_ts = min(nodes_ts)
        lgraph = lineage_graph(self.cg, self.root_ids, timestamp_past=min_ts)
        future_root_ids_d = defaultdict(lambda: empty_1d)
        for id_ in self.root_ids:
            node_descendants = nx.descendants(lgraph, id_)
            node_descendants = np.unique(
                np.array(list(node_descendants), dtype=np.uint64)
            )
            future_root_ids_d[id_] = node_descendants

        self.lock_acquired, self.locked_root_ids = self.cg.client.lock_roots(
            root_ids=self.root_ids,
            operation_id=self.operation_id,
            future_root_ids_d=future_root_ids_d,
            max_tries=7,
        )
        if not self.lock_acquired:
            raise exceptions.LockingError("Could not acquire root lock")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.lock_acquired:
            max_workers = max(1, len(self.locked_root_ids) // 2)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unlock_futures = [
                    executor.submit(
                        self.cg.client.unlock_root, root_id, self.operation_id
                    )
                    for root_id in self.locked_root_ids
                ]
                for future in as_completed(unlock_futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.warning(f"Failed to unlock root: {e}")


class IndefiniteRootLock:
    """
    Attempts to lock the requested root IDs using a unique operation ID.
    Assumes the root IDs have already been locked temporally.
    Also renews temporal lock before creating locking indefinitely,
    fails to lock indefinitely if the temporal lock cannot be re-acquired.

    :raises exceptions.LockingError:
    when a root ID lock cannot be renewed
    or when it has already been locked indefinitely.
    """

    __slots__ = ["cg", "root_ids", "acquired", "operation_id", "privileged_mode"]

    def __init__(
        self,
        cg,
        operation_id: np.uint64,
        root_ids: Union[np.uint64, Sequence[np.uint64]],
        privileged_mode: bool = False,
    ) -> None:
        self.cg = cg
        self.operation_id = operation_id
        self.root_ids = np.atleast_1d(root_ids)
        self.acquired = False
        # `privileged_mode` if True, override locking.
        # This is intended to be used in extremely rare cases to fix errors
        # caused by failed writes.
        self.privileged_mode = privileged_mode

    def __enter__(self):
        if self.privileged_mode:
            from warnings import warn

            warn("Warning: Privileged mode without acquiring indefinite lock.")
            return self
        if not self.cg.client.renew_locks(self.root_ids, self.operation_id):
            raise exceptions.LockingError("Could not renew locks before writing.")

        nodes_ts = self.cg.get_node_timestamps(self.root_ids, return_numpy=0)
        min_ts = min(nodes_ts)
        lgraph = lineage_graph(self.cg, self.root_ids, timestamp_past=min_ts)
        future_root_ids_d = defaultdict(lambda: empty_1d)
        for id_ in self.root_ids:
            node_descendants = nx.descendants(lgraph, id_)
            node_descendants = np.unique(
                np.array(list(node_descendants), dtype=np.uint64)
            )
            future_root_ids_d[id_] = node_descendants

        self.acquired, self.root_ids, failed = self.cg.client.lock_roots_indefinitely(
            root_ids=self.root_ids,
            operation_id=self.operation_id,
            future_root_ids_d=future_root_ids_d,
        )
        if not self.acquired:
            raise exceptions.LockingError(f"{failed} have been locked indefinitely.")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.acquired:
            max_workers = max(1, len(self.root_ids) // 2)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unlock_futures = [
                    executor.submit(
                        self.cg.client.unlock_indefinitely_locked_root,
                        root_id,
                        self.operation_id,
                    )
                    for root_id in self.root_ids
                ]
                for future in as_completed(unlock_futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.warning(f"Failed to unlock root: {e}")
