import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Union
from collections import defaultdict

import networkx as nx
import numpy as np

from pychunkedgraph import get_logger

from . import exceptions
from .types import empty_1d
from .lineage import lineage_graph

logger = get_logger(__name__)


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
        "future_root_ids_d",
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
        self.future_root_ids_d = defaultdict(lambda: empty_1d)

    def __enter__(self):
        if not self.operation_id:
            self.operation_id = self.cg.id_client.create_operation_id()

        if self.privileged_mode:
            return self

        nodes_ts = self.cg.get_node_timestamps(self.root_ids, return_numpy=0)
        min_ts = min(nodes_ts)
        lgraph = lineage_graph(self.cg, self.root_ids, timestamp_past=min_ts)
        self.future_root_ids_d = defaultdict(lambda: empty_1d)
        for id_ in self.root_ids:
            node_descendants = nx.descendants(lgraph, id_)
            node_descendants = np.unique(
                np.array(list(node_descendants), dtype=np.uint64)
            )
            self.future_root_ids_d[id_] = node_descendants

        self.lock_acquired, self.locked_root_ids = self.cg.client.lock_roots(
            root_ids=self.root_ids,
            operation_id=self.operation_id,
            future_root_ids_d=self.future_root_ids_d,
            max_tries=7,
        )
        if not self.lock_acquired:
            raise exceptions.LockingError("Could not acquire root lock")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.lock_acquired:
            max_workers = min(8, max(1, len(self.locked_root_ids)))
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
                        logger.warning(f"Failed to unlock root: {e}")


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

    __slots__ = [
        "cg",
        "root_ids",
        "acquired",
        "operation_id",
        "privileged_mode",
        "future_root_ids_d",
    ]

    def __init__(
        self,
        cg,
        operation_id: np.uint64,
        root_ids: Union[np.uint64, Sequence[np.uint64]],
        privileged_mode: bool = False,
        future_root_ids_d=None,
    ) -> None:
        self.cg = cg
        self.operation_id = operation_id
        self.root_ids = np.atleast_1d(root_ids)
        self.acquired = False
        # `privileged_mode` if True, override locking.
        # This is intended to be used in extremely rare cases to fix errors
        # caused by failed writes.
        self.privileged_mode = privileged_mode
        self.future_root_ids_d = future_root_ids_d

    def __enter__(self):
        if self.privileged_mode:
            return self
        if not self.cg.client.renew_locks(self.root_ids, self.operation_id):
            raise exceptions.LockingError("Could not renew locks before writing.")

        if self.future_root_ids_d is None:
            nodes_ts = self.cg.get_node_timestamps(self.root_ids, return_numpy=0)
            min_ts = min(nodes_ts)
            lgraph = lineage_graph(self.cg, self.root_ids, timestamp_past=min_ts)
            self.future_root_ids_d = defaultdict(lambda: empty_1d)
            for id_ in self.root_ids:
                node_descendants = nx.descendants(lgraph, id_)
                node_descendants = np.unique(
                    np.array(list(node_descendants), dtype=np.uint64)
                )
                self.future_root_ids_d[id_] = node_descendants

        self.acquired, self.root_ids, failed = self.cg.client.lock_roots_indefinitely(
            root_ids=self.root_ids,
            operation_id=self.operation_id,
            future_root_ids_d=self.future_root_ids_d,
        )
        if not self.acquired:
            raise exceptions.LockingError(f"{failed} have been locked indefinitely.")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.acquired:
            max_workers = min(8, max(1, len(self.root_ids)))
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
                        logger.warning(f"Failed to unlock root: {e}")


def _downsample_block_lock_row_key(block_coord) -> bytes:
    """Row key for one pyramid_block's downsample lock cell.

    Hash-prefixed so spatially-clustered block coords — common when a
    team edits the same region — scatter across bigtable tablets instead
    of piling up in one lexicographic range, which would hot-spot a
    single tablet under concurrent load.

    26 bytes total:
      - 2-byte blake2b hash of the packed coord (tablet distribution).
      - 24 bytes of packed coord (big-endian uint64 per axis).
    uint64 per axis tracks the existing node-id width and puts no cap on
    the block grid. The full coord in the key guarantees uniqueness even
    if two coords share the 2-byte hash prefix.
    """
    bx, by, bz = (int(c) for c in block_coord)
    packed = (
        bx.to_bytes(8, "big", signed=False)
        + by.to_bytes(8, "big", signed=False)
        + bz.to_bytes(8, "big", signed=False)
    )
    return hashlib.blake2b(packed, digest_size=2).digest() + packed


class DownsampleBlockLock:
    """Lock a set of pyramid_blocks for the lifetime of a downsample task.

    The downsample worker holds one across read → tinybrain → write for
    every block it touches. All-or-nothing: on partial acquisition we
    release what we got and retry with backoff; on repeated failure we
    raise so the pubsub message ends up un-acked and redelivered.

    Uses `cg.client.lock_by_row_key` with hash-prefixed row keys — the
    generic row-key lock primitive in kvdbclient — so these rows never
    collide with node-id-keyed root locks even though both use the same
    `Concurrency.Lock` column.
    """

    __slots__ = ["cg", "block_coords", "operation_id", "acquired_keys"]

    # Retry budget for partial-acquire failures. Each attempt releases
    # anything it got in the previous pass, then re-acquires from scratch.
    _MAX_ACQUIRE_ATTEMPTS = 7
    _ACQUIRE_BACKOFF_BASE_SEC = 0.5

    def __init__(
        self,
        cg,
        block_coords: Sequence,
        operation_id: np.uint64,
    ) -> None:
        self.cg = cg
        # Sort so every `__enter__` uses a consistent acquisition order
        # across workers — reduces contention between workers whose block
        # sets overlap. Sort is on the coord tuple (not the hashed row
        # key) so the order is stable and debuggable.
        self.block_coords = sorted(
            (int(bx), int(by), int(bz)) for bx, by, bz in block_coords
        )
        self.operation_id = np.uint64(operation_id)
        self.acquired_keys: list = []

    def __enter__(self):
        for attempt in range(self._MAX_ACQUIRE_ATTEMPTS):
            self.acquired_keys = []
            all_ok = True
            for coord in self.block_coords:
                row_key = _downsample_block_lock_row_key(coord)
                if self.cg.client.lock_by_row_key(row_key, self.operation_id):
                    self.acquired_keys.append(row_key)
                else:
                    all_ok = False
                    break
            if all_ok:
                return self
            self._release_acquired()
            time.sleep(self._ACQUIRE_BACKOFF_BASE_SEC * (2**attempt))
        raise exceptions.LockingError(
            f"Could not acquire downsample block locks for coords "
            f"{self.block_coords} after {self._MAX_ACQUIRE_ATTEMPTS} attempts"
        )

    def __exit__(self, exception_type, exception_value, traceback):
        self._release_acquired()

    def _release_acquired(self):
        if not self.acquired_keys:
            return
        max_workers = min(8, max(1, len(self.acquired_keys)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.cg.client.unlock_by_row_key, key, self.operation_id
                )
                for key in self.acquired_keys
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Failed to unlock downsample block: {e}")
        self.acquired_keys = []

    def renew(self) -> bool:
        """Extend expiry on every held lock. Returns False if any failed."""
        ok = True
        for key in self.acquired_keys:
            if not self.cg.client.renew_lock_by_row_key(key, self.operation_id):
                logger.warning(f"Failed to renew downsample block lock {key!r}")
                ok = False
        return ok


def _l2_chunk_lock_row_key(chunk_id) -> bytes:
    """Row key for one L2 chunk's spatial lock cell.

    Hash-prefixed so spatially-clustered chunk IDs scatter across
    bigtable tablets instead of piling up in one lexicographic range,
    which would hot-spot a single tablet under concurrent load.

    10 bytes total:
      - 2-byte blake2b hash of the chunk_id (tablet distribution).
      - 8 bytes of big-endian uint64 chunk_id.
    chunk_id already encodes layer+xyz in its bits, so the full key is
    unique per L2 chunk.
    """
    packed = int(chunk_id).to_bytes(8, "big", signed=False)
    return hashlib.blake2b(packed, digest_size=2).digest() + packed


class L2ChunkLock:
    """Lock a set of L2 chunks to serialize SV splits that touch them.

    Closes the cross-root spatial race: two SV splits on overlapping L2
    chunks but distinct roots acquire disjoint root-lock sets and would
    otherwise race on seg state. This lock is held across the
    `split_supervoxel` loop (seg write + SV-level hierarchy row write)
    so the pair commits atomically.

    All-or-nothing: on partial acquisition we release what we got and
    retry with backoff; on repeated failure we raise `LockingError`.

    Uses `cg.client.lock_by_row_key` — the generic row-key lock in
    kvdbclient — with a row-key namespace distinct from root and
    downsample block locks (all three share `attributes.Concurrency.Lock`
    under the hood; the row key disambiguates).
    """

    __slots__ = ["cg", "chunk_ids", "operation_id", "acquired_keys"]

    # Retry budget for partial-acquire failures. Each attempt releases
    # anything it got in the previous pass, then re-acquires from scratch.
    _MAX_ACQUIRE_ATTEMPTS = 7
    _ACQUIRE_BACKOFF_BASE_SEC = 0.5

    def __init__(
        self,
        cg,
        chunk_ids: Sequence[np.uint64],
        operation_id: np.uint64,
    ) -> None:
        self.cg = cg
        # Sort so every `__enter__` uses a consistent acquisition order
        # across workers — reduces contention when overlapping lock sets
        # would otherwise race AB/BA.
        self.chunk_ids = sorted(int(c) for c in chunk_ids)
        self.operation_id = np.uint64(operation_id)
        self.acquired_keys: list = []

    def __enter__(self):
        for attempt in range(self._MAX_ACQUIRE_ATTEMPTS):
            self.acquired_keys = []
            all_ok = True
            for chunk_id in self.chunk_ids:
                row_key = _l2_chunk_lock_row_key(chunk_id)
                if self.cg.client.lock_by_row_key(row_key, self.operation_id):
                    self.acquired_keys.append(row_key)
                else:
                    all_ok = False
                    break
            if all_ok:
                return self
            self._release_acquired()
            time.sleep(self._ACQUIRE_BACKOFF_BASE_SEC * (2**attempt))
        raise exceptions.LockingError(
            f"Could not acquire L2 chunk locks for chunks {self.chunk_ids} "
            f"after {self._MAX_ACQUIRE_ATTEMPTS} attempts"
        )

    def __exit__(self, exception_type, exception_value, traceback):
        self._release_acquired()

    def _release_acquired(self):
        if not self.acquired_keys:
            return
        max_workers = min(8, max(1, len(self.acquired_keys)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.cg.client.unlock_by_row_key, key, self.operation_id
                )
                for key in self.acquired_keys
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Failed to unlock L2 chunk: {e}")
        self.acquired_keys = []

    def renew(self) -> bool:
        """Extend expiry on every held lock. Returns False if any failed."""
        ok = True
        for key in self.acquired_keys:
            if not self.cg.client.renew_lock_by_row_key(key, self.operation_id):
                logger.warning(f"Failed to renew L2 chunk lock {key!r}")
                ok = False
        return ok
