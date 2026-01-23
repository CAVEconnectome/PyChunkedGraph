# pylint: disable=invalid-name, missing-docstring, too-many-lines, protected-access, broad-exception-raised

import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Sequence
from functools import reduce

import numpy as np
from google.cloud import bigtable

logger = logging.getLogger(__name__)

from . import locks
from . import edits
from . import types
from . import attributes
from .edges import Edges
from .edges.utils import get_edges_status
from .utils import basetypes
from .utils import serializers
from .cache import CacheService
from .cutting import run_multicut
from .exceptions import PreconditionError
from .exceptions import PostconditionError
from .utils.generic import get_bounding_box as get_bbox
from ..logging.log_db import TimeIt


if TYPE_CHECKING:
    from .chunkedgraph import ChunkedGraph


class GraphEditOperation(ABC):
    __slots__ = [
        "cg",
        "user_id",
        "source_coords",
        "sink_coords",
        "parent_ts",
        "privileged_mode",
        "do_sanity_check",
    ]
    Result = namedtuple("Result", ["operation_id", "new_root_ids", "new_lvl2_ids"])

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        source_coords: Optional[Sequence[Sequence[int]]] = None,
        sink_coords: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.cg = cg
        self.user_id = user_id
        self.source_coords = None
        self.sink_coords = None
        # `parent_ts` is the timestamp to get parents/roots
        # after an operation fails while persisting changes.
        # When that happens, parents/roots before the operation must be used to fix it.
        # it is passed as an argument to `GraphEditOperation.execute()`
        self.parent_ts = None
        # `privileged_mode` if True, override locking.
        # This is intended to be used in extremely rare cases to fix errors
        # caused by failed writes.
        self.privileged_mode = False

        if source_coords is not None:
            self.source_coords = np.atleast_2d(source_coords).astype(
                basetypes.COORDINATES
            )
            if self.source_coords.size == 0:
                self.source_coords = None
        if sink_coords is not None:
            self.sink_coords = np.atleast_2d(sink_coords).astype(basetypes.COORDINATES)
            if self.sink_coords.size == 0:
                self.sink_coords = None

    @classmethod
    def _resolve_undo_chain(
        cls,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        operation_id: np.uint64,
        is_undo: bool,
        multicut_as_split: bool,
    ):
        log_record, _ = cg.client.read_log_entry(operation_id)
        log_record_type = cls.get_log_record_type(log_record)

        while log_record_type in (RedoOperation, UndoOperation):
            if log_record_type is RedoOperation:
                operation_id = log_record[attributes.OperationLogs.RedoOperationID]
            else:
                is_undo = not is_undo
                operation_id = log_record[attributes.OperationLogs.UndoOperationID]
            log_record, _ = cg.client.read_log_entry(operation_id)
            log_record_type = cls.get_log_record_type(log_record)

        if is_undo:
            return UndoOperation(
                cg,
                user_id=user_id,
                superseded_operation_id=operation_id,
                multicut_as_split=multicut_as_split,
            )
        else:
            return RedoOperation(
                cg,
                user_id=user_id,
                superseded_operation_id=operation_id,
                multicut_as_split=multicut_as_split,
            )

    @staticmethod
    def get_log_record_type(
        log_record: Dict[attributes._Attribute, Union[np.ndarray, np.number]],
        *,
        multicut_as_split=True,
    ) -> Type["GraphEditOperation"]:
        """Guesses the type of GraphEditOperation given a log record dictionary.
        :param log_record: log record dictionary
        :type log_record: Dict[attributes._Attribute, Union[np.ndarray, np.number]]
        :param multicut_as_split: If true, treat MulticutOperation as SplitOperation

        :return: The type of the matching GraphEditOperation subclass
        :rtype: Type["GraphEditOperation"]
        """
        if attributes.OperationLogs.UndoOperationID in log_record:
            return UndoOperation
        if attributes.OperationLogs.RedoOperationID in log_record:
            return RedoOperation
        if attributes.OperationLogs.AddedEdge in log_record:
            return MergeOperation
        if attributes.OperationLogs.RemovedEdge in log_record:
            if (
                multicut_as_split
                or attributes.OperationLogs.BoundingBoxOffset not in log_record
            ):
                return SplitOperation
            return MulticutOperation
        if attributes.OperationLogs.BoundingBoxOffset in log_record:
            return MulticutOperation
        raise TypeError("Could not determine graph operation type.")

    @classmethod
    def from_log_record(
        cls,
        cg: "ChunkedGraph",
        log_record: Dict[attributes._Attribute, Union[np.ndarray, np.number]],
        *,
        multicut_as_split: bool = True,
    ) -> "GraphEditOperation":
        """Generates the correct GraphEditOperation given a log record dictionary.
        :param cg: The "ChunkedGraph" instance
        :type cg: "ChunkedGraph"
        :param log_record: log record dictionary
        :type log_record: Dict[attributes._Attribute, Union[np.ndarray, np.number]]
        :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).
        :type multicut_as_split: bool

        :return: The matching GraphEditOperation subclass
        :rtype: "GraphEditOperation"
        """

        def _optional(column):
            try:
                return log_record[column]
            except KeyError:
                return None

        log_record_type = cls.get_log_record_type(
            log_record, multicut_as_split=multicut_as_split
        )
        user_id = log_record[attributes.OperationLogs.UserID]

        if log_record_type is UndoOperation:
            superseded_operation_id = log_record[
                attributes.OperationLogs.UndoOperationID
            ]
            return cls.undo_operation(
                cg,
                user_id=user_id,
                operation_id=superseded_operation_id,
                multicut_as_split=multicut_as_split,
            )

        if log_record_type is RedoOperation:
            superseded_operation_id = log_record[
                attributes.OperationLogs.RedoOperationID
            ]
            return cls.redo_operation(
                cg,
                user_id=user_id,
                operation_id=superseded_operation_id,
                multicut_as_split=multicut_as_split,
            )

        source_coords = _optional(attributes.OperationLogs.SourceCoordinate)
        sink_coords = _optional(attributes.OperationLogs.SinkCoordinate)

        if log_record_type is MergeOperation:
            added_edges = log_record[attributes.OperationLogs.AddedEdge]
            affinities = _optional(attributes.OperationLogs.Affinity)
            return MergeOperation(
                cg,
                user_id=user_id,
                source_coords=source_coords,
                sink_coords=sink_coords,
                added_edges=added_edges,
                affinities=affinities,
            )

        if log_record_type is SplitOperation:
            removed_edges = log_record[attributes.OperationLogs.RemovedEdge]
            return SplitOperation(
                cg,
                user_id=user_id,
                source_coords=source_coords,
                sink_coords=sink_coords,
                removed_edges=removed_edges,
            )

        if log_record_type is MulticutOperation:
            bbox_offset = log_record[attributes.OperationLogs.BoundingBoxOffset]
            source_ids = log_record[attributes.OperationLogs.SourceID]
            sink_ids = log_record[attributes.OperationLogs.SinkID]
            removed_edges = log_record[attributes.OperationLogs.RemovedEdge]
            return MulticutOperation(
                cg,
                user_id=user_id,
                source_coords=source_coords,
                sink_coords=sink_coords,
                bbox_offset=bbox_offset,
                source_ids=source_ids,
                sink_ids=sink_ids,
                removed_edges=removed_edges,
            )

        raise TypeError("Could not determine graph operation type.")

    @classmethod
    def from_operation_id(
        cls,
        cg: "ChunkedGraph",
        operation_id: np.uint64,
        *,
        multicut_as_split: bool = True,
        privileged_mode: Optional[bool] = False,
    ):
        """Generates the correct GraphEditOperation given a operation ID.
        :param cg: The "ChunkedGraph" instance
        :type cg: "ChunkedGraph"
        :param operation_id: The operation ID
        :type operation_id: np.uint64
        :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).
        :type multicut_as_split: bool

        `privileged_mode` if True, override locking.
        This is intended to be used in extremely rare cases to fix errors
        caused by failed writes.

        :return: The matching GraphEditOperation subclass
        :rtype: "GraphEditOperation"
        """
        log, _ = cg.client.read_log_entry(operation_id)
        operation = cls.from_log_record(cg, log, multicut_as_split=multicut_as_split)
        operation.privileged_mode = privileged_mode
        return operation

    @classmethod
    def undo_operation(
        cls,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        operation_id: np.uint64,
        multicut_as_split: bool = True,
    ) -> Union["UndoOperation", "RedoOperation"]:
        """Create a GraphEditOperation that, if executed, would undo the changes introduced by
            operation_id.

        NOTE: If operation_id is an UndoOperation, this function might return an instance of
              RedoOperation instead (depending on how the Undo/Redo chain unrolls)

        :param cg: The "ChunkedGraph" instance
        :type cg: "ChunkedGraph"
        :param user_id: User that should be associated with this undo operation
        :type user_id: str
        :param operation_id: The operation ID to be undone
        :type operation_id: np.uint64
        :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).
        :type multicut_as_split: bool

        :return: A GraphEditOperation that, if executed, will undo the change introduced by
            operation_id.
        :rtype: Union["UndoOperation", "RedoOperation"]
        """
        return cls._resolve_undo_chain(
            cg,
            user_id=user_id,
            operation_id=operation_id,
            is_undo=True,
            multicut_as_split=multicut_as_split,
        )

    @classmethod
    def redo_operation(
        cls,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        operation_id: np.uint64,
        multicut_as_split=True,
    ) -> Union["UndoOperation", "RedoOperation"]:
        """Create a GraphEditOperation that, if executed, would redo the changes introduced by
            operation_id.

        NOTE: If operation_id is an UndoOperation, this function might return an instance of
              UndoOperation instead (depending on how the Undo/Redo chain unrolls)

        :param cg: The "ChunkedGraph" instance
        :type cg: "ChunkedGraph"
        :param user_id: User that should be associated with this redo operation
        :type user_id: str
        :param operation_id: The operation ID to be redone
        :type operation_id: np.uint64
        :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).
        :type multicut_as_split: bool

        :return: A GraphEditOperation that, if executed, will redo the changes introduced by
            operation_id.
        :rtype: Union["UndoOperation", "RedoOperation"]
        """
        return cls._resolve_undo_chain(
            cg,
            user_id=user_id,
            operation_id=operation_id,
            is_undo=False,
            multicut_as_split=multicut_as_split,
        )

    @abstractmethod
    def _update_root_ids(self) -> np.ndarray:
        """Retrieves and validates the most recent root IDs affected by this GraphEditOperation.
        :return: New most recent root IDs
        :rtype: np.ndarray
        """

    @abstractmethod
    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        """Initiates the graph operation calculation.
        :return: New root IDs, new Lvl2 node IDs, and affected records
        :rtype: Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]
        """

    @abstractmethod
    def _create_log_record(
        self,
        *,
        operation_id,
        timestamp,
        operation_ts,
        new_root_ids,
        status=1,
        exception="",
    ) -> "bigtable.row.Row":
        """Creates a log record with all necessary information to replay the current
            GraphEditOperation
        :return: Bigtable row containing the log record
        :rtype: bigtable.row.Row
        """

    @abstractmethod
    def invert(self) -> "GraphEditOperation":
        """Creates a GraphEditOperation that would cancel out changes introduced by the current
            GraphEditOperation
        :return: The inverse of GraphEditOperation
        :rtype: GraphEditOperation
        """

    def execute(
        self, *, operation_id=None, parent_ts=None, override_ts=None
    ) -> "GraphEditOperation.Result":
        """
        Executes current GraphEditOperation:
        * Calls the subclass's _update_root_ids method
        * Locks root IDs normally
        * Calls the subclass's _apply method
        * Calls the subclass's _create_log_record method
        * Lock roots indefinitely to prevent corruption in case persisting changes fails
          Such cases are retired in a cron job.
        * Persist changes
        * Release indefinite locks
        * Releases normal root ID lock

        `parent_ts` is the timestamp to get parents/roots
        for normal edits it is None, which means latest parents/roots
        But after an operation fails while persisting changes,
        parents/roots before the operation must be used to fix it.
        `override_ts` can be used to preserve proper timestamp in such cases.
        """
        is_merge = isinstance(self, MergeOperation)
        op_type = "merge" if is_merge else "split"
        self.parent_ts = parent_ts
        root_ids = self._update_root_ids()
        with locks.RootLock(
            self.cg,
            root_ids,
            operation_id=operation_id,
            privileged_mode=self.privileged_mode,
        ) as lock:
            self.cg.cache = CacheService(self.cg)
            self.cg.meta.custom_data["operation_id"] = operation_id
            timestamp = self.cg.client.get_consolidated_lock_timestamp(
                lock.locked_root_ids,
                np.array([lock.operation_id] * len(lock.locked_root_ids)),
            )

            log_record_before_edit = self._create_log_record(
                operation_id=lock.operation_id,
                new_root_ids=types.empty_1d,
                timestamp=timestamp,
                operation_ts=override_ts if override_ts else timestamp,
                status=attributes.OperationLogs.StatusCodes.CREATED.value,
            )
            self.cg.client.write([log_record_before_edit])

            try:
                with TimeIt(f"{op_type}.apply", self.cg.graph_id, lock.operation_id):
                    new_root_ids, new_lvl2_ids, affected_records = self._apply(
                        operation_id=lock.operation_id,
                        timestamp=override_ts if override_ts else timestamp,
                    )
                if self.cg.meta.READ_ONLY:
                    # return without persisting changes
                    return GraphEditOperation.Result(
                        operation_id=lock.operation_id,
                        new_root_ids=new_root_ids,
                        new_lvl2_ids=new_lvl2_ids,
                    )
            except PreconditionError as err:
                self.cg.cache = None
                raise PreconditionError(err) from err
            except PostconditionError as err:
                self.cg.cache = None
                raise PostconditionError(err) from err
            except (AssertionError, RuntimeError) as err:
                self.cg.cache = None
                raise RuntimeError(err) from err
            except Exception as err:
                # unknown exception, update log record with error
                self.cg.cache = None
                log_record_error = self._create_log_record(
                    operation_id=lock.operation_id,
                    new_root_ids=types.empty_1d,
                    timestamp=None,
                    operation_ts=override_ts if override_ts else timestamp,
                    status=attributes.OperationLogs.StatusCodes.EXCEPTION.value,
                    exception=repr(err),
                )
                self.cg.client.write([log_record_error])
                raise Exception(err) from err

            with TimeIt(f"{op_type}.write", self.cg.graph_id, lock.operation_id):
                result = self._write(
                    lock,
                    override_ts if override_ts else timestamp,
                    new_root_ids,
                    new_lvl2_ids,
                    affected_records,
                )
                return result

    def _write(self, lock, timestamp, new_root_ids, new_lvl2_ids, affected_records):
        """Helper to persist changes after an edit."""
        new_root_ids = np.array(new_root_ids, dtype=basetypes.NODE_ID)
        new_lvl2_ids = np.array(new_lvl2_ids, dtype=basetypes.NODE_ID)

        # this must be written first to indicate writing has started.
        log_record_after_edit = self._create_log_record(
            operation_id=lock.operation_id,
            new_root_ids=new_root_ids,
            timestamp=None,
            operation_ts=timestamp,
            status=attributes.OperationLogs.StatusCodes.WRITE_STARTED.value,
        )

        with locks.IndefiniteRootLock(
            self.cg,
            lock.operation_id,
            lock.locked_root_ids,
            privileged_mode=lock.privileged_mode,
        ):
            # indefinite lock for writing, if a node instance or pod dies during this
            # the roots must stay locked indefinitely to prevent further corruption.
            self.cg.client.write(
                [log_record_after_edit] + affected_records,
                lock.locked_root_ids,
                operation_id=lock.operation_id,
                slow_retry=False,
            )
            log_record_success = self._create_log_record(
                operation_id=lock.operation_id,
                new_root_ids=new_root_ids,
                timestamp=None,
                operation_ts=timestamp,
                status=attributes.OperationLogs.StatusCodes.SUCCESS.value,
            )
            self.cg.client.write([log_record_success])
        self.cg.cache = None
        return GraphEditOperation.Result(
            operation_id=lock.operation_id,
            new_root_ids=new_root_ids,
            new_lvl2_ids=new_lvl2_ids,
        )


class MergeOperation(GraphEditOperation):
    """Merge Operation: Connect *known* pairs of supervoxels by adding a (weighted) edge.

    :param cg: The "ChunkedGraph" object
    :type cg: "ChunkedGraph"
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param added_edges: Supervoxel IDs of all added edges [[source, sink]]
    :type added_edges: Sequence[Sequence[np.uint64]]
    :param source_coords: world space coordinates in nm,
        corresponding to IDs in added_edges[:,0], defaults to None
    :type source_coords: Optional[Sequence[Sequence[int]]], optional
    :param sink_coords: world space coordinates in nm,
        corresponding to IDs in added_edges[:,1], defaults to None
    :type sink_coords: Optional[Sequence[Sequence[int]]], optional
    :param affinities: edge weights for newly added edges,
        entries corresponding to added_edges, defaults to None
    :type affinities: Optional[Sequence[np.float32]], optional
    """

    __slots__ = [
        "source_ids",
        "sink_ids",
        "added_edges",
        "affinities",
        "bbox_offset",
        "allow_same_segment_merge",
        "do_sanity_check",
        "stitch_mode",
    ]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        added_edges: Sequence[Sequence[np.uint64]],
        source_coords: Sequence[Sequence[int]],
        sink_coords: Sequence[Sequence[int]],
        bbox_offset: Tuple[int, int, int] = (240, 240, 24),
        affinities: Optional[Sequence[np.float32]] = None,
        allow_same_segment_merge: Optional[bool] = False,
        do_sanity_check: Optional[bool] = True,
        stitch_mode: bool = False,
    ) -> None:
        super().__init__(
            cg, user_id=user_id, source_coords=source_coords, sink_coords=sink_coords
        )
        self.added_edges = np.atleast_2d(added_edges).astype(basetypes.NODE_ID)
        self.bbox_offset = np.atleast_1d(bbox_offset).astype(basetypes.COORDINATES)
        self.allow_same_segment_merge = allow_same_segment_merge
        self.do_sanity_check = do_sanity_check
        self.stitch_mode = stitch_mode

        self.affinities = None
        if affinities is not None:
            self.affinities = np.atleast_1d(affinities).astype(basetypes.EDGE_AFFINITY)
            if self.affinities.size == 0:
                self.affinities = None

        if np.any(np.equal(self.added_edges[:, 0], self.added_edges[:, 1])):
            raise PreconditionError("Requested merge contains at least 1 self-loop.")

        layers = self.cg.get_chunk_layers(self.added_edges.ravel())
        assert np.sum(layers) == layers.size, "Supervoxels expected."

    def _update_root_ids(self) -> np.ndarray:
        root_ids = np.unique(
            self.cg.get_roots(
                self.added_edges.ravel(), assert_roots=True, time_stamp=self.parent_ts
            )
        )
        return root_ids

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        root_ids = set(
            self.cg.get_roots(
                self.added_edges.ravel(), assert_roots=True, time_stamp=self.parent_ts
            )
        )
        if len(root_ids) < 2 and not self.allow_same_segment_merge:
            raise PreconditionError(
                "Supervoxels must belong to different objects."
                f" Tried to merge {self.added_edges.ravel()},"
                f" which all belong to {tuple(root_ids)[0]}."
            )

        atomic_edges = self.added_edges
        fake_edge_rows = []
        if not self.stitch_mode:
            bbox = get_bbox(self.source_coords, self.sink_coords, self.bbox_offset)
            with TimeIt("subgraph", self.cg.graph_id, operation_id):
                edges = self.cg.get_subgraph(
                    root_ids,
                    bbox=bbox,
                    bbox_is_coordinate=True,
                    edges_only=True,
                )

            if self.allow_same_segment_merge:
                inactive_edges = types.empty_2d
            else:
                with TimeIt("preprocess", self.cg.graph_id, operation_id):
                    inactive_edges = edits.merge_preprocess(
                        self.cg,
                        subgraph_edges=edges,
                        supervoxels=self.added_edges.ravel(),
                        parent_ts=self.parent_ts,
                    )

            atomic_edges, fake_edge_rows = edits.check_fake_edges(
                self.cg,
                atomic_edges=self.added_edges,
                inactive_edges=inactive_edges,
                time_stamp=timestamp,
                parent_ts=self.parent_ts,
            )

        with TimeIt("add_edges", self.cg.graph_id, operation_id):
            new_roots, new_l2_ids, new_entries = edits.add_edges(
                self.cg,
                atomic_edges=atomic_edges,
                operation_id=operation_id,
                time_stamp=timestamp,
                parent_ts=self.parent_ts,
                allow_same_segment_merge=self.allow_same_segment_merge,
                do_sanity_check=self.do_sanity_check,
                stitch_mode=self.stitch_mode,
            )
        return new_roots, new_l2_ids, fake_edge_rows + new_entries

    def _create_log_record(
        self,
        *,
        operation_id: np.uint64,
        timestamp: datetime,
        operation_ts: datetime,
        new_root_ids: Sequence[np.uint64],
        status: int = 1,
        exception: str = "",
    ) -> "bigtable.row.Row":
        val_dict = {
            attributes.OperationLogs.UserID: self.user_id,
            attributes.OperationLogs.RootID: new_root_ids,
            attributes.OperationLogs.AddedEdge: self.added_edges,
            attributes.OperationLogs.Status: status,
            attributes.OperationLogs.OperationException: exception,
            attributes.OperationLogs.OperationTimeStamp: operation_ts,
        }
        if self.source_coords is not None:
            val_dict[attributes.OperationLogs.SourceCoordinate] = self.source_coords
        if self.sink_coords is not None:
            val_dict[attributes.OperationLogs.SinkCoordinate] = self.sink_coords
        if self.affinities is not None:
            val_dict[attributes.OperationLogs.Affinity] = self.affinities
        return self.cg.client.mutate_row(
            serializers.serialize_uint64(operation_id), val_dict, timestamp
        )

    def invert(self) -> "SplitOperation":
        return SplitOperation(
            self.cg,
            user_id=self.user_id,
            removed_edges=self.added_edges,
            source_coords=self.source_coords,
            sink_coords=self.sink_coords,
        )


class SplitOperation(GraphEditOperation):
    """Split Operation: Cut *known* pairs of supervoxel that are directly connected by an edge.

    :param cg: The "ChunkedGraph" object
    :type cg: "ChunkedGraph"
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param removed_edges: Supervoxel IDs of all removed edges [[source, sink]]
    :type removed_edges: Sequence[Sequence[np.uint64]]
    :param source_coords: world space coordinates in nm, corresponding to IDs in
        removed_edges[:,0], defaults to None
    :type source_coords: Optional[Sequence[Sequence[int]]], optional
    :param sink_coords: world space coordinates in nm, corresponding to IDs in
        removed_edges[:,1], defaults to None
    :type sink_coords: Optional[Sequence[Sequence[int]]], optional
    """

    __slots__ = ["removed_edges", "bbox_offset", "do_sanity_check"]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        removed_edges: Sequence[Sequence[np.uint64]],
        source_coords: Optional[Sequence[Sequence[int]]] = None,
        sink_coords: Optional[Sequence[Sequence[int]]] = None,
        bbox_offset: Tuple[int] = (240, 240, 24),
        do_sanity_check: Optional[bool] = True,
    ) -> None:
        super().__init__(
            cg, user_id=user_id, source_coords=source_coords, sink_coords=sink_coords
        )
        self.removed_edges = np.atleast_2d(removed_edges).astype(basetypes.NODE_ID)
        self.bbox_offset = np.atleast_1d(bbox_offset).astype(basetypes.COORDINATES)
        self.do_sanity_check = do_sanity_check
        if np.any(np.equal(self.removed_edges[:, 0], self.removed_edges[:, 1])):
            raise PreconditionError("Requested split contains at least 1 self-loop.")

        layers = self.cg.get_chunk_layers(self.removed_edges.ravel())
        assert np.sum(layers) == layers.size, "IDs must be supervoxels."

    def _update_root_ids(self) -> np.ndarray:
        root_ids = np.unique(
            self.cg.get_roots(
                self.removed_edges.ravel(),
                assert_roots=True,
                time_stamp=self.parent_ts,
            )
        )
        if len(root_ids) > 1:
            raise PreconditionError("Supervoxels must belong to the same object.")
        return root_ids

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        if (
            len(
                set(
                    self.cg.get_roots(
                        self.removed_edges.ravel(),
                        assert_roots=True,
                        time_stamp=self.parent_ts,
                    )
                )
            )
            > 1
        ):
            raise PreconditionError("Supervoxels must belong to the same object.")

        with TimeIt("remove_edges", self.cg.graph_id, operation_id):
            return edits.remove_edges(
                self.cg,
                operation_id=operation_id,
                atomic_edges=self.removed_edges,
                time_stamp=timestamp,
                parent_ts=self.parent_ts,
                do_sanity_check=self.do_sanity_check,
            )

    def _create_log_record(
        self,
        *,
        operation_id: np.uint64,
        timestamp: datetime,
        operation_ts: datetime,
        new_root_ids: Sequence[np.uint64],
        status: int = 1,
        exception: str = "",
    ) -> "bigtable.row.Row":
        val_dict = {
            attributes.OperationLogs.UserID: self.user_id,
            attributes.OperationLogs.RootID: new_root_ids,
            attributes.OperationLogs.RemovedEdge: self.removed_edges,
            attributes.OperationLogs.Status: status,
            attributes.OperationLogs.OperationException: exception,
            attributes.OperationLogs.OperationTimeStamp: operation_ts,
        }
        if self.source_coords is not None:
            val_dict[attributes.OperationLogs.SourceCoordinate] = self.source_coords
        if self.sink_coords is not None:
            val_dict[attributes.OperationLogs.SinkCoordinate] = self.sink_coords

        return self.cg.client.mutate_row(
            serializers.serialize_uint64(operation_id), val_dict, timestamp
        )

    def invert(self) -> "MergeOperation":
        return MergeOperation(
            self.cg,
            user_id=self.user_id,
            added_edges=self.removed_edges,
            source_coords=self.source_coords,
            sink_coords=self.sink_coords,
        )


class MulticutOperation(GraphEditOperation):
    """
    Multicut Operation: Apply min-cut algorithm to identify suitable edges for removal
        in order to separate two groups of supervoxels.

    :param cg: The "ChunkedGraph" object
    :type cg: "ChunkedGraph"
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param source_ids: Supervoxel IDs that should be separated from supervoxel IDs in sink_ids
    :type souce_ids: Sequence[np.uint64]
    :param sink_ids: Supervoxel IDs that should be separated from supervoxel IDs in source_ids
    :type sink_ids: Sequence[np.uint64]
    :param source_coords: world space coordinates in nm, corresponding to IDs in source_ids
    :type source_coords: Sequence[Sequence[int]]
    :param sink_coords: world space coordinates in nm, corresponding to IDs in sink_ids
    :type sink_coords: Sequence[Sequence[int]]
    :param bbox_offset: Padding for min-cut bounding box, applied to min/max coordinates
        retrieved from source_coords and sink_coords, defaults to None
    :type bbox_offset: Sequence[int]
    """

    __slots__ = [
        "source_ids",
        "sink_ids",
        "removed_edges",
        "bbox_offset",
        "path_augment",
        "disallow_isolating_cut",
        "do_sanity_check",
    ]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        source_ids: Sequence[np.uint64],
        sink_ids: Sequence[np.uint64],
        source_coords: Sequence[Sequence[int]],
        sink_coords: Sequence[Sequence[int]],
        bbox_offset: Sequence[int],
        removed_edges: Sequence[Sequence[np.uint64]] = types.empty_2d,
        path_augment: bool = True,
        disallow_isolating_cut: bool = True,
        do_sanity_check: Optional[bool] = True,
    ) -> None:
        super().__init__(
            cg, user_id=user_id, source_coords=source_coords, sink_coords=sink_coords
        )
        self.removed_edges = removed_edges
        self.source_ids = np.atleast_1d(source_ids).astype(basetypes.NODE_ID)
        self.sink_ids = np.atleast_1d(sink_ids).astype(basetypes.NODE_ID)
        self.bbox_offset = np.atleast_1d(bbox_offset).astype(basetypes.COORDINATES)
        self.path_augment = path_augment
        self.disallow_isolating_cut = disallow_isolating_cut
        self.do_sanity_check = do_sanity_check
        if np.any(np.isin(self.sink_ids, self.source_ids)):
            raise PreconditionError(
                "Supervoxels exist in both sink and source, "
                "try placing the points further apart."
            )

        ids = np.concatenate([self.source_ids, self.sink_ids]).astype(basetypes.NODE_ID)
        layers = self.cg.get_chunk_layers(ids)
        assert np.sum(layers) == layers.size, "IDs must be supervoxels."

    def _update_root_ids(self) -> np.ndarray:
        sink_and_source_ids = np.concatenate((self.source_ids, self.sink_ids)).astype(
            basetypes.NODE_ID
        )
        root_ids = np.unique(
            self.cg.get_roots(
                sink_and_source_ids, assert_roots=True, time_stamp=self.parent_ts
            )
        )
        if len(root_ids) > 1:
            raise PreconditionError("Supervoxels must belong to the same segment.")
        return root_ids

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        # Verify that sink and source are from the same root object
        root_ids = set(
            self.cg.get_roots(
                np.concatenate([self.source_ids, self.sink_ids]).astype(
                    basetypes.NODE_ID
                ),
                assert_roots=True,
                time_stamp=self.parent_ts,
            )
        )
        if len(root_ids) > 1:
            raise PreconditionError("Supervoxels must belong to the same object.")

        bbox = get_bbox(
            self.source_coords,
            self.sink_coords,
            self.cg.meta.split_bounding_offset,
        )
        with TimeIt("get_subgraph", self.cg.graph_id, operation_id):
            l2id_agglomeration_d, edges_tuple = self.cg.get_subgraph(
                root_ids.pop(), bbox=bbox, bbox_is_coordinate=True
            )

            edges = reduce(lambda x, y: x + y, edges_tuple, Edges([], []))
            supervoxels = np.concatenate(
                [agg.supervoxels for agg in l2id_agglomeration_d.values()]
            ).astype(basetypes.NODE_ID)
            mask0 = np.isin(edges.node_ids1, supervoxels)
            mask1 = np.isin(edges.node_ids2, supervoxels)
            edges = edges[mask0 & mask1]
        if len(edges) == 0:
            raise PreconditionError("No local edges found.")

        with TimeIt("multicut", self.cg.graph_id, operation_id):
            self.removed_edges = run_multicut(
                edges,
                self.source_ids,
                self.sink_ids,
                path_augment=self.path_augment,
                disallow_isolating_cut=self.disallow_isolating_cut,
            )
        if not self.removed_edges.size:
            raise PostconditionError("Mincut could not find any edges to remove.")

        with TimeIt("remove_edges", self.cg.graph_id, operation_id):
            return edits.remove_edges(
                self.cg,
                operation_id=operation_id,
                atomic_edges=self.removed_edges,
                time_stamp=timestamp,
                parent_ts=self.parent_ts,
                do_sanity_check=self.do_sanity_check,
            )

    def _create_log_record(
        self,
        *,
        operation_id: np.uint64,
        timestamp: datetime,
        operation_ts: datetime,
        new_root_ids: Sequence[np.uint64],
        status: int = 1,
        exception: str = "",
    ) -> "bigtable.row.Row":
        val_dict = {
            attributes.OperationLogs.UserID: self.user_id,
            attributes.OperationLogs.RootID: new_root_ids,
            attributes.OperationLogs.SourceCoordinate: self.source_coords,
            attributes.OperationLogs.SinkCoordinate: self.sink_coords,
            attributes.OperationLogs.SourceID: self.source_ids,
            attributes.OperationLogs.SinkID: self.sink_ids,
            attributes.OperationLogs.BoundingBoxOffset: self.bbox_offset,
            attributes.OperationLogs.RemovedEdge: self.removed_edges,
            attributes.OperationLogs.Status: status,
            attributes.OperationLogs.OperationException: exception,
            attributes.OperationLogs.OperationTimeStamp: operation_ts,
        }
        return self.cg.client.mutate_row(
            serializers.serialize_uint64(operation_id), val_dict, timestamp
        )

    def invert(self) -> "MergeOperation":
        return MergeOperation(
            self.cg,
            user_id=self.user_id,
            added_edges=self.removed_edges,
            source_coords=self.source_coords,
            sink_coords=self.sink_coords,
        )


class RedoOperation(GraphEditOperation):
    """
    RedoOperation: Used to apply a previous graph edit operation. In contrast to a
        "coincidental" redo (e.g. merging an edge added by a previous merge operation), a
        RedoOperation is linked to an earlier operation ID to enable its correct repetition.
        Acts as counterpart to UndoOperation.

    NOTE: Avoid instantiating a RedoOperation directly, if possible. The class method
          GraphEditOperation.redo_operation() is in general preferred as it will correctly
          unroll Undo/Redo chains.

    :param cg: The "ChunkedGraph" object
    :type cg: "ChunkedGraph"
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param superseded_operation_id: Operation ID to be redone
    :type superseded_operation_id: np.uint64
    :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).
    :type multicut_as_split: bool
    """

    __slots__ = [
        "superseded_operation_id",
        "superseded_operation",
        "added_edges",
        "removed_edges",
        "operation_status",
    ]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        superseded_operation_id: np.uint64,
        multicut_as_split: bool,
    ) -> None:
        super().__init__(cg, user_id=user_id)
        log_record, _ = cg.client.read_log_entry(superseded_operation_id)
        log_record_type = GraphEditOperation.get_log_record_type(log_record)
        if log_record_type in (RedoOperation, UndoOperation):
            raise ValueError(
                (
                    f"RedoOperation received {log_record_type.__name__} as target operation, "
                    "which is not allowed. Use GraphEditOperation.redo_operation() instead."
                )
            )

        self.superseded_operation_id = superseded_operation_id
        self.operation_status = log_record[attributes.OperationLogs.Status]
        if self.operation_status != attributes.OperationLogs.StatusCodes.SUCCESS.value:
            return
        self.superseded_operation = GraphEditOperation.from_log_record(
            cg, log_record=log_record, multicut_as_split=multicut_as_split
        )
        if hasattr(self.superseded_operation, "added_edges"):
            self.added_edges = self.superseded_operation.added_edges
        if hasattr(self.superseded_operation, "removed_edges"):
            self.removed_edges = self.superseded_operation.removed_edges

    def _update_root_ids(self):
        if self.operation_status != attributes.OperationLogs.StatusCodes.SUCCESS.value:
            return types.empty_1d
        return self.superseded_operation._update_root_ids()

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        return self.superseded_operation._apply(
            operation_id=operation_id, timestamp=timestamp
        )

    def _create_log_record(
        self,
        *,
        operation_id: np.uint64,
        timestamp: datetime,
        operation_ts: datetime,
        new_root_ids: Sequence[np.uint64],
        status: int = 1,
        exception: str = "",
    ) -> "bigtable.row.Row":
        val_dict = {
            attributes.OperationLogs.UserID: self.user_id,
            attributes.OperationLogs.RedoOperationID: self.superseded_operation_id,
            attributes.OperationLogs.RootID: new_root_ids,
            attributes.OperationLogs.OperationTimeStamp: operation_ts,
            attributes.OperationLogs.Status: status,
            attributes.OperationLogs.OperationException: exception,
        }
        if hasattr(self, "added_edges"):
            val_dict[attributes.OperationLogs.AddedEdge] = self.added_edges
        if hasattr(self, "removed_edges"):
            val_dict[attributes.OperationLogs.RemovedEdge] = self.removed_edges
        return self.cg.client.mutate_row(
            serializers.serialize_uint64(operation_id), val_dict, timestamp
        )

    def invert(self) -> "GraphEditOperation":
        """
        Inverts a RedoOperation. Treated as Undoing the original operation
        """
        return UndoOperation(
            self.cg,
            user_id=self.user_id,
            superseded_operation_id=self.superseded_operation_id,
            multicut_as_split=True,
        )

    def execute(
        self, *, operation_id=None, parent_ts=None, override_ts=None
    ) -> "GraphEditOperation.Result":
        if self.operation_status != attributes.OperationLogs.StatusCodes.SUCCESS.value:
            # Don't redo failed operations
            return GraphEditOperation.Result(
                operation_id=operation_id,
                new_root_ids=types.empty_1d,
                new_lvl2_ids=types.empty_1d,
            )
        return super().execute(
            operation_id=operation_id, parent_ts=parent_ts, override_ts=override_ts
        )


class UndoOperation(GraphEditOperation):
    """
    UndoOperation: Used to apply the inverse of a previous graph edit operation. In contrast
        to a "coincidental" undo (e.g. merging an edge previously removed by a split operation), an
        UndoOperation is linked to an earlier operation ID to enable its correct reversal.

    NOTE: Avoid instantiating an UndoOperation directly, if possible. The class method
          GraphEditOperation.undo_operation() is in general preferred as it will correctly
          unroll Undo/Redo chains.

    :param cg: The "ChunkedGraph" object
    :type cg: "ChunkedGraph"
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param superseded_operation_id: Operation ID to be undone
    :type superseded_operation_id: np.uint64
    :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).
    :type multicut_as_split: bool
    """

    __slots__ = [
        "superseded_operation_id",
        "inverse_superseded_operation",
        "added_edges",
        "removed_edges",
        "operation_status",
    ]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        superseded_operation_id: np.uint64,
        multicut_as_split: bool,
    ) -> None:
        super().__init__(cg, user_id=user_id)
        log_record, _ = cg.client.read_log_entry(superseded_operation_id)
        log_record_type = GraphEditOperation.get_log_record_type(log_record)
        if log_record_type in (RedoOperation, UndoOperation):
            raise ValueError(
                (
                    f"UndoOperation received {log_record_type.__name__} as target operation, "
                    "which is not allowed. Use GraphEditOperation.undo_operation() instead."
                )
            )

        self.superseded_operation_id = superseded_operation_id
        self.operation_status = log_record[attributes.OperationLogs.Status]
        if self.operation_status != attributes.OperationLogs.StatusCodes.SUCCESS.value:
            return
        superseded_operation = GraphEditOperation.from_log_record(
            cg, log_record=log_record, multicut_as_split=multicut_as_split
        )
        if log_record_type is MergeOperation:
            # account for additional activated edges so merge can be properly undone
            from .misc import get_activated_edges

            activated_edges = get_activated_edges(cg, superseded_operation_id)
            if len(activated_edges) > 0:
                superseded_operation.added_edges = activated_edges
        self.inverse_superseded_operation = superseded_operation.invert()
        if hasattr(self.inverse_superseded_operation, "added_edges"):
            self.added_edges = self.inverse_superseded_operation.added_edges
        if hasattr(self.inverse_superseded_operation, "removed_edges"):
            self.removed_edges = self.inverse_superseded_operation.removed_edges

    def _update_root_ids(self):
        if self.operation_status != attributes.OperationLogs.StatusCodes.SUCCESS.value:
            return types.empty_1d
        return self.inverse_superseded_operation._update_root_ids()

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        if isinstance(self.inverse_superseded_operation, MergeOperation):
            return edits.add_edges(
                self.inverse_superseded_operation.cg,
                atomic_edges=self.inverse_superseded_operation.added_edges,
                operation_id=operation_id,
                time_stamp=timestamp,
                parent_ts=self.inverse_superseded_operation.parent_ts,
                allow_same_segment_merge=True,
            )
        return self.inverse_superseded_operation._apply(
            operation_id=operation_id, timestamp=timestamp
        )

    def _create_log_record(
        self,
        *,
        operation_id: np.uint64,
        timestamp: datetime,
        operation_ts: datetime,
        new_root_ids: Sequence[np.uint64],
        status: int = 1,
        exception: str = "",
    ) -> "bigtable.row.Row":
        val_dict = {
            attributes.OperationLogs.UserID: self.user_id,
            attributes.OperationLogs.UndoOperationID: self.superseded_operation_id,
            attributes.OperationLogs.RootID: new_root_ids,
            attributes.OperationLogs.OperationTimeStamp: operation_ts,
            attributes.OperationLogs.Status: status,
            attributes.OperationLogs.OperationException: exception,
        }
        if hasattr(self, "added_edges"):
            val_dict[attributes.OperationLogs.AddedEdge] = self.added_edges
        if hasattr(self, "removed_edges"):
            val_dict[attributes.OperationLogs.RemovedEdge] = self.removed_edges
        return self.cg.client.mutate_row(
            serializers.serialize_uint64(operation_id), val_dict, timestamp
        )

    def invert(self) -> "GraphEditOperation":
        """
        Inverts an UndoOperation. Treated as Redoing the original operation
        """
        return RedoOperation(
            self.cg,
            user_id=self.user_id,
            superseded_operation_id=self.superseded_operation_id,
            multicut_as_split=True,
        )

    def execute(
        self, *, operation_id=None, parent_ts=None, override_ts=None
    ) -> "GraphEditOperation.Result":
        if self.operation_status != attributes.OperationLogs.StatusCodes.SUCCESS.value:
            # Don't undo failed operations
            return GraphEditOperation.Result(
                operation_id=operation_id,
                new_root_ids=types.empty_1d,
                new_lvl2_ids=types.empty_1d,
            )
        if isinstance(self.inverse_superseded_operation, MergeOperation):
            # in case we are undoing a partial split (with only one resulting root id)
            e, a = get_edges_status(
                self.inverse_superseded_operation.cg,
                self.inverse_superseded_operation.added_edges,
            )
            if np.any(~e):
                raise PreconditionError("All edges must exist.")
            if np.all(a):
                return GraphEditOperation.Result(
                    operation_id=operation_id,
                    new_root_ids=types.empty_1d,
                    new_lvl2_ids=types.empty_1d,
                )
        if isinstance(self.inverse_superseded_operation, SplitOperation):
            e, a = get_edges_status(
                self.inverse_superseded_operation.cg,
                self.inverse_superseded_operation.removed_edges,
            )
            if np.any(~e):
                raise PreconditionError("All edges must exist.")
            if np.all(~a):
                return GraphEditOperation.Result(
                    operation_id=operation_id,
                    new_root_ids=types.empty_1d,
                    new_lvl2_ids=types.empty_1d,
                )
        return super().execute(
            operation_id=operation_id, parent_ts=parent_ts, override_ts=override_ts
        )
