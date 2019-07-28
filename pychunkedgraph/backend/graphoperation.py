import itertools
from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pychunkedgraph.backend import chunkedgraph_edits as cg_edits
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.backend.root_lock import RootLock
from pychunkedgraph.backend.utils import basetypes, column_keys, serializers

if TYPE_CHECKING:
    from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
    from google.cloud import bigtable


class GraphEditOperation(ABC):
    __slots__ = ["cg", "user_id", "source_coords", "sink_coords"]
    Result = namedtuple("Result", ["operation_id", "new_root_ids", "new_lvl2_ids"])

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        source_coords: Optional[Sequence[Sequence[np.int]]] = None,
        sink_coords: Optional[Sequence[Sequence[np.int]]] = None,
    ) -> None:
        super().__init__()
        self.cg = cg
        self.user_id = user_id
        self.source_coords = None
        self.sink_coords = None

        if source_coords is not None:
            self.source_coords = np.atleast_2d(source_coords).astype(basetypes.COORDINATES)
        if sink_coords is not None:
            self.sink_coords = np.atleast_2d(sink_coords).astype(basetypes.COORDINATES)

    @staticmethod
    def from_log_record(
        cg: "ChunkedGraph",
        log_record: Dict[column_keys._Column, Union[np.ndarray, np.number]],
        *,
        multicut_as_split=True,
    ) -> "GraphEditOperation":
        """Generates the correct GraphEditOperation given a log record dictionary.
        :param cg: The ChunkedGraph instance
        :type cg: "ChunkedGraph"
        :param log_record: log record dictionary
        :type log_record: Dict[column_keys._Column, Union[np.ndarray, np.number]]
        :param multicut_as_split: If true, don't recalculate MultiCutOperation, just
            use the resulting removed edges and generate SplitOperation instead (faster).

        :return: The matching GraphEditOperation subclass
        :rtype: "GraphEditOperation"
        """
        user_id = log_record[column_keys.OperationLogs.UserID]

        if column_keys.OperationLogs.UndoOperationID in log_record:
            superseded_operation_id = log_record[column_keys.OperationLogs.UndoOperationID]
            return UndoOperation(
                cg, user_id=user_id, superseded_operation_id=superseded_operation_id
            )

        if column_keys.OperationLogs.RedoOperationID in log_record:
            superseded_operation_id = log_record[column_keys.OperationLogs.RedoOperationID]
            return RedoOperation(
                cg, user_id=user_id, superseded_operation_id=superseded_operation_id
            )

        source_ids = log_record[column_keys.OperationLogs.SourceID]
        sink_ids = log_record[column_keys.OperationLogs.SinkID]
        source_coords = log_record[column_keys.OperationLogs.SourceCoordinate]
        sink_coords = log_record[column_keys.OperationLogs.SinkCoordinate]

        if column_keys.OperationLogs.AddedEdge in log_record:
            added_edges = log_record[column_keys.OperationLogs.AddedEdge]
            affinities = log_record[column_keys.OperationLogs.Affinity]
            return MergeOperation(
                cg,
                user_id=user_id,
                source_coords=source_coords,
                sink_coords=sink_coords,
                added_edges=added_edges,
                affinities=affinities,
            )

        if column_keys.OperationLogs.RemovedEdge in log_record:
            removed_edges = log_record[column_keys.OperationLogs.RemovedEdge]
            if multicut_as_split or column_keys.OperationLogs.BoundingBoxOffset not in log_record:
                return SplitOperation(
                    cg,
                    user_id=user_id,
                    source_coords=source_coords,
                    sink_coords=sink_coords,
                    removed_edges=removed_edges,
                )

            bbox_offset = log_record[column_keys.OperationLogs.BoundingBoxOffset]
            return MulticutOperation(
                cg,
                user_id=user_id,
                source_coords=source_coords,
                sink_coords=sink_coords,
                bbox_offset=bbox_offset,
                source_ids=source_ids,
                sink_ids=sink_ids,
            )

        raise cg_exceptions.PreconditionError(
            f"Log record contains neither added nor removed edges."
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
        :return: New root IDs, new Lvl2 node IDs, and affected Bigtable rows
        :rtype: Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]
        """

    @abstractmethod
    def _create_log_record(self, *, operation_id, timestamp, new_root_ids) -> "bigtable.row.Row":
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

    def execute(self) -> "GraphEditOperation.Result":
        """Executes current GraphEditOperation:
            * Calls the subclass's _update_root_ids method
            * Locks root IDs
            * Calls the subclass's _apply method
            * Calls the subclass's _create_log_record method
            * Writes all new rows to Bigtable
            * Releases root ID lock
        :return: Result of successful graph operation
        :rtype: GraphEditOperation.Result
        """
        root_ids = self._update_root_ids()

        with RootLock(self.cg, root_ids) as root_lock:
            lock_operation_ids = np.array([root_lock.operation_id] * len(root_lock.locked_root_ids))
            timestamp = self.cg.read_consolidated_lock_timestamp(
                root_lock.locked_root_ids, lock_operation_ids
            )

            new_root_ids, new_lvl2_ids, rows = self._apply(
                operation_id=root_lock.operation_id, timestamp=timestamp
            )

            # FIXME: Remove once cg_edits.remove_edges/cg_edits.add_edges return consistent type
            new_root_ids = np.array(new_root_ids, dtype=basetypes.NODE_ID)
            new_lvl2_ids = np.array(new_lvl2_ids, dtype=basetypes.NODE_ID)

            # Add a row to the log
            log_row = self._create_log_record(
                operation_id=root_lock.operation_id, new_root_ids=new_root_ids, timestamp=timestamp
            )

            # Put log row first!
            rows = [log_row] + rows

            # Execute write (makes sure that we are still owning the lock)
            self.cg.bulk_write(
                rows,
                root_lock.locked_root_ids,
                operation_id=root_lock.operation_id,
                slow_retry=False,
            )
            return GraphEditOperation.Result(
                operation_id=root_lock.operation_id,
                new_root_ids=new_root_ids,
                new_lvl2_ids=new_lvl2_ids,
            )


class MergeOperation(GraphEditOperation):
    """Merge Operation: Connect *known* pairs of supervoxels by adding a (weighted) edge.

    :param cg: The ChunkedGraph object
    :type cg: ChunkedGraph
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param added_edges: Supervoxel IDs of all added edges [[source, sink]]
    :type added_edges: Sequence[Sequence[np.uint64]]
    :param source_coords: world space coordinates in nm, corresponding to IDs in added_edges[:,0], defaults to None
    :type source_coords: Optional[Sequence[Sequence[np.int]]], optional
    :param sink_coords: world space coordinates in nm, corresponding to IDs in added_edges[:,1], defaults to None
    :type sink_coords: Optional[Sequence[Sequence[np.int]]], optional
    :param affinities: edge weights for newly added edges, entries corresponding to added_edges, defaults to None
    :type affinities: Optional[Sequence[np.float32]], optional
    """

    __slots__ = ["added_edges", "affinities"]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        added_edges: Sequence[Sequence[np.uint64]],
        source_coords: Optional[Sequence[Sequence[np.int]]] = None,
        sink_coords: Optional[Sequence[Sequence[np.int]]] = None,
        affinities: Optional[Sequence[np.float32]] = None,
    ) -> None:
        super().__init__(cg, user_id=user_id, source_coords=source_coords, sink_coords=sink_coords)
        self.added_edges = np.atleast_2d(added_edges).astype(basetypes.NODE_ID)
        self.affinities = None

        if affinities is not None:
            self.affinities = np.atleast_1d(affinities).astype(basetypes.EDGE_AFFINITY)

        if np.any(np.in1d(self.added_edges[:, 0], self.added_edges[:, 1])):
            raise cg_exceptions.PreconditionError(
                f"One or more supervoxel exists as both, sink and source."
            )

        for supervoxel_id in self.added_edges.ravel():
            layer = self.cg.get_chunk_layer(supervoxel_id)
            if layer != 1:
                raise cg_exceptions.PreconditionError(
                    f"Supervoxel expected, but {supervoxel_id} is a layer {layer} node."
                )

    def _update_root_ids(self) -> np.ndarray:
        root_ids = np.unique(self.cg.get_roots(self.added_edges.ravel()))
        return root_ids

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        new_root_ids, new_lvl2_ids, rows = cg_edits.add_edges(
            self.cg,
            operation_id,
            atomic_edges=self.added_edges,
            time_stamp=timestamp,
            affinities=self.affinities,
        )
        return new_root_ids, new_lvl2_ids, rows

    def _create_log_record(self, *, operation_id, timestamp, new_root_ids) -> "bigtable.row.Row":
        val_dict = {
            column_keys.OperationLogs.UserID: self.user_id,
            column_keys.OperationLogs.RootID: new_root_ids,
            column_keys.OperationLogs.AddedEdge: self.added_edges,
        }
        if self.source_coords is not None:
            val_dict[column_keys.OperationLogs.SourceCoordinate] = self.source_coords
        if self.sink_coords is not None:
            val_dict[column_keys.OperationLogs.SinkCoordinate] = self.sink_coords
        if self.affinities is not None:
            val_dict[column_keys.OperationLogs.Affinity] = self.affinities

        return self.cg.mutate_row(serializers.serialize_uint64(operation_id), val_dict, timestamp)

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

    :param cg: The ChunkedGraph object
    :type cg: ChunkedGraph
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param removed_edges: Supervoxel IDs of all removed edges [[source, sink]]
    :type removed_edges: Sequence[Sequence[np.uint64]]
    :param source_coords: world space coordinates in nm, corresponding to IDs in
        removed_edges[:,0], defaults to None
    :type source_coords: Optional[Sequence[Sequence[np.int]]], optional
    :param sink_coords: world space coordinates in nm, corresponding to IDs in
        removed_edges[:,1], defaults to None
    :type sink_coords: Optional[Sequence[Sequence[np.int]]], optional
    """

    __slots__ = ["removed_edges"]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        removed_edges: Sequence[Sequence[np.uint64]],
        source_coords: Optional[Sequence[Sequence[np.int]]] = None,
        sink_coords: Optional[Sequence[Sequence[np.int]]] = None,
    ) -> None:
        super().__init__(cg, user_id=user_id, source_coords=source_coords, sink_coords=sink_coords)
        self.removed_edges = np.atleast_2d(removed_edges).astype(basetypes.NODE_ID)

        if np.any(np.in1d(self.removed_edges[:, 0], self.removed_edges[:, 1])):
            raise cg_exceptions.PreconditionError(
                f"One or more supervoxel exists as both, sink and source."
            )

        for supervoxel_id in self.removed_edges.ravel():
            layer = self.cg.get_chunk_layer(supervoxel_id)
            if layer != 1:
                raise cg_exceptions.PreconditionError(
                    f"Supervoxel expected, but {supervoxel_id} is a layer {layer} node."
                )

    def _update_root_ids(self) -> np.ndarray:
        root_ids = np.unique(self.cg.get_roots(self.removed_edges.ravel()))
        if len(root_ids) > 1:
            raise cg_exceptions.PreconditionError(
                f"All supervoxel must belong to the same object. Already split?"
            )
        return root_ids

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        new_root_ids, new_lvl2_ids, rows = cg_edits.remove_edges(
            self.cg, operation_id, atomic_edges=self.removed_edges, time_stamp=timestamp
        )
        return new_root_ids, new_lvl2_ids, rows

    def _create_log_record(
        self, *, operation_id: np.uint64, timestamp: datetime, new_root_ids: Sequence[np.uint64]
    ) -> "bigtable.row.Row":
        val_dict = {
            column_keys.OperationLogs.UserID: self.user_id,
            column_keys.OperationLogs.RootID: new_root_ids,
            column_keys.OperationLogs.RemovedEdge: self.removed_edges,
        }
        if self.source_coords is not None:
            val_dict[column_keys.OperationLogs.SourceCoordinate] = self.source_coords
        if self.sink_coords is not None:
            val_dict[column_keys.OperationLogs.SinkCoordinate] = self.sink_coords

        return self.cg.mutate_row(serializers.serialize_uint64(operation_id), val_dict, timestamp)

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

    :param cg: The ChunkedGraph object
    :type cg: ChunkedGraph
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param source_ids: Supervoxel IDs that should be separated from supervoxel IDs in sink_ids
    :type souce_ids: Sequence[np.uint64]
    :param sink_ids: Supervoxel IDs that should be separated from supervoxel IDs in source_ids
    :type sink_ids: Sequence[np.uint64]
    :param source_coords: world space coordinates in nm, corresponding to IDs in source_ids
    :type source_coords: Sequence[Sequence[np.int]]
    :param sink_coords: world space coordinates in nm, corresponding to IDs in sink_ids
    :type sink_coords: Sequence[Sequence[np.int]]
    :param bbox_offset: Padding for min-cut bounding box, applied to min/max coordinates
        retrieved from source_coords and sink_coords, defaults to None
    :type bbox_offset: Optional[Sequence[np.int]], optional
    """

    __slots__ = ["source_ids", "sink_ids", "removed_edges", "bbox_offset"]

    def __init__(
        self,
        cg: "ChunkedGraph",
        *,
        user_id: str,
        source_ids: Sequence[np.uint64],
        sink_ids: Sequence[np.uint64],
        source_coords: Sequence[Sequence[np.int]],
        sink_coords: Sequence[Sequence[np.int]],
        bbox_offset: Optional[Sequence[np.int]] = None,
    ) -> None:
        super().__init__(cg, user_id=user_id, source_coords=source_coords, sink_coords=sink_coords)
        self.removed_edges = None  # Calculated from coordinates and IDs
        self.source_ids = np.atleast_1d(source_ids).astype(basetypes.NODE_ID)
        self.sink_ids = np.atleast_1d(sink_ids).astype(basetypes.NODE_ID)
        self.bbox_offset = None

        if bbox_offset is not None:
            self.bbox_offset = np.atleast_1d(bbox_offset).astype(basetypes.COORDINATES)

        if np.any(np.in1d(self.sink_ids, self.source_ids)):
            raise cg_exceptions.PreconditionError(
                f"One or more supervoxel exists as both, sink and source."
            )

        for supervoxel_id in itertools.chain(self.source_ids, self.sink_ids):
            layer = self.cg.get_chunk_layer(supervoxel_id)
            if layer != 1:
                raise cg_exceptions.PreconditionError(
                    f"Supervoxel expected, but {supervoxel_id} is a layer {layer} node."
                )

    def _update_root_ids(self) -> np.ndarray:
        sink_and_source_ids = np.concatenate((self.source_ids, self.sink_ids))
        root_ids = np.unique(self.cg.get_roots(sink_and_source_ids))
        if len(root_ids) > 1:
            raise cg_exceptions.PreconditionError(
                f"All supervoxel must belong to the same object. Already split?"
            )
        return root_ids

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        self.removed_edges = self.cg._run_multicut(
            self.source_ids, self.sink_ids, self.source_coords, self.sink_coords, self.bbox_offset
        )

        if self.removed_edges.size == 0:
            raise cg_exceptions.PostconditionError(
                "Mincut could not find any edges to remove - weird!"
            )

        new_root_ids, new_lvl2_ids, rows = cg_edits.remove_edges(
            self.cg, operation_id, atomic_edges=self.removed_edges, time_stamp=timestamp
        )
        return new_root_ids, new_lvl2_ids, rows

    def _create_log_record(
        self, *, operation_id: np.uint64, timestamp: datetime, new_root_ids: Sequence[np.uint64]
    ) -> "bigtable.row.Row":
        val_dict = {
            column_keys.OperationLogs.UserID: self.user_id,
            column_keys.OperationLogs.RootID: new_root_ids,
            column_keys.OperationLogs.SourceCoordinate: self.source_coords,
            column_keys.OperationLogs.SinkCoordinate: self.sink_coords,
            column_keys.OperationLogs.SourceID: self.source_ids,
            column_keys.OperationLogs.SinkID: self.sink_ids,
        }
        if self.bbox_offset is not None:
            val_dict[column_keys.OperationLogs.BoundingBoxOffset] = self.bbox_offset

        return self.cg.mutate_row(serializers.serialize_uint64(operation_id), val_dict, timestamp)

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

    :param cg: The ChunkedGraph object
    :type cg: ChunkedGraph
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param superseded_operation_id: Operation ID to be redone
    :type superseded_operation_id: np.uint64
    """

    __slots__ = ["superseded_operation_id", "superseded_operation"]

    def __new__(cls, cg: "ChunkedGraph", *, user_id: str, superseded_operation_id: np.uint64):
        # Path compression:
        # RedoOperation[RedoOperation[x]] -> RedoOperation[x]
        # RedoOperation[UndoOperation[x]] -> UndoOperation[x]
        log_record = cg.read_log_row(superseded_operation_id)
        superseded_operation = GraphEditOperation.from_log_record(cg, log_record)
        if isinstance(superseded_operation, RedoOperation):
            original_operation_id = log_record[column_keys.OperationLogs.RedoOperationID]
            return RedoOperation(cg, user_id=user_id, superseded_operation_id=original_operation_id)
        if isinstance(superseded_operation, UndoOperation):
            original_operation_id = log_record[column_keys.OperationLogs.UndoOperationID]
            return UndoOperation(cg, user_id=user_id, superseded_operation_id=original_operation_id)
        return super().__new__(cls)

    def __init__(
        self, cg: "ChunkedGraph", *, user_id: str, superseded_operation_id: np.uint64
    ) -> None:
        super().__init__(cg, user_id=user_id)
        superseded_operation = GraphEditOperation.from_log_record(
            cg, cg.read_log_row(superseded_operation_id)
        )
        self.superseded_operation = superseded_operation
        self.superseded_operation_id = superseded_operation_id

    def _update_root_ids(self):
        return self.superseded_operation._update_root_ids()

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        return self.superseded_operation._apply(operation_id=operation_id, timestamp=timestamp)

    def _create_log_record(
        self, *, operation_id: np.uint64, timestamp: datetime, new_root_ids: Sequence[np.uint64]
    ) -> "bigtable.row.Row":
        val_dict = {
            column_keys.OperationLogs.UserID: self.user_id,
            column_keys.OperationLogs.RedoOperationID: self.superseded_operation_id,
            column_keys.OperationLogs.RootID: new_root_ids,
        }
        return self.cg.mutate_row(serializers.serialize_uint64(operation_id), val_dict, timestamp)

    def invert(self) -> "GraphEditOperation":
        """
        Inverts a RedoOperation. Results in the inverse of the original GraphEditOperation
        """
        return self.superseded_operation.invert()


class UndoOperation(GraphEditOperation):
    """
    UndoOperation: Used to apply the inverse of a previous graph edit operation. In contrast
        to a "coincidental" undo (e.g. merging an edge previously removed by a split operation), an
        UndoOperation is linked to an earlier operation ID to enable its correct reversal.

    :param cg: The ChunkedGraph object
    :type cg: ChunkedGraph
    :param user_id: User ID that will be assigned to this operation
    :type user_id: str
    :param superseded_operation_id: Operation ID to be undone
    :type superseded_operation_id: np.uint64
    """

    __slots__ = ["superseded_operation_id", "superseded_operation", "inverse_superseded_operation"]

    def __new__(cls, cg: "ChunkedGraph", *, user_id: str, superseded_operation_id: np.uint64):
        # Path compression:
        # UndoOperation[RedoOperation[x]] -> UndoOperation[x]
        # UndoOperation[UndoOperation[x]] -> RedoOperation[x]
        log_record = cg.read_log_row(superseded_operation_id)
        superseded_operation = GraphEditOperation.from_log_record(cg, log_record)
        if isinstance(superseded_operation, RedoOperation):
            original_operation_id = log_record[column_keys.OperationLogs.RedoOperationID]
            return UndoOperation(cg, user_id=user_id, superseded_operation_id=original_operation_id)
        if isinstance(superseded_operation, UndoOperation):
            original_operation_id = log_record[column_keys.OperationLogs.UndoOperationID]
            return RedoOperation(cg, user_id=user_id, superseded_operation_id=original_operation_id)
        return super().__new__(cls)

    def __init__(
        self, cg: "ChunkedGraph", *, user_id: str, superseded_operation_id: np.uint64
    ) -> None:
        super().__init__(cg, user_id=user_id)
        superseded_operation = GraphEditOperation.from_log_record(
            cg, cg.read_log_row(superseded_operation_id)
        )
        self.superseded_operation = superseded_operation
        self.inverse_superseded_operation = superseded_operation.invert()
        self.superseded_operation_id = superseded_operation_id

    def _update_root_ids(self):
        return self.inverse_superseded_operation._update_root_ids()

    def _apply(
        self, *, operation_id, timestamp
    ) -> Tuple[np.ndarray, np.ndarray, List["bigtable.row.Row"]]:
        return self.inverse_superseded_operation._apply(
            operation_id=operation_id, timestamp=timestamp
        )

    def _create_log_record(
        self, *, operation_id: np.uint64, timestamp: datetime, new_root_ids: Sequence[np.uint64]
    ) -> "bigtable.row.Row":
        val_dict = {
            column_keys.OperationLogs.UserID: self.user_id,
            column_keys.OperationLogs.UndoOperationID: self.superseded_operation_id,
            column_keys.OperationLogs.RootID: new_root_ids,
        }
        return self.cg.mutate_row(serializers.serialize_uint64(operation_id), val_dict, timestamp)

    def invert(self) -> "GraphEditOperation":
        """
        Inverts an UndoOperation. Results in the original GraphEditOperation itself
        """
        return self.superseded_operation
