# TODO design to use these attributes across different clients
# `family_id` is specific to bigtable

from typing import NamedTuple

from .utils import serializers
from .utils import basetypes


class _AttributeType(NamedTuple):
    key: bytes
    family_id: str
    serializer: serializers._Serializer


class _Attribute(_AttributeType):
    __slots__ = ()
    _attributes = {}

    def __init__(self, **kwargs):
        super().__init__()
        _Attribute._attributes[(kwargs["family_id"], kwargs["key"])] = self

    def serialize(self, obj):
        return self.serializer.serialize(obj)

    def deserialize(self, stream):
        return self.serializer.deserialize(stream)

    @property
    def basetype(self):
        return self.serializer.basetype

    @property
    def index(self):
        return int(self.key.decode("utf-8").split("_")[-1])


class _AttributeArray:
    _attributearrays = {}

    def __init__(self, pattern, family_id, serializer):
        self._pattern = pattern
        self._family_id = family_id
        self._serializer = serializer
        _AttributeArray._attributearrays[(family_id, pattern)] = self

        # TODO: Add missing check in `fromkey(family_id, key)` and remove this
        #       loop (pre-creates `_Attributes`, so that the inverse lookup works)
        for i in range(20):
            self[i]  # pylint: disable=W0104

    def __getitem__(self, item):
        return _Attribute(
            key=self.pattern % item,
            family_id=self._family_id,
            serializer=self._serializer,
        )

    @property
    def pattern(self):
        return self._pattern

    @property
    def serialize(self):
        return self._serializer.serialize

    @property
    def deserialize(self):
        return self._serializer.deserialize

    @property
    def basetype(self):
        return self._serializer.basetype


class Concurrency:
    Counter = _Attribute(
        key=b"counter",
        family_id="1",
        serializer=serializers.NumPyValue(dtype=basetypes.COUNTER),
    )

    Lock = _Attribute(key=b"lock", family_id="0", serializer=serializers.UInt64String())


class Connectivity:
    Affinity = _Attribute(
        key=b"affinities",
        family_id="0",
        serializer=serializers.NumPyArray(dtype=basetypes.EDGE_AFFINITY),
    )

    Area = _Attribute(
        key=b"areas",
        family_id="0",
        serializer=serializers.NumPyArray(dtype=basetypes.EDGE_AREA),
    )

    CrossChunkEdge = _AttributeArray(
        pattern=b"atomic_cross_edges_%d",
        family_id="3",
        serializer=serializers.NumPyArray(
            dtype=basetypes.NODE_ID, shape=(-1, 2), compression_level=22
        ),
    )

    FakeEdges = _Attribute(
        key=b"fake_edges",
        family_id="3",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2)),
    )


class Hierarchy:
    Child = _Attribute(
        key=b"children",
        family_id="0",
        serializer=serializers.NumPyArray(
            dtype=basetypes.NODE_ID, compression_level=22
        ),
    )

    FormerParent = _Attribute(
        key=b"former_parents",
        family_id="0",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID),
    )

    NewParent = _Attribute(
        key=b"new_parents",
        family_id="0",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID),
    )

    Parent = _Attribute(
        key=b"parents",
        family_id="0",
        serializer=serializers.NumPyValue(dtype=basetypes.NODE_ID),
    )


class GraphMeta:
    key = b"meta"
    Meta = _Attribute(key=b"meta", family_id="0", serializer=serializers.Pickle())


class GraphProvenance:
    key = b"provenance"
    Provenance = _Attribute(
        key=b"provenance", family_id="0", serializer=serializers.Pickle()
    )


class OperationLogs:
    key = b"ioperations"
    OperationID = _Attribute(
        key=b"operation_id", family_id="0", serializer=serializers.UInt64String()
    )

    UndoOperationID = _Attribute(
        key=b"undo_operation_id", family_id="2", serializer=serializers.UInt64String()
    )

    RedoOperationID = _Attribute(
        key=b"redo_operation_id", family_id="2", serializer=serializers.UInt64String()
    )

    UserID = _Attribute(
        key=b"user", family_id="2", serializer=serializers.String("utf-8")
    )

    RootID = _Attribute(
        key=b"roots",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID),
    )

    SourceID = _Attribute(
        key=b"source_ids",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID),
    )

    SinkID = _Attribute(
        key=b"sink_ids",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID),
    )

    SourceCoordinate = _Attribute(
        key=b"source_coords",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.COORDINATES, shape=(-1, 3)),
    )

    SinkCoordinate = _Attribute(
        key=b"sink_coords",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.COORDINATES, shape=(-1, 3)),
    )

    BoundingBoxOffset = _Attribute(
        key=b"bb_offset",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.COORDINATES),
    )

    AddedEdge = _Attribute(
        key=b"added_edges",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2)),
    )

    RemovedEdge = _Attribute(
        key=b"removed_edges",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2)),
    )

    Affinity = _Attribute(
        key=b"affinities",
        family_id="2",
        serializer=serializers.NumPyArray(dtype=basetypes.EDGE_AFFINITY),
    )

    @staticmethod
    def all():
        return [
            OperationLogs.OperationID,
            OperationLogs.UndoOperationID,
            OperationLogs.RedoOperationID,
            OperationLogs.UserID,
            OperationLogs.RootID,
            OperationLogs.SourceID,
            OperationLogs.SinkID,
            OperationLogs.SourceCoordinate,
            OperationLogs.SinkCoordinate,
            OperationLogs.BoundingBoxOffset,
            OperationLogs.AddedEdge,
            OperationLogs.RemovedEdge,
            OperationLogs.Affinity,
        ]


def from_key(family_id: str, key: bytes):
    try:
        return _Attribute._attributes[(family_id, key)]
    except KeyError:
        # FIXME: Look if the key matches a columnarray pattern and
        #        remove loop initialization in _AttributeArray.__init__()
        raise KeyError(f"Unknown key {family_id}:{key.decode()}")
