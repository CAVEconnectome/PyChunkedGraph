from typing import NamedTuple
from pychunkedgraph.backend.utils import basetypes, serializers


class _ColumnType(NamedTuple):
    key: bytes
    family_id: str
    serializer: serializers._Serializer


class _Column(_ColumnType):
    __slots__ = ()
    _columns = {}

    def __init__(self, **kwargs):
        super().__init__()
        _Column._columns[(kwargs['family_id'], kwargs['key'])] = self

    def serialize(self, obj):
        return self.serializer.serialize(obj)

    def deserialize(self, stream):
        return self.serializer.deserialize(stream)

    @property
    def basetype(self):
        return self.serializer.basetype


class _ColumnArray():
    _columnarrays = {}

    def __init__(self, pattern, family_id, serializer):
        self._pattern = pattern
        self._family_id = family_id
        self._serializer = serializer
        _ColumnArray._columnarrays[(family_id, pattern)] = self

        # TODO: Add missing check in `fromkey(family_id, key)` and remove this
        #       loop (pre-creates `_Columns`, so that the inverse lookup works)
        for i in range(20):
            self[i]  # pylint: disable=W0104

    def __getitem__(self, item):
        return _Column(key=self.pattern % item,
                       family_id=self.family_id,
                       serializer=self._serializer)

    @property
    def pattern(self):
        return self._pattern

    @property
    def family_id(self):
        return self._family_id

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
    CounterID = _Column(
        key=b'counter',
        family_id='1',
        serializer=serializers.NumPyValue(dtype=basetypes.COUNTER))

    Lock = _Column(
        key=b'lock',
        family_id='0',
        serializer=serializers.UInt64String())


class Connectivity:
    Affinity = _Column(
        key=b'affinities',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.EDGE_AFFINITY))

    Area = _Column(
        key=b'areas',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.EDGE_AREA))

    Connected = _Column(
        key=b'connected',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    Disconnected = _Column(
        key=b'disconnected',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    Partner = _Column(
        key=b'atomic_partners',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    CrossChunkEdge = _ColumnArray(
        pattern=b'atomic_cross_edges_%d',
        family_id='3',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2)))


class Hierarchy:
    Child = _Column(
        key=b'children',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    FormerParent = _Column(
        key=b'former_parents',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    NewParent = _Column(
        key=b'new_parents',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    Parent = _Column(
        key=b'parents',
        family_id='0',
        serializer=serializers.NumPyValue(dtype=basetypes.NODE_ID))


class GraphSettings:
    DatasetInfo = _Column(
        key=b'dataset_info',
        family_id='0',
        serializer=serializers.JSON())

    ChunkSize = _Column(
        key=b'chunk_size',
        family_id='0',
        serializer=serializers.NumPyArray(dtype=basetypes.CHUNKSIZE))

    FanOut = _Column(
        key=b'fan_out',
        family_id='0',
        serializer=serializers.NumPyValue(dtype=basetypes.FANOUT))

    LayerCount = _Column(
        key=b'n_layers',
        family_id='0',
        serializer=serializers.NumPyValue(dtype=basetypes.LAYERCOUNT))

    SegmentationPath = _Column(
        key=b'cv_path',
        family_id='0',
        serializer=serializers.String('utf-8'))

    MeshDir = _Column(
        key=b'mesh_dir',
        family_id='0',
        serializer=serializers.String('utf-8'))


class OperationLogs:
    OperationID = _Column(
        key=b'operation_id',
        family_id='0',
        serializer=serializers.UInt64String())

    UserID = _Column(
        key=b'user',
        family_id='2',
        serializer=serializers.String('utf-8'))

    RootID = _Column(
        key=b'roots',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    SourceID = _Column(
        key=b'source_ids',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    SinkID = _Column(
        key=b'sink_ids',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID))

    SourceCoordinate = _Column(
        key=b'source_coords',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.COORDINATES, shape=(-1, 3)))

    SinkCoordinate = _Column(
        key=b'sink_coords',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.COORDINATES, shape=(-1, 3)))

    BoundingBoxOffset = _Column(
        key=b'bb_offset',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.COORDINATES))

    AddedEdge = _Column(
        key=b'added_edges',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2)))

    RemovedEdge = _Column(
        key=b'removed_edges',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.NODE_ID, shape=(-1, 2)))

    Affinity = _Column(
        key=b'affinities',
        family_id='2',
        serializer=serializers.NumPyArray(dtype=basetypes.EDGE_AFFINITY))


def from_key(family_id: str, key: bytes):
    try:
        return _Column._columns[(family_id, key)]
    except KeyError:
        # FIXME: Look if the key matches a columnarray pattern and
        #        remove loop initialization in _ColumnArray.__init__()
        raise KeyError(f"Unknown key {family_id}:{key.decode()}")
