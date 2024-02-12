"""
Classes and types for edges
"""

from collections import namedtuple
from os import environ
from typing import Optional

import numpy as np
import tensorstore as ts
import zstandard as zstd
from graph_tool import Graph

from ..utils import basetypes


_edge_type_fileds = ("in_chunk", "between_chunk", "cross_chunk")
_edge_type_defaults = ("in", "between", "cross")

EdgeTypes = namedtuple("EdgeTypes", _edge_type_fileds, defaults=_edge_type_defaults)
EDGE_TYPES = EdgeTypes()

DEFAULT_AFFINITY = np.finfo(np.float32).tiny
DEFAULT_AREA = np.finfo(np.float32).tiny
ADJACENCY_DTYPE = np.dtype(
    [
        ("node", basetypes.NODE_ID),
        ("aff", basetypes.EDGE_AFFINITY),
        ("area", basetypes.EDGE_AREA),
    ]
)
ZSTD_EDGE_COMPRESSION = 17


class Edges:
    def __init__(
        self,
        node_ids1: np.ndarray,
        node_ids2: np.ndarray,
        *,
        affinities: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
    ):
        self.node_ids1 = np.array(node_ids1, dtype=basetypes.NODE_ID, copy=False)
        self.node_ids2 = np.array(node_ids2, dtype=basetypes.NODE_ID, copy=False)
        assert self.node_ids1.size == self.node_ids2.size

        self._as_pairs = None

        if affinities is not None and len(affinities) > 0:
            self._affinities = np.array(
                affinities, dtype=basetypes.EDGE_AFFINITY, copy=False
            )
            assert self.node_ids1.size == self._affinities.size
        else:
            self._affinities = np.full(len(self.node_ids1), DEFAULT_AFFINITY)

        if areas is not None and len(areas) > 0:
            self._areas = np.array(areas, dtype=basetypes.EDGE_AREA, copy=False)
            assert self.node_ids1.size == self._areas.size
        else:
            self._areas = np.full(len(self.node_ids1), DEFAULT_AREA)

    @property
    def affinities(self) -> np.ndarray:
        return self._affinities

    @affinities.setter
    def affinities(self, affinities):
        self._affinities = affinities

    @property
    def areas(self) -> np.ndarray:
        return self._areas

    @areas.setter
    def areas(self, areas):
        self._areas = areas

    def __add__(self, other):
        """add two Edges instances"""
        node_ids1 = np.concatenate([self.node_ids1, other.node_ids1])
        node_ids2 = np.concatenate([self.node_ids2, other.node_ids2])
        affinities = np.concatenate([self.affinities, other.affinities])
        areas = np.concatenate([self.areas, other.areas])
        return Edges(node_ids1, node_ids2, affinities=affinities, areas=areas)

    def __iadd__(self, other):
        self.node_ids1 = np.concatenate([self.node_ids1, other.node_ids1])
        self.node_ids2 = np.concatenate([self.node_ids2, other.node_ids2])
        self.affinities = np.concatenate([self.affinities, other.affinities])
        self.areas = np.concatenate([self.areas, other.areas])
        return self

    def __len__(self):
        return self.node_ids1.size

    def __getitem__(self, key):
        """`key` must be a boolean numpy array."""
        try:
            return Edges(
                self.node_ids1[key],
                self.node_ids2[key],
                affinities=self.affinities[key],
                areas=self.areas[key],
            )
        except Exception as err:
            raise (err)

    def get_pairs(self) -> np.ndarray:
        """
        return numpy array of edge pairs [[sv1, sv2] ... ]
        """
        if not self._as_pairs is None:
            return self._as_pairs
        self._as_pairs = np.column_stack((self.node_ids1, self.node_ids2))
        return self._as_pairs


def put_edges(destination: str, nodes: np.ndarray, edges: Edges) -> None:
    graph_ids, _edges = np.unique(edges.get_pairs(), return_inverse=True)
    graph_ids_reverse = {n: i for i, n in enumerate(graph_ids)}
    _edges = _edges.reshape(-1, 2)

    graph = Graph(directed=False)
    graph.add_edge_list(_edges)
    e_aff = graph.new_edge_property("double", vals=edges.affinities)
    e_area = graph.new_edge_property("int", vals=edges.areas)
    cctx = zstd.ZstdCompressor(level=ZSTD_EDGE_COMPRESSION)
    ocdbt_host = environ["OCDBT_COORDINATOR_HOST"]
    ocdbt_port = environ["OCDBT_COORDINATOR_PORT"]

    spec = {
        "driver": "ocdbt",
        "base": destination,
        "coordinator": {"address": f"{ocdbt_host}:{ocdbt_port}"},
    }
    dataset = ts.KvStore.open(spec).result()
    with ts.Transaction() as txn:
        for _node in nodes:
            node = graph_ids_reverse[_node]
            neighbors = graph.get_all_neighbors(node)
            adjacency_list = np.zeros(neighbors.size, dtype=ADJACENCY_DTYPE)
            adjacency_list["node"] = graph_ids[neighbors]
            adjacency_list["aff"] = [e_aff[(node, neighbor)] for neighbor in neighbors]
            adjacency_list["area"] = [
                e_area[(node, neighbor)] for neighbor in neighbors
            ]
            dataset.with_transaction(txn)[str(graph_ids[node])] = cctx.compress(
                adjacency_list.tobytes()
            )


def get_edges(source: str, nodes: np.ndarray) -> Edges:
    spec = {"driver": "ocdbt", "base": source}
    dataset = ts.KvStore.open(spec).result()
    zdc = zstd.ZstdDecompressor()

    read_futures = [dataset.read(str(n)) for n in nodes]
    read_results = [rf.result() for rf in read_futures]
    compressed = [rr.value for rr in read_results]

    try:
        n_threads = int(environ.get("ZSTD_THREADS", 1))
    except ValueError:
        n_threads = 1

    decompressed = []
    try:
        decompressed = zdc.multi_decompress_to_buffer(compressed, threads=n_threads)
    except ValueError:
        for content in compressed:
            decompressed.append(zdc.decompressobj().decompress(content))

    node_ids1 = [np.empty(0, dtype=basetypes.NODE_ID)]
    node_ids2 = [np.empty(0, dtype=basetypes.NODE_ID)]
    affinities = [np.empty(0, dtype=basetypes.EDGE_AFFINITY)]
    areas = [np.empty(0, dtype=basetypes.EDGE_AREA)]
    for n, content in zip(nodes, compressed):
        adjacency_list = np.frombuffer(content, dtype=ADJACENCY_DTYPE)
        node_ids1.append([n] * adjacency_list.size)
        node_ids2.append(adjacency_list["node"])
        affinities.append(adjacency_list["aff"])
        areas.append(adjacency_list["area"])

    return Edges(
        np.concatenate(node_ids1),
        np.concatenate(node_ids2),
        affinities=np.concatenate(affinities),
        areas=np.concatenate(areas),
    )
