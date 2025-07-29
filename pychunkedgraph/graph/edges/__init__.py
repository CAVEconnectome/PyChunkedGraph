"""
Classes and types for edges
"""

from collections import namedtuple
import datetime
from os import environ
from copy import copy
from typing import Iterable, Optional

import numpy as np
import tensorstore as ts
import zstandard as zstd
from graph_tool import Graph

from pychunkedgraph.graph import types
from pychunkedgraph.graph.chunks import utils as chunk_utils
from pychunkedgraph.graph.utils import basetypes

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


def get_stale_nodes(
    cg, edge_nodes: Iterable[basetypes.NODE_ID], parent_ts: datetime.datetime = None
):
    """
    Checks to see if partner nodes in edges (edges[:,1]) are stale.
    This is done by getting a supervoxel of the node and check
    if it has a new parent at the same layer as the node.
    """
    edge_supervoxels = cg.get_single_leaf_multiple(edge_nodes)
    # nodes can be at different layers due to skip connections
    edge_nodes_layers = cg.get_chunk_layers(edge_nodes)
    stale_nodes = [types.empty_1d]
    for layer in np.unique(edge_nodes_layers):
        _mask = edge_nodes_layers == layer
        layer_nodes = edge_nodes[_mask]
        _nodes = cg.get_roots(
            edge_supervoxels[_mask],
            stop_layer=layer,
            ceil=False,
            time_stamp=parent_ts,
        )
        stale_mask = layer_nodes != _nodes
        stale_nodes.append(layer_nodes[stale_mask])
    return np.concatenate(stale_nodes), edge_supervoxels


def get_latest_edges(
    cg,
    stale_edges: Iterable,
    edge_layers: Iterable,
    parent_ts: datetime.datetime = None,
) -> dict:
    """
    For each of stale_edges [[`node`, `partner`]], get their L2 edge equivalent.
    Then get supervoxels of those L2 IDs and get parent(s) at `node` level.
    These parents would be the new identities for the stale `partner`.
    """
    _nodes = np.unique(stale_edges[:, 1])
    nodes_ts_map = dict(zip(_nodes, cg.get_node_timestamps(_nodes, return_numpy=False)))
    _nodes = np.unique(stale_edges)
    layers, coords = cg.get_chunk_layers_and_coordinates(_nodes)
    layers_d = dict(zip(_nodes, layers))
    coords_d = dict(zip(_nodes, coords))

    def _get_normalized_coords(node_a, node_b) -> tuple:
        max_layer = layers_d[node_a]
        coord_a, coord_b = coords_d[node_a], coords_d[node_b]
        if layers_d[node_a] != layers_d[node_b]:
            # normalize if nodes are not from the same layer
            max_layer = max(layers_d[node_a], layers_d[node_b])
            chunk_a = cg.get_parent_chunk_id(node_a, parent_layer=max_layer)
            chunk_b = cg.get_parent_chunk_id(node_b, parent_layer=max_layer)
            coord_a, coord_b = cg.get_chunk_coordinates_multiple([chunk_a, chunk_b])
        return max_layer, coord_a, coord_b

    def _get_l2chunkids_along_boundary(max_layer, coord_a, coord_b):
        direction = coord_a - coord_b
        axis = np.flatnonzero(direction)
        assert len(axis) == 1, f"{direction}, {coord_a}, {coord_b}"
        axis = axis[0]
        children_a = chunk_utils.get_bounding_children_chunks(
            cg.meta, max_layer, coord_a, children_layer=2
        )
        children_b = chunk_utils.get_bounding_children_chunks(
            cg.meta, max_layer, coord_b, children_layer=2
        )
        if direction[axis] > 0:
            mid = coord_a[axis] * 2 ** (max_layer - 2)
            l2chunks_a = children_a[children_a[:, axis] == mid]
            l2chunks_b = children_b[children_b[:, axis] == mid - 1]
        else:
            mid = coord_b[axis] * 2 ** (max_layer - 2)
            l2chunks_a = children_a[children_a[:, axis] == mid - 1]
            l2chunks_b = children_b[children_b[:, axis] == mid]

        l2chunk_ids_a = chunk_utils.get_chunk_ids_from_coords(cg.meta, 2, l2chunks_a)
        l2chunk_ids_b = chunk_utils.get_chunk_ids_from_coords(cg.meta, 2, l2chunks_b)
        return l2chunk_ids_a, l2chunk_ids_b

    def _get_filtered_l2ids(node_a, node_b, chunks_map):
        def _filter(node):
            result = []
            children = cg.get_children(node)
            while True:
                chunk_ids = cg.get_chunk_ids_from_node_ids(children)
                mask = np.isin(chunk_ids, chunks_map[node])
                children = children[mask]

                mask = cg.get_chunk_layers(children) == 2
                result.append(children[mask])

                mask = cg.get_chunk_layers(children) > 2
                if children[mask].size == 0:
                    break
                children = cg.get_children(children[mask], flatten=True)
            return np.concatenate(result)

        return _filter(node_a), _filter(node_b)

    result = []
    chunks_map = {}
    for edge_layer, _edge in zip(edge_layers, stale_edges):
        node_a, node_b = _edge
        mlayer, coord_a, coord_b = _get_normalized_coords(node_a, node_b)
        chunks_a, chunks_b = _get_l2chunkids_along_boundary(mlayer, coord_a, coord_b)

        chunks_map[node_a] = []
        chunks_map[node_b] = []
        _layer = 2
        while _layer < mlayer:
            chunks_map[node_a].append(chunks_a)
            chunks_map[node_b].append(chunks_b)
            chunks_a = np.unique(cg.get_parent_chunk_id_multiple(chunks_a))
            chunks_b = np.unique(cg.get_parent_chunk_id_multiple(chunks_b))
            _layer += 1
        chunks_map[node_a] = np.concatenate(chunks_map[node_a])
        chunks_map[node_b] = np.concatenate(chunks_map[node_b])

        l2ids_a, l2ids_b = _get_filtered_l2ids(node_a, node_b, chunks_map)
        edges_d = cg.get_cross_chunk_edges(
            node_ids=l2ids_a, time_stamp=nodes_ts_map[node_b], raw_only=True
        )

        _edges = []
        for v in edges_d.values():
            _edges.append(v.get(edge_layer, types.empty_2d))
        _edges = np.concatenate(_edges)
        mask = np.isin(_edges[:, 1], l2ids_b)

        children_a = cg.get_children(_edges[mask][:, 0], flatten=True)
        children_b = cg.get_children(_edges[mask][:, 1], flatten=True)
        if 85431849467249595 in children_a and 85502218144317440 in children_b:
            print("woohoo0")
            continue

        if 85502218144317440 in children_a and 85431849467249595 in children_b:
            print("woohoo1")
            continue
        parents_a = np.unique(
            cg.get_roots(
                children_a, stop_layer=mlayer, ceil=False, time_stamp=parent_ts
            )
        )
        assert parents_a.size == 1 and parents_a[0] == node_a, (
            node_a,
            parents_a,
            children_a,
        )

        parents_b = np.unique(
            cg.get_roots(
                children_b, stop_layer=mlayer, ceil=False, time_stamp=parent_ts
            )
        )

        parents_a = np.array([node_a] * parents_b.size, dtype=basetypes.NODE_ID)
        result.append(np.column_stack((parents_a, parents_b)))
    return np.concatenate(result)
