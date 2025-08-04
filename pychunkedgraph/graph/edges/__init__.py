"""
Classes and types for edges
"""

from collections import namedtuple
import datetime, logging
from os import environ
from copy import copy
from typing import Iterable, Optional

import numpy as np
import tensorstore as ts
import zstandard as zstd
from graph_tool import Graph

from pychunkedgraph.graph import types
from pychunkedgraph.graph.chunks.utils import (
    get_bounding_children_chunks,
    get_chunk_ids_from_coords,
)
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
    return np.concatenate(stale_nodes)


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
    chunks_map = {}

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

    def _get_l2chunkids_along_boundary(mlayer: int, coord_a, coord_b, padding: int = 0):
        """
        Gets L2 Chunk IDs along opposing faces for larger chunks.
        If padding is enabled, more faces of L2 chunks are padded on both sides.
        This is necessary to find fake edges that can span more than 2 L2 chunks.
        """
        direction = coord_a - coord_b
        major_axis = np.argmax(np.abs(direction))
        bounds_a = get_bounding_children_chunks(cg.meta, mlayer, tuple(coord_a), 2)
        bounds_b = get_bounding_children_chunks(cg.meta, mlayer, tuple(coord_b), 2)

        l2chunk_count = 2 ** (mlayer - 2)
        max_coord = coord_a if direction[major_axis] > 0 else coord_b

        skip = abs(direction[major_axis]) - 1
        l2_skip = skip * l2chunk_count

        mid = max_coord[major_axis] * l2chunk_count
        face_a = mid if direction[major_axis] > 0 else (mid - l2_skip - 1)
        face_b = mid if direction[major_axis] < 0 else (mid - l2_skip - 1)

        l2chunks_a = [bounds_a[bounds_a[:, major_axis] == face_a]]
        l2chunks_b = [bounds_b[bounds_b[:, major_axis] == face_b]]

        step_a, step_b = (1, -1) if direction[major_axis] > 0 else (-1, 1)
        for _ in range(padding):
            _l2_chunks_a = copy(l2chunks_a[-1])
            _l2_chunks_b = copy(l2chunks_b[-1])
            _l2_chunks_a[:, major_axis] += step_a
            _l2_chunks_b[:, major_axis] += step_b
            l2chunks_a.append(_l2_chunks_a)
            l2chunks_b.append(_l2_chunks_b)

        l2chunks_a = np.concatenate(l2chunks_a)
        l2chunks_b = np.concatenate(l2chunks_b)

        l2chunk_ids_a = get_chunk_ids_from_coords(cg.meta, 2, l2chunks_a)
        l2chunk_ids_b = get_chunk_ids_from_coords(cg.meta, 2, l2chunks_b)
        return l2chunk_ids_a, l2chunk_ids_b

    def _get_filtered_l2ids(node_a, node_b, padding: int):
        """
        Finds L2 IDs along opposing faces for given nodes.
        Filterting is done by first finding L2 chunks along these faces.
        Then get their parent chunks iteratively.
        Then filter children iteratively using these chunks.
        """

        def _filter(node):
            result = []
            children = np.array([node], dtype=basetypes.NODE_ID)
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

        mlayer, coord_a, coord_b = _get_normalized_coords(node_a, node_b)
        chunks_a, chunks_b = _get_l2chunkids_along_boundary(
            mlayer, coord_a, coord_b, padding
        )

        chunks_map[node_a] = [[cg.get_chunk_id(node_a)]]
        chunks_map[node_b] = [[cg.get_chunk_id(node_b)]]
        _layer = 2
        while _layer < mlayer:
            chunks_map[node_a].append(chunks_a)
            chunks_map[node_b].append(chunks_b)
            chunks_a = np.unique(cg.get_parent_chunk_id_multiple(chunks_a))
            chunks_b = np.unique(cg.get_parent_chunk_id_multiple(chunks_b))
            _layer += 1
        chunks_map[node_a] = np.concatenate(chunks_map[node_a])
        chunks_map[node_b] = np.concatenate(chunks_map[node_b])
        return int(mlayer), _filter(node_a), _filter(node_b)

    result = [types.empty_2d]
    for edge_layer, _edge in zip(edge_layers, stale_edges):
        node_a, node_b = _edge
        mlayer, l2ids_a, l2ids_b = _get_filtered_l2ids(node_a, node_b, padding=0)
        if l2ids_a.size == 0 or l2ids_b.size == 0:
            logging.info(f"{node_a}, {node_b}, expanding search with padding.")
            mlayer, l2ids_a, l2ids_b = _get_filtered_l2ids(node_a, node_b, padding=2)
            logging.info(f"Found {l2ids_a} and {l2ids_b}.")

        _edges = []
        edges_d = cg.get_cross_chunk_edges(
            node_ids=l2ids_a, time_stamp=nodes_ts_map[node_b], raw_only=True
        )
        for v in edges_d.values():
            _edges.append(v.get(edge_layer, types.empty_2d))

        try:
            _edges = np.concatenate(_edges)
        except ValueError as exc:
            logging.warning(f"No edges found for {node_a}, {node_b}")
            raise ValueError from exc

        mask = np.isin(_edges[:, 1], l2ids_b)
        parents_a = _edges[mask][:, 0]
        children_b = cg.get_children(_edges[mask][:, 1], flatten=True)
        parents_b = np.unique(cg.get_parents(children_b, time_stamp=parent_ts))
        _cx_edges_d = cg.get_cross_chunk_edges(parents_b)
        parents_b = []
        for _node, _edges_d in _cx_edges_d.items():
            for _edges in _edges_d.values():
                _mask = np.isin(_edges[:, 1], parents_a)
                if np.any(_mask):
                    parents_b.append(_node)

        parents_b = np.array(parents_b, dtype=basetypes.NODE_ID)
        parents_b = np.unique(
            cg.get_roots(parents_b, stop_layer=mlayer, ceil=False, time_stamp=parent_ts)
        )

        parents_a = np.array([node_a] * parents_b.size, dtype=basetypes.NODE_ID)
        result.append(np.column_stack((parents_a, parents_b)))
    return np.concatenate(result)


def get_latest_edges_wrapper(
    cg,
    cx_edges_d: dict,
    parent_ts: datetime.datetime = None,
) -> np.ndarray:
    """Helper function to filter stale edges and replace with latest edges."""
    _cx_edges = [types.empty_2d]
    _edge_layers = [types.empty_1d]
    for k, v in cx_edges_d.items():
        _cx_edges.append(v)
        _edge_layers.append([k] * len(v))
    _cx_edges = np.concatenate(_cx_edges)
    _edge_layers = np.concatenate(_edge_layers, dtype=int)

    edge_nodes = np.unique(_cx_edges)
    stale_nodes = get_stale_nodes(cg, edge_nodes, parent_ts=parent_ts)
    stale_nodes_mask = np.isin(edge_nodes, stale_nodes)

    latest_edges = types.empty_2d.copy()
    if np.any(stale_nodes_mask):
        stalte_edges_mask = np.isin(_cx_edges[:, 1], stale_nodes)
        stale_edges = _cx_edges[stalte_edges_mask]
        stale_edge_layers = _edge_layers[stalte_edges_mask]
        latest_edges = get_latest_edges(
            cg,
            stale_edges,
            stale_edge_layers,
            parent_ts=parent_ts,
        )
    return np.concatenate([_cx_edges, latest_edges])
