"""
Classes and types for edges
"""

from collections import defaultdict, namedtuple
import datetime, logging
from os import environ
from typing import Iterable, Optional

import numpy as np
import tensorstore as ts
import zstandard as zstd
from graph_tool import Graph
from cachetools import LRUCache

from pychunkedgraph.graph import types
from pychunkedgraph.graph.chunks.utils import get_l2chunkids_along_boundary
from pychunkedgraph.graph.utils import basetypes

from ..utils import basetypes
from ..utils.generic import get_parents_at_timestamp


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
PARENTS_CACHE: LRUCache = None
CHILDREN_CACHE: LRUCache = None


class Edges:
    def __init__(
        self,
        node_ids1: np.ndarray,
        node_ids2: np.ndarray,
        *,
        affinities: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
    ):
        self.node_ids1 = np.array(node_ids1, dtype=basetypes.NODE_ID)
        self.node_ids2 = np.array(node_ids2, dtype=basetypes.NODE_ID)
        assert self.node_ids1.size == self.node_ids2.size

        self._as_pairs = None

        if affinities is not None and len(affinities) > 0:
            self._affinities = np.array(affinities, dtype=basetypes.EDGE_AFFINITY)
            assert self.node_ids1.size == self._affinities.size
        else:
            self._affinities = np.full(len(self.node_ids1), DEFAULT_AFFINITY)

        if areas is not None and len(areas) > 0:
            self._areas = np.array(areas, dtype=basetypes.EDGE_AREA)
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
    cg, nodes: Iterable[basetypes.NODE_ID], parent_ts: datetime.datetime = None
):
    """
    Checks to see if given nodes are stale.
    This is done by getting a supervoxel of a node and checking
    if it has a new parent at the same layer as the node.
    """
    nodes = np.array(nodes, dtype=basetypes.NODE_ID)
    supervoxels = cg.get_single_leaf_multiple(nodes)
    # nodes can be at different layers due to skip connections
    node_layers = cg.get_chunk_layers(nodes)
    stale_nodes = [types.empty_1d]
    for layer in np.unique(node_layers):
        _mask = node_layers == layer
        layer_nodes = nodes[_mask]
        _nodes = cg.get_roots(
            supervoxels[_mask],
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
    _nodes = np.unique(stale_edges)
    nodes_ts_map = dict(
        zip(_nodes, cg.get_node_timestamps(_nodes, return_numpy=False, normalize=True))
    )
    layers, coords = cg.get_chunk_layers_and_coordinates(_nodes)
    layers_d = dict(zip(_nodes, layers))
    coords_d = dict(zip(_nodes, coords))

    def _get_children_from_cache(nodes):
        children = []
        non_cached = []
        for node in nodes:
            try:
                v = CHILDREN_CACHE[node]
                children.append(v)
            except KeyError:
                non_cached.append(node)

        children_map = cg.get_children(non_cached)
        for k, v in children_map.items():
            CHILDREN_CACHE[k] = v
            children.append(v)
        return np.concatenate(children)

    def _get_normalized_coords(node_a, node_b) -> tuple:
        max_layer = layers_d[node_a]
        coord_a, coord_b = coords_d[node_a], coords_d[node_b]
        if layers_d[node_a] != layers_d[node_b]:
            # normalize if nodes are not from the same layer
            max_layer = max(layers_d[node_a], layers_d[node_b])
            chunk_a = cg.get_parent_chunk_id(node_a, parent_layer=max_layer)
            chunk_b = cg.get_parent_chunk_id(node_b, parent_layer=max_layer)
            coord_a, coord_b = cg.get_chunk_coordinates_multiple([chunk_a, chunk_b])
        return max_layer, tuple(coord_a), tuple(coord_b)

    def _get_filtered_l2ids(node_a, node_b, padding: int):
        """
        Finds L2 IDs along opposing faces for given nodes.
        Filterting is done by first finding L2 chunks along these faces.
        Then get their parent chunks iteratively.
        Then filter children iteratively using these chunks.
        """
        chunks_map = {}

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
                if PARENTS_CACHE is None:
                    children = cg.get_children(children[mask], flatten=True)
                else:
                    children = _get_children_from_cache(children[mask])
            return np.concatenate(result)

        mlayer, coord_a, coord_b = _get_normalized_coords(node_a, node_b)
        chunks_a, chunks_b = get_l2chunkids_along_boundary(
            cg.meta, mlayer, coord_a, coord_b, padding
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

    def _populate_parents_cache(children: np.ndarray):
        global PARENTS_CACHE

        not_cached = []
        for child in children:
            try:
                # reset lru index, these will be needed soon
                _ = PARENTS_CACHE[child]
            except KeyError:
                not_cached.append(child)

        all_parents = cg.get_parents(not_cached, current=False)
        for child, parents in zip(not_cached, all_parents):
            PARENTS_CACHE[child] = {}
            for parent, ts in parents:
                PARENTS_CACHE[child][ts] = parent

    def _check_cross_edges_from_a(node_b, nodes_a, layer, parent_ts):
        """
        Checks to match cross edges from partners_a
        to hierarchy of potential node from partner b.
        """
        _node_hierarchy = cg.get_root(
            node_b,
            time_stamp=parent_ts,
            stop_layer=layer,
            get_all_parents=True,
            ceil=False,
        )
        _node_hierarchy = np.append(_node_hierarchy, node_b)
        _cx_edges_d_from_a = cg.get_cross_chunk_edges(nodes_a, time_stamp=parent_ts)
        for _edges_d_from_a in _cx_edges_d_from_a.values():
            _edges_from_a = _edges_d_from_a.get(layer, types.empty_2d)
            _mask = np.isin(_edges_from_a[:, 1], _node_hierarchy)
            if np.any(_mask):
                return True
        return False

    def _check_hierarchy_a_from_b(nodes_a, hierarchy_a, layer, parent_ts):
        """
        Checks for overlap between hierarchy of a,
        and hierarchy of a identified from partners of b.
        """
        _hierarchy_a_from_b = [nodes_a]
        for _a in nodes_a:
            _hierarchy_a_from_b.append(
                cg.get_root(
                    _a,
                    time_stamp=parent_ts,
                    stop_layer=layer,
                    get_all_parents=True,
                    ceil=False,
                )
            )
            _children = cg.get_children(_a)
            _children_layers = cg.get_chunk_layers(_children)
            _hierarchy_a_from_b.append(_children[_children_layers == 2])
            _children = _children[_children_layers > 2]
            while _children.size:
                _hierarchy_a_from_b.append(_children)
                _children = cg.get_children(_children, flatten=True)
                _children_layers = cg.get_chunk_layers(_children)
                _hierarchy_a_from_b.append(_children[_children_layers == 2])
                _children = _children[_children_layers > 2]

        _hierarchy_a_from_b = np.concatenate(_hierarchy_a_from_b)
        return np.any(np.isin(_hierarchy_a_from_b, hierarchy_a))

    def _get_parents_b(edges, parent_ts, layer, fallback: bool = False):
        """
        Attempts to find new partner side nodes.
        Gets new partners at parent_ts using supervoxels, at `parent_ts`.
        Searches for new partners that may have any edges to `edges[:,0]`.
        """
        if PARENTS_CACHE is None:
            children_b = cg.get_children(edges[:, 1], flatten=True)
            parents_b = np.unique(cg.get_parents(children_b, time_stamp=parent_ts))
        else:
            children_b = _get_children_from_cache(edges[:, 1])
            _populate_parents_cache(children_b)
            _parents_b, missing = get_parents_at_timestamp(
                children_b, PARENTS_CACHE, time_stamp=parent_ts, unique=True
            )
            # handle cache miss cases
            _parents_b_missed = np.unique(cg.get_parents(missing, time_stamp=parent_ts))
            parents_b = np.concatenate([_parents_b, _parents_b_missed])

        parents_a = np.unique(edges[:, 0])
        stale_a = get_stale_nodes(cg, parents_a, parent_ts=parent_ts)
        if stale_a.size == parents_a.size or fallback:
            # this is applicable only for v2 to v3 migration
            # handle cases when source nodes in `edges[:,0]` are stale
            atomic_edges_d = cg.get_atomic_cross_edges(stale_a)
            partners = [types.empty_1d]
            for _edges_d in atomic_edges_d.values():
                _edges = _edges_d.get(layer, types.empty_2d)
                partners.append(_edges[:, 1])
            partners = np.concatenate(partners)
            return np.unique(cg.get_parents(partners, time_stamp=parent_ts))

        _cx_edges_d = cg.get_cross_chunk_edges(parents_b, time_stamp=parent_ts)
        _hierarchy_a = [parents_a]
        for _a in parents_a:
            _hierarchy_a.append(
                cg.get_root(
                    _a,
                    time_stamp=parent_ts,
                    stop_layer=layer,
                    get_all_parents=True,
                    ceil=False,
                )
            )
        _hierarchy_a = np.concatenate(_hierarchy_a)

        _parents_b = []
        for _node, _edges_d in _cx_edges_d.items():
            _edges = _edges_d.get(layer, types.empty_2d)
            if _check_cross_edges_from_a(_node, _edges[:, 1], layer, parent_ts):
                _parents_b.append(_node)
            elif _check_hierarchy_a_from_b(
                _edges[:, 1], _hierarchy_a, layer, parent_ts
            ):
                _parents_b.append(_node)
        return np.array(_parents_b, dtype=basetypes.NODE_ID)

    def _get_parents_b_with_chunk_mask(
        l2ids_b: np.ndarray, parents_b: np.ndarray, max_ts: datetime.datetime, edge
    ):
        chunks_old = cg.get_chunk_ids_from_node_ids(l2ids_b)
        chunks_new = cg.get_chunk_ids_from_node_ids(parents_b)
        chunk_mask = np.isin(chunks_new, chunks_old)
        parents_b = parents_b[chunk_mask]
        _stale_nodes = get_stale_nodes(cg, parents_b, parent_ts=max_ts)
        assert _stale_nodes.size == 0, f"{edge}, {_stale_nodes}, {parent_ts}"
        return parents_b

    def _get_cx_edges(l2ids_a, max_node_ts, raw_only: bool = True):
        _edges_d = cg.get_cross_chunk_edges(
            node_ids=l2ids_a, time_stamp=max_node_ts, raw_only=raw_only
        )
        _edges = []
        for v in _edges_d.values():
            if edge_layer in v:
                _edges.append(v[edge_layer])
        return np.concatenate(_edges)

    def _get_dilated_edges(edges):
        layers_b = cg.get_chunk_layers(edges[:, 1])
        _mask = layers_b == 2
        _l2_edges = [edges[_mask]]
        for _edge in edges[~_mask]:
            _node_a, _node_b = _edge
            _nodes_b = cg.get_l2children([_node_b])
            _l2_edges.append(
                np.array([[_node_a, _b] for _b in _nodes_b], dtype=basetypes.NODE_ID)
            )
        return np.unique(np.concatenate(_l2_edges), axis=0)

    def _get_new_edge(edge, edge_layer, parent_ts, padding, fallback: bool = False):
        """
        Attempts to find new edge(s) for the stale `edge`.
            * Find L2 IDs on opposite sides of the face in L2 chunks along the face.
            * Find new edges between them (before the given timestamp).
            * If none found, expand search by adding another layer of L2 chunks.
        """
        node_a, node_b = edge
        mlayer, l2ids_a, l2ids_b = _get_filtered_l2ids(node_a, node_b, padding=padding)
        if l2ids_a.size == 0 or l2ids_b.size == 0:
            return types.empty_2d.copy()

        max_node_ts = max(nodes_ts_map[node_a], nodes_ts_map[node_b])
        try:
            _edges = _get_cx_edges(l2ids_a, max_node_ts)
        except ValueError:
            _edges = _get_cx_edges(l2ids_a, max_node_ts, raw_only=False)
        except ValueError:
            return types.empty_2d.copy()

        mask = np.isin(_edges[:, 1], l2ids_b)
        if np.any(mask):
            parents_b = _get_parents_b(_edges[mask], parent_ts, edge_layer)
        else:
            # partner edges likely lifted, dilate and retry
            _edges = _get_dilated_edges(_edges)
            mask = np.isin(_edges[:, 1], l2ids_b)
            if np.any(mask):
                parents_b = _get_parents_b(_edges[mask], parent_ts, edge_layer)
            else:
                # if none of `l2ids_b` were found in edges, `l2ids_a` already have new edges
                # so get the new identities of `l2ids_b` by using chunk mask
                try:
                    parents_b = _get_parents_b_with_chunk_mask(
                        l2ids_b, _edges[:, 1], max_node_ts, edge
                    )
                except AssertionError:
                    parents_b = []
                    if fallback:
                        parents_b = _get_parents_b(_edges, parent_ts, edge_layer, True)

        parents_b = np.unique(
            cg.get_roots(parents_b, stop_layer=mlayer, ceil=False, time_stamp=parent_ts)
        )

        parents_a = np.array([node_a] * parents_b.size, dtype=basetypes.NODE_ID)
        return np.column_stack((parents_a, parents_b))

    result = [types.empty_2d]
    for edge_layer, _edge in zip(edge_layers, stale_edges):
        max_chebyshev_distance = int(environ.get("MAX_CHEBYSHEV_DISTANCE", 3))
        for pad in range(0, max_chebyshev_distance + 1):
            fallback = pad == max_chebyshev_distance
            _new_edges = _get_new_edge(
                _edge, edge_layer, parent_ts, padding=pad, fallback=fallback
            )
            if _new_edges.size:
                break
            logging.info(f"{_edge}, expanding search with padding {pad+1}.")
        assert _new_edges.size, f"No new edge found {_edge}; {edge_layer}, {parent_ts}"
        result.append(_new_edges)
    return np.concatenate(result)


class _BatchEdgeContext:
    """Context object holding pre-computed data for batched edge processing."""

    def __init__(self, cg, stale_edges: np.ndarray, edge_layers: np.ndarray):
        self.cg = cg
        self.stale_edges = stale_edges
        self.edge_layers = edge_layers

        # Pre-compute node data
        _nodes = np.unique(stale_edges)
        self.nodes_ts_map = dict(
            zip(_nodes, cg.get_node_timestamps(_nodes, return_numpy=False, normalize=True))
        )
        layers, coords = cg.get_chunk_layers_and_coordinates(_nodes)
        self.layers_d = dict(zip(_nodes, layers))
        self.coords_d = dict(zip(_nodes, coords))

    def get_normalized_coords(self, node_a, node_b) -> tuple:
        """Get normalized coordinates for two nodes at their common layer."""
        max_layer = self.layers_d[node_a]
        coord_a, coord_b = self.coords_d[node_a], self.coords_d[node_b]
        if self.layers_d[node_a] != self.layers_d[node_b]:
            max_layer = max(self.layers_d[node_a], self.layers_d[node_b])
            chunk_a = self.cg.get_parent_chunk_id(node_a, parent_layer=max_layer)
            chunk_b = self.cg.get_parent_chunk_id(node_b, parent_layer=max_layer)
            coord_a, coord_b = self.cg.get_chunk_coordinates_multiple([chunk_a, chunk_b])
        return max_layer, tuple(coord_a), tuple(coord_b)


def _group_edges_by_boundary(ctx: _BatchEdgeContext) -> tuple[dict, dict]:
    """Group edges by their chunk boundary for shared L2 chunk computation."""
    boundary_to_edges = defaultdict(list)
    edge_to_boundary = {}
    for idx, (edge, _) in enumerate(zip(ctx.stale_edges, ctx.edge_layers)):
        node_a, node_b = edge
        mlayer, coord_a, coord_b = ctx.get_normalized_coords(node_a, node_b)
        boundary_key = (mlayer, coord_a, coord_b)
        boundary_to_edges[boundary_key].append(idx)
        edge_to_boundary[idx] = boundary_key
    return boundary_to_edges, edge_to_boundary


def _compute_boundary_l2chunks(
    ctx: _BatchEdgeContext,
    remaining_indices: set,
    edge_to_boundary: dict,
    padding: int,
) -> dict:
    """Compute L2 boundary chunks for all unique boundaries at given padding."""
    boundary_l2chunks = {}
    remaining_boundaries = set(edge_to_boundary[idx] for idx in remaining_indices)
    for boundary_key in remaining_boundaries:
        mlayer, coord_a, coord_b = boundary_key
        chunks_a, chunks_b = get_l2chunkids_along_boundary(
            ctx.cg.meta, mlayer, coord_a, coord_b, padding
        )
        boundary_l2chunks[boundary_key] = (chunks_a, chunks_b)
    return boundary_l2chunks


def _build_chunks_map_for_node(
    ctx: _BatchEdgeContext,
    node,
    boundary_key: tuple,
    boundary_l2chunks: dict,
) -> np.ndarray:
    """Build chunks map for a node to filter its children to boundary L2 chunks."""
    mlayer, coord_a, _ = boundary_key
    chunks_a, chunks_b = boundary_l2chunks[boundary_key]

    # Determine if node is on side a or b based on coordinates
    node_coord = tuple(ctx.coords_d[node])
    if ctx.layers_d[node] != mlayer:
        node_chunk = ctx.cg.get_parent_chunk_id(node, parent_layer=mlayer)
        node_coord = tuple(ctx.cg.get_chunk_coordinates(node_chunk))

    chunks = chunks_a if node_coord == coord_a else chunks_b

    chunks_list = [[ctx.cg.get_chunk_id(node)]]
    _chunks = chunks
    _layer = 2
    while _layer < mlayer:
        chunks_list.append(_chunks)
        _chunks = np.unique(ctx.cg.get_parent_chunk_id_multiple(_chunks))
        _layer += 1
    return np.concatenate(chunks_list)


def _filter_node_to_l2(
    ctx: _BatchEdgeContext,
    node,
    chunks_map_node: np.ndarray,
    all_children_d: dict,
) -> np.ndarray:
    """Filter node down to L2 children within chunk boundary."""
    result = []
    children = np.array([node], dtype=basetypes.NODE_ID)

    while True:
        chunk_ids = ctx.cg.get_chunk_ids_from_node_ids(children)
        mask = np.isin(chunk_ids, chunks_map_node)
        children = children[mask]

        child_layers = ctx.cg.get_chunk_layers(children)
        mask_l2 = child_layers == 2
        result.append(children[mask_l2])

        mask_higher = child_layers > 2
        if children[mask_higher].size == 0:
            break

        # Batch get children - use pre-fetched or fetch new
        to_expand = children[mask_higher]
        new_children = []
        for c in to_expand:
            if c in all_children_d:
                new_children.append(all_children_d[c])
            else:
                fetched = ctx.cg.get_children(c)
                new_children.append(fetched.get(c, types.empty_1d))
        children = np.concatenate(new_children) if new_children else types.empty_1d

    return np.concatenate(result) if result else types.empty_1d


def _compute_edge_l2_data(
    ctx: _BatchEdgeContext,
    remaining_indices: set,
    edge_to_boundary: dict,
    boundary_l2chunks: dict,
    all_children_d: dict,
) -> dict:
    """Compute L2 IDs for all remaining edges."""
    edge_l2_data = {}
    for idx in remaining_indices:
        edge = ctx.stale_edges[idx]
        node_a, node_b = edge
        boundary_key = edge_to_boundary[idx]
        mlayer = boundary_key[0]

        chunks_map_a = _build_chunks_map_for_node(ctx, node_a, boundary_key, boundary_l2chunks)
        chunks_map_b = _build_chunks_map_for_node(ctx, node_b, boundary_key, boundary_l2chunks)

        l2ids_a = _filter_node_to_l2(ctx, node_a, chunks_map_a, all_children_d)
        l2ids_b = _filter_node_to_l2(ctx, node_b, chunks_map_b, all_children_d)

        edge_l2_data[idx] = (mlayer, l2ids_a, l2ids_b)
    return edge_l2_data


def _batch_fetch_cross_edges(
    ctx: _BatchEdgeContext,
    remaining_indices: set,
    edge_l2_data: dict,
    parent_ts: datetime.datetime,
) -> tuple[dict, dict]:
    """Batch fetch cross-chunk edges for all L2 source nodes."""
    all_l2ids_a = []
    max_ts_map = {}

    for idx in remaining_indices:
        _, l2ids_a, _ = edge_l2_data[idx]
        if l2ids_a.size > 0:
            all_l2ids_a.append(l2ids_a)
            edge = ctx.stale_edges[idx]
            max_ts_map[idx] = max(ctx.nodes_ts_map[edge[0]], ctx.nodes_ts_map[edge[1]])

    if all_l2ids_a:
        all_l2ids_a_unique = np.unique(np.concatenate(all_l2ids_a))
        min_max_ts = min(max_ts_map.values()) if max_ts_map else parent_ts
        all_cx_edges_d = ctx.cg.get_cross_chunk_edges(
            all_l2ids_a_unique, time_stamp=min_max_ts, raw_only=True
        )
    else:
        all_cx_edges_d = {}

    return all_cx_edges_d, max_ts_map


def _match_and_dilate_edges(
    ctx: _BatchEdgeContext,
    l2ids_a: np.ndarray,
    l2ids_b: np.ndarray,
    edge_layer: int,
    all_cx_edges_d: dict,
) -> tuple[np.ndarray, bool]:
    """Match edges to partners and dilate if needed. Returns (matched_edges, success)."""
    # Get cross edges for this edge's L2 source nodes
    _edges = []
    for l2id in l2ids_a:
        if l2id in all_cx_edges_d:
            edges_d = all_cx_edges_d[l2id]
            if edge_layer in edges_d:
                _edges.append(edges_d[edge_layer])

    if not _edges:
        return types.empty_2d, False

    _edges = np.concatenate(_edges)
    mask = np.isin(_edges[:, 1], l2ids_b)

    if np.any(mask):
        return _edges[mask], True

    # Try dilating edges (partner edges may be lifted)
    layers_b = ctx.cg.get_chunk_layers(_edges[:, 1])
    _mask_l2 = layers_b == 2
    dilated = [_edges[_mask_l2]]

    for _e in _edges[~_mask_l2]:
        _nodes_b = ctx.cg.get_l2children([_e[1]])
        dilated.append(
            np.array([[_e[0], _b] for _b in _nodes_b], dtype=basetypes.NODE_ID)
        )

    if not dilated:
        return types.empty_2d, False

    _edges = np.unique(np.concatenate(dilated), axis=0)
    mask = np.isin(_edges[:, 1], l2ids_b)

    if np.any(mask):
        return _edges[mask], True

    # Fallback: use chunk mask
    chunks_old = ctx.cg.get_chunk_ids_from_node_ids(l2ids_b)
    chunks_new = ctx.cg.get_chunk_ids_from_node_ids(_edges[:, 1])
    chunk_mask = np.isin(chunks_new, chunks_old)

    if np.any(chunk_mask):
        return _edges[chunk_mask], True

    return types.empty_2d, False


def _verify_and_get_parents_b(
    ctx: _BatchEdgeContext,
    matched_edges: np.ndarray,
    edge_layer: int,
    parent_ts: datetime.datetime,
    fallback: bool,
) -> np.ndarray:
    """Get and verify parents for matched partner nodes."""
    # Get parents for matched partner nodes
    children_b = ctx.cg.get_children(matched_edges[:, 1], flatten=True)
    parents_b = np.unique(ctx.cg.get_parents(children_b, time_stamp=parent_ts))

    # Verify parents_b have edges back to node_a hierarchy
    parents_a = np.unique(matched_edges[:, 0])
    stale_a = get_stale_nodes(ctx.cg, parents_a, parent_ts=parent_ts)

    if stale_a.size == parents_a.size or fallback:
        # Fallback for v2->v3 migration or max padding
        atomic_edges_d = ctx.cg.get_atomic_cross_edges(stale_a if stale_a.size else parents_a)
        partners = [types.empty_1d]
        for _edges_d in atomic_edges_d.values():
            _e = _edges_d.get(edge_layer, types.empty_2d)
            partners.append(_e[:, 1])
        partners = np.concatenate(partners)
        if partners.size:
            parents_b = np.unique(ctx.cg.get_parents(partners, time_stamp=parent_ts))
        return parents_b

    # Verify via cross-edge check
    _cx_edges_d = ctx.cg.get_cross_chunk_edges(parents_b, time_stamp=parent_ts)
    _hierarchy_a = [parents_a]
    for _a in parents_a:
        _hierarchy_a.append(
            ctx.cg.get_root(
                _a, time_stamp=parent_ts, stop_layer=edge_layer,
                get_all_parents=True, ceil=False,
            )
        )
    _hierarchy_a = np.concatenate(_hierarchy_a)

    verified_parents_b = []
    for _node, _edges_d in _cx_edges_d.items():
        _e = _edges_d.get(edge_layer, types.empty_2d)
        if _e.size == 0:
            continue

        # Check if any edges from _node point back to hierarchy_a
        _node_hierarchy = ctx.cg.get_root(
            _node, time_stamp=parent_ts, stop_layer=edge_layer,
            get_all_parents=True, ceil=False,
        )
        _node_hierarchy = np.append(_node_hierarchy, _node)
        _cx_from_partners = ctx.cg.get_cross_chunk_edges(_e[:, 1], time_stamp=parent_ts)

        found = False
        for _partner_edges_d in _cx_from_partners.values():
            _partner_edges = _partner_edges_d.get(edge_layer, types.empty_2d)
            if np.any(np.isin(_partner_edges[:, 1], _node_hierarchy)):
                found = True
                break

        if found:
            verified_parents_b.append(_node)
        else:
            # Check hierarchy overlap
            _hierarchy_a_from_b = [_e[:, 1]]
            for _a in _e[:, 1]:
                _hierarchy_a_from_b.append(
                    ctx.cg.get_root(
                        _a, time_stamp=parent_ts, stop_layer=edge_layer,
                        get_all_parents=True, ceil=False,
                    )
                )
            _hierarchy_a_from_b = np.concatenate(_hierarchy_a_from_b)
            if np.any(np.isin(_hierarchy_a_from_b, _hierarchy_a)):
                verified_parents_b.append(_node)

    if verified_parents_b:
        return np.array(verified_parents_b, dtype=basetypes.NODE_ID)
    return parents_b


def _process_single_edge_batched(
    ctx: _BatchEdgeContext,
    idx: int,
    edge_l2_data: dict,
    all_cx_edges_d: dict,
    parent_ts: datetime.datetime,
    fallback: bool,
) -> np.ndarray:
    """Process a single edge using pre-fetched data. Returns new edges or None."""
    edge = ctx.stale_edges[idx]
    edge_layer = ctx.edge_layers[idx]
    node_a = edge[0]
    mlayer, l2ids_a, l2ids_b = edge_l2_data[idx]

    if l2ids_a.size == 0 or l2ids_b.size == 0:
        return None

    matched_edges, success = _match_and_dilate_edges(
        ctx, l2ids_a, l2ids_b, edge_layer, all_cx_edges_d
    )
    if not success:
        return None

    parents_b = _verify_and_get_parents_b(
        ctx, matched_edges, edge_layer, parent_ts, fallback
    )

    # Get final parents at target layer
    parents_b = np.unique(
        ctx.cg.get_roots(parents_b, stop_layer=mlayer, ceil=False, time_stamp=parent_ts)
    )

    if parents_b.size > 0:
        parents_a_final = np.array([node_a] * parents_b.size, dtype=basetypes.NODE_ID)
        return np.column_stack((parents_a_final, parents_b))
    return None


def get_latest_edges_batched(
    cg,
    stale_edges: np.ndarray,
    edge_layers: np.ndarray,
    parent_ts: datetime.datetime = None,
) -> np.ndarray:
    """
    Batched version of get_latest_edges for processing many stale edges efficiently.

    Instead of processing edges one-by-one, this function:
    1. Groups edges by boundary to share L2 chunk computations
    2. Processes all edges at each padding level before expanding search
    3. Batches database operations (children, parents, cross-edges) across all edges

    Args:
        cg: ChunkedGraph instance
        stale_edges: Array of shape (n, 2) with stale edges [[node_a, node_b], ...]
        edge_layers: Array of layer numbers for each edge
        parent_ts: Timestamp for parent lookups

    Returns:
        Array of new edges replacing the stale ones
    """
    if stale_edges.size == 0:
        return types.empty_2d.copy()

    max_chebyshev_distance = int(environ.get("MAX_CHEBYSHEV_DISTANCE", 3))

    # Initialize context with pre-computed node data
    ctx = _BatchEdgeContext(cg, stale_edges, edge_layers)

    # Group edges by boundary for shared L2 chunk computation
    _, edge_to_boundary = _group_edges_by_boundary(ctx)

    # Progressive padding with batch processing
    remaining_indices = set(range(len(stale_edges)))
    results = [None] * len(stale_edges)

    for pad in range(0, max_chebyshev_distance + 1):
        if not remaining_indices:
            break

        fallback = pad == max_chebyshev_distance

        # Compute L2 boundary chunks for all unique boundaries at this padding
        boundary_l2chunks = _compute_boundary_l2chunks(
            ctx, remaining_indices, edge_to_boundary, pad
        )

        # Batch: Pre-fetch children for all nodes
        all_nodes_for_children = set()
        for idx in remaining_indices:
            all_nodes_for_children.add(stale_edges[idx, 0])
            all_nodes_for_children.add(stale_edges[idx, 1])
        all_children_d = cg.get_children(list(all_nodes_for_children))

        # Compute L2 IDs for all remaining edges
        edge_l2_data = _compute_edge_l2_data(
            ctx, remaining_indices, edge_to_boundary, boundary_l2chunks, all_children_d
        )

        # Batch: Get cross-chunk edges for all L2 source nodes
        all_cx_edges_d, _ = _batch_fetch_cross_edges(
            ctx, remaining_indices, edge_l2_data, parent_ts
        )

        # Process each remaining edge using pre-fetched data
        still_remaining = set()
        for idx in remaining_indices:
            result = _process_single_edge_batched(
                ctx, idx, edge_l2_data, all_cx_edges_d, parent_ts, fallback
            )
            if result is not None:
                results[idx] = result
            else:
                still_remaining.add(idx)

        remaining_indices = still_remaining

        if remaining_indices:
            logging.info(
                f"Batch: {len(remaining_indices)} edges expanding search with padding {pad+1}."
            )

    # Handle any edges that couldn't be resolved
    for idx in remaining_indices:
        edge = stale_edges[idx]
        edge_layer = edge_layers[idx]
        logging.warning(f"No new edge found for {edge}; layer={edge_layer}, ts={parent_ts}")
        results[idx] = types.empty_2d.copy()

    # Concatenate all results
    valid_results = [r for r in results if r is not None and r.size > 0]
    if valid_results:
        return np.concatenate(valid_results)
    return types.empty_2d.copy()


def get_latest_edges_wrapper(
    cg, cx_edges_d: dict, parent_ts: datetime.datetime = None
) -> tuple[dict, np.ndarray]:
    """
    Helper function to filter stale edges and replace with latest edges.
    Filters out edges with nodes stale in source, edges[:,0], at given timestamp.

    Uses batched processing when there are many stale edges (>=10) for better
    performance with reduced database round-trips.
    """
    nodes = [types.empty_1d]
    new_cx_edges_d = {0: types.empty_2d}

    all_edges = np.concatenate(list(cx_edges_d.values()))
    all_edge_nodes = np.unique(all_edges)
    all_stale_nodes = get_stale_nodes(cg, all_edge_nodes, parent_ts=parent_ts)
    if all_stale_nodes.size == 0:
        return cx_edges_d, all_edge_nodes

    # Collect all stale destination edges across all layers
    all_stale_edges = []
    all_stale_edge_layers = []
    layer_non_stale_edges = {}  # layer -> [stale_source_edges, healthy_edges]

    for layer, _cx_edges in cx_edges_d.items():
        if _cx_edges.size == 0:
            continue

        stale_source_mask = np.isin(_cx_edges[:, 0], all_stale_nodes)
        stale_source_edges = _cx_edges[stale_source_mask]

        remaining_edges = _cx_edges[~stale_source_mask]
        stale_destination_mask = np.isin(remaining_edges[:, 1], all_stale_nodes)
        healthy_edges = remaining_edges[~stale_destination_mask]

        layer_non_stale_edges[layer] = (stale_source_edges, healthy_edges)

        if np.any(stale_destination_mask):
            stale_dest_edges = remaining_edges[stale_destination_mask]
            all_stale_edges.append(stale_dest_edges)
            all_stale_edge_layers.extend([layer] * len(stale_dest_edges))

    # Process stale edges
    total_stale_count = sum(len(e) for e in all_stale_edges) if all_stale_edges else 0
    BATCH_THRESHOLD = int(environ.get("STALE_EDGE_BATCH_THRESHOLD", 10))

    if total_stale_count > 0:
        combined_stale_edges = np.concatenate(all_stale_edges)
        combined_stale_layers = np.array(all_stale_edge_layers, dtype=int)

        if total_stale_count >= BATCH_THRESHOLD:
            # Use batched processing for many stale edges
            logging.debug(f"Using batched processing for {total_stale_count} stale edges")
            latest_edges = get_latest_edges_batched(
                cg,
                combined_stale_edges,
                combined_stale_layers,
                parent_ts=parent_ts,
            )
        else:
            # Use original sequential processing for few stale edges
            latest_edges = get_latest_edges(
                cg,
                combined_stale_edges,
                combined_stale_layers,
                parent_ts=parent_ts,
            )

        logging.debug(f"{combined_stale_edges} -> {latest_edges}; {parent_ts}")

        # Distribute latest edges back to their respective layers
        # Build a mapping from source node to its layer
        source_to_layer = {}
        for edge, layer in zip(combined_stale_edges, combined_stale_layers):
            source_to_layer[edge[0]] = layer

        latest_edges_by_layer = defaultdict(list)
        if latest_edges.size > 0:
            for edge in latest_edges:
                source_node = edge[0]
                if source_node in source_to_layer:
                    layer = source_to_layer[source_node]
                    latest_edges_by_layer[layer].append(edge)
    else:
        latest_edges_by_layer = defaultdict(list)

    # Build final result dictionary
    for layer in cx_edges_d.keys():
        if layer not in layer_non_stale_edges:
            continue

        stale_source_edges, healthy_edges = layer_non_stale_edges[layer]
        _new_cx_edges = [types.empty_2d, stale_source_edges, healthy_edges]

        if latest_edges_by_layer[layer]:
            _new_cx_edges.append(np.array(latest_edges_by_layer[layer], dtype=basetypes.NODE_ID))

        new_cx_edges_d[layer] = np.concatenate(_new_cx_edges)
        nodes.append(np.unique(new_cx_edges_d[layer]))

    return new_cx_edges_d, np.concatenate(nodes)
