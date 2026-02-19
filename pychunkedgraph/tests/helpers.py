from functools import reduce

import numpy as np

from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.utils import basetypes
from ..ingest.create.atomic_layer import add_atomic_chunk


class CloudVolumeBounds(object):
    def __init__(self, bounds=[[0, 0, 0], [0, 0, 0]]):
        self._bounds = np.array(bounds)

    @property
    def bounds(self):
        return self._bounds

    def __repr__(self):
        return self.bounds

    def to_list(self):
        return list(np.array(self.bounds).flatten())


class CloudVolumeMock(object):
    def __init__(self):
        self.resolution = np.array([1, 1, 1], dtype=int)
        self.bounds = CloudVolumeBounds()


def create_chunk(cg, vertices=None, edges=None, timestamp=None):
    """
    Helper function to add vertices and edges to the chunkedgraph - no safety checks!
    """
    edges = edges if edges else []
    vertices = vertices if vertices else []
    vertices = np.unique(np.array(vertices, dtype=np.uint64))
    edges = [(np.uint64(v1), np.uint64(v2), np.float32(aff)) for v1, v2, aff in edges]
    isolated_ids = [
        x
        for x in vertices
        if (x not in [edges[i][0] for i in range(len(edges))])
        and (x not in [edges[i][1] for i in range(len(edges))])
    ]

    chunk_edges_active = {}
    for edge_type in EDGE_TYPES:
        chunk_edges_active[edge_type] = Edges([], [])

    for e in edges:
        if cg.get_chunk_id(e[0]) == cg.get_chunk_id(e[1]):
            sv1s = np.array([e[0]], dtype=basetypes.NODE_ID)
            sv2s = np.array([e[1]], dtype=basetypes.NODE_ID)
            affs = np.array([e[2]], dtype=basetypes.EDGE_AFFINITY)
            chunk_edges_active[EDGE_TYPES.in_chunk] += Edges(
                sv1s, sv2s, affinities=affs
            )

    chunk_id = None
    if len(chunk_edges_active[EDGE_TYPES.in_chunk]):
        chunk_id = cg.get_chunk_id(chunk_edges_active[EDGE_TYPES.in_chunk].node_ids1[0])
    elif len(vertices):
        chunk_id = cg.get_chunk_id(vertices[0])

    for e in edges:
        if not cg.get_chunk_id(e[0]) == cg.get_chunk_id(e[1]):
            # Ensure proper order
            if chunk_id is not None:
                if not chunk_id == cg.get_chunk_id(e[0]):
                    e = [e[1], e[0], e[2]]
            sv1s = np.array([e[0]], dtype=basetypes.NODE_ID)
            sv2s = np.array([e[1]], dtype=basetypes.NODE_ID)
            affs = np.array([e[2]], dtype=basetypes.EDGE_AFFINITY)
            if np.isinf(e[2]):
                chunk_edges_active[EDGE_TYPES.cross_chunk] += Edges(
                    sv1s, sv2s, affinities=affs
                )
            else:
                chunk_edges_active[EDGE_TYPES.between_chunk] += Edges(
                    sv1s, sv2s, affinities=affs
                )

    all_edges = reduce(lambda x, y: x + y, chunk_edges_active.values())
    cg.mock_edges += all_edges

    isolated_ids = np.array(isolated_ids, dtype=np.uint64)
    add_atomic_chunk(
        cg,
        cg.get_chunk_coordinates(chunk_id),
        chunk_edges_active,
        isolated=isolated_ids,
        time_stamp=timestamp,
    )


def to_label(cg, l, x, y, z, segment_id):
    return cg.get_node_id(np.uint64(segment_id), layer=l, x=x, y=y, z=z)


def get_layer_chunk_bounds(
    n_layers: int, atomic_chunk_bounds: np.ndarray = np.array([])
) -> dict:
    if atomic_chunk_bounds.size == 0:
        limit = 2 ** (n_layers - 2)
        atomic_chunk_bounds = np.array([limit, limit, limit])
    layer_bounds_d = {}
    for layer in range(2, n_layers):
        layer_bounds = atomic_chunk_bounds / (2 ** (layer - 2))
        layer_bounds_d[layer] = np.ceil(layer_bounds).astype(int)
    return layer_bounds_d
