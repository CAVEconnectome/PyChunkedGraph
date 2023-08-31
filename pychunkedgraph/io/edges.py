# pylint: disable=invalid-name, missing-docstring
"""
Functions for reading and writing edges from cloud storage.
"""

from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import zstandard as zstd
from cloudfiles import CloudFiles

from .protobuf.chunkEdges_pb2 import EdgesMsg
from .protobuf.chunkEdges_pb2 import ChunkEdgesMsg
from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.utils import basetypes
from ..graph.edges.utils import concatenate_chunk_edges


def serialize(edges: Edges) -> EdgesMsg:
    edges_proto = EdgesMsg()
    edges_proto.node_ids1 = edges.node_ids1.astype(basetypes.NODE_ID).tobytes()
    edges_proto.node_ids2 = edges.node_ids2.astype(basetypes.NODE_ID).tobytes()
    edges_proto.affinities = edges.affinities.astype(basetypes.EDGE_AFFINITY).tobytes()
    edges_proto.areas = edges.areas.astype(basetypes.EDGE_AREA).tobytes()
    return edges_proto


def deserialize(edges_message: EdgesMsg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sv_ids1 = np.frombuffer(edges_message.node_ids1, basetypes.NODE_ID)
    sv_ids2 = np.frombuffer(edges_message.node_ids2, basetypes.NODE_ID)
    affinities = np.frombuffer(edges_message.affinities, basetypes.EDGE_AFFINITY)
    areas = np.frombuffer(edges_message.areas, basetypes.EDGE_AREA)
    return Edges(sv_ids1, sv_ids2, affinities=affinities, areas=areas)


def _decompress_edges(content: bytes) -> Dict:
    zdc = zstd.ZstdDecompressor()
    chunk_edges = ChunkEdgesMsg()
    chunk_edges.ParseFromString(zdc.multi_decompress_to_buffer(content, threads=4))

    # in, between and cross
    edges_dict = {}
    edges_dict[EDGE_TYPES.in_chunk] = deserialize(chunk_edges.in_chunk)
    edges_dict[EDGE_TYPES.between_chunk] = deserialize(chunk_edges.between_chunk)
    edges_dict[EDGE_TYPES.cross_chunk] = deserialize(chunk_edges.cross_chunk)
    return edges_dict


def get_chunk_edges(edges_dir: str, chunks_coordinates: List[np.ndarray]) -> Dict:
    """Read edges from GCS."""
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    cf = CloudFiles(edges_dir, num_threads=4)
    files = cf.get(fnames, raw=True)

    edges = []
    for f in files:
        if not f["content"]:
            continue
        edges.append(_decompress_edges(f["content"]))
    return concatenate_chunk_edges(edges)


def put_chunk_edges(
    edges_dir: str, chunk_coordinates: np.ndarray, edges_d, compression_level: int
) -> None:
    """Write edges to GCS."""
    chunk_edges = ChunkEdgesMsg()
    chunk_edges.in_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.in_chunk]))
    chunk_edges.between_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.between_chunk]))
    chunk_edges.cross_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.cross_chunk]))

    cctx = zstd.ZstdCompressor(level=compression_level)
    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)

    # filename format - edges_x_y_z.serialization.compression
    filename = f"edges_{chunk_str}.proto.zst"
    cf = CloudFiles(edges_dir)
    cf.put(
        filename,
        content=cctx.compress(chunk_edges.SerializeToString()),
        compress=None,
        cache_control="no-cache",
    )
