"""
Functions for reading and writing edges from cloud storage.
"""

from typing import List, Dict, Tuple, Union

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from cloudvolume.storage import SimpleStorage

from .protobuf.chunkEdges_pb2 import EdgesMsg
from .protobuf.chunkEdges_pb2 import ChunkEdgesMsg
from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.utils import basetypes
from ..graph.utils.context_managers import TimeIt
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
    chunk_edges = ChunkEdgesMsg()
    zstd_decompressor_obj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstd_decompressor_obj.decompress(content)
    chunk_edges.ParseFromString(file_content)

    # in, between and cross
    edges_dict = {}
    edges_dict[EDGE_TYPES.in_chunk] = deserialize(chunk_edges.in_chunk)
    edges_dict[EDGE_TYPES.between_chunk] = deserialize(chunk_edges.between_chunk)
    edges_dict[EDGE_TYPES.cross_chunk] = deserialize(chunk_edges.cross_chunk)
    return edges_dict


def get_chunk_edges(
    edges_dir: str, chunks_coordinates: List[np.ndarray], cv_threads: int = 1
) -> Dict:
    """ Read edges from GCS. """
    from cloudfiles import CloudFiles

    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    with TimeIt("cloud files get"):
        cf = CloudFiles(edges_dir)
        cf.get(fnames, raw=True)
    return concatenate_chunk_edges([_decompress_edges(cf[name]) for name in fnames])


def put_chunk_edges(
    edges_dir: str, chunk_coordinates: np.ndarray, edges_d, compression_level: int
) -> None:
    """ Write edges to GCS. """
    chunk_edges = ChunkEdgesMsg()
    chunk_edges.in_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.in_chunk]))
    chunk_edges.between_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.between_chunk]))
    chunk_edges.cross_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.cross_chunk]))

    cctx = zstd.ZstdCompressor(level=compression_level)
    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)

    # filename format - edges_x_y_z.serialization.compression
    file = f"edges_{chunk_str}.proto.zst"
    with Storage(edges_dir) as storage:  # pylint: disable=not-context-manager
        storage.put_file(
            file_path=file,
            content=cctx.compress(chunk_edges.SerializeToString()),
            compress=None,
            cache_control="no-cache",
        )
