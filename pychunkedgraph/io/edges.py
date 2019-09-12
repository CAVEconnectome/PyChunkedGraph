"""
Functions for reading and writing edges 
to (slow) storage with CloudVolume
"""

from typing import List, Dict, Tuple, Union

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from cloudvolume.storage import SimpleStorage

from .protobuf.chunkEdges_pb2 import EdgesMsg, ChunkEdgesMsg
from ..backend.utils.edge_utils import concatenate_chunk_edges
from ..backend.definitions.edges import Edges, IN_CHUNK, BT_CHUNK, CX_CHUNK
from ..backend.utils import basetypes


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
    """
    :param content: zstd compressed bytes
    :type bytes:
    :return: edges_dict with keys "in", "cross", "between"
    :rtype: dict
    """

    chunk_edges = ChunkEdgesMsg()
    zstd_decompressor_obj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstd_decompressor_obj.decompress(content)
    chunk_edges.ParseFromString(file_content)

    # in, between and cross
    edges_dict = {}
    edges_dict[IN_CHUNK] = deserialize(chunk_edges.in_chunk)
    edges_dict[BT_CHUNK] = deserialize(chunk_edges.between_chunk)
    edges_dict[CX_CHUNK] = deserialize(chunk_edges.cross_chunk)
    return edges_dict


def get_chunk_edges(
    edges_dir: str, chunks_coordinates: List[np.ndarray], cv_threads: int = 1
) -> Dict:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunks_coordinates: list of chunk coords for which to load edges
    :type List[np.ndarray]:
    :param cv_threads: cloudvolume storage client thread count
    :type int:     
    :return: dictionary {"edge_type": Edges}
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    storage = (
        Storage(edges_dir, n_threads=cv_threads)
        if cv_threads > 1
        else SimpleStorage(edges_dir)
    )

    chunk_edge_dicts = []

    with storage:
        files = storage.get_files(fnames)
        for _file in files:
            # cv error
            if _file["error"]:
                raise ValueError(_file["error"])
            # empty chunk
            if not _file["content"]:
                continue
            edges_dict = _decompress_edges(_file["content"])
            chunk_edge_dicts.append(edges_dict)
    return concatenate_chunk_edges(chunk_edge_dicts)


def put_chunk_edges(
    edges_dir: str, chunk_coordinates: np.ndarray, edges_d, compression_level: int
) -> None:
    """
    :param edges_dir: cloudvolume storage path
    :type str:
    :param chunk_coordinates: chunk coords x,y,z
    :type np.ndarray:
    :param edges_d: edges_d with keys "in", "cross", "between"
    :type dict:
    :param compression_level: zstandard compression level (1-22, higher - better ratio)
    :type int:
    :return None:
    """

    chunk_edges = ChunkEdgesMsg()
    chunk_edges.in_chunk.CopyFrom(serialize(edges_d[IN_CHUNK]))
    chunk_edges.between_chunk.CopyFrom(serialize(edges_d[BT_CHUNK]))
    chunk_edges.cross_chunk.CopyFrom(serialize(edges_d[CX_CHUNK]))

    cctx = zstd.ZstdCompressor(level=compression_level)
    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)

    # filename format - edges_x_y_z.serialization.compression
    file = f"edges_{chunk_str}.proto.zst"
    with Storage(edges_dir) as storage:
        storage.put_file(
            file_path=file,
            content=cctx.compress(chunk_edges.SerializeToString()),
            compress=None,
            cache_control="no-cache",
        )
