"""
Functions for reading and writing edges 
to secondary storage with CloudVolume
"""

import os
from typing import List, Dict, Tuple, Union

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from cloudvolume.storage import SimpleStorage

from ..backend.utils import basetypes
from .protobuf.chunkEdges_pb2 import Edges, ChunkEdges


def _decompress_edges(content: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param content: zstd compressed bytes
    :type bytes:
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    def _get_edges(
        edge_type: str, edgesMessage: Edges
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if edge_type == "cross":
            edges = np.frombuffer(edgesMessage.crossChunk, dtype="<u8").reshape(-1, 2)
            affinities = float("inf") * np.ones(edges.shape[0], dtype="<f4")
            areas = np.zeros(edges.shape[0], dtype="<u8")
            return edges, affinities, areas

        edges = np.frombuffer(edgesMessage.edgeList, dtype="<u8").reshape(-1, 2)
        affinities = np.frombuffer(edgesMessage.affinities, dtype="<f4")
        areas = np.frombuffer(edgesMessage.areas, dtype="<u8")
        return edges, affinities, areas

    chunkEdgesMessage = ChunkEdges()

    zstdDecompressorObj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstdDecompressorObj.decompress(content)
    chunkEdgesMessage.ParseFromString(file_content)

    in_edges, in_affinities, in_areas = _get_edges("in", chunkEdgesMessage.inChunk)
    between_edges, between_affinities, between_areas = _get_edges(
        "between", chunkEdgesMessage.betweenChunk
    )
    cross_edges, cross_affinities, cross_areas = _get_edges("cross", chunkEdgesMessage)

    edges = np.concatenate([in_edges, between_edges, cross_edges])
    affinities = np.concatenate([in_affinities, between_affinities, cross_affinities])
    areas = np.concatenate([in_areas, between_areas, cross_areas])

    return edges, affinities, areas


def get_chunk_edges(
    edges_dir: str, chunks_coordinates: List[np.ndarray], cv_threads
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunks_coordinates:
    :type np.ndarray:
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # TODO change filename format
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    edges = np.array([], dtype=np.uint64).reshape(0, 2)
    affinities = np.array([], dtype=np.float32)
    areas = np.array([], dtype=np.uint64)

    if cv_threads > 1:
        st = Storage(edges_dir, n_threads=cv_threads)
    else:
        st = SimpleStorage(edges_dir)

    files = []
    with st:
        files = st.get_files(fnames)
        for _file in files:
            if not _file["content"]:
                continue
            _edges, _affinities, _areas = _decompress_edges(_file["content"])
            edges = np.concatenate([edges, _edges])
            affinities = np.concatenate([affinities, _affinities])
            areas = np.concatenate([areas, _areas])

    return edges, affinities, areas


def put_chunk_edges(
    edges_dir: str,
    chunk_coordinates: np.ndarray,
    chunk_edges_raw: dict,
    compression_level: int,
) -> None:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunk_coordinates: chunk coords x,y,z
    :type np.ndarray:
    :param chunk_edges_raw: chunk_edges_raw with keys "in", "cross", "between"
    :type dict:
    :param compression_level: zstandard compression level (1-22, higher - better ratio)
    :type int:
    :return None:
    """

    def _get_edges(edge_type: str) -> Edges:

        edges = Edges()
        edges.node_ids1[:] = chunk_edges_raw[edge_type]["sv1"]
        edges.node_ids2[:] = chunk_edges_raw[edge_type]["sv2"]

        n_edges = len(chunk_edges_raw[edge_type]["sv1"])

        if edge_type == "cross":
            edges.affinities[:] = float("inf") * np.ones(
                n_edges, basetypes.EDGE_AFFINITY
            )
            edges.areas[:] = np.ones(n_edges, basetypes.EDGE_AREA)
        else:
            edges.affinities[:] = chunk_edges_raw[edge_type]["aff"].astype(np.float32)
            edges.areas[:] = chunk_edges_raw[edge_type]["area"].astype(np.uint64)

        return edges

    chunkEdgesMessage = ChunkEdges()
    chunkEdgesMessage.in_chunk.CopyFrom(_get_edges("in"))
    chunkEdgesMessage.between_chunk.CopyFrom(_get_edges("between"))
    chunkEdgesMessage.cross_chunk.CopyFrom(_get_edges("cross"))

    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_proto = cctx.compress(chunkEdgesMessage.SerializeToString())

    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)
    # filename format - edges_x_y_z.serialization.compression

    file = f"edges_{chunk_str}.proto.zst"
    with Storage(edges_dir) as st:
        st.put_file(
            file_path=file,
            content=compressed_proto,
            compress=None,
            cache_control="no-cache",
        )
