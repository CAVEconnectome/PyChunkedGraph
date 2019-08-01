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
from .protobuf.chunkEdges_pb2 import Edge, Edges, ChunkEdges


def _decompress_edges(content: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param content: zstd compressed bytes
    :type bytes:
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    def _get_edges(
        edge_type: str, edges_message: Union[ChunkEdges, ChunkEdges.Edges]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if edge_type == "cross":
            edges = np.frombuffer(edges_message.crossChunk, dtype="<u8").reshape(-1, 2)
            affinities = float("inf") * np.ones(edges.shape[0], dtype="<f4")
            areas = np.ones(edges.shape[0], dtype="<u8")
            return edges, affinities, areas

        edges = np.frombuffer(edges_message.edgeList, dtype="<u8").reshape(-1, 2)
        affinities = np.frombuffer(edges_message.affinities, dtype="<f4")
        areas = np.frombuffer(edges_message.areas, dtype="<u8")
        return edges, affinities, areas

    chunk_edges_message = ChunkEdges()

    zstdDecompressorObj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstdDecompressorObj.decompress(content)
    chunk_edges_message.ParseFromString(file_content)

    in_edges, in_affinities, in_areas = _get_edges("in", chunk_edges_message.inChunk)
    between_edges, between_affinities, between_areas = _get_edges(
        "between", chunk_edges_message.betweenChunk
    )
    cross_edges, cross_affinities, cross_areas = _get_edges("cross", chunk_edges_message)

    edges = np.concatenate([in_edges, between_edges, cross_edges])
    affinities = np.concatenate([in_affinities, between_affinities, cross_affinities])
    areas = np.concatenate([in_areas, between_areas, cross_areas])

    return edges, affinities, areas


def get_chunk_edges(
    edges_dir: str, chunks_coordinates: List[np.ndarray], cv_threads: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunks_coordinates:
    :type np.ndarray:
    :return: edges, affinities, areas
    :param cv_threads: number of threads for cloudvolume Storage
    :type int:
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    edges = np.array([], dtype=np.uint64).reshape(0, 2)
    affinities = np.array([], dtype=np.float32)
    areas = np.array([], dtype=np.uint64)

    st = SimpleStorage(edges_dir)
    if cv_threads > 1:
        st = Storage(edges_dir, n_threads=cv_threads)

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
    chunk_edges: dict,
    compression_level: int,
) -> None:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunk_coordinates: chunk coords x,y,z
    :type np.ndarray:
    :param chunk_edges: chunk_edges with keys "in", "cross", "between"
    :type dict:
    :param compression_level: zstandard compression level (1-22, higher - better ratio)
    :type int:
    :return None:
    """

    def _get_edges(edge_type: str) -> Union[bytes, ChunkEdges.Edges]:
        # convert two numpy arrays to edge list
        # arr1 = [1, 2, 3]
        # arr2 = [4, 5, 6]
        # edges = [[1,4],[2,5],[3,6]]
        # this is faster than numpy.dstack
        edges = np.concatenate(
            [
                chunk_edges[edge_type]["sv1"][:, None],
                chunk_edges[edge_type]["sv2"][:, None],
            ],
            axis=1,
        )
        edges_bytes = edges.astype(np.uint64).tobytes()
        if edge_type == "cross":
            return edges_bytes

        edges_message = ChunkEdges.Edges()
        edges_message.edgeList = edges_bytes
        edges_message.affinities = (
            chunk_edges[edge_type]["aff"].astype(np.float32).tobytes()
        )
        edges_message.areas = chunk_edges[edge_type]["area"].astype(np.uint64).tobytes()

        return edges_message

    chunk_edges_message = ChunkEdges()
    chunk_edges_message.inChunk.CopyFrom(_get_edges("in"))
    chunk_edges_message.betweenChunk.CopyFrom(_get_edges("between"))
    chunk_edges_message.crossChunk = _get_edges("cross")

    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_proto = cctx.compress(chunk_edges_message.SerializeToString())

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
